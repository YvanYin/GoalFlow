import math
import torch
import torch.nn as nn
from navsim.agents.goalflow.multihead_custom_attention import MultiheadCustomAttention


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()
        self.feature_dim = feature_dim
        self.pe_type = pe_type
    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x
    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=x_position.device)
            * (-math.log(10000.0) / (self.feature_dim)))
        div_term = div_term.view(1, 1, -1) # [1, 1, d]
        sinx = torch.sin(x_position[...,None] * div_term)  # [B, N, d]
        cosx = torch.cos(x_position[...,None] * div_term)
        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)
        if position_code.requires_grad:
            position_code = position_code.detach()
        return position_code
    
class ParallelAttentionLayer(nn.Module):
    """Self-/Cross-attention between two sequences."""

    def __init__(self, d_model=256, dropout=0.1, n_heads=8, pre_norm=False,
                 self_attention1=True, self_attention2=True,
                 cross_attention1=True, cross_attention2=True,
                 apply_ffn=True,
                 slot_attention12=False, slot_attention21=False,
                 rotary_pe=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attention1 = self_attention1
        self.self_attention2 = self_attention2
        self.cross_attention1 = cross_attention1
        self.cross_attention2 = cross_attention2
        self.apply_ffn = apply_ffn
        self.rotary_pe = rotary_pe

        # Self-attention for seq1
        if self.self_attention1:
            self.sa1 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout
            )
            self.dropout_1 = nn.Dropout(dropout)
            self.norm_1 = nn.LayerNorm(d_model)

        # Self-attention for seq2
        if self.self_attention2:
            self.sa2 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout
            )
            self.dropout_2 = nn.Dropout(dropout)
            self.norm_2 = nn.LayerNorm(d_model)

        # Cross attention from seq1 to seq2
        self.norm_12 = None
        if cross_attention1:
            self.cross_12 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout,
                slot_competition=slot_attention12
            )
            self.dropout_12 = nn.Dropout(dropout)
            self.norm_12 = nn.LayerNorm(d_model)

        # Cross attention from seq2 to seq1
        self.norm_21 = None
        if cross_attention2:
            self.cross_21 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout,
                slot_competition=slot_attention21
            )
            self.dropout_21 = nn.Dropout(dropout)
            self.norm_21 = nn.LayerNorm(d_model)

        # FFN-1
        if self_attention1 or cross_attention1:
            self.ffn_12 = nn.Sequential(
                nn.Linear(d_model, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, d_model),
                nn.Dropout(dropout)
            )
            self.norm_122 = nn.LayerNorm(d_model)

        # FFN-2
        if self_attention2 or cross_attention2:
            self.ffn_21 = nn.Sequential(
                nn.Linear(d_model, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, d_model),
                nn.Dropout(dropout)
            )
            self.norm_212 = nn.LayerNorm(d_model)

    def _norm(self, x, layer, normalize=True):
        if normalize and layer is not None:
            return layer(x)
        return x

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, seq1, seq1_key_padding_mask, seq2,
                seq2_key_padding_mask,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None,
                attn_mask_11=None, attn_mask_22=None,
                attn_mask_12=None, attn_mask_21=None):
        """Forward pass, seq1 (B, S1, F), seq2 (B, S2, F)."""
        rot_args = {}

        # Create key, query, value for seq1, seq2
        q1 = k1 = v1 = self._norm(seq1, self.norm_12, self.pre_norm)
        q2 = k2 = v2 = self._norm(seq2, self.norm_21, self.pre_norm)
        if not self.rotary_pe:
            q1 = k1 = self.with_pos_embed(seq1, seq1_pos)
            q2 = k2 = self.with_pos_embed(seq2, seq2_pos)
        q1 = self.with_pos_embed(q1, seq1_sem_pos)
        k1 = self.with_pos_embed(k1, seq1_sem_pos)
        q2 = self.with_pos_embed(q2, seq2_sem_pos)
        k2 = self.with_pos_embed(k2, seq2_sem_pos)

        # Cross-attention from seq1 to seq2
        if self.cross_attention1:
            if self.rotary_pe:
                rot_args['rotary_pe'] = (seq1_pos, seq2_pos)
            seq1b = self.cross_12(
                query=q1.transpose(0, 1),
                key=k2.transpose(0, 1),
                value=v2.transpose(0, 1),
                attn_mask=attn_mask_12,
                key_padding_mask=seq2_key_padding_mask,  # (B, S2)
                **rot_args,
            )[0].transpose(0, 1)
            seq1 = seq1 + self.dropout_12(seq1b)
            seq1 = self._norm(seq1, self.norm_12, not self.pre_norm)

        # Cross-attention from seq2 to seq1
        if self.cross_attention2:
            if self.rotary_pe:
                rot_args['rotary_pe'] = (seq2_pos, seq1_pos)
            seq2b = self.cross_21(
                query=q2.transpose(0, 1),
                key=k1.transpose(0, 1),
                value=v1.transpose(0, 1),
                attn_mask=attn_mask_21,
                key_padding_mask=seq1_key_padding_mask,  # (B, S1)
                **rot_args,
            )[0].transpose(0, 1)
            seq2 = seq2 + self.dropout_21(seq2b)
            seq2 = self._norm(seq2, self.norm_21, not self.pre_norm)

        # Self-attention for seq1
        if self.self_attention1:
            q1 = k1 = v1 = self._norm(seq1, self.norm_1, self.pre_norm)
            if self.rotary_pe:
                rot_args['rotary_pe'] = (seq1_pos, seq1_pos)
            else:
                q1 = k1 = self.with_pos_embed(seq1, seq1_pos)
            q1 = self.with_pos_embed(q1, seq1_sem_pos)
            k1 = self.with_pos_embed(k1, seq1_sem_pos)
            seq1b = self.sa1(
                query=q1.transpose(0, 1),
                key=k1.transpose(0, 1),
                value=v1.transpose(0, 1),
                attn_mask=attn_mask_11,
                key_padding_mask=seq1_key_padding_mask,  # (B, S1)
                **rot_args,
            )[0].transpose(0, 1)
            seq1 = seq1 + self.dropout_1(seq1b)
            seq1 = self._norm(seq1, self.norm_1, not self.pre_norm)

        # Self-attention for seq2
        if self.self_attention2:
            q2 = k2 = v2 = self._norm(seq2, self.norm_2, self.pre_norm)
            if self.rotary_pe:
                rot_args['rotary_pe'] = (seq2_pos, seq2_pos)
            else:
                q2 = k2 = self.with_pos_embed(seq2, seq2_pos)
            q2 = self.with_pos_embed(q2, seq2_sem_pos)
            k2 = self.with_pos_embed(k2, seq2_sem_pos)
            seq2b = self.sa2(
                query=q2.transpose(0, 1),
                key=k2.transpose(0, 1),
                value=v2.transpose(0, 1),
                attn_mask=attn_mask_22,
                key_padding_mask=seq2_key_padding_mask,  # (B, S2)
                **rot_args,
            )[0].transpose(0, 1)
            seq2 = seq2 + self.dropout_2(seq2b)
            seq2 = self._norm(seq2, self.norm_2, not self.pre_norm)

        # FFN-1
        if (self.self_attention1 or self.cross_attention1) and self.apply_ffn:
            seq1 = self._norm(seq1, self.norm_122, self.pre_norm)
            seq1 = seq1 + self.ffn_12(seq1)
            seq1 = self._norm(seq1, self.norm_122, not self.pre_norm)

        # FFN-2
        if (self.self_attention2 or self.cross_attention2) and self.apply_ffn:
            seq2 = self._norm(seq2, self.norm_212, self.pre_norm)
            seq2 = seq2 + self.ffn_21(seq2)
            seq2 = self._norm(seq2, self.norm_212, not self.pre_norm)

        return seq1, seq2