import math

import torch
import torch.nn as nn
import numpy as np


class VolumetricPositionEncoding(nn.Module):
    # Relative positional encoding (ROPE inspired rotary pe)

    def __init__(self, feature_dim):
        super().__init__()
        self.feat_dim = feature_dim
        self.vol_bnds = [[0., 0., 0.], [1., 1., 1.]]
        self.voxel_size = 0.08
        self.vol_origin = self.vol_bnds[0]
        self.pe_type = 'rotary'

    def voxelize(self, xyz):
        '''
        @param xyz: B,N,3
        @return: B,N,3
        '''
        if type(self.vol_origin) == list:
            self.vol_origin = torch.FloatTensor(
                self.vol_origin
            ).view(1, 1, -1).to(xyz.device)
        return (xyz - self.vol_origin) / self.voxel_size

    @staticmethod
    def embed_rotary(x, cos, sin):
        '''
        @param x: [B,N,d]
        @param cos: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @param sin: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @return:
        '''
        x2 = torch.stack(
            [-x[..., 1::2], x[..., ::2]], dim=-1
        ).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    @staticmethod
    def embed_pos(pe_type, x, pe):
        """Combine feature and position code."""
        if pe_type == 'rotary':
            return VolumetricPositionEncoding.embed_rotary(
                x, pe[..., 0], pe[..., 1]
            )
        elif pe_type == 'sinusoidal':
            return x + pe
        else:
            raise KeyError()

    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return: [B, N, F, 2]
        '''
        original_shape = XYZ.shape
        bsize = original_shape[0]
        XYZ = XYZ.reshape(bsize, -1, original_shape[-1])
        npoint = XYZ.shape[1]

        # bsize, npoint, _ = XYZ.shape
        vox = XYZ
        # vox = self.voxelize(XYZ) # no need voxelize for partnet for now
        x_position, y_position, z_position = (
            vox[..., :1], vox[..., 1:2], vox[..., 2:3]
        )
        div_term = torch.exp(
            torch.arange(0, self.feat_dim // 3, 2, device=XYZ.device).float()
            * (-math.log(10000.0) / (self.feat_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        if self.pe_type == 'sinusoidal':
            position_code = torch.cat([sinx, cosx, siny, cosy, sinz, cosz], -1)

        elif self.pe_type == "rotary":
            # sin/cos [θ0,θ1,θ2......θd/6-1]
            # -> sin/cos [θ0,θ0,θ1,θ1,θ2,θ2......θd/6-1,θd/6-1]
            sinx, cosx, siny, cosy, sinz, cosz = map(
                lambda feat: torch.stack([feat, feat], dim=-1).view(
                    bsize, npoint, -1
                ),
                [sinx, cosx, siny, cosy, sinz, cosz]
            )
            sin_pos = torch.cat([sinx, siny, sinz], dim=-1)
            cos_pos = torch.cat([cosx, cosy, cosz], dim=-1)
            position_code = torch.stack([cos_pos, sin_pos], dim=-1)

        else:
            raise KeyError()

        if position_code.requires_grad:
            position_code = position_code.detach()
        position_code = position_code.reshape(*original_shape[:-1], self.feat_dim, 2)

        return position_code
    

class VolumetricPositionEncoding2D(nn.Module):
    # Relative positional encoding (ROPE inspired rotary pe)

    def __init__(self, feature_dim):
        super().__init__()
        self.feat_dim = feature_dim
        self.pe_type = 'rotary'

    @staticmethod
    def embed_rotary(x, cos, sin):
        '''
        @param x: [B,N,d]
        @param cos: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @param sin: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @return:
        '''
        x2 = torch.stack(
            [-x[..., 1::2], x[..., ::2]], dim=-1
        ).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    @staticmethod
    def embed_pos(pe_type, x, pe):
        """Combine feature and position code."""
        if pe_type == 'rotary':
            return VolumetricPositionEncoding.embed_rotary(
                x, pe[..., 0], pe[..., 1]
            )
        elif pe_type == 'sinusoidal':
            return x + pe
        else:
            raise KeyError()

    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return: [B, N, F, 2]
        '''
        original_shape = XYZ.shape
        bsize = original_shape[0]
        XYZ = XYZ.reshape(bsize, -1, original_shape[-1])
        npoint = XYZ.shape[1]

        # bsize, npoint, _ = XYZ.shape
        vox = XYZ
        # vox = self.voxelize(XYZ) # no need voxelize for partnet for now
        x_position, y_position = (
            vox[..., :1], vox[..., 1:2]
        )
        div_term = torch.exp(
            torch.arange(0, self.feat_dim // 2, 2, device=XYZ.device).float()
            * (-math.log(10000.0) / (self.feat_dim // 2))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)

        if self.pe_type == 'sinusoidal':
            position_code = torch.cat([sinx, cosx, siny, cosy], -1)

        elif self.pe_type == "rotary":
            # sin/cos [θ0,θ1,θ2......θd/6-1]
            # -> sin/cos [θ0,θ0,θ1,θ1,θ2,θ2......θd/6-1,θd/6-1]
            sinx, cosx, siny, cosy = map(
                lambda feat: torch.stack([feat, feat], dim=-1).view(
                    bsize, npoint, -1
                ),
                [sinx, cosx, siny, cosy]
            )
            sin_pos = torch.cat([sinx, siny], dim=-1)
            cos_pos = torch.cat([cosx, cosy], dim=-1)
            position_code = torch.stack([cos_pos, sin_pos], dim=-1)

        else:
            raise KeyError()

        if position_code.requires_grad:
            position_code = position_code.detach()
        position_code = position_code.reshape(*original_shape[:-1], self.feat_dim, 2)

        return position_code


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288, fourier_xyz=False):
        super().__init__()
        self.fourier_xyz = fourier_xyz
        if self.fourier_xyz:
            self.pos_embed_fourier = PositionEmbeddingCoordsSine(
                d_pos=num_pos_feats, pos_type="fourier",
                normalize=False, d_in=input_channel
            )
            input_channel = num_pos_feats

        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        if self.fourier_xyz:
            xyz = self.pos_embed_fourier(xyz).transpose(1, 2)
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding.transpose(1, 2).contiguous()


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        # clone coords so that shift/scale operations
        # do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        # ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert ndim % 2 == 0  # obviously, we fixed that 2 lines above

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(
                    cdim,
                    dtype=torch.float32, device=xyz.device
                )
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale does not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        # ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    @torch.no_grad()
    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            return self.get_sine_embeddings(xyz, num_channels, input_range)
        if self.pos_type == "fourier":
            return self.get_fourier_embeddings(xyz, num_channels, input_range)
        raise ValueError(f"Unknown {self.pos_type}")


def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        _device = src_range[0].device
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=_device),
            torch.ones((src_range[0].shape[0], 3), device=_device)
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz


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
