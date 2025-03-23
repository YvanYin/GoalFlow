import torch
import math
import numpy as np
from typing import Sequence

def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    """
    Convert 2D position into positional embeddings.

    Args:
        pos (torch.Tensor): Input 2D position tensor.
        num_pos_feats (int, optional): Number of positional features. Default is 128.
        temperature (int, optional): Temperature factor for positional embeddings. Default is 10000.

    Returns:
        torch.Tensor: Positional embeddings tensor.
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.floor(dim_t / 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb

def get_vcs2bev_img_mat(vcs_range: Sequence, bev_size: Sequence):
    """Return a 3x3 homo matrix from vcs coord system to bev img coord system.

    Args:
        vcs_range: vcs range.(order is (bottom,right,top,left))
        bev_size: bev size.(order is (h,w))
    Notes:
        vcs coord defined as:
                  x
                  ^
                  |
                  |
         y <------. z

    """

    scope_H = vcs_range[2] - vcs_range[0]
    scope_W = vcs_range[3] - vcs_range[1]
    bev_height, bev_width = bev_size

    vcs2bev_img = [
        [0, -bev_width / scope_W, vcs_range[3] / scope_W * bev_width],
        [-bev_height / scope_H, 0, vcs_range[2] / scope_H * bev_height],
        [0, 0, 1],
    ]
    vcs2bev_img = np.array(vcs2bev_img, dtype="float16").reshape((3, 3))

    return vcs2bev_img