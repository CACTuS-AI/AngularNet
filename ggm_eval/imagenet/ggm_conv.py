"""GGM convolution, forward pass only."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..projections import make_G_conv2d


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (int(v), int(v))


class Conv2dGGM(nn.Module):
    """GGM convolution (inference). G is rebuilt from G_seed whenever weights are loaded."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, N_factor: float = 1.0,
                 stride=1, padding=0, groups: int = 1, bias: bool = False):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.in_channels, self.out_channels, self.groups = int(in_channels), int(out_channels), int(groups)
        self.kh, self.kw = _pair(kernel_size)
        self.stride, self.padding = _pair(stride), _pair(padding)
        self.cin_g = self.in_channels // self.groups
        self.cout_g = self.out_channels // self.groups
        self.K = self.cin_g * self.kh * self.kw
        self.N_factor = float(N_factor)
        self.N = int(self.K * self.N_factor)

        self.weight = nn.Parameter(torch.empty(self.out_channels, self.cin_g, self.kh, self.kw))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.register_buffer("G_seed", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("G", torch.empty(0), persistent=False)
        self.register_load_state_dict_post_hook(lambda module, _: module.regenerate_G())

    @torch.no_grad()
    def regenerate_G(self):
        self.G = make_G_conv2d(int(self.G_seed.item()), self.groups, self.N, self.cin_g, self.kh, self.kw
                               ).to(self.weight.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, N, K = self.groups, self.N, self.K
        x = x - x.mean(dim=(1, 2, 3), keepdim=True)

        G_conv = self.G.to(device=x.device, dtype=x.dtype).reshape(g * N, self.cin_g, self.kh, self.kw)
        z = F.conv2d(x, G_conv, bias=None, stride=self.stride, padding=self.padding, groups=g)
        z_b = torch.where(z >= 0, torch.ones((), device=z.device, dtype=z.dtype),
                          -torch.ones((), device=z.device, dtype=z.dtype))

        W_flat = self.weight.reshape(g, self.cout_g, K)
        Gw = self.G.to(device=x.device, dtype=W_flat.dtype).reshape(g, N, K)
        W_proj = torch.bmm(Gw, W_flat.transpose(1, 2))                          # [g, N, cout_g]
        W_b = torch.where(W_proj >= 0, torch.ones((), device=W_proj.device, dtype=W_proj.dtype),
                          -torch.ones((), device=W_proj.device, dtype=W_proj.dtype))
        W_b = W_b.transpose(1, 2).reshape(self.out_channels, N, 1, 1)

        y = F.conv2d(z_b, W_b.to(dtype=z_b.dtype), bias=None, stride=1, padding=0, groups=g) / float(N)
        if self.bias is not None:
            y = y + self.bias.reshape(1, -1, 1, 1)
        return y

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size=({self.kh}, {self.kw}), "
                f"stride={self.stride}, padding={self.padding}, groups={self.groups}, "
                f"N={self.N} (N_factor={self.N_factor:g})")
