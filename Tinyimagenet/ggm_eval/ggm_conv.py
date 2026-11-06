"""GGM convolution, forward pass only."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (int(v), int(v))


def _autocast_state(device_type: str) -> tuple:
    try:
        return (torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type))
    except (TypeError, AttributeError):  # torch < 2.4
        if device_type == "cuda":
            return (torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype())
        return (torch.is_autocast_cpu_enabled(), torch.get_autocast_cpu_dtype())


def quantize(z: torch.Tensor, k_bits: int, s0: float = 1.0) -> torch.Tensor:
    """1 bit: sign (+1 at zero). k bits: s0 * round(clamp(z/s0, -1, 1) * Q) / Q, Q = 2^(k-1) - 1."""
    if k_bits == 1:
        return torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    q = (2 ** (k_bits - 1)) - 1
    s0_t = torch.as_tensor(s0, device=z.device, dtype=z.dtype)
    return torch.round((z / s0_t).clamp(-1.0, 1.0) * q) / q * s0_t


def quantize_and_scale(z: torch.Tensor, k_bits: int, s0: float, dim: int,
                       log_scale_mult: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Quantize, then scale by mean(|z|) over `dim` times exp(log_scale_mult)."""
    q0 = quantize(z, k_bits, s0)
    scale = z.abs().mean(dim=dim, keepdim=True).clamp_min(eps)
    scale = scale * torch.exp(log_scale_mult).to(device=z.device, dtype=z.dtype)
    return q0 * scale


class GGMConv2d(nn.Module):
    """GGM convolution (inference)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, *, N_factor: float,
                 k_bits: int, stride=1, padding=0, dilation=1, groups: int = 1, bias: bool = False,
                 s0: float = 1.0):
        super().__init__()
        if k_bits < 1 or k_bits >= 16:
            raise ValueError(f"k_bits must be in [1, 15], got {k_bits}")
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")

        self.in_channels, self.out_channels, self.groups = int(in_channels), int(out_channels), int(groups)
        self.kh, self.kw = _pair(kernel_size)
        self.stride, self.padding, self.dilation = _pair(stride), _pair(padding), _pair(dilation)
        self.in_per_group = self.in_channels // self.groups
        self.out_per_group = self.out_channels // self.groups
        self.d_group = self.in_per_group * self.kh * self.kw
        self.N_factor = float(N_factor)
        self.N = int(self.N_factor * self.d_group)
        self.k_bits = int(k_bits)
        self.s0 = float(s0)

        self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_per_group, self.kh, self.kw))
        self.gain = nn.Parameter(torch.ones(self.out_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.x_log_scale_mult = nn.Parameter(torch.tensor(0.0))
        self.w_log_scale_mult = nn.Parameter(torch.zeros(self.groups, 1, self.out_per_group))
        self.register_buffer("G", torch.empty(self.groups, self.N, self.in_per_group, self.kh, self.kw))
        self.register_buffer("rho_eps_log", torch.tensor(-27.631021))  # log(1e-12)
        self._w_cache = None

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, kernel_size=({self.kh}, {self.kw}), "
                f"stride={self.stride}, padding={self.padding}, groups={self.groups}, "
                f"N={self.N} (N_factor={self.N_factor:g}), k_bits={self.k_bits}")

    def _apply(self, fn, *args, **kwargs):
        self._w_cache = None  # .to()/.cuda()/.half() invalidate the cached weight side
        return super()._apply(fn, *args, **kwargs)

    def train(self, mode: bool = True):
        self._w_cache = None
        return super().train(mode)

    def _weight_side(self, groups: int, N: int, d: int) -> torch.Tensor:
        Wg = self.weight.reshape(groups, self.out_per_group, self.in_per_group, self.kh, self.kw)
        W_sq = Wg.pow(2).sum(dim=(2, 3, 4), keepdim=True)
        W_hat = Wg / torch.sqrt(W_sq + torch.exp(self.rho_eps_log) + 1e-12)
        W_p = torch.einsum("gnd,god->gno", self.G.reshape(groups, N, d),
                           W_hat.reshape(groups, self.out_per_group, d))
        return quantize_and_scale(W_p, self.k_bits, self.s0, dim=1,
                                  log_scale_mult=self.w_log_scale_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        groups, N, d = self.groups, self.G.shape[1], self.d_group
        B = x.shape[0]
        conv = dict(bias=None, stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=groups)

        # input patches: centred, normalised, projected onto G, quantized
        ones = x.new_ones((groups, self.in_per_group, self.kh, self.kw))
        patch_mean = F.conv2d(x, ones, **conv) / float(d)                      # [B, g, H, W]
        x_proj = F.conv2d(x, self.G.reshape(groups * N, self.in_per_group, self.kh, self.kw), **conv)
        Hout, Wout = x_proj.shape[-2:]
        x_proj = x_proj.reshape(B, groups, N, Hout, Wout)
        G_sum = self.G.reshape(groups, N, -1).sum(dim=-1).reshape(1, groups, N, 1, 1)
        x_proj = x_proj - patch_mean.reshape(B, groups, 1, Hout, Wout) * G_sum

        patch_sq = F.conv2d(x.pow(2), ones, **conv)
        patch_sq = (patch_sq - float(d) * patch_mean.pow(2)).clamp_min(0.0)
        x_norm = torch.sqrt(patch_sq + 1e-12)
        x_p = x_proj / x_norm.reshape(B, groups, 1, Hout, Wout)
        x_q = quantize_and_scale(x_p, self.k_bits, self.s0, dim=2,
                                 log_scale_mult=self.x_log_scale_mult)          # [B, g, N, H, W]

        # filters: normalised, projected onto G, quantized (input independent, cached)
        if self.training:
            W_q = self._weight_side(groups, N, d)
        else:
            key = (x.device, x.dtype) + _autocast_state(x.device.type)
            if self._w_cache is None or self._w_cache[0] != key:
                self._w_cache = (key, self._weight_side(groups, N, d))
            W_q = self._w_cache[1]

        # average over the N projection directions
        out = torch.einsum("bgnhw,gno->bgohw", x_q, W_q) / float(N)
        out = out.reshape(B, self.out_channels, Hout, Wout)
        out = out * self.gain.reshape(1, -1, 1, 1)
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        return out
