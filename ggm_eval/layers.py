"""GGM layers with packed (1-bit or ternary) weights."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projections import make_G_conv2d


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (int(v), int(v))


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


def unpack_weights(packed: torch.Tensor, count: int, k_bits: int = 1) -> torch.Tensor:
    """uint8 [..., bytes] -> float32 [..., count]. 1 bit: 8 values per byte, 0 -> -1, 1 -> +1.
    2 bits (ternary): 4 values per byte, 0 -> -1, 1 -> 0, 2 -> +1. Little-endian within a byte."""
    if k_bits not in (1, 2):
        raise ValueError(f"packed weights have 1 or 2 bits, got {k_bits}")
    shifts = torch.arange(0, 8, k_bits, dtype=torch.uint8, device=packed.device)
    codes = (packed.unsqueeze(-1) >> shifts) & ((1 << k_bits) - 1)
    codes = codes.reshape(*packed.shape[:-1], -1)[..., :count].to(torch.float32)
    return codes.mul_(2).sub_(1) if k_bits == 1 else codes.sub_(1)


def packed_bytes(count: int, k_bits: int = 1) -> int:
    per_byte = 8 // k_bits
    return (count + per_byte - 1) // per_byte


class _PackedWeights(nn.Module):
    """Unpacks the stored weight side once; the float copy is dropped when the module is moved."""

    _w_cache = None

    def _apply(self, fn, *args, **kwargs):
        self._w_cache = None
        return super()._apply(fn, *args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._w_cache = None
        return super()._load_from_state_dict(*args, **kwargs)

    def unpacked(self) -> torch.Tensor:
        if self._w_cache is None:
            self._w_cache = self._unpack()
        return self._w_cache


class ScaledConv2dGGM(_PackedWeights):
    """GGM convolution with normalised inputs, learned scales and 1- or 1.58-bit quantization
    (TinyImageNet models). G is set by the checkpoint loader."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, *, N_factor: float,
                 k_bits: int, stride=1, padding=0, dilation=1, groups: int = 1, bias: bool = False,
                 s0: float = 1.0):
        super().__init__()
        if k_bits not in (1, 2):
            raise ValueError(f"k_bits must be 1 or 2 (1.58-bit), got {k_bits}")
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

        self.gain = nn.Parameter(torch.ones(self.out_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.x_log_scale_mult = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("G", torch.empty(self.groups, self.N, self.in_per_group, self.kh, self.kw))
        # weight side: quantized G W (1 bit: ±1, 2 bits: -1/0/+1) and one scale per output channel
        self.register_buffer("w_packed", torch.zeros(self.groups, self.out_per_group,
                                                     packed_bytes(self.N, self.k_bits), dtype=torch.uint8))
        self.register_buffer("w_scale", torch.ones(self.groups, self.out_per_group))

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, kernel_size=({self.kh}, {self.kw}), "
                f"stride={self.stride}, padding={self.padding}, groups={self.groups}, "
                f"N={self.N} (N_factor={self.N_factor:g}), k_bits={self.k_bits}")

    def _unpack(self) -> torch.Tensor:  # [groups, N, out_per_group]
        w = unpack_weights(self.w_packed, self.N, self.k_bits) * self.w_scale.unsqueeze(-1)
        return w.transpose(1, 2).contiguous()

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

        # average over the N projection directions
        out = torch.einsum("bgnhw,gno->bgohw", x_q, self.unpacked()) / float(N)
        out = out.reshape(B, self.out_channels, Hout, Wout)
        out = out * self.gain.reshape(1, -1, 1, 1)
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        return out


class Conv2dGGM(_PackedWeights):
    """Sign GGM convolution (ImageNet and CIFAR-10 models).
    G is rebuilt from G_seed whenever weights are loaded."""

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

        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.register_buffer("w_packed", torch.zeros(self.out_channels, packed_bytes(self.N), dtype=torch.uint8))
        self.register_buffer("G_seed", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("G", torch.empty(0), persistent=False)
        self.register_load_state_dict_post_hook(lambda module, _: module.regenerate_G())

    @torch.no_grad()
    def regenerate_G(self):
        self.G = make_G_conv2d(int(self.G_seed.item()), self.groups, self.N, self.cin_g, self.kh, self.kw
                               ).to(self.w_packed.device)

    def _unpack(self) -> torch.Tensor:  # sign(G W): [out_channels, N, 1, 1]
        return unpack_weights(self.w_packed, self.N).reshape(self.out_channels, self.N, 1, 1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, N = self.groups, self.N
        x = x - x.mean(dim=(1, 2, 3), keepdim=True)

        G_conv = self.G.to(device=x.device, dtype=x.dtype).reshape(g * N, self.cin_g, self.kh, self.kw)
        z = F.conv2d(x, G_conv, bias=None, stride=self.stride, padding=self.padding, groups=g)
        z_b = torch.where(z >= 0, torch.ones((), device=z.device, dtype=z.dtype),
                          -torch.ones((), device=z.device, dtype=z.dtype))

        y = F.conv2d(z_b, self.unpacked().to(dtype=z_b.dtype), bias=None, stride=1, padding=0, groups=g) / float(N)
        if self.bias is not None:
            y = y + self.bias.reshape(1, -1, 1, 1)
        return y

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size=({self.kh}, {self.kw}), "
                f"stride={self.stride}, padding={self.padding}, groups={self.groups}, "
                f"N={self.N} (N_factor={self.N_factor:g})")


class LinearGGM(_PackedWeights):
    """Sign GGM linear layer (Medformer). G is set by the checkpoint loader."""

    def __init__(self, in_features: int, out_features: int, ratio: float = 1.0):
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.N = int(float(ratio) * self.in_features)
        self.register_buffer("w_packed", torch.zeros(self.out_features, packed_bytes(self.N), dtype=torch.uint8))
        self.register_buffer("G", torch.empty(self.N, self.in_features))

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, N={self.N}"

    def _unpack(self) -> torch.Tensor:  # sign(G W^T): [N, out_features]
        return unpack_weights(self.w_packed, self.N).t().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_b = (x.float() @ self.G.transpose(-1, -2)).sign()
        return (x_b @ self.unpacked()) / self.N


class Conv1dGGM(_PackedWeights):
    """Sign GGM 1-D convolution (Medformer). G is set by the checkpoint loader."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 N_factor: float = 1.0, groups: int = 1):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.kernel_size, self.groups = int(kernel_size), int(groups)
        self.cin_g = self.in_channels // self.groups
        self.cout_g = self.out_channels // self.groups
        self.K = self.cin_g * self.kernel_size
        self.N = int(self.K * float(N_factor))
        self.register_buffer("w_packed", torch.zeros(self.out_channels, packed_bytes(self.N), dtype=torch.uint8))
        self.register_buffer("G", torch.empty(self.groups, self.N, self.cin_g, self.kernel_size))

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"groups={self.groups}, N={self.N}")

    def _unpack(self) -> torch.Tensor:  # sign(G W): [out_channels, N, 1]
        return unpack_weights(self.w_packed, self.N).reshape(self.out_channels, self.N, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, N = self.groups, self.N
        x = x - x.mean(dim=(1, 2), keepdim=True)
        G = self.G.to(device=x.device, dtype=x.dtype)
        z = F.conv1d(x, G.reshape(g * N, self.cin_g, self.kernel_size), groups=g)
        z_b = torch.where(z >= 0, 1.0, -1.0)
        return F.conv1d(z_b, self.unpacked().to(dtype=z_b.dtype), groups=g) / float(N)


class PackedLinearGGM(_PackedWeights):
    """GGM linear layer with centred, normalised inputs and learned scales (BERT models).
    G is set by the checkpoint loader."""

    def __init__(self, in_features: int, out_features: int, N_factor: float = 1.0, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.N = int(float(N_factor) * self.in_features)
        self.register_buffer("G", torch.empty(self.N, self.in_features))
        self.register_buffer("w_packed", torch.zeros(self.out_features, packed_bytes(self.N), dtype=torch.uint8))
        self.register_buffer("w_scale", torch.ones(self.out_features))
        self.register_buffer("gain", torch.ones(self.out_features))
        self.register_buffer("x_log_scale_mult", torch.zeros(()))
        self.register_buffer("bias", torch.zeros(self.out_features) if bias else None)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, N={self.N}"

    def _unpack(self) -> torch.Tensor:  # [N, out]: ±1 weights times their per-output scale
        return unpack_weights(self.w_packed, self.N).t() * self.w_scale.reshape(1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1]).float()
        x = x - x.mean(dim=-1, keepdim=True)
        x_hat = x / torch.sqrt(x.square().sum(dim=-1, keepdim=True) + 1e-12)
        x_p = x_hat @ self.G.t()
        x_scale = x_p.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6) * self.x_log_scale_mult.exp()
        x_q = torch.where(x_p >= 0, 1.0, -1.0) * x_scale
        out = (x_q @ self.unpacked()) / float(self.N) * self.gain
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.out_features)


class BinaryNorm(nn.Module):
    """Mean-centring followed by a per-feature affine map."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return self.weight * (x - x.mean(dim=-1, keepdim=True)) + self.bias
