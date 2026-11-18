"""Projection matrices G, rebuilt from seeds."""
from __future__ import annotations

import contextlib
import math

import torch

G_THREADS = 6  # CPU threads used when building G


def _portable_randn(n: int, generator: torch.Generator) -> torch.Tensor:
    """torch.randn(n) on the CPU, rebuilt from torch.rand with each Box-Muller step in float64."""
    f32 = torch.float32

    def fill(u):  # u [k, 16] uniforms -> 16 normals per row
        u1, u2 = (1.0 - u[:, :8]).to(f32), u[:, 8:]
        r = torch.sqrt((-2.0 * torch.log(u1.double()).to(f32)).double()).to(f32).double()
        th = (2.0 * math.pi * u2.double()).to(f32).double()
        c, s = torch.cos(th).to(f32).double(), torch.sin(th).to(f32).double()
        return torch.cat([(r * c).to(f32), (r * s).to(f32)], dim=1)

    u = torch.rand(n, generator=generator, dtype=f32)
    out = torch.empty(n, dtype=f32)
    full = (n // 16) * 16
    if full:
        out[:full] = fill(u[:full].reshape(-1, 16)).reshape(-1)
    if n % 16:  # PyTorch redraws the last 16 values
        out[n - 16:] = fill(torch.rand(16, generator=generator, dtype=f32).reshape(1, 16)).reshape(-1)
    return out


def make_G(seed: int, N: int, d: int) -> torch.Tensor:
    """Seeded Gaussian -> float64 QR -> sign fix -> scale by sqrt(d), rounded to float32 once."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    blocks, remaining = [], int(N)
    while remaining > 0:
        A = _portable_randn(d * d, g).reshape(d, d).double()  # QR in float64
        Q, R = torch.linalg.qr(A)
        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs.view(1, -1)
        take = min(remaining, d)
        blocks.append(Q[:take])
        remaining -= take
    return (torch.cat(blocks, dim=0) * math.sqrt(d)).float()


def make_G_float32(seed: int, N: int, d: int) -> torch.Tensor:
    """The float32 construction (torch.randn + float32 QR) that the BERT models were packed with."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    blocks, remaining = [], int(N)
    while remaining > 0:
        A = torch.randn(d, d, generator=g, device="cpu")
        Q, R = torch.linalg.qr(A)
        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs.view(1, -1)
        take = min(remaining, d)
        blocks.append(Q[:take])
        remaining -= take
    return torch.cat(blocks, dim=0) * math.sqrt(d)


def make_G_conv2d(seed: int, groups: int, N: int, cin_g: int, kh: int, kw: int) -> torch.Tensor:
    """[groups, N, cin_g, kh, kw]; group i uses seed + 10007 * i."""
    d = int(cin_g) * int(kh) * int(kw)
    return torch.stack([make_G(int(seed) + 10007 * i, N, d).reshape(N, cin_g, kh, kw)
                        for i in range(int(groups))], dim=0)


def make_G_conv1d(seed: int, groups: int, N: int, cin_g: int, k: int) -> torch.Tensor:
    """[groups, N, cin_g, k]; group i uses seed + 10007 * i."""
    d = int(cin_g) * int(k)
    return torch.stack([make_G(int(seed) + 10007 * i, N, d).reshape(N, cin_g, k)
                        for i in range(int(groups))], dim=0)


@contextlib.contextmanager
def cpu_threads(n: int):
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(old)
