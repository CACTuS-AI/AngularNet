"""Projection matrices G, rebuilt from their seeds."""
from __future__ import annotations

import contextlib
import math

import torch

G_THREADS = 6  # CPU threads used when building G


def make_G(seed: int, N: int, d: int) -> torch.Tensor:
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


@contextlib.contextmanager
def cpu_threads(n: int):
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(old)
