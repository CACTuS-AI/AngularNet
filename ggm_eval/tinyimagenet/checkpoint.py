"""Loading seed checkpoints: weights plus one seed per GGM layer for its projection matrix G."""
from __future__ import annotations

import time

import torch

from .ggm_conv import GGMConv2d
from ..projections import G_THREADS, cpu_threads, make_G_conv2d, sha256_of
from .resnet import GGMResNet

FORMAT = "ggm-seed-checkpoint-v1"


def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    return ckpt


def build_model(ckpt: dict, log=print):
    """Build the model, regenerate every G from its seed, load the weights strictly.

    Returns (model in eval mode, info dict with g_layers / g_exact counts)."""
    meta = ckpt["meta"]
    model = GGMResNet(meta["arch"], N_factor=meta["N_factor"], k_bits=meta["k_bits"])
    convs = {n: m for n, m in model.named_modules() if isinstance(m, GGMConv2d)}
    seeds, hashes = meta["g_seeds"], meta.get("g_sha256", {})
    if set(seeds) != set(convs):
        raise ValueError("checkpoint seeds do not match the model's GGM layers")

    state = dict(ckpt["state_dict"])
    exact, t0 = 0, time.time()
    with cpu_threads(G_THREADS):
        for i, (name, m) in enumerate(convs.items(), 1):
            G = make_G_conv2d(seeds[name], m.groups, m.N, m.in_per_group, m.kh, m.kw)
            exact += hashes.get(name) == sha256_of(G)
            state[name + ".G"] = G
            log(f"\r  regenerating G from seeds: {i}/{len(convs)} layers", end="", flush=True)
    log(f" ({time.time() - t0:.0f} s)", flush=True)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {**meta, "g_layers": len(convs), "g_exact": int(exact)}
