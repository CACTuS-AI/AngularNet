"""Loading seed checkpoints: weights plus one seed per GGM layer for its projection matrix G."""
from __future__ import annotations

import torch

from .ggm_conv import GGMConv2d
from ..projections import G_THREADS, cpu_threads, make_G_conv2d
from .resnet import GGMResNet

FORMAT = "ggm-seed-checkpoint-v1"


def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    return ckpt


def build_model(ckpt: dict):
    """Build the model, regenerate every G from its seed, load the weights strictly."""
    meta = ckpt["meta"]
    model = GGMResNet(meta["arch"], N_factor=meta["N_factor"], k_bits=meta["k_bits"])
    convs = {n: m for n, m in model.named_modules() if isinstance(m, GGMConv2d)}
    seeds = meta["g_seeds"]
    if set(seeds) != set(convs):
        raise ValueError("checkpoint seeds do not match the model's GGM layers")

    state = dict(ckpt["state_dict"])
    with cpu_threads(G_THREADS):
        for name, m in convs.items():
            state[name + ".G"] = make_G_conv2d(seeds[name], m.groups, m.N, m.in_per_group, m.kh, m.kw)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model
