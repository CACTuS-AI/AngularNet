"""Loading checkpoints: model weights plus one G_seed per GGM layer."""
from __future__ import annotations

import torch

from .models import MODELS
from .vgg import build_model


def load_model(path: str, name: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state dict")
    state = {k.replace("_orig_mod.", "").removeprefix("module."): v for k, v in state.items()}
    m = MODELS[name]
    model = build_model(m["arch"], N_factor=m["N"])
    model.load_state_dict(state, strict=True)  # rebuilds every G from its seed
    return model.eval()
