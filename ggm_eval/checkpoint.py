"""Checkpoint loading."""
from __future__ import annotations

import torch

from .bert import BertGGM
from .layers import Conv1dGGM, LinearGGM, PackedLinearGGM, ScaledConv2dGGM
from .medformer import MedformerGGM
from .models import MODELS
from .projections import G_THREADS, cpu_threads, make_G, make_G_conv1d, make_G_conv2d
from .resnet import TinyImageNetResNet, build_cifar_resnet, build_imagenet_resnet
from .vgg import build_vgg

SEED_FORMAT = "ggm-seed-checkpoint-v1"


def _seed_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != SEED_FORMAT:
        raise ValueError(f"{path} is not a {SEED_FORMAT} file")
    return ckpt


def _load_seed_model(model, ckpt: dict, layer_types):
    """Regenerate every G from the seeds in the meta data, then load the weights strictly."""
    layers = {n: m for n, m in model.named_modules() if isinstance(m, layer_types)}
    seeds = ckpt["meta"]["g_seeds"]
    if set(seeds) != set(layers):
        raise ValueError("checkpoint seeds do not match the model's GGM layers")
    state = dict(ckpt["state_dict"])
    cache = {}
    with cpu_threads(G_THREADS):
        for name, m in layers.items():
            if isinstance(m, ScaledConv2dGGM):
                key = ("conv2d", seeds[name], m.groups, m.N, m.in_per_group, m.kh, m.kw)
                make = lambda: make_G_conv2d(seeds[name], m.groups, m.N, m.in_per_group, m.kh, m.kw)
            elif isinstance(m, Conv1dGGM):
                key = ("conv1d", seeds[name], m.groups, m.N, m.cin_g, m.kernel_size)
                make = lambda: make_G_conv1d(seeds[name], m.groups, m.N, m.cin_g, m.kernel_size)
            else:  # LinearGGM, PackedLinearGGM
                key = ("linear", seeds[name], m.N, m.in_features)
                make = lambda: make_G(seeds[name], m.N, m.in_features)
            if key not in cache:
                cache[key] = make()
            state[name + ".G"] = cache[key].clone()
    model.load_state_dict(state, strict=True)
    return model.eval()


def _load_seed_buffer_model(model, path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state dict")
    state = {k.replace("_orig_mod.", "").removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)  # rebuilds every G from its G_seed
    return model.eval()


def load_model(dataset: str, path: str, name: str):
    """Build the model of a released checkpoint and load its weights."""
    if dataset == "tinyimagenet":
        ckpt = _seed_checkpoint(path)
        meta = ckpt["meta"]
        model = TinyImageNetResNet(meta["arch"], N_factor=meta["N_factor"], k_bits=meta["k_bits"])
        return _load_seed_model(model, ckpt, ScaledConv2dGGM)
    if dataset in ("adftd", "ptbxl"):
        ckpt = _seed_checkpoint(path)
        return _load_seed_model(MedformerGGM(**ckpt["meta"]["model"]), ckpt, (LinearGGM, Conv1dGGM))
    if dataset in ("sst2", "qnli"):  # also returns the tokenizer settings stored with the model
        ckpt = _seed_checkpoint(path)
        model = _load_seed_model(BertGGM(**ckpt["meta"]["model"]), ckpt, PackedLinearGGM)
        model.vocab, model.max_seq_length = ckpt["meta"]["vocab"], ckpt["meta"]["max_seq_length"]
        return model
    m = MODELS[dataset][name]
    if dataset == "imagenet":
        model = build_imagenet_resnet(m["arch"], N_factor=m["N"])
    elif m["arch"].startswith("resnet"):
        model = build_cifar_resnet(m["arch"], N_factor=m["N"])
    else:
        model = build_vgg(m["arch"], N_factor=m["N"])
    return _load_seed_buffer_model(model, path)
