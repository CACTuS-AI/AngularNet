"""timm ResNet-18/34 with PReLU activations and GGM convolutions (stem, downsample, BatchNorm and classifier full precision)."""
from __future__ import annotations

import timm
import torch.nn as nn

from .ggm_conv import Conv2dGGM

SKIP_TOP = {"conv1", "bn1", "act1", "maxpool", "fc", "head", "stem"}


def _set_prelu(model: nn.Module, init: float = 0.25) -> None:
    if isinstance(getattr(model, "bn1", None), nn.BatchNorm2d) and hasattr(model, "act1"):
        model.act1 = nn.PReLU(num_parameters=model.bn1.num_features, init=init)
    for m in model.modules():
        for i in (1, 2, 3):
            bn, act = f"bn{i}", f"act{i}"
            if hasattr(m, act) and isinstance(getattr(m, bn, None), nn.BatchNorm2d):
                setattr(m, act, nn.PReLU(num_parameters=getattr(m, bn).num_features, init=init))


def _replace_convs(parent: nn.Module, N_factor: float, path: str) -> None:
    for name, module in parent.named_children():
        full = f"{path}.{name}"
        if ".downsample" in full or full.endswith("downsample"):
            continue
        if isinstance(module, nn.Conv2d):
            setattr(parent, name, Conv2dGGM(
                module.in_channels, module.out_channels, module.kernel_size[0], N_factor=N_factor,
                stride=module.stride[0], padding=module.padding[0], groups=module.groups,
                bias=module.bias is not None))
        else:
            _replace_convs(module, N_factor, full)


def build_model(arch: str, N_factor: float = 1.0, num_classes: int = 1000) -> nn.Module:
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes, in_chans=3)
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    _set_prelu(model)
    for name, module in model.named_children():
        if name not in SKIP_TOP:
            _replace_convs(module, N_factor, name)
    return model
