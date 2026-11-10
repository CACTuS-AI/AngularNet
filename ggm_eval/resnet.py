"""ResNets with GGM convolutions."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .layers import Conv2dGGM, ScaledConv2dGGM


# ---------------------------------------------------------------- TinyImageNet

TINY_STAGES = {"resnet18": (2, 2, 2, 2), "resnet34": (3, 4, 6, 3)}


class TinyBasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int, conv_kw: dict):
        super().__init__()
        self.conv1 = ScaledConv2dGGM(inplanes, planes, 3, stride=stride, padding=1, **conv_kw)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = ScaledConv2dGGM(planes, planes, 3, stride=1, padding=1, **conv_kw)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                ScaledConv2dGGM(inplanes, planes, 1, stride=stride, padding=0, **conv_kw),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act2(x + shortcut)


class TinyImageNetResNet(nn.Module):
    """ResNet-18/34 for 64x64 inputs (3x3 stem, no max-pool), ReLU."""

    def __init__(self, arch: str = "resnet18", *, N_factor: float = 1.0, k_bits: int = 1,
                 num_classes: int = 200):
        super().__init__()
        if arch not in TINY_STAGES:
            raise ValueError(f"arch must be one of {sorted(TINY_STAGES)}, got {arch!r}")
        self.arch, self.N_factor, self.k_bits = arch, float(N_factor), int(k_bits)
        conv_kw = dict(N_factor=N_factor, k_bits=k_bits)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)

        inplanes = 64
        for i, (planes, blocks) in enumerate(zip((64, 128, 256, 512), TINY_STAGES[arch]), start=1):
            stride = 1 if i == 1 else 2
            layer = []
            for b in range(blocks):
                layer.append(TinyBasicBlock(inplanes, planes, stride if b == 0 else 1, conv_kw))
                inplanes = planes
            setattr(self, f"layer{i}", nn.Sequential(*layer))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.global_pool(x), 1))


# ---------------------------------------------------------------- ImageNet-1k

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


def build_imagenet_resnet(arch: str, N_factor: float = 1.0, num_classes: int = 1000) -> nn.Module:
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes, in_chans=3)
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    _set_prelu(model)
    for name, module in model.named_children():
        if name not in SKIP_TOP:
            _replace_convs(module, N_factor, name)
    return model


# ---------------------------------------------------------------- CIFAR-10

CIFAR_STAGES = {"resnet18": [2, 2, 2, 2]}
CIFAR_CHANNELS = [64, 128, 256, 512]


class CifarBasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, N_factor: float):
        super().__init__()
        self.conv1 = Conv2dGGM(cin, cout, 3, N_factor=N_factor, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.act1 = nn.PReLU(num_parameters=cout)
        self.conv2 = Conv2dGGM(cout, cout, 3, N_factor=N_factor, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act2 = nn.PReLU(num_parameters=cout)
        self.downsample = nn.Identity()
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(Conv2dGGM(cin, cout, 1, N_factor=N_factor, stride=stride),
                                            nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        return self.act2(self.bn2(self.conv2(out)) + self.downsample(x))


class CifarResNet(nn.Module):
    """ResNet-18 for 32x32 inputs (3x3 stem, no max-pool), PReLU, GGM downsample convolutions."""

    def __init__(self, arch: str = "resnet18", N_factor: float = 1.0, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, CIFAR_CHANNELS[0], 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(CIFAR_CHANNELS[0])
        self.act1 = nn.PReLU(num_parameters=CIFAR_CHANNELS[0])
        cin = CIFAR_CHANNELS[0]
        for i, (blocks, cout) in enumerate(zip(CIFAR_STAGES[arch], CIFAR_CHANNELS), start=1):
            layer = []
            for b in range(blocks):
                layer.append(CifarBasicBlock(cin, cout, 2 if (b == 0 and i > 1) else 1, N_factor))
                cin = cout
            setattr(self, f"layer{i}", nn.Sequential(*layer))
        self.fc = nn.Linear(CIFAR_CHANNELS[-1], num_classes)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(x.mean(dim=(2, 3)), 1))


def build_cifar_resnet(arch: str, N_factor: float = 1.0, num_classes: int = 10) -> nn.Module:
    return CifarResNet(arch, N_factor=N_factor, num_classes=num_classes)
