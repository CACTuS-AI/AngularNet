"""ResNet-18/34 for 64x64 inputs with GGM convolutions (stem, BatchNorm and classifier full precision)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .ggm_conv import GGMConv2d

STAGES = {"resnet18": (2, 2, 2, 2), "resnet34": (3, 4, 6, 3)}


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int, conv_kw: dict):
        super().__init__()
        self.conv1 = GGMConv2d(inplanes, planes, 3, stride=stride, padding=1, **conv_kw)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = GGMConv2d(planes, planes, 3, stride=1, padding=1, **conv_kw)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                GGMConv2d(inplanes, planes, 1, stride=stride, padding=0, **conv_kw),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act2(x + shortcut)


class GGMResNet(nn.Module):
    def __init__(self, arch: str = "resnet18", *, N_factor: float = 1.0, k_bits: int = 1,
                 num_classes: int = 200):
        super().__init__()
        if arch not in STAGES:
            raise ValueError(f"arch must be one of {sorted(STAGES)}, got {arch!r}")
        self.arch, self.N_factor, self.k_bits = arch, float(N_factor), int(k_bits)
        conv_kw = dict(N_factor=N_factor, k_bits=k_bits)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)

        inplanes = 64
        for i, (planes, blocks) in enumerate(zip((64, 128, 256, 512), STAGES[arch]), start=1):
            stride = 1 if i == 1 else 2
            layer = []
            for b in range(blocks):
                layer.append(BasicBlock(inplanes, planes, stride if b == 0 else 1, conv_kw))
                inplanes = planes
            setattr(self, f"layer{i}", nn.Sequential(*layer))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.global_pool(x), 1))
