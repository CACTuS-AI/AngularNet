"""CIFAR ResNet-18 with PReLU activations and GGM convolutions (stem, BatchNorm and classifier full precision)."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..imagenet.ggm_conv import Conv2dGGM

LAYERS = {"resnet18": [2, 2, 2, 2]}
CHANNELS = [64, 128, 256, 512]


class BasicBlock(nn.Module):
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


class ResNetCIFAR(nn.Module):
    def __init__(self, arch: str = "resnet18", N_factor: float = 1.0, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, CHANNELS[0], 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(CHANNELS[0])
        self.act1 = nn.PReLU(num_parameters=CHANNELS[0])
        cin = CHANNELS[0]
        for i, (blocks, cout) in enumerate(zip(LAYERS[arch], CHANNELS), start=1):
            layer = []
            for b in range(blocks):
                layer.append(BasicBlock(cin, cout, 2 if (b == 0 and i > 1) else 1, N_factor))
                cin = cout
            setattr(self, f"layer{i}", nn.Sequential(*layer))
        self.fc = nn.Linear(CHANNELS[-1], num_classes)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(x.mean(dim=(2, 3)), 1))


def build_model(arch: str, N_factor: float = 1.0, num_classes: int = 10) -> nn.Module:
    return ResNetCIFAR(arch, N_factor=N_factor, num_classes=num_classes)
