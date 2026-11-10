"""VGG-16 with GGM convolutions."""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import Conv2dGGM

CFG = {"vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]}


class VGG(nn.Module):
    def __init__(self, arch: str = "vgg16", N_factor: float = 1.0, num_classes: int = 10):
        super().__init__()
        layers, cin = [], 3
        for v in CFG[arch]:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            conv = (nn.Conv2d(cin, v, 3, padding=1, bias=False) if cin == 3
                    else Conv2dGGM(cin, v, 3, N_factor=N_factor, padding=1))
            # nn.Identity keeps the layer indices of the trained model (a dropout layer there)
            layers += [conv, nn.BatchNorm2d(v), nn.PReLU(num_parameters=v), nn.Identity()]
            cin = v
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(512 * 2 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.features(x), 1))


def build_vgg(arch: str, N_factor: float = 1.0, num_classes: int = 10) -> nn.Module:
    return VGG(arch, N_factor=N_factor, num_classes=num_classes)
