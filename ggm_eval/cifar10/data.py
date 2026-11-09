"""CIFAR-10 test set (10,000 images): download and loading."""
from __future__ import annotations

import os

from torchvision import datasets, transforms

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
NUM_IMAGES = 10_000

test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def build_test_set(root: str, download: bool = True):
    if not download and not os.path.isdir(os.path.join(root, "cifar-10-batches-py")):
        raise FileNotFoundError(f"CIFAR-10 not found: expected {os.path.join(root, 'cifar-10-batches-py')}")
    ds = datasets.CIFAR10(root, train=False, download=download, transform=test_transform)
    if len(ds) != NUM_IMAGES:
        print(f"[data] warning: found {len(ds):,} test images (expected {NUM_IMAGES:,})")
    return ds
