"""ImageNet-1k validation set (used as the test set): locating and loading."""
from __future__ import annotations

import os

from torchvision import datasets, transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NUM_CLASSES, NUM_IMAGES = 1000, 50_000

test_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

LAYOUT_HELP = (
    "Expected <data_root>/val/<wnid>/<image>.JPEG (1000 class folders, 50,000 images), or the two\n"
    "official archives ILSVRC2012_img_val.tar and ILSVRC2012_devkit_t12.tar.gz in <data_root>\n"
    "(they are unpacked into that layout on first use). If an earlier unpacking was interrupted, delete\n"
    "<data_root>/val and run again. Download instructions: see README (ImageNet-1k)."
)


def _has_class_folders(val_dir: str) -> bool:
    return os.path.isdir(val_dir) and sum(
        1 for e in os.scandir(val_dir) if e.is_dir()) == NUM_CLASSES


def build_test_set(root: str):
    val_dir = os.path.join(root, "val")
    if not _has_class_folders(val_dir):
        if os.path.isfile(os.path.join(root, "ILSVRC2012_img_val.tar")) and \
           os.path.isfile(os.path.join(root, "ILSVRC2012_devkit_t12.tar.gz")):
            print("[data] unpacking the ImageNet validation archive (one-time)")
            datasets.ImageNet(root, split="val")
        if not _has_class_folders(val_dir):
            raise FileNotFoundError(f"ImageNet validation set not found under {root!r}.\n{LAYOUT_HELP}")
    ds = datasets.ImageFolder(val_dir, transform=test_transform)  # classes and files in sorted order
    if len(ds) != NUM_IMAGES:
        print(f"[data] warning: found {len(ds):,} images in {val_dir} (the full validation set has {NUM_IMAGES:,})")
    return ds
