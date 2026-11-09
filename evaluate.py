#!/usr/bin/env python
"""Evaluate GGM checkpoints on the TinyImageNet-200, ImageNet-1k or CIFAR-10 test set.

usage:
  python evaluate.py tinyimagenet resnet18_ggm_1bit_N1   one model
  python evaluate.py imagenet all                        every checkpoint of a dataset
  python evaluate.py cifar10 vgg16_ggm_1bit_N1
  python evaluate.py list                                all available models
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.error

_missing = [m for m in ("torch", "torchvision", "PIL", "numpy") if importlib.util.find_spec(m) is None]
if _missing:
    sys.exit(f"error: missing Python packages ({', '.join(_missing)}). Install them with:\n"
             f"  pip install -r requirements.txt\n(see Setup in README.md)")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ("tinyimagenet", "imagenet", "cifar10")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", nargs="*", metavar="DATASET [MODEL]",
                   help=f"dataset ({', '.join(DATASETS)}) and model name, 'all' or 'list'; or just 'list'")
    p.add_argument("--dataset", choices=DATASETS, help=argparse.SUPPRESS)
    p.add_argument("--model", help=argparse.SUPPRESS)
    p.add_argument("--list", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--checkpoint_dir", default=None, help="default: checkpoints/<dataset>")
    p.add_argument("--data_root", default=None, help="default: data/<dataset>")
    p.add_argument("--no_download", action="store_true", help="tinyimagenet, cifar10: fail instead of downloading the test set")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--limit_batches", type=int, default=0, help="smoke test: stop after this many batches")
    p.add_argument("--json_out", default="", help="also write the results to this JSON file")
    args = p.parse_args()

    t = list(args.target)
    if len(t) > 2:
        p.error("give at most a dataset and a model")
    if t and t[0].lower() in ("list", "--list"):
        if len(t) > 1:
            p.error("'list' takes no model name")
        args.list = True
    elif t:
        if t[0].lower() not in DATASETS:
            p.error(f"unknown dataset {t[0]!r} (use {', '.join(DATASETS)})")
        args.dataset = t[0].lower()
        if len(t) == 1 or t[1].lower() in ("list", "--list"):
            args.list = True
        else:
            args.model = t[1]
    elif not (args.dataset or args.list):
        p.print_usage()
        args.list = True
    return args


def registry(dataset: str):
    if dataset == "tinyimagenet":
        from ggm_eval.tinyimagenet import models
    elif dataset == "cifar10":
        from ggm_eval.cifar10 import models
    else:
        from ggm_eval.imagenet import models
    return models


def checkpoint_path(ckpt_dir: str, name: str) -> str:
    return os.path.join(ckpt_dir, name + ".pt")


def list_models(dataset: str, ckpt_dir: str):
    MODELS = registry(dataset).MODELS
    print(f"{dataset}  (checkpoints: {ckpt_dir})")
    print(f"  {'model':<26} {'arch':<9} {'bits':<10} {'N':>2}  checkpoint")
    for name, m in MODELS.items():
        have = "present" if os.path.isfile(checkpoint_path(ckpt_dir, name)) else "missing"
        print(f"  {name:<26} {m['arch']:<9} {m['bits']:<10} {m['N']:>2}  {have}")


def load_test_set(dataset: str, root: str, download: bool):
    if dataset == "tinyimagenet":
        from ggm_eval.tinyimagenet.data import TinyImageNetTest, ensure_test_set
        return TinyImageNetTest(ensure_test_set(root, download=download))
    if dataset == "cifar10":
        from ggm_eval.cifar10.data import build_test_set as build_cifar10
        return build_cifar10(root, download=download)
    from ggm_eval.imagenet.data import build_test_set
    return build_test_set(root)


def load_model(dataset: str, path: str, name: str):
    """Build the model and load its weights; every G is rebuilt from its seed."""
    if dataset == "tinyimagenet":
        from ggm_eval.tinyimagenet.checkpoint import build_model, load_checkpoint
        return build_model(load_checkpoint(path))
    if dataset == "cifar10":
        from ggm_eval.cifar10.checkpoint import load_model as load_cifar10
        return load_cifar10(path, name)
    from ggm_eval.imagenet.checkpoint import load_model as load_imagenet
    return load_imagenet(path, name)


@torch.inference_mode()
def evaluate(model, loader, device, amp_dtype, channels_last, limit_batches=0):
    n = correct1 = correct5 = 0
    loss_sum = 0.0
    for i, (imgs, labels) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        correct1 += (logits.argmax(1) == labels).sum().item()
        correct5 += (logits.float().topk(5, dim=1).indices == labels[:, None]).any(dim=1).sum().item()
        loss_sum += loss.item() * imgs.size(0)
        n += imgs.size(0)
        print(f"\r  evaluating: {n:,}/{len(loader.dataset):,} images", end="", flush=True)
    print()
    return {"top1": 100 * correct1 / n, "top5": 100 * correct5 / n, "loss": loss_sum / n, "images": n}


def main():
    args = parse_args()
    if args.list and not args.dataset:
        for d in DATASETS:
            list_models(d, os.path.join(HERE, "checkpoints", d))
            print()
        return 0
    if not args.dataset:
        print(f"error: give a dataset ({', '.join(DATASETS)})", file=sys.stderr)
        return 2
    ckpt_dir = args.checkpoint_dir or os.path.join(HERE, "checkpoints", args.dataset)
    data_root = args.data_root or os.path.join(HERE, "data", args.dataset)
    if args.list:
        list_models(args.dataset, ckpt_dir)
        return 0
    if not args.model:
        print("error: give a model name or 'all' (see: python evaluate.py list)", file=sys.stderr)
        return 2
    try:
        reg = registry(args.dataset)
    except ImportError as e:
        print(f"error: {e}. Install the requirements: pip install -r requirements.txt", file=sys.stderr)
        return 1
    if args.model.lower() == "all":
        names = [m for m in reg.MODELS if os.path.isfile(checkpoint_path(ckpt_dir, m))]
        if not names:
            print(f"error: no checkpoints found in {ckpt_dir}", file=sys.stderr)
            return 1
    else:
        try:
            names = [reg.resolve(args.model)]
        except KeyError:
            print(f"error: unknown {args.dataset} model {args.model!r}. Available models:\n", file=sys.stderr)
            list_models(args.dataset, ckpt_dir)
            return 2
        if not os.path.isfile(checkpoint_path(ckpt_dir, names[0])):
            print(f"error: checkpoint not found: {checkpoint_path(ckpt_dir, names[0])}", file=sys.stderr)
            return 1

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "cpu":
            print("note: PyTorch does not see a GPU, so the CPU is used "
                  "(for a GPU, see Setup in README.md; to silence this note, pass --device cpu)")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("error: --device cuda requested but no CUDA GPU is available", file=sys.stderr)
        return 1
    # per-dataset settings: tinyimagenet bf16 + TF32, cifar10 fp32 + TF32, imagenet fp32
    amp_dtype = torch.bfloat16 if (args.dataset == "tinyimagenet" and device.type == "cuda") else None
    channels_last_inputs = device.type == "cuda" or args.dataset == "imagenet"
    if device.type == "cuda":
        if args.dataset in ("tinyimagenet", "cifar10"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    try:
        dataset = load_test_set(args.dataset, data_root, download=not args.no_download)
    except FileNotFoundError as e:
        hint = (". Remove --no_download to download it, or pass --data_root <dir>."
                if args.dataset in ("tinyimagenet", "cifar10") else "")
        print(f"error: {e}{hint}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, RuntimeError) as e:
        print(f"error: could not prepare the {args.dataset} test set: {e}", file=sys.stderr)
        return 1
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    print(f"{args.dataset} test set: {len(dataset):,} images in {dataset.root}\n"
          f"device: {device}{gpu} | batch: {args.batch_size} | torch {torch.__version__}")

    results = []
    for name in names:
        print(f"\n== {args.dataset} / {name}")
        t0 = time.time()
        try:
            model = load_model(args.dataset, checkpoint_path(ckpt_dir, name), name)
        except Exception as e:  # truncated download, wrong file, ...
            print(f"error: could not load {checkpoint_path(ckpt_dir, name)}: {e}", file=sys.stderr)
            return 1
        print("  G rebuilt from seeds")
        model.to(device)
        if device.type == "cuda":  # 4-D tensors only: 5-D projection buffers stay as they are
            for t in list(model.parameters()) + list(model.buffers()):
                if t.dim() == 4:
                    t.data = t.data.contiguous(memory_format=torch.channels_last)
        r = evaluate(model, loader, device, amp_dtype, channels_last_inputs, args.limit_batches)
        part = f"(partial run on {r['images']:,} of {len(dataset):,} images)   " if r["images"] < len(dataset) else ""
        print(f"  top-1 {r['top1']:.2f} %   top-5 {r['top5']:.2f} %   loss {r['loss']:.4f}   {part}"
              f"{time.time() - t0:.0f} s")
        results.append({"dataset": args.dataset, "model": name, **r, "device": str(device),
                        "batch_size": args.batch_size,
                        "torch": torch.__version__})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(results) > 1:
        print(f"\n{'model':<26} {'top-1':>7} {'top-5':>7}  images")
        for r in results:
            print(f"{r['model']:<26} {r['top1']:>6.2f}% {r['top5']:>6.2f}%  {r['images']:,}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
