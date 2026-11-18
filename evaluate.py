#!/usr/bin/env python
"""Evaluate GGM checkpoints on their test sets."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.error

_missing = [m for m in ("torch", "torchvision", "PIL", "numpy", "sklearn") if importlib.util.find_spec(m) is None]
if _missing:
    sys.exit(f"error: missing Python packages ({', '.join(_missing)}). Install them with:\n"
             f"  pip install -r requirements.txt\n(see Setup in README.md)")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DATASETS = ("tinyimagenet", "imagenet", "cifar10")
TIMESERIES_DATASETS = ("adftd", "ptbxl")
TEXT_DATASETS = ("sst2", "qnli")
DATASETS = IMAGE_DATASETS + TIMESERIES_DATASETS + TEXT_DATASETS
ALIASES = {"sst-2": "sst2"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", nargs="*", metavar="DATASET [MODEL]",
                   help=f"dataset ({', '.join(DATASETS)}) and model name, 'all' or 'list'; or just 'list'")
    p.add_argument("--dataset", choices=DATASETS, help=argparse.SUPPRESS)
    p.add_argument("--model", help=argparse.SUPPRESS)
    p.add_argument("--list", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--checkpoint_dir", default=None, help="default: checkpoints/<dataset>")
    p.add_argument("--data_root", default=None, help="default: data/<dataset>")
    p.add_argument("--no_download", action="store_true",
                   help="fail instead of downloading the test set (all datasets except imagenet)")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="default: 64 for images, 128 for time series, 32 for text")
    p.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1), help="image datasets only")
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
        t[0] = ALIASES.get(t[0].lower(), t[0])
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
    if args.batch_size is None:
        args.batch_size = (128 if args.dataset in TIMESERIES_DATASETS else
                           32 if args.dataset in TEXT_DATASETS else 64)
    return args


def checkpoint_path(ckpt_dir: str, name: str) -> str:
    return os.path.join(ckpt_dir, name + ".pt")


def list_models(dataset: str, ckpt_dir: str):
    from ggm_eval.models import MODELS
    print(f"{dataset}  (checkpoints: {ckpt_dir})")
    print(f"  {'model':<26} {'arch':<9} {'bits':<10} {'N':>2}  checkpoint")
    for name, m in MODELS[dataset].items():
        have = "present" if os.path.isfile(checkpoint_path(ckpt_dir, name)) else "missing"
        print(f"  {name:<26} {m['arch']:<9} {m['bits']:<10} {m['N']:>2}  {have}")


@torch.inference_mode()
def evaluate_images(model, loader, device, channels_last, limit_batches=0):
    n = correct1 = correct5 = 0
    loss_sum = 0.0
    for i, (imgs, labels) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        correct1 += (logits.argmax(1) == labels).sum().item()
        correct5 += (logits.float().topk(5, dim=1).indices == labels[:, None]).any(dim=1).sum().item()
        loss_sum += loss.item() * imgs.size(0)
        n += imgs.size(0)
        print(f"\r  evaluating: {n:,}/{len(loader.dataset):,} images", end="", flush=True)
    print()
    return {"top1": 100 * correct1 / n, "top5": 100 * correct5 / n, "loss": loss_sum / n, "images": n}


@torch.inference_mode()
def evaluate_text(model, loader, device, limit_batches=0):
    n = correct = 0
    loss_sum = 0.0
    for i, (ids, segments, mask, labels) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        logits = model(ids.to(device), segments.to(device), mask.to(device)).float().cpu()
        correct += (logits.argmax(1) == labels).sum().item()
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        n += labels.size(0)
        print(f"\r  evaluating: {n:,}/{len(loader.dataset):,} examples", end="", flush=True)
    print()
    return {"accuracy": 100 * correct / n, "loss": loss_sum / n, "examples": n}


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
    timeseries = args.dataset in TIMESERIES_DATASETS
    text = args.dataset in TEXT_DATASETS
    ckpt_dir = args.checkpoint_dir or os.path.join(HERE, "checkpoints", args.dataset)
    data_root = args.data_root or os.path.join(HERE, "data", args.dataset)
    if args.list:
        list_models(args.dataset, ckpt_dir)
        return 0
    if not args.model:
        print("error: give a model name or 'all' (see: python evaluate.py list)", file=sys.stderr)
        return 2
    from ggm_eval.models import MODELS, resolve
    if args.model.lower() == "all":
        names = [m for m in MODELS[args.dataset] if os.path.isfile(checkpoint_path(ckpt_dir, m))]
        if not names:
            print(f"error: no checkpoints found in {ckpt_dir}", file=sys.stderr)
            return 1
    else:
        try:
            names = [resolve(args.dataset, args.model)]
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
    channels_last = args.dataset in IMAGE_DATASETS and (device.type == "cuda" or args.dataset == "imagenet")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False  # every model runs in plain fp32
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    from ggm_eval.checkpoint import load_model
    from ggm_eval.data import load_test_set
    try:
        dataset = load_test_set(args.dataset, data_root, download=not args.no_download)
    except FileNotFoundError as e:
        hint = "" if args.dataset == "imagenet" else ". Remove --no_download to download it, or pass --data_root <dir>."
        print(f"error: {e}{hint}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, RuntimeError, ValueError) as e:
        hint = (". It can also be downloaded by hand (see 'Test sets' in README.md) "
                "and passed with --data_root <dir>." if timeseries or text else "")
        print(f"error: could not prepare the {args.dataset} test set: {e}{hint}", file=sys.stderr)
        return 1
    if timeseries:  # the test set is already in memory
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        what = f"{len(dataset):,} samples from {dataset.subjects} subjects"
    elif text:  # tokenized per model, with the vocabulary stored in its checkpoint
        loader = None
        what = f"{len(dataset):,} examples (GLUE dev split)"
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        what = f"{len(dataset):,} images"
    gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    print(f"{args.dataset} test set: {what} in {dataset.root}\n"
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
        if timeseries:
            from ggm_eval.metrics import evaluate as evaluate_timeseries
            r = evaluate_timeseries(model, loader, device, len(dataset.classes), args.limit_batches)
            n = r["samples"]
            line = "   ".join(f"{k} {r[k]:.2f} %" for k in ("accuracy", "precision", "recall", "f1", "auroc", "auprc")
                             if k in r)
            unit = "samples"
        elif text:
            from ggm_eval.tokenization import BertTokenizer
            encoded = dataset.encode(BertTokenizer(model.vocab), model.max_seq_length)
            loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=False)
            r = evaluate_text(model, loader, device, args.limit_batches)
            n = r["examples"]
            line = f"accuracy {r['accuracy']:.2f} %"
            unit = "examples"
        else:
            if device.type == "cuda":  # 4-D tensors only: 5-D projection buffers stay as they are
                for t in list(model.parameters()) + list(model.buffers()):
                    if t.dim() == 4:
                        t.data = t.data.contiguous(memory_format=torch.channels_last)
            r = evaluate_images(model, loader, device, channels_last, args.limit_batches)
            n = r["images"]
            line = f"top-1 {r['top1']:.2f} %   top-5 {r['top5']:.2f} %"
            unit = "images"
        part = f"(partial run on {n:,} of {len(dataset):,} {unit})   " if n < len(dataset) else ""
        print(f"  {line}   loss {r['loss']:.4f}   {part}{time.time() - t0:.0f} s")
        results.append({"dataset": args.dataset, "model": name, **r, "device": str(device),
                        "batch_size": args.batch_size, "torch": torch.__version__})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(results) > 1 and args.dataset in IMAGE_DATASETS:
        print(f"\n{'model':<26} {'top-1':>7} {'top-5':>7}  images")
        for r in results:
            print(f"{r['model']:<26} {r['top1']:>6.2f}% {r['top5']:>6.2f}%  {r['images']:,}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
