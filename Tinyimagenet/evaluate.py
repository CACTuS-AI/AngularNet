#!/usr/bin/env python
"""Evaluate GGM ResNet checkpoints on the TinyImageNet-200 test set."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import urllib.error

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ggm_eval.checkpoint import build_model, load_checkpoint
from ggm_eval.data import TinyImageNetTest, ensure_test_set
from ggm_eval.models import MODELS, resolve

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="model name (see --list), or 'all'")
    p.add_argument("--list", action="store_true", help="list the models and which checkpoints are present")
    p.add_argument("--checkpoint_dir", default=os.path.join(HERE, "checkpoints"))
    p.add_argument("--data_root", default=os.path.join(HERE, "data"),
                   help="directory holding tiny-imagenet-200/ (downloaded there if missing)")
    p.add_argument("--no_download", action="store_true", help="fail instead of downloading the test set")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", default=None, choices=["off", "bf16", "fp16"],
                   help="autocast dtype (default: bf16 on cuda, off on cpu)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--limit_batches", type=int, default=0, help="smoke test: stop after this many batches")
    p.add_argument("--json_out", default="", help="also write the results to this JSON file")
    return p.parse_args()


def checkpoint_path(ckpt_dir: str, name: str) -> str:
    return os.path.join(ckpt_dir, name + ".pt")


def list_models(ckpt_dir: str):
    print(f"checkpoint directory: {ckpt_dir}\n")
    print(f"  {'model':<26} {'arch':<9} {'bits':<10} {'N':>2}  checkpoint")
    for name, m in MODELS.items():
        have = "present" if os.path.isfile(checkpoint_path(ckpt_dir, name)) else "missing"
        print(f"  {name:<26} {m['arch']:<9} {m['bits']:<10} {m['N']:>2}  {have}")


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
    if args.list:
        list_models(args.checkpoint_dir)
        return 0
    if not args.model:
        print("error: give --model <name> or --model all (see --list)", file=sys.stderr)
        return 2
    if args.model.lower() == "all":
        names = [m for m in MODELS if os.path.isfile(checkpoint_path(args.checkpoint_dir, m))]
        if not names:
            print(f"error: no checkpoints found in {args.checkpoint_dir}", file=sys.stderr)
            return 1
    else:
        try:
            names = [resolve(args.model)]
        except KeyError:
            print(f"error: unknown model {args.model!r}. Available models:\n", file=sys.stderr)
            list_models(args.checkpoint_dir)
            return 2
        if not os.path.isfile(checkpoint_path(args.checkpoint_dir, names[0])):
            print(f"error: checkpoint not found: {checkpoint_path(args.checkpoint_dir, names[0])}", file=sys.stderr)
            return 1

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("error: --device cuda requested but no CUDA GPU is available", file=sys.stderr)
        return 1
    amp = args.amp or ("bf16" if device.type == "cuda" else "off")
    amp_dtype = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}[amp]
    channels_last = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    try:
        data_dir = ensure_test_set(args.data_root, download=not args.no_download)
        dataset = TinyImageNetTest(data_dir)
    except FileNotFoundError as e:
        print(f"error: {e}. Remove --no_download to download it, or pass --data_root <dir>.", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, RuntimeError) as e:
        print(f"error: could not prepare the TinyImageNet test set: {e}\n"
              "Check the internet connection, or download https://cs231n.stanford.edu/tiny-imagenet-200.zip "
              "yourself, extract it, and pass --data_root <directory containing tiny-imagenet-200>.", file=sys.stderr)
        return 1
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    print(f"test set: {len(dataset):,} images in {data_dir}\n"
          f"device: {device}{gpu} | amp: {amp} | batch: {args.batch_size} | torch {torch.__version__}")
    if device.type != "cuda":
        print("note: running on CPU in fp32; this is slow")

    results = []
    for name in names:
        print(f"\n== {name}")
        t0 = time.time()
        try:
            ckpt = load_checkpoint(checkpoint_path(args.checkpoint_dir, name))
        except Exception as e:  # truncated download, wrong file, ...
            print(f"error: could not read {checkpoint_path(args.checkpoint_dir, name)}: {e}", file=sys.stderr)
            return 1
        model, info = build_model(ckpt)
        print(f"  G: {info['g_exact']}/{info['g_layers']} layers match the stored checksums")
        model.to(device)
        if channels_last:  # 4-D tensors only: the 5-D projection buffers G cannot be channels-last
            for t in list(model.parameters()) + list(model.buffers()):
                if t.dim() == 4:
                    t.data = t.data.contiguous(memory_format=torch.channels_last)
        r = evaluate(model, loader, device, amp_dtype, channels_last, args.limit_batches)
        note = f"(partial run on {r['images']:,} of {len(dataset):,} images)   " if r["images"] < len(dataset) else ""
        print(f"  top-1 {r['top1']:.2f} %   top-5 {r['top5']:.2f} %   loss {r['loss']:.4f}   {note}"
              f"{time.time() - t0:.0f} s")
        results.append({"model": name, **r, "g_exact_layers": info["g_exact"],
                        "g_layers": info["g_layers"], "device": str(device), "amp": amp,
                        "batch_size": args.batch_size, "torch": torch.__version__})
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
