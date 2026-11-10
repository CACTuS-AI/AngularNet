"""Evaluation loop and metrics for the time-series models."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def metrics(logits: torch.Tensor, labels: torch.Tensor, num_class: int) -> dict:
    """Accuracy and macro precision / recall / F1 / AUROC / AUPRC (in %), plus cross-entropy loss."""
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    y, pred = labels.numpy(), logits.argmax(1).numpy()
    out = {
        "accuracy": 100 * accuracy_score(y, pred),
        "precision": 100 * precision_score(y, pred, average="macro", zero_division=0),
        "recall": 100 * recall_score(y, pred, average="macro", zero_division=0),
        "f1": 100 * f1_score(y, pred, average="macro", zero_division=0),
    }
    if len(np.unique(y)) == num_class:  # AUROC/AUPRC need every class among the samples
        probs = torch.softmax(logits, dim=1).numpy()
        onehot = np.eye(num_class, dtype=np.float32)[y]
        out["auroc"] = 100 * roc_auc_score(onehot, probs, multi_class="ovr")
        out["auprc"] = 100 * average_precision_score(onehot, probs, average="macro")
    else:
        print("  note: not every class occurs in the evaluated samples; AUROC/AUPRC are skipped")
    out["loss"] = F.cross_entropy(logits, labels).item()
    return out


@torch.inference_mode()
def evaluate(model, loader, device, num_class: int, limit_batches: int = 0) -> dict:
    all_logits, all_labels = [], []
    n = 0
    for i, (x, y) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        all_logits.append(model(x.to(device)).float().cpu())
        all_labels.append(y)
        n += x.size(0)
        print(f"\r  evaluating: {n:,}/{len(loader.dataset):,} samples", end="", flush=True)
    print()
    return {"samples": n, **metrics(torch.cat(all_logits), torch.cat(all_labels), num_class)}
