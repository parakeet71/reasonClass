#!/usr/bin/env python3
"""Train an improved router classifier on prompt embeddings.

Enhancements over v1:
  - Residual blocks with projection shortcuts for deeper gradient flow.
  - Cosine-annealing LR schedule with warm restarts.
  - Label smoothing in the loss.
  - Gradient clipping.
  - Deeper & wider architecture (384 -> 256 -> 128 -> 64).
  - Higher dropout for stronger regularisation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── easy-edit hyperparameters ───────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 150
DEFAULT_SEED = 42
WEIGHT_DECAY = 5e-4
GRAD_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 25

TRAIN_FRACTION = 0.85
VAL_FRACTION = 0.075
TEST_FRACTION = 0.075

HIDDEN_DIMS = [256, 128, 64]
DROPOUTS = [0.35, 0.25, 0.15]

LR_CYCLE_STEPS = 30
LR_CYCLE_DECAY = 0.85

THRESHOLD_START = 0.10
THRESHOLD_STOP = 0.90
THRESHOLD_STEP = 0.025
PREFER_ZERO_MIN_RECALL = 0.65

CHECKPOINT_NAME = "best_model.pt"
METRICS_NAME = "metrics.json"
THRESHOLD_SWEEP_NAME = "threshold_sweep.json"


def architecture_config() -> dict:
    return {
        "hidden_dims": HIDDEN_DIMS,
        "dropouts": DROPOUTS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an improved binary MLP router on prompt embeddings.",
    )
    parser.add_argument("--npz_path", required=True, help="Input embeddings .npz file.")
    parser.add_argument("--out_dir", required=True, help="Directory for outputs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def validate_hyperparameters() -> None:
    split_total = TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION
    if not np.isclose(split_total, 1.0):
        raise ValueError(f"split fractions must sum to 1.0, got {split_total}")
    if min(TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION) <= 0:
        raise ValueError("split fractions must be positive")


# ── model ───────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Two-layer residual block with optional dimension projection."""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.linear_in = nn.Linear(dim_in, dim_hidden)
        self.dropout_in = nn.Dropout(dropout)
        self.linear_out = nn.Linear(dim_hidden, dim_out)
        self.dropout_out = nn.Dropout(dropout * 0.7)
        self.shortcut = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        h = self.norm(x)
        h = self.linear_in(h)
        h = F.silu(h)
        h = self.dropout_in(h)
        h = self.linear_out(h)
        h = self.dropout_out(h)
        return h + residual


class RouterMLPv2(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dims: list[int] | None = None,
        dropouts: list[float] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS
        if dropouts is None:
            dropouts = DROPOUTS

        blocks: list[nn.Module] = []
        prev = dim
        for hd, dr in zip(hidden_dims, dropouts):
            blocks.append(ResBlock(prev, hd, hd, dr))
            prev = hd
        blocks.append(nn.LayerNorm(prev))
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


# ── data loading ────────────────────────────────────────────────────────────

def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(path) as data:
        if "embeddings" not in data or "reasoning" not in data:
            raise ValueError("npz must contain 'embeddings' and 'reasoning' arrays")
        embeddings = data["embeddings"].astype(np.float32)
        labels = data["reasoning"].astype(np.int64)
        ids = data["ids"].astype(np.int64) if "ids" in data else None
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [N, D], got {embeddings.shape}")
    if labels.ndim != 1:
        raise ValueError(f"reasoning must be [N], got {labels.shape}")
    if len(embeddings) != len(labels):
        raise ValueError("embeddings and reasoning must have same length")
    if ids is not None and len(ids) != len(labels):
        raise ValueError("ids and reasoning must have same length")
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("reasoning labels must be 0, 1, or 2")
    binary_mask = np.isin(labels, [0, 1])
    if not binary_mask.all():
        filtered = binary_mask.sum()
        if filtered < 10:
            raise ValueError(f"only {filtered} binary examples after filtering label 2")
        embeddings = embeddings[binary_mask]
        labels = labels[binary_mask]
        if ids is not None:
            ids = ids[binary_mask]
    class_counts = np.bincount(labels, minlength=2)
    if class_counts.min() < 3:
        raise ValueError("need at least 3 examples per class")
    return embeddings, labels, ids


def load_npz_metadata(path: Path) -> dict:
    with np.load(path) as data:
        metadata = {}
        for key in ("embedding_max_length", "embedding_normalized"):
            if key in data:
                val = data[key]
                if isinstance(val, np.ndarray):
                    metadata[key] = val.item()
                else:
                    metadata[key] = val
    return metadata


# ── stratified split (exactly as in v1, seed-rng) ───────────────────────────

def split_class_indices(
    indices: np.ndarray, rng: np.random.Generator,
    train_frac: float, val_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_n = int(round(n * train_frac))
    val_n = int(round(n * val_frac))
    if n >= 3:
        train_n = min(max(train_n, 1), n - 2)
        val_n = min(max(val_n, 1), n - train_n - 1)
    elif n >= 2:
        train_n = 1
        val_n = 0
    else:
        train_n = 1
        val_n = 0
    return shuffled[:train_n], shuffled[train_n : train_n + val_n], shuffled[train_n + val_n :]


def stratified_split(
    labels: np.ndarray, seed: int,
    train_frac: float = TRAIN_FRACTION,
    val_frac: float = VAL_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        tr, va, te = split_class_indices(idx, rng, train_frac, val_frac)
        parts["train"].append(tr)
        parts["val"].append(va)
        parts["test"].append(te)
    train_idx = np.concatenate(parts["train"])
    val_idx = np.concatenate(parts["val"])
    test_idx = np.concatenate(parts["test"])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    if len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("val and test splits must be non-empty")
    return train_idx, val_idx, test_idx


def make_loader(
    embeddings: np.ndarray, labels: np.ndarray, indices: np.ndarray,
    batch_size: int, shuffle: bool,
) -> DataLoader:
    x = torch.from_numpy(embeddings[indices]).float()
    y = torch.from_numpy(labels[indices]).float()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


# ── metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    y_pred = (probs >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = max(len(y_true), 1)
    accuracy = (tp + tn) / n
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


# ── training ────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    criterion: nn.Module, device: torch.device, grad_clip: float | None,
    label_smoothing: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if label_smoothing > 0:
            y_smooth = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            y_smooth = y
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y_smooth)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * len(y)
        total_examples += len(y)
    return total_loss / max(total_examples, 1)


# ── threshold selection ────────────────────────────────────────────────────

def choose_balanced_threshold(sweep: list[dict]) -> dict:
    return max(sweep, key=lambda r: (r["f1"], r["recall"], r["precision"], r["accuracy"]))


def choose_prefer_zero_threshold(sweep: list[dict], min_recall: float) -> dict:
    candidates = [r for r in sweep if r["recall"] >= min_recall]
    if not candidates:
        candidates = sweep
    return max(candidates, key=lambda r: (r["precision"], r["f1"], r["accuracy"], r["threshold"]))


# ── io ──────────────────────────────────────────────────────────────────────

def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def print_metrics(name: str, metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    print(
        f"{name}: acc={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"threshold={metrics['threshold']:.2f}"
    )
    print(
        f"{name} confusion matrix [[tn, fp], [fn, tp]]: "
        f"[[{cm['tn']}, {cm['fp']}], [{cm['fn']}, {cm['tp']}]]"
    )


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    npz_path = Path(args.npz_path)
    out_dir = Path(args.out_dir)

    try:
        validate_hyperparameters()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for param, val in (("batch_size", args.batch_size), ("lr", args.lr), ("epochs", args.epochs)):
        if val <= 0:
            print(f"error: --{param} must be positive", file=sys.stderr)
            return 2

    if not npz_path.is_file():
        print(f"error: --npz_path not found: {npz_path}", file=sys.stderr)
        return 2

    set_seed(args.seed)
    embeddings, labels, ids = load_npz(npz_path)
    embedding_metadata = load_npz_metadata(npz_path)
    train_idx, val_idx, test_idx = stratified_split(labels, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(embeddings, labels, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(embeddings, labels, val_idx, args.batch_size, shuffle=False)
    test_loader = make_loader(embeddings, labels, test_idx, args.batch_size, shuffle=False)

    train_labels = labels[train_idx]
    num_pos = int((train_labels == 1).sum())
    num_neg = int((train_labels == 0).sum())
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32, device=device)

    model = RouterMLPv2(embeddings.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    lr_steps = max(LR_CYCLE_STEPS, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=lr_steps, T_mult=1, eta_min=args.lr * 1e-3,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"loaded {len(labels):,} examples with dim={embeddings.shape[1]}")
    print(f"split: train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
    print(f"classes: neg={(labels == 0).sum():,}  pos={(labels == 1).sum():,}")
    print(f"params: {param_count:,}  device: {device}")
    print(f"architecture: {HIDDEN_DIMS}  dropouts: {DROPOUTS}")
    print(f"label_smoothing={LABEL_SMOOTHING}  grad_clip={GRAD_CLIP_NORM}")

    progress = tqdm(range(1, args.epochs + 1), desc="training", unit="epoch")
    for epoch in progress:
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, GRAD_CLIP_NORM,
            label_smoothing=LABEL_SMOOTHING,
        )
        scheduler.step()

        val_y, val_probs = predict_probs(model, val_loader, device)
        val_metrics = compute_metrics(val_y, val_probs, threshold=0.5)
        val_f1 = val_metrics["f1"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_f1,
            "lr": float(scheduler.get_last_lr()[0]),
        })
        progress.set_postfix(loss=f"{train_loss:.4f}", val_f1=f"{val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": int(embeddings.shape[1]),
                "best_epoch": best_epoch,
                "best_val_f1": float(best_val_f1),
                "threshold": 0.5,
                "seed": args.seed,
                "npz_path": str(npz_path),
                "embedding_metadata": embedding_metadata,
                "architecture": architecture_config(),
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_y, val_probs = predict_probs(model, val_loader, device)
    test_y, test_probs = predict_probs(model, test_loader, device)

    thresholds = [round(float(x), 2)
                  for x in np.arange(THRESHOLD_START, THRESHOLD_STOP + THRESHOLD_STEP / 2, THRESHOLD_STEP)]
    threshold_sweep = [compute_metrics(val_y, val_probs, t) for t in thresholds]

    balanced = choose_balanced_threshold(threshold_sweep)
    prefer_zero = choose_prefer_zero_threshold(threshold_sweep, PREFER_ZERO_MIN_RECALL)

    train_y, train_probs = predict_probs(model, train_loader, device)
    balanced_metrics = {
        "train": compute_metrics(train_y, train_probs, balanced["threshold"]),
        "val": compute_metrics(val_y, val_probs, balanced["threshold"]),
        "test": compute_metrics(test_y, test_probs, balanced["threshold"]),
    }
    prefer_zero_metrics = {
        "train": compute_metrics(train_y, train_probs, prefer_zero["threshold"]),
        "val": compute_metrics(val_y, val_probs, prefer_zero["threshold"]),
        "test": compute_metrics(test_y, test_probs, prefer_zero["threshold"]),
    }

    checkpoint["balanced_threshold"] = balanced["threshold"]
    checkpoint["prefer_zero_threshold"] = prefer_zero["threshold"]
    checkpoint["threshold_selection"] = {
        "balanced": "max validation F1, then recall",
        "prefer_zero": f"max precision with recall >= {PREFER_ZERO_MIN_RECALL}",
    }
    torch.save(checkpoint, checkpoint_path)

    print("\n=== balanced threshold ===")
    print_metrics("train", balanced_metrics["train"])
    print_metrics("val",   balanced_metrics["val"])
    print_metrics("test",  balanced_metrics["test"])
    print(f"balanced threshold: {balanced['threshold']:.2f}")
    print("\n=== prefer-zero threshold ===")
    print_metrics("train", prefer_zero_metrics["train"])
    print_metrics("val",   prefer_zero_metrics["val"])
    print_metrics("test",  prefer_zero_metrics["test"])
    print(f"prefer-zero threshold: {prefer_zero['threshold']:.2f}")

    metrics = {
        "npz_path": str(npz_path),
        "num_examples": int(len(labels)),
        "input_dim": int(embeddings.shape[1]),
        "embedding_metadata": embedding_metadata,
        "has_ids": ids is not None,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_epochs": args.epochs,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "label_smoothing": LABEL_SMOOTHING,
        "patience": EARLY_STOPPING_PATIENCE,
        "prefer_zero_min_recall": PREFER_ZERO_MIN_RECALL,
        "split_fractions": {"train": TRAIN_FRACTION, "val": VAL_FRACTION, "test": TEST_FRACTION},
        "architecture": architecture_config(),
        "best_epoch": best_epoch,
        "best_val_f1_at_0_5": float(best_val_f1),
        "weighted_f1_at_0_5": float(compute_metrics(val_y, val_probs, 0.5)["f1"]),
        "balanced_threshold": balanced["threshold"],
        "prefer_zero_threshold": prefer_zero["threshold"],
        "class_counts": {
            "negative": int((labels == 0).sum()),
            "positive": int((labels == 1).sum()),
            "train_negative": num_neg,
            "train_positive": num_pos,
        },
        "split_sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "balanced": balanced_metrics,
        "prefer_zero": prefer_zero_metrics,
        "history": history,
    }

    save_json(out_dir / METRICS_NAME, metrics)
    save_json(out_dir / THRESHOLD_SWEEP_NAME, threshold_sweep)
    print(f"\nsaved checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
