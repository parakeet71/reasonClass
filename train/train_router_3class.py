#!/usr/bin/env python3
"""Train a 3-class router classifier on prompt embeddings.

Classes: 0=Direct, 1=Reasoning, 2=Context-dependent.
Uses residual blocks with softmax output.
"""

from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── hyperparameters ─────────────────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 200
DEFAULT_SEED = 42
WEIGHT_DECAY = 5e-4
GRAD_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.03
EARLY_STOPPING_PATIENCE = 30
TRAIN_FRACTION = 0.85
VAL_FRACTION = 0.075
TEST_FRACTION = 0.075
HIDDEN_DIMS = [256, 128, 64]
DROPOUTS = [0.35, 0.25, 0.15]
NUM_CLASSES = 3
LR_CYCLE_STEPS = 40

CHECKPOINT_NAME = "best_model.pt"
METRICS_NAME = "metrics.json"


def architecture_config() -> dict:
    return {"hidden_dims": HIDDEN_DIMS, "dropouts": DROPOUTS, "num_classes": NUM_CLASSES}


def parse_args():
    p = argparse.ArgumentParser(description="Train 3-class router on prompt embeddings.")
    p.add_argument("--npz_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

# ── model ───────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.linear_in = nn.Linear(dim_in, dim_hidden)
        self.dropout_in = nn.Dropout(dropout)
        self.linear_out = nn.Linear(dim_hidden, dim_out)
        self.dropout_out = nn.Dropout(dropout * 0.7)
        self.shortcut = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x):
        r = self.shortcut(x)
        h = self.norm(x)
        h = F.silu(self.linear_in(h))
        h = self.dropout_in(h)
        h = self.linear_out(h)
        h = self.dropout_out(h)
        return h + r


class RouterMLP3Class(nn.Module):
    def __init__(self, dim, hidden_dims=None, dropouts=None, num_classes=NUM_CLASSES):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS
        if dropouts is None:
            dropouts = DROPOUTS
        blocks = []
        prev = dim
        for hd, dr in zip(hidden_dims, dropouts):
            blocks.append(ResBlock(prev, hd, hd, dr))
            prev = hd
        blocks.append(nn.LayerNorm(prev))
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x):
        return self.head(self.body(x))

# ── data loading ────────────────────────────────────────────────────────────

def load_npz(path):
    with np.load(path) as data:
        embeddings = data["embeddings"].astype(np.float32)
        labels = data["reasoning"].astype(np.int64)
        ids = data["ids"].astype(np.int64) if "ids" in data else None
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [N,D], got {embeddings.shape}")
    if labels.ndim != 1:
        raise ValueError(f"reasoning must be [N], got {labels.shape}")
    if len(embeddings) != len(labels):
        raise ValueError("embeddings and reasoning length mismatch")
    valid = np.isin(labels, [0, 1, 2])
    if not valid.all():
        raise ValueError(f"labels must be 0,1,2; got {np.unique(labels).tolist()}")
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    if class_counts.min() < 3:
        raise ValueError("need >= 3 examples per class")
    return embeddings, labels, ids


def load_npz_metadata(path):
    with np.load(path) as data:
        m = {}
        for k in ("embedding_max_length", "embedding_normalized"):
            if k in data:
                v = data[k]
                m[k] = v.item() if isinstance(v, np.ndarray) else v
    return m


# ── stratified split ───────────────────────────────────────────────────────

def split_class_indices(indices, rng, train_frac, val_frac):
    sh = indices.copy()
    rng.shuffle(sh)
    n = len(sh)
    tn = int(round(n * train_frac))
    vn = int(round(n * val_frac))
    if n >= 3:
        tn = min(max(tn, 1), n - 2)
        vn = min(max(vn, 1), n - tn - 1)
    return sh[:tn], sh[tn:tn + vn], sh[tn + vn:]


def stratified_split(labels, seed, train_frac=TRAIN_FRACTION, val_frac=VAL_FRACTION):
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for cls in range(NUM_CLASSES):
        idx = np.flatnonzero(labels == cls)
        tr, va, te = split_class_indices(idx, rng, train_frac, val_frac)
        parts["train"].append(tr)
        parts["val"].append(va)
        parts["test"].append(te)
    train_idx = np.concatenate(parts["train"])
    val_idx = np.concatenate(parts["val"])
    test_idx = np.concatenate(parts["test"])
    for a in (train_idx, val_idx, test_idx):
        rng.shuffle(a)
    if len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("val/test must be non-empty")
    return train_idx, val_idx, test_idx


def make_loader(embeddings, labels, indices, batch_size, shuffle):
    x = torch.from_numpy(embeddings[indices]).float()
    y = torch.from_numpy(labels[indices]).long()  # CrossEntropy expects Long
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

# ── metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    """y_true, y_pred are both integer class labels."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    for c in range(NUM_CLASSES):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        per_class[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(cm[c, :].sum())}

    n = max(len(y_true), 1)
    accuracy = float(np.diag(cm).sum() / n)
    macro_f1 = float(np.mean([per_class[c]["f1"] for c in range(NUM_CLASSES)]))
    weights = np.array([per_class[c]["support"] for c in range(NUM_CLASSES)], dtype=np.float64)
    weights /= weights.sum()
    weighted_f1 = float(np.sum([per_class[c]["f1"] * weights[c] for c in range(NUM_CLASSES)]))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {str(c): per_class[c] for c in range(NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    return labels.numpy(), probs, preds

# ── training ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip, label_smoothing=0.0):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if label_smoothing > 0:
            loss = criterion(logits, y)
            # Apply label smoothing manually: mix uniform distribution
            smooth_loss = 0.0
            for c in range(NUM_CLASSES):
                mask = (y == c)
                if mask.any():
                    targets = torch.full_like(logits[mask], label_smoothing / (NUM_CLASSES - 1))
                    targets[:, c] = 1.0 - label_smoothing
                    smooth_loss += -(targets * F.log_softmax(logits[mask], dim=1)).sum(dim=1).mean() * mask.float().mean()
            loss = smooth_loss
        else:
            loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / max(n, 1)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def class_name(c):
    return ["Direct", "Reason", "Context"][c]


def print_metrics(name, m):
    print(f"\n{name}:")
    print(f"  accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  weighted_f1={m['weighted_f1']:.4f}")
    for c in range(NUM_CLASSES):
        pc = m["per_class"][str(c)]
        print(f"  {class_name(c):8s}: prec={pc['precision']:.4f}  rec={pc['recall']:.4f}  f1={pc['f1']:.4f}  sup={pc['support']}")
    print(f"  confusion_matrix: {m['confusion_matrix']}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    npz_path = Path(args.npz_path)
    out_dir = Path(args.out_dir)
    set_seed(args.seed)

    embeddings, labels, ids = load_npz(npz_path)
    embedding_metadata = load_npz_metadata(npz_path)
    train_idx, val_idx, test_idx = stratified_split(labels, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(embeddings, labels, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(embeddings, labels, val_idx, args.batch_size, shuffle=False)
    test_loader = make_loader(embeddings, labels, test_idx, args.batch_size, shuffle=False)

    class_counts = np.bincount(labels[train_idx], minlength=NUM_CLASSES)
    class_weights = torch.tensor(
        [len(train_idx) / (NUM_CLASSES * max(c, 1)) for c in class_counts],
        dtype=torch.float32, device=device,
    )

    model = RouterMLP3Class(embeddings.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    lr_steps = max(LR_CYCLE_STEPS, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=lr_steps, T_mult=1, eta_min=args.lr * 1e-4,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME

    best_val_wf1, best_epoch, es_counter = -1.0, 0, 0
    history = []

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"examples: {len(labels):,}  dim: {embeddings.shape[1]}  params: {param_count:,}")
    print(f"split: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")
    for c in range(NUM_CLASSES):
        print(f"  class {c} ({class_name(c):8s}): total={(labels==c).sum():,}  train={class_counts[c]:,}")
    print(f"arch: {HIDDEN_DIMS}  dropouts: {DROPOUTS}")
    print(f"lr={args.lr}  wd={WEIGHT_DECAY}  label_smoothing={LABEL_SMOOTHING}")

    progress = tqdm(range(1, args.epochs + 1), desc="training", unit="epoch")
    for epoch in progress:
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, GRAD_CLIP_NORM, LABEL_SMOOTHING)
        scheduler.step()

        val_y, val_probs, val_preds = predict_probs(model, val_loader, device)
        val_m = compute_metrics(val_y, val_preds)
        val_wf1 = val_m["weighted_f1"]

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "val_weighted_f1": val_wf1,
            "lr": float(scheduler.get_last_lr()[0]),
        })
        progress.set_postfix(loss=f"{train_loss:.4f}", wf1=f"{val_wf1:.4f}")

        if val_wf1 > best_val_wf1:
            best_val_wf1, best_epoch, es_counter = val_wf1, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": int(embeddings.shape[1]),
                "num_classes": NUM_CLASSES,
                "best_epoch": best_epoch,
                "best_val_weighted_f1": float(best_val_wf1),
                "seed": args.seed,
                "architecture": architecture_config(),
            }, checkpoint_path)
        else:
            es_counter += 1
            if es_counter >= EARLY_STOPPING_PATIENCE:
                print(f"early stopping at epoch {epoch}")
                break

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Evaluate on all splits
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        y, probs, preds = predict_probs(model, loader, device)
        m = compute_metrics(y, preds)
        print_metrics(name, m)

    # Final metrics from test set
    test_y, test_probs, test_preds = predict_probs(model, test_loader, device)
    test_m = compute_metrics(test_y, test_preds)

    metrics = {
        "npz_path": str(npz_path),
        "num_examples": int(len(labels)),
        "num_classes": NUM_CLASSES,
        "input_dim": int(embeddings.shape[1]),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_epochs": args.epochs,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "label_smoothing": LABEL_SMOOTHING,
        "patience": EARLY_STOPPING_PATIENCE,
        "architecture": architecture_config(),
        "best_epoch": best_epoch,
        "best_val_weighted_f1": float(best_val_wf1),
        "class_counts": {str(c): int((labels == c).sum()) for c in range(NUM_CLASSES)},
        "split_sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "test_metrics": test_m,
        "history": history,
    }
    save_json(out_dir / METRICS_NAME, metrics)
    print(f"\nsaved: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
