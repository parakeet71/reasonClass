#!/usr/bin/env python3
"""Visualize the 0/1/2 class distribution in a labeled JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = "data/labels_00001.jsonl"
DEFAULT_OUTPUT = "class_distribution.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reasoning class distribution.")
    parser.add_argument("--input_path", default=DEFAULT_INPUT, help="Labeled JSONL file.")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT, help="Output chart path.")
    return parser.parse_args()


def load_counts(input_path: Path) -> tuple[Counter, int, int]:
    counts = Counter()
    total = 0
    bad_lines = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                entry = json.loads(line)
                label = int(entry["reasoning"])
                if label not in {0, 1, 2}:
                    raise ValueError(f"invalid reasoning label: {label}")
            except Exception as exc:
                bad_lines += 1
                print(f"warning: skipping line {line_number}: {exc}", file=sys.stderr)
                continue

            counts[label] += 1
            total += 1

    return counts, total, bad_lines


def plot_counts(counts: Counter, total: int, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    label_names = ["0 (Direct)", "1 (Reason)", "2 (Context)"]
    values = [counts.get(i, 0) for i in range(3)]
    colors = ["#4c78a8", "#f58518", "#72b66b"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(label_names, values, color=colors)
    ax.set_title("Prompt Classification Distribution")
    ax.set_ylabel("Count")

    for bar, value in zip(bars, values):
        pct = (value / total * 100) if total else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n{pct:.1f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.is_file():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 2

    counts, total, bad_lines = load_counts(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_counts(counts, total, output_path)

    print(f"input: {input_path}")
    print(f"class 0 (Direct):  {counts.get(0, 0):,} ({(counts.get(0, 0) / total * 100) if total else 0:.2f}%)")
    print(f"class 1 (Reason):  {counts.get(1, 0):,} ({(counts.get(1, 0) / total * 100) if total else 0:.2f}%)")
    print(f"class 2 (Context): {counts.get(2, 0):,} ({(counts.get(2, 0) / total * 100) if total else 0:.2f}%)")
    print(f"valid rows: {total:,}")
    print(f"bad lines skipped: {bad_lines:,}")
    print(f"chart saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
