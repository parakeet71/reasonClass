#!/usr/bin/env python3
"""Clean the labeled JSONL dataset by removing context-dependent / noisy prompts."""

import json, sys, re
from pathlib import Path

# ── rules ───────────────────────────────────────────────────────────────────
MIN_PROMPT_LENGTH = 20  # shorter than this → almost certainly context-dependent

# Prompts that are clearly continuations, corrections, or reference previous turns
CORRECTION_PREFIXES = [
    "no,", "nope,", "wrong,", "incorrect,", "not that,",
    "i meant", "i actually meant", "that's not",
    "actually,", "actually no",
]

CORRECTION_CONTAINS = [
    "instead of", "not like that", "not that one",
    "change that to", "make it", "redo it",
    "try again", "do it again",
]

# Prompts that directly reference prior context
REFERENCE_WORDS = [
    "above", "previous response", "earlier you", "as mentioned",
    "the same way", "like that one", "that one again",
    "as above", "as before", "like before", "same as",
    "what about the", "and what about",
]

# Very specific short follow-ups
SHORT_FOLLOWUP_PREFIXES = [
    "also,", "and then", "and also", "then also",
    "now also", "next,", "next up",
]


def is_noisy(prompt: str) -> tuple[bool, str]:
    p = prompt.strip()
    p_lower = p.lower()

    if len(p) < MIN_PROMPT_LENGTH:
        return True, "too_short"

    if not p:
        return True, "empty"

    # Correction prefixes (check start of prompt)
    for prefix in CORRECTION_PREFIXES:
        if p_lower.startswith(prefix):
            return True, f"correction_prefix:{prefix}"

    for pattern in CORRECTION_CONTAINS:
        if pattern in p_lower:
            return True, f"correction_contains:{pattern}"

    # Short follow-up prefixes
    for prefix in SHORT_FOLLOWUP_PREFIXES:
        if p_lower.startswith(prefix):
            return True, f"followup_prefix:{prefix}"

    # Reference words
    for word in REFERENCE_WORDS:
        if word in p_lower:
            return True, f"reference:{word}"

    return False, ""


def main():
    input_path = Path("data/labels_00001.jsonl")
    output_path = Path("data/labels_clean.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept, removed = 0, 0
    reasons = {}
    samples_removed = {k: [] for k in ["too_short", "correction_prefix",
                                         "correction_contains", "followup_prefix",
                                         "reference"]}

    with open(input_path) as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                print(f"skipping invalid JSON at line {i}", file=sys.stderr)
                continue

            noisy, reason = is_noisy(d["prompt"])
            if noisy:
                removed += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                cat = reason.split(":")[0]
                if cat in samples_removed and len(samples_removed[cat]) < 3:
                    samples_removed[cat].append(d["prompt"][:120])
            else:
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Kept:   {kept:,}")
    print(f"Removed: {removed:,} ({removed/(kept+removed)*100:.1f}%)")
    print(f"Total:  {kept+removed:,}")
    print()
    print("Removal reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:30s}: {count:>6,}")
    print()
    print("Sample removed prompts:")
    for cat, samples in samples_removed.items():
        if samples:
            print(f"  [{cat}]")
            for s in samples:
                print(f"    - {s}")
    print()
    print(f"Saved clean dataset to: {output_path}")

    # Check class distribution of clean data
    c0, c1 = 0, 0
    with open(output_path) as f:
        for line in f:
            d = json.loads(line)
            if d["reasoning"] == 0: c0 += 1
            else: c1 += 1
    print(f"Clean class distribution: 0={c0:,}, 1={c1:,} ({c1/(c0+c1)*100:.1f}% reasoning)")


if __name__ == "__main__":
    main()
