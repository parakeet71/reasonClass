#!/usr/bin/env python3
"""Classify prompt JSONL chunks using the DeepSeek API.

3-class labeling: 0=Direct, 1=Reasoning, 2=Context-dependent.

Usage:
  python deepseek_classify_prompt.py --n_samples 64 --batch_size 16
  python deepseek_classify_prompt.py --n_samples 1000 --batch_size 16 --model deepseek-reasoner --reasoning_effort low
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import openai
from dotenv import load_dotenv
from tqdm import tqdm

# ── configurable defaults ──────────────────────────────────────────────────
DEEPSEEK_MODEL = "deepseek-v4-flash"
REASONING_EFFORT: str | None = "high"         # "minimal", "low", "medium", "high" (reasoner only)
DEFAULT_MAX_TOKENS = 1024                     # reasoning + answer (high effort uses most of this)
DEFAULT_TEMPERATURE = 0.0
LINES_PER_CHUNK = 50_000

THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
LABEL_RE = re.compile(r"(?<!\d)[012](?!\d)")

CLASSIFIER_PREFIX = (
    "Classify this prompt. Output ONLY one digit: 0, 1, or 2. "
    ""
    "2 (CONTEXT-DEPENDENT): Unanswerable without prior conversation. "
    "Signals: very short (\"go on\", \"again\", \"the second one\"), "
    "corrections (\"no, change that\"), references to missing context "
    "(\"using the code above\", \"in the same style\", \"this part\", \"rewrite it\"), "
    "raw error dumps with no question. "
    "Do NOT use 2 for role-play or system-instruction prompts that include the full task. "
    "\"You are a translator. Translate: hello\" is self-contained, not 2. "
    "If the prompt contains conversation history (\"Assistant:\", \"User:\") "
    "ignore it — base your decision only on the final user message. "
    ""
    "1 (NEEDS REASONING): Self-contained AND requires significant reasoning. "
    "Analysis, judgment, planning, debugging, troubleshooting, multi-step math, "
    "code behavior or architecture, tradeoffs, "
    "high-stakes advice (medical/legal/financial/safety), "
    "academic/technical writing with strict formatting or length constraints, "
    "content generation requiring domain expertise or factual accuracy, "
    "error checking, proofreading, interpretation, synthesis, comparison, "
    "causal explanation, or any task where a thoughtful answer differs from a hasty one. "
    ""
    "0 (DIRECT): Self-contained AND answerable with minimal reasoning. "
    "Simple facts, definitions, basic arithmetic, translation, formatting, "
    "extraction, simple summarization, simple rewriting or grammar fixes, "
    "simple code generation (CRUD, file operations, basic API calls), "
    "code explanation, simple how-to questions, "
    "straightforward creative generation (poems, stories, captions, tweets, slogans, "
    "image prompts, marketing copy, roleplay), product definitions without advice. "
    ""
    "Key rules: 1) If the prompt does NOT stand on its own → 2. "
    "2) Technical/academic writing with constraints → 1, not 0. "
    "3) Translation → 0 unless it also requires analysis, rewriting, or checking. "
    "4) Between 0 and 1, prefer 1 when reasoning improves correctness or safety. "
    "Prompt: "
)

# ── argument parsing ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify prompts with DeepSeek API.")
    p.add_argument(
        "--n_samples", type=int, required=True,
        help="Number of prompts to classify.",
    )
    p.add_argument(
        "--batch_size", type=int, default=1,
        help="Number of parallel API requests. (default: 1)",
    )
    p.add_argument(
        "--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens in API response. (default: {DEFAULT_MAX_TOKENS})",
    )
    p.add_argument(
        "--model", default=DEEPSEEK_MODEL,
        help=f"DeepSeek model name. (default: {DEEPSEEK_MODEL})",
    )
    p.add_argument(
        "--reasoning_effort", default=REASONING_EFFORT,
        help="Reasoning effort (minimal/low/medium/high), only for deepseek-reasoner.",
    )
    p.add_argument(
        "--in_dir", default="../data/WildChat_clean_chunks",
    )
    p.add_argument(
        "--out_dir", default="data/deepseek_labels",
        help="Directory for labeled output.",
    )
    p.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    return p.parse_args()


# ── DeepSeek API ────────────────────────────────────────────────────────────

class DeepSeekClassifier:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        reasoning_effort: str | None = None,
    ):
        self.client = openai.OpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_key,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def _build_request_kwargs(self, prompt: str, reasoning_effort: str | None = None) -> dict:
        eff = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        kwargs: dict = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": f"{CLASSIFIER_PREFIX}{prompt}",
            }],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if eff is not None:
            kwargs["extra_body"] = {"reasoning_effort": eff}
        return kwargs

    def classify_single(self, prompt: str, reasoning_effort: str | None = None) -> str:
        kwargs = self._build_request_kwargs(prompt, reasoning_effort)
        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        return content.strip()

    def classify_batch(self, prompts: list[str]) -> list[str]:
        results: list[str | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            futures = {
                ex.submit(self._classify_with_retry, i, p): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = ""
                    tqdm.write(f"  warning: prompt {idx} failed after retries: {exc}", file=sys.stderr)
        return [r or "" for r in results]

    def _classify_with_retry(self, idx: int, prompt: str, max_retries: int = 3) -> str:
        last_exc = None
        for attempt in range(max_retries):
            try:
                eff = self.reasoning_effort if attempt == 0 else None
                result = self.classify_single(prompt, reasoning_effort=eff)
                if result:
                    return result
            except Exception as exc:
                last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        raise last_exc or ValueError("empty response after retries")


# ── label extraction ────────────────────────────────────────────────────────

def extract_label(output: str) -> int:
    """Extract 0, 1, or 2 from model output."""
    cleaned = THINK_BLOCK_RE.sub(" ", output)
    cleaned = THINK_TAG_RE.sub(" ", cleaned)
    matches = LABEL_RE.findall(cleaned)
    if not matches:
        raise ValueError(f"no 0/1/2 label found in: {output!r}")
    label = int(matches[-1])
    if label not in (0, 1, 2):
        raise ValueError(f"invalid label {label} in: {output!r}")
    return label


# ── I/O helpers ─────────────────────────────────────────────────────────────

def load_existing_ids(out_dir: Path) -> tuple[set[str], int, int, int]:
    seen: set[str] = set()
    valid, bad, dup = 0, 0, 0
    for path in sorted(out_dir.glob("labels_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    eid = str(entry["id"])
                except Exception:
                    bad += 1
                    continue
                if eid in seen:
                    dup += 1
                    continue
                seen.add(eid)
                valid += 1
    return seen, valid, bad, dup


def file_ends_with_newline(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return True
    with path.open("rb") as f:
        f.seek(-1, 2)
        return f.read(1) == b"\n"


def open_output_chunk(out_dir: Path, chunk_idx: int):
    path = out_dir / f"labels_{chunk_idx:05d}.jsonl"
    needs_newline = not file_ends_with_newline(path)
    f = path.open("a", encoding="utf-8")
    if needs_newline:
        f.write("\n")
    return path, f


def sample_line_from_file(input_file: Path) -> tuple[int, str]:
    """Reservoir-sampling a single random line from a file."""
    sampled_num = 0
    sampled_line: str | None = None
    with input_file.open(encoding="utf-8") as src:
        for num, line in enumerate(src, 1):
            if random.randrange(num) == 0:
                sampled_num = num
                sampled_line = line
    if sampled_line is None:
        raise ValueError(f"empty file: {input_file}")
    return sampled_num, sampled_line


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("error: DEEPSEEK_API_KEY not set in environment or .env", file=sys.stderr)
        return 1

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.is_dir():
        print(f"error: --in_dir not found: {in_dir}", file=sys.stderr)
        return 2
    if args.n_samples <= 0:
        print("error: --n_samples must be positive", file=sys.stderr)
        return 2
    if args.batch_size <= 0:
        print("error: --batch_size must be positive", file=sys.stderr)
        return 2

    input_files = sorted(in_dir.rglob("*.jsonl"))
    if not input_files:
        print(f"error: no .jsonl files in {in_dir}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    seen_ids, existing_valid, existing_bad, existing_dup = load_existing_ids(out_dir)
    if existing_valid:
        print(f"resume: {existing_valid:,} existing labeled rows in {out_dir}")

    # Build classifier
    model = args.model
    reason_effort = args.reasoning_effort

    print(f"model: {model}  reasoning_effort: {reason_effort or 'N/A'}")
    print(f"max_tokens: {args.max_tokens}  temperature: {args.temperature}")
    print(f"target: {args.n_samples} samples  batch_size: {args.batch_size}")

    classifier = DeepSeekClassifier(
        api_key=api_key,
        model=model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=reason_effort,
    )

    # Sampling loop
    classified = 0
    bad_outputs = 0
    duplicate_skipped = 0
    attempts = 0
    max_attempts = max(args.n_samples * 100, 10_000)
    total_valid = existing_valid

    label_counts: Counter = Counter()

    out_file = None
    out_chunk_idx: int | None = None

    def save_labeled(entry: dict, input_file: Path, label: int) -> None:
        nonlocal total_valid, out_file, out_chunk_idx
        seen_ids.add(str(entry["id"]))
        chunk_idx = total_valid // LINES_PER_CHUNK + 1
        if out_file is None or chunk_idx != out_chunk_idx:
            if out_file is not None:
                out_file.close()
            out_path, out_file = open_output_chunk(out_dir, chunk_idx)
            out_chunk_idx = chunk_idx
            print(f"writing {out_path}")
        labeled = {
            "source": entry.get("source_file", str(input_file)),
            "id": entry["id"],
            "prompt": entry["prompt"],
            "reasoning": label,
        }
        json.dump(labeled, out_file, ensure_ascii=False)
        out_file.write("\n")
        total_valid += 1

    try:
        pbar = tqdm(total=args.n_samples, desc="classifying", unit="sample")
        while classified < args.n_samples and attempts < max_attempts:
            # Build a batch by sampling random lines
            batch_entries: list[tuple[Path, dict]] = []
            batch_attempts = 0
            while len(batch_entries) < args.batch_size and attempts < max_attempts:
                attempts += 1
                batch_attempts += 1
                input_file = random.choice(input_files)
                try:
                    _, line = sample_line_from_file(input_file)
                except ValueError:
                    continue
                try:
                    entry = json.loads(line.strip())
                    pid = str(entry["id"])
                    prompt = entry.get("prompt", "")
                    if not isinstance(prompt, str) or not prompt.strip():
                        continue
                except (json.JSONDecodeError, KeyError):
                    continue
                if pid in seen_ids:
                    duplicate_skipped += 1
                    continue
                batch_entries.append((input_file, entry))
                seen_ids.add(pid)  # reserve

            if not batch_entries:
                continue

            # Classify the batch in parallel
            prompts = [e["prompt"] for _, e in batch_entries]
            outputs = classifier.classify_batch(prompts)

            for (input_file, entry), output in zip(batch_entries, outputs):
                if classified >= args.n_samples:
                    break
                try:
                    label = extract_label(output)
                except ValueError as exc:
                    bad_outputs += 1
                    tqdm.write(f"  bad output: {exc}", file=sys.stderr)
                    # Remove from seen_ids so it can be retried
                    seen_ids.discard(str(entry["id"]))
                    continue
                save_labeled(entry, input_file, label)
                label_counts[label] += 1
                classified += 1
                pbar.update(1)

                if classified % 50 == 0:
                    tqdm.write(
                        f"  progress: {classified}/{args.n_samples}  "
                        f"dist: 0={label_counts.get(0,0)} 1={label_counts.get(1,0)} 2={label_counts.get(2,0)}"
                    )

        pbar.close()
    finally:
        if out_file is not None:
            out_file.close()

    print()
    print(f"input files searched: {len(input_files)}")
    print(f"sampling attempts:   {attempts:,}")
    print(f"duplicate skipped:   {duplicate_skipped:,}")
    print(f"bad outputs skipped: {bad_outputs:,}")
    print(f"classified:          {classified:,}")
    print(f"class distribution:  0={label_counts.get(0,0):,}  1={label_counts.get(1,0):,}  2={label_counts.get(2,0):,}")
    print(f"output dir:          {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
