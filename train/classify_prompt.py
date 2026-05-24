#!/usr/bin/env python3
"""Classify prompt JSONL chunks with a local OpenAI-compatible model."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import openai
import requests


SERVER = "http://127.0.0.1:8080"
LINES_PER_CHUNK = 50_000
CLASSIFIER_MAX_TOKENS = 32
CLASSIFIER_TEMPERATURE = 0.0
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
LABEL_RE = re.compile(r"(?<!\d)[012](?!\d)")

CLASSIFIER_PREFIX = (
    "Classify this prompt into one of three categories. "
    "Output ONLY a single digit: 0, 1, or 2. Do not explain. "
    ""
    "2 (CONTEXT-DEPENDENT): The prompt cannot be understood or answered without prior conversation context. "
    "This includes: continuations (\"go on\", \"next\", \"more\"), corrections (\"no, change that\", \"instead\"), "
    "revisions (\"rewrite it shorter\", \"fix the bug\"), follow-ups referencing something already discussed "
    "(\"what about the second one\", \"in the same style\", \"using the code above\"), "
    "very short prompts that are clearly responding to something prior (\"in 90 words only\", \"enough\", \"again\"), "
    "or any prompt that contains pronouns/demonstratives pointing at unseen context "
    "(\"that one\", \"this part\", \"the same\", \"as before\", \"like you did\"). "
    "Also use 2 when the prompt opens with a raw code block, stack trace, or data dump "
    "with no standalone question, since it is likely a follow-up to an ongoing debugging session. "
    ""
    "1 (NEEDS REASONING): The prompt is self-contained AND answering well requires significant reasoning. "
    "This includes: judgment calls, ambiguity, planning, troubleshooting, debugging, analysis, "
    "interpretation, comparison, nontrivial math, probability, code behavior or integration, "
    "causal or technical explanation, tradeoffs, multi-step problem solving, "
    "medical/legal/financial/safety advice with risk assessment, "
    "literary/philosophical/scientific/academic analysis asking for meaning, arguments, or synthesis, "
    "complex content generation with strict or conflicting constraints, "
    "or any self-contained task where a thoughtful answer meaningfully differs from a hasty one. "
    ""
    "0 (DIRECT): The prompt is self-contained AND can be answered well with minimal reasoning. "
    "This includes: simple facts, definitions, basic arithmetic, straightforward translation, "
    "formatting, extraction, simple summarization, simple rewriting or grammar fixes, "
    "simple coding or API lookups, straightforward creative generation (poems, stories, "
    "captions, tweets, slogans, image prompts, marketing copy, roleplay), "
    "health/beauty/product definitions without advice or diagnosis, "
    "or any prompt where the answer is mainly about following instructions rather than figuring things out. "
    ""
    "Key rule: if the prompt does NOT stand on its own, answer 2 regardless of difficulty. "
    "Only choose 0 or 1 for prompts that would make sense to a stranger with no conversation history. "
    "If unsure between 0 and 1, choose 1 only when reasoning is likely to improve correctness, safety, or usefulness. "
    "Prompt: "
)

def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify prompt JSONL chunks with a local model."
    )
    parser.add_argument(
        "--in_dir",
        default="../data/WildChat_clean_chunks",
    )
    p.add_argument(
        "--out_dir",
        default="data/local_labels",
        help="Directory where labeled chunks will be written.",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=None,
        help="Optional maximum number of examples to classify.",
    )
    parser.add_argument(
        "--random_sample",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Randomly choose input files and sample random entries. Defaults to false.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of prompts to send to the model per request. Defaults to 1.",
    )
    return parser.parse_args()


def get_model_name(server: str) -> str:
    response = requests.get(f"{server}/v1/models", timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "models" in payload and payload["models"]:
        return payload["models"][0].get("name") or payload["models"][0].get("id")
    if "data" in payload and payload["data"]:
        return payload["data"][0].get("id") or payload["data"][0].get("name")

    raise RuntimeError("no model found at /v1/models")


def classify_prompt(client: openai.OpenAI, model_name: str, prompt: str) -> str:
    completion = client.completions.create(
        model=model_name,
        prompt=f"{CLASSIFIER_PREFIX}{prompt}",
        max_tokens=CLASSIFIER_MAX_TOKENS,
        temperature=CLASSIFIER_TEMPERATURE,
    )
    output = completion.choices[0].text.strip()
    return " ".join(output.split())


def classify_prompts(client: openai.OpenAI, model_name: str, prompts: list[str]) -> list[str]:
    completion = client.completions.create(
        model=model_name,
        prompt=[f"{CLASSIFIER_PREFIX}{prompt}" for prompt in prompts],
        max_tokens=CLASSIFIER_MAX_TOKENS,
        temperature=CLASSIFIER_TEMPERATURE,
    )
    outputs = [""] * len(prompts)
    for choice_idx, choice in enumerate(completion.choices):
        prompt_idx = getattr(choice, "index", choice_idx)
        outputs[prompt_idx] = " ".join(choice.text.strip().split())

    if any(output == "" for output in outputs):
        raise ValueError("model returned fewer outputs than prompts")
    return outputs


def extract_label(output: str) -> int:
    """Extract 0, 1, or 2 label from model output."""
    output = THINK_BLOCK_RE.sub(" ", output)
    output = THINK_TAG_RE.sub(" ", output)
    matches = LABEL_RE.findall(output)
    if not matches:
        raise ValueError(f"model output did not contain a 0/1/2 label: {output!r}")
    label = int(matches[-1])
    if label not in (0, 1, 2):
        raise ValueError(f"model output contained invalid label {label}: {output!r}")
    return label


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


def load_existing_ids(out_dir: Path) -> tuple[set[str], int, int, int]:
    seen_ids = set()
    valid_rows = 0
    bad_rows = 0
    duplicate_rows = 0

    for path in sorted(out_dir.glob("labels_*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    entry = json.loads(line)
                    entry_id_key = str(entry["id"])
                except Exception as exc:
                    bad_rows += 1
                    print(
                        f"warning: ignoring existing malformed row {path}:{line_number}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                if entry_id_key in seen_ids:
                    duplicate_rows += 1
                    continue

                seen_ids.add(entry_id_key)
                valid_rows += 1

    return seen_ids, valid_rows, bad_rows, duplicate_rows


def iter_entries(input_files: list[Path]) -> Iterable[tuple[Path, int, str]]:
    for input_file in input_files:
        print(f"processing {input_file}", file=sys.stderr)
        with input_file.open("r", encoding="utf-8") as src:
            for line_number, line in enumerate(src, start=1):
                yield input_file, line_number, line


def sample_line_from_file(input_file: Path) -> tuple[int, str]:
    sampled_line_number = 0
    sampled_line = None

    with input_file.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, start=1):
            if random.randrange(line_number) == 0:
                sampled_line_number = line_number
                sampled_line = line

    if sampled_line is None:
        raise ValueError("file is empty")
    return sampled_line_number, sampled_line


def iter_random_entries(input_files: list[Path], target_count: int) -> Iterable[tuple[Path, int, str]]:
    attempts = 0
    max_attempts = max(target_count * 20, 100)

    while attempts < max_attempts:
        attempts += 1
        input_file = random.choice(input_files)
        print(f"sampling {input_file}", file=sys.stderr)
        line_number, line = sample_line_from_file(input_file)
        yield input_file, line_number, line


def process_batch(
    batch: list[dict],
    client: openai.OpenAI,
    model_name: str,
) -> tuple[list[tuple[dict, int]], int]:
    prompts = [item["prompt"] for item in batch]

    try:
        outputs = classify_prompts(client, model_name, prompts)
    except Exception as exc:
        print(
            f"warning: batched request failed for {len(batch)} prompts, "
            f"falling back to single prompts: {exc}",
            file=sys.stderr,
        )
        outputs = []
        for item in batch:
            try:
                outputs.append(classify_prompt(client, model_name, item["prompt"]))
            except Exception as single_exc:
                outputs.append("")
                print(
                    f"warning: skipping {item['input_file']}:{item['line_number']}: {single_exc}",
                    file=sys.stderr,
                )

    labeled = []
    bad_outputs = 0
    for item, output in zip(batch, outputs):
        try:
            reasoning = extract_label(output)
        except Exception as exc:
            bad_outputs += 1
            print(
                f"warning: skipping {item['input_file']}:{item['line_number']}: {exc}",
                file=sys.stderr,
            )
            continue

        labeled.append((item, reasoning))

    return labeled, bad_outputs


def main() -> int:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.is_dir():
        print(f"error: --in_dir is not a directory: {in_dir}", file=sys.stderr)
        return 2
    if args.n_examples is not None and args.n_examples < 0:
        print("error: --n_examples must be non-negative", file=sys.stderr)
        return 2
    if args.batch_size <= 0:
        print("error: --batch_size must be positive", file=sys.stderr)
        return 2
    if args.random_sample and args.n_examples is None:
        args.n_examples = 1

    input_files = sorted(in_dir.rglob("*.jsonl"))
    if not input_files:
        print(f"error: no .jsonl files found in {in_dir}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    seen_ids, existing_valid_rows, existing_bad_rows, existing_duplicate_rows = load_existing_ids(
        out_dir
    )
    if existing_valid_rows:
        print(
            f"resume: found {existing_valid_rows} existing labeled rows in {out_dir}",
            file=sys.stderr,
        )

    try:
        model_name = get_model_name(SERVER)
    except Exception as exc:
        print(f"error: could not reach local model server at {SERVER}: {exc}", file=sys.stderr)
        return 1
    client = openai.OpenAI(
        base_url=f"{SERVER}/v1",
        api_key="sk-no-key-required",
        timeout=120,
    )

    classified = 0
    lines_seen = 0
    bad_lines = 0
    bad_outputs = 0
    duplicate_ids = 0
    total_valid_rows = existing_valid_rows
    out_file = None
    out_chunk_idx = None
    pending_batch = []
    pending_ids = set()

    try:
        if args.random_sample:
            entries = iter_random_entries(input_files, args.n_examples)
        else:
            entries = iter_entries(input_files)

        def write_labeled(item: dict, reasoning: int) -> None:
            nonlocal classified, total_valid_rows, out_file, out_chunk_idx

            seen_ids.add(str(item["id"]))
            chunk_idx = total_valid_rows // LINES_PER_CHUNK + 1
            if out_file is None or chunk_idx != out_chunk_idx:
                if out_file is not None:
                    out_file.close()
                out_path, out_file = open_output_chunk(out_dir, chunk_idx)
                out_chunk_idx = chunk_idx
                print(f"writing {out_path}", file=sys.stderr)

            labeled_entry = {
                "source": item["entry"].get("source_file", str(item["input_file"])),
                "id": item["id"],
                "prompt": item["prompt"],
                "reasoning": reasoning,
            }
            json.dump(labeled_entry, out_file, ensure_ascii=False)
            out_file.write("\n")
            classified += 1
            total_valid_rows += 1

            if classified % 100 == 0:
                print(f"classified {classified}", file=sys.stderr)

        def flush_batch() -> None:
            nonlocal bad_outputs, pending_batch, pending_ids

            if not pending_batch:
                return

            labeled_items, batch_bad_outputs = process_batch(
                pending_batch,
                client,
                model_name,
            )
            bad_outputs += batch_bad_outputs
            for item, reasoning in labeled_items:
                write_labeled(item, reasoning)
            pending_batch = []
            pending_ids = set()

        for input_file, line_number, line in entries:
            if args.n_examples is not None and classified >= args.n_examples:
                break

            lines_seen += 1
            try:
                entry = json.loads(line)
                prompt = entry["prompt"]
                if not isinstance(prompt, str):
                    raise ValueError("'prompt' is not a string")
                entry_id = entry["id"]
            except Exception as exc:
                bad_lines += 1
                print(
                    f"warning: skipping {input_file}:{line_number}: {exc}",
                    file=sys.stderr,
                )
                continue

            entry_id_key = str(entry_id)
            if entry_id_key in seen_ids or entry_id_key in pending_ids:
                duplicate_ids += 1
                continue

            pending_batch.append(
                {
                    "entry": entry,
                    "id": entry_id,
                    "prompt": prompt,
                    "input_file": input_file,
                    "line_number": line_number,
                }
            )
            pending_ids.add(entry_id_key)
            if (
                len(pending_batch) >= args.batch_size
                or (
                    args.n_examples is not None
                    and classified + len(pending_batch) >= args.n_examples
                )
            ):
                flush_batch()

        flush_batch()
    finally:
        if out_file is not None:
            out_file.close()

    if args.random_sample and args.n_examples is not None and classified < args.n_examples:
        print(
            f"warning: random sampling stopped after {classified}/{args.n_examples} examples",
            file=sys.stderr,
        )

    print(f"input files found: {len(input_files)}")
    print(f"input lines seen: {lines_seen}")
    print(f"existing valid rows found: {existing_valid_rows}")
    print(f"existing malformed rows ignored: {existing_bad_rows}")
    print(f"existing duplicate ids ignored: {existing_duplicate_rows}")
    print(f"bad lines skipped: {bad_lines}")
    print(f"bad model outputs skipped: {bad_outputs}")
    print(f"duplicate ids skipped: {duplicate_ids}")
    print(f"examples classified: {classified}")
    print(f"output dir: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
