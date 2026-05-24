#!/usr/bin/env python3
"""Embed prompt fields from JSONL files into compressed NPZ files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm


DEFAULT_TOKENIZER = "sentence-transformers/all-MiniLM-L12-v2"
EMBEDDING_MAX_LENGTH = 256
EMBEDDING_DTYPE = np.float32

"""
python3 embed_prompts.py --in_dir data/deepseek_labels --out_dir data/embeddings --model_path ../gte_small_onnx 
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed JSONL prompt fields and save compressed NPZ files."
    )
    parser.add_argument(
        "--in_dir",
        required=True,
        help="Directory containing input .jsonl files.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory where .npz embedding files will be written.",
    )
    parser.add_argument(
        "--model_path",
        default="../gte_small_onnx",
        help="Embedding model directory (ONNX or SentenceTransformer). Defaults to gte-small.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Prompts to embed per batch. Defaults to 256.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: skipping {path}:{line_number}: {exc}", file=sys.stderr)


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def clean_prompt_text(prompt: object) -> str:
    return str(prompt).strip()


def mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)


class SentenceTransformerEncoder:
    def __init__(self, model_path: Path):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(model_path))
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = EMBEDDING_MAX_LENGTH

    def encode(self, prompts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            [clean_prompt_text(prompt) for prompt in prompts],
            batch_size=len(prompts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(EMBEDDING_DTYPE)


class OnnxMiniLMEncoder:
    def __init__(self, model_path: Path):
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ONNX embedding requires onnxruntime and transformers. "
                "Install onnxruntime in the environment or pass a full "
                "SentenceTransformer model directory."
            ) from exc

        onnx_path = self.resolve_onnx_path(model_path)
        tokenizer_path = model_path if model_path.is_dir() else model_path.parent

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path), local_files_only=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)

        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    @staticmethod
    def resolve_onnx_path(model_path: Path) -> Path:
        if model_path.is_file() and model_path.suffix == ".onnx":
            return model_path

        candidates = sorted(model_path.glob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"no .onnx file found in {model_path}")

        int8_candidates = [path for path in candidates if "int8" in path.name.lower()]
        return int8_candidates[0] if int8_candidates else candidates[0]

    def encode(self, prompts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            [clean_prompt_text(prompt) for prompt in prompts],
            padding=True,
            truncation=True,
            max_length=EMBEDDING_MAX_LENGTH,
            return_tensors="np",
        )
        feed = {
            name: encoded[name]
            for name in self.input_names
            if name in encoded
        }
        outputs = self.session.run(None, feed)
        embeddings = outputs[0]
        if embeddings.ndim == 3:
            embeddings = mean_pool(embeddings, encoded["attention_mask"])
        return normalize(embeddings.astype(EMBEDDING_DTYPE))


def load_encoder(model_path: Path) -> SentenceTransformerEncoder | OnnxMiniLMEncoder | OptimumONNXEncoder:
    # Check for optimum-style ONNX (has config.json + model.onnx)
    if model_path.is_dir() and (model_path / "config.json").exists():
        config = json.load((model_path / "config.json").open())
        if any("Bert" in a or "Roberta" in a or "XLMRoberta" in a for a in config.get("architectures", [])):
            return OptimumONNXEncoder(model_path)
    if model_path.is_file() and model_path.suffix == ".onnx":
        return OnnxMiniLMEncoder(model_path)
    if model_path.is_dir() and list(model_path.glob("*.onnx")):
        return OnnxMiniLMEncoder(model_path)
    return SentenceTransformerEncoder(model_path)


class OptimumONNXEncoder:
    """Encoder for optimum-exported ONNX models (gte-small, etc.)."""
    def __init__(self, model_path: Path):
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            str(model_path), provider="CPUExecutionProvider"
        )

    def encode(self, prompts: list[str]) -> np.ndarray:
        texts = [clean_prompt_text(p) for p in prompts]
        tokens = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=EMBEDDING_MAX_LENGTH, return_tensors="np",
        )
        outputs = self.model(**tokens)
        mask = tokens["attention_mask"][..., None].astype(np.float32)
        emb = (outputs.last_hidden_state * mask).sum(axis=1) / np.clip(
            mask.sum(axis=1), 1e-9, None
        )
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        return emb.astype(EMBEDDING_DTYPE)


def load_file(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    prompts: list[str] = []
    ids: list[int] = []
    labels: list[int] = []
    skipped = 0

    for line_number, row in iter_jsonl(path):
        try:
            prompts.append(clean_prompt_text(row["prompt"]))
            ids.append(int(row["id"]))
            labels.append(int(row["reasoning"]))
        except Exception as exc:
            skipped += 1
            print(f"warning: skipping {path}:{line_number}: {exc}", file=sys.stderr)

    return prompts, np.asarray(ids), np.asarray(labels), skipped


def embed_file(
    input_path: Path,
    output_path: Path,
    encoder: SentenceTransformerEncoder | OnnxMiniLMEncoder,
    batch_size: int,
) -> tuple[int, int]:
    prompts, ids, labels, skipped = load_file(input_path)
    if not prompts:
        print(f"warning: no valid prompts in {input_path}", file=sys.stderr)
        return 0, skipped

    batches = list(batched(prompts, batch_size))
    embeddings = []
    progress = tqdm(
        batches,
        desc=f"embedding {input_path.name}",
        unit="batch",
        leave=False,
    )
    for batch in progress:
        embeddings.append(encoder.encode(batch))

    stacked = np.vstack(embeddings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=stacked.astype(EMBEDDING_DTYPE),
        ids=ids.astype("int64"),
        reasoning=labels.astype("int8"),
        embedding_max_length=np.asarray(EMBEDDING_MAX_LENGTH, dtype="int64"),
        embedding_normalized=np.asarray(True),
    )
    return len(prompts), skipped


def main() -> int:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    model_path = Path(args.model_path)

    if not in_dir.is_dir():
        print(f"error: --in_dir is not a directory: {in_dir}", file=sys.stderr)
        return 2
    if not model_path.exists():
        print(f"error: --model_path does not exist: {model_path}", file=sys.stderr)
        return 2
    if args.batch_size <= 0:
        print("error: --batch_size must be positive", file=sys.stderr)
        return 2

    input_files = sorted(in_dir.rglob("*.jsonl"))
    if not input_files:
        print(f"error: no .jsonl files found in {in_dir}", file=sys.stderr)
        return 2

    try:
        encoder = load_encoder(model_path)
    except Exception as exc:
        print(f"error: failed to load model: {exc}", file=sys.stderr)
        return 1

    total_rows = 0
    total_skipped = 0
    for input_path in tqdm(input_files, desc="files", unit="file"):
        relative = input_path.relative_to(in_dir)
        output_path = (out_dir / relative).with_suffix(".npz")
        rows, skipped = embed_file(input_path, output_path, encoder, args.batch_size)
        total_rows += rows
        total_skipped += skipped
        print(f"saved {rows:,} embeddings: {output_path}")

    print(f"embedded rows: {total_rows:,}")
    print(f"skipped rows: {total_skipped:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
