#!/usr/bin/env python3
"""Router: classify prompt, then send to LLM with/without reasoning prompt.

Loads gte-small ONNX + RouterMLPv2 checkpoint.
Classifies prompt → below threshold: direct prompt, above threshold: reasoning prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch import nn
from dotenv import load_dotenv
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# ── model architecture (inline, no training deps) ────────────────────────────

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


class RouterMLP(nn.Module):
    def __init__(self, dim, hidden_dims=None, dropouts=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        if dropouts is None:
            dropouts = [0.35, 0.25, 0.15]
        blocks = []
        prev = dim
        for hd, dr in zip(hidden_dims, dropouts):
            blocks.append(ResBlock(prev, hd, hd, dr))
            prev = hd
        blocks.append(nn.LayerNorm(prev))
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)

# ── defaults ────────────────────────────────────────────────────────────────
LLM_SERVER = "http://127.0.0.1:8080"
BALANCED_THRESHOLD = 0.50
PREFER_ZERO_THRESHOLD = 0.60
DEFAULT_THRESHOLD = "balanced"

REASONING_SYSTEM_PROMPT = (
    "Think through this carefully step by step before answering."
)
DIRECT_SYSTEM_PROMPT = (
    "Answer directly and concisely. Do not think out loud."
)

# ── router ──────────────────────────────────────────────────────────────────

class Router:
    def __init__(self, onnx_dir: str, checkpoint_path: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        self.ort = ORTModelForFeatureExtraction.from_pretrained(
            onnx_dir, provider="CPUExecutionProvider"
        )

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = RouterMLP(384)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.balanced_threshold = float(ckpt.get("balanced_threshold", BALANCED_THRESHOLD))
        self.prefer_zero_threshold = float(
            ckpt.get("prefer_zero_threshold", PREFER_ZERO_THRESHOLD)
        )

    def embed(self, prompt: str) -> np.ndarray:
        tokens = self.tokenizer(
            [prompt], padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        out = self.ort(**tokens)
        mask = tokens["attention_mask"][..., None].astype(np.float32)
        emb = (out.last_hidden_state * mask).sum(axis=1) / np.clip(
            mask.sum(axis=1), 1e-9, None
        )
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        return emb.astype(np.float32)

    def classify(self, prompt: str) -> float:
        emb = self.embed(prompt)
        x = torch.from_numpy(emb).float()
        with torch.no_grad():
            score = torch.sigmoid(self.model(x)).item()
        return score

    def needs_reasoning(self, prompt: str, mode: str = "balanced") -> tuple[float, bool]:
        score = self.classify(prompt)
        threshold = (
            self.balanced_threshold if mode == "balanced" else self.prefer_zero_threshold
        )
        return score, score >= threshold


# ── LLM client ──────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, server: str, model_name: str | None = None):
        self.server = server.rstrip("/")
        self.model_name = model_name or self._detect_model()

    def _detect_model(self) -> str:
        resp = requests.get(f"{self.server}/v1/models", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "data" in data and data["data"]:
            return data["data"][0]["id"]
        if "models" in data and data["models"]:
            return data["models"][0].get("name") or data["models"][0].get("id")
        raise RuntimeError("No model found at /v1/models")

    def chat(self, prompt: str, system: str, max_tokens: int = 512) -> dict:
        resp = requests.post(
            f"{self.server}/v1/chat/completions",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()


# ── routing ─────────────────────────────────────────────────────────────────

def route_prompt(
    router: Router,
    llm: LLMClient,
    prompt: str,
    threshold_mode: str = "balanced",
    verbose: bool = True,
) -> dict:
    score, needs = router.needs_reasoning(prompt, threshold_mode)
    system = REASONING_SYSTEM_PROMPT if needs else DIRECT_SYSTEM_PROMPT
    label = "REASON" if needs else "DIRECT"

    if verbose:
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        print(f"[{label:7s}] {score:.3f} {bar}")

    response = llm.chat(prompt, system)
    content = response["choices"][0]["message"]["content"]

    if verbose:
        print(f"  → {content[:200]}")

    return {
        "score": score,
        "needs_reasoning": needs,
        "label": label,
        "system": system,
        "response": content,
    }


# ── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Router: classify + route to LLM.")
    p.add_argument("--onnx_dir", default="gte_small_onnx", help="gte-small ONNX model dir")
    p.add_argument("--checkpoint", default="runs/gte_24k/best_model.pt", help="Router checkpoint")
    p.add_argument("--server", default=LLM_SERVER, help="LLM server URL")
    p.add_argument("--threshold", default=DEFAULT_THRESHOLD, choices=["balanced", "prefer_zero"])
    p.add_argument("--prompt", default=None, help="Single prompt to route")
    p.add_argument("--test", action="store_true", help="Run built-in test suite")
    return p.parse_args()


TEST_PROMPTS = [
    # Clear reasoning
    ("REASON", "Debug this race condition: two goroutines write to the same map without a mutex."),
    ("REASON", "Compare quicksort vs mergesort for an already-sorted array."),
    ("REASON", "A patient has chest pain, shortness of breath, and radiating arm pain. What is the most likely diagnosis?"),
    ("REASON", "Design a database schema for a hospital patient management system."),
    ("REASON", "Plan a 7-day itinerary for a family of four visiting Tokyo on a budget."),
    ("DIRECT", "What is the capital of France?"),
    ("DIRECT", "Translate hello to Spanish."),
    ("DIRECT", "Write a haiku about autumn leaves."),
    ("DIRECT", "What year did World War II end?"),
    ("DIRECT", "Define ephemeral in one sentence."),
    # Nuanced
    ("REASON", "Why might a Kubernetes pod keep restarting despite clean logs?"),
    ("REASON", "Is it better to use a list comprehension or a for loop for filtering in Python when performance matters?"),
    ("REASON", "Explain the CAP theorem to a product manager with no technical background."),
    ("REASON", "My PyTorch model gets NaN losses on GPU but works fine on CPU. What could cause this?"),
    ("REASON", "Should I use a microservices architecture for a simple e-commerce site with 100 daily users?"),
    ("DIRECT", "List three species of penguins."),
    ("DIRECT", "Tell me a knock-knock joke."),
    ("DIRECT", "What is 15 times 27?"),
    ("DIRECT", "Convert 100 dollars to Japanese yen."),
    ("DIRECT", "Summarize: the quick brown fox jumps over the lazy dog."),
]


def run_tests(router: Router, llm: LLMClient, threshold_mode: str):
    print(f"Threshold: {threshold_mode} | Model: {llm.model_name}")
    print()

    correct = 0
    total = 0
    for expected, prompt in TEST_PROMPTS:
        result = route_prompt(router, llm, prompt, threshold_mode, verbose=True)
        total += 1
        if (expected == "REASON" and result["needs_reasoning"]) or (
            expected == "DIRECT" and not result["needs_reasoning"]
        ):
            correct += 1
        print()

    print(f"Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"Server: {llm.server}")


def main():
    load_dotenv()
    args = parse_args()

    print(f"Loading router...")
    router = Router(args.onnx_dir, args.checkpoint)
    llm = LLMClient(args.server)

    if args.prompt:
        result = route_prompt(router, llm, args.prompt, args.threshold)
        print(f"\nScore: {result['score']:.4f}")
        print(f"Decision: {result['label']}")
        print(f"Response:\n{result['response']}")
    elif args.test:
        run_tests(router, llm, args.threshold)
    else:
        run_tests(router, llm, args.threshold)


if __name__ == "__main__":
    main()
