#!/usr/bin/env python3
"""Transparent proxy that auto-classifies prompts for llama.cpp.

Run this instead of llama-server. Listens on port 8081, classifies every
prompt, injects the right system prompt, and forwards to port 8080.

Start the backend server with --reasoning off to avoid <think> tokens:
  llama-server -m model.gguf --reasoning off --port 8080 ...

Then:
  python proxy.py --port 8081 --backend http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# ── model ───────────────────────────────────────────────────────────────────

HIDDEN_DIMS = [256, 128, 64]
DROPOUTS = [0.35, 0.25, 0.15]


class ResBlock(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.linear_in = nn.Linear(d_in, d_hidden)
        self.dropout_in = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_hidden, d_out)
        self.dropout_out = nn.Dropout(dropout * 0.7)
        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        r = self.shortcut(x)
        h = self.norm(x)
        h = F.silu(self.linear_in(h))
        h = self.dropout_in(h)
        h = self.linear_out(h)
        h = self.dropout_out(h)
        return h + r


class RouterMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        blocks = []
        prev = dim
        for hd, dr in zip(HIDDEN_DIMS, DROPOUTS):
            blocks.append(ResBlock(prev, hd, hd, dr))
            prev = hd
        blocks.append(nn.LayerNorm(prev))
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


# ── classifier ─────────────────────────────────────────────────────────────

class Classifier:
    def __init__(self, onnx_dir: str, checkpoint: str, threshold: float):
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        self.ort = ORTModelForFeatureExtraction.from_pretrained(
            onnx_dir, provider="CPUExecutionProvider"
        )
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = RouterMLP(384)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.threshold = threshold

    def classify(self, prompt: str) -> tuple[float, bool]:
        tokens = self.tokenizer(
            [prompt], padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        out = self.ort(**tokens)
        mask = tokens["attention_mask"][..., None].astype(np.float32)
        emb = (out.last_hidden_state * mask).sum(axis=1) / np.clip(
            mask.sum(axis=1), 1e-9, None
        )
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        x = torch.from_numpy(emb).float()
        with torch.no_grad():
            score = torch.sigmoid(self.model(x)).item()
        return score, score >= self.threshold


REASON_SYSTEM = "Think through this carefully step by step before answering."
DIRECT_SYSTEM = "Answer directly and concisely. Do not think out loud."

# ── proxy ───────────────────────────────────────────────────────────────────

THINK_RE = re.compile(r"<\s*think\b[^>]*>.*?<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)


def strip_thinking(content: str) -> str:
    """Strip <think>...</think> blocks from Qwen responses."""
    return THINK_RE.sub("", content).strip()


class ProxyHandler(BaseHTTPRequestHandler):
    classifier: Classifier = None
    backend: str = ""
    verbose: bool = False

    def log_message(self, fmt, *args):
        if self.verbose:
            super().log_message(fmt, *args)

    def _proxy_get(self, method, path, body=None):
        url = f"{self.backend}{path}"
        hdrs = {k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "transfer-encoding")}
        if body:
            hdrs["Content-Length"] = str(len(body))
        else:
            hdrs["Content-Length"] = "0"

        req = Request(url, data=body, headers=hdrs, method=method)
        with urlopen(req, timeout=300) as resp:
            self.send_response(resp.status)
            # Strip Hop-by-hop headers
            skip = {k.lower() for k in ("transfer-encoding", "content-encoding", "server", "date", "connection")}
            for k, v in resp.getheaders():
                if k.lower() in skip:
                    continue
                self.send_header(k, v)
            self.end_headers()

            # Stream the response in chunks
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break

    def do_POST(self):
        try:
            self._do_post()
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Proxy error: {e}".encode())
            if self.verbose:
                import traceback
                traceback.print_exc()

    def _do_post(self):
        if self.path != "/v1/chat/completions":
            self._proxy_get("POST", self.path, self._read_body())
            return

        body = self._read_body()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._proxy_get("POST", self.path, body)
            return

        messages = data.get("messages", [])
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        if not user_msgs:
            self._proxy_get("POST", self.path, body)
            return

        prompt = user_msgs[-1]
        t0 = time.perf_counter()
        score, needs_reason = self.classifier.classify(prompt)
        elapsed = (time.perf_counter() - t0) * 1000

        system = REASON_SYSTEM if needs_reason else DIRECT_SYSTEM
        label = "REASON" if needs_reason else "DIRECT"

        if self.verbose:
            bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
            print(f"[{label:7s}] {score:.3f} {bar}  {elapsed:.0f}ms  {prompt[:80]}")

        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages.insert(0, {"role": "system", "content": system})
            data["messages"] = messages
            body = json.dumps(data).encode("utf-8")

        self._proxy_get("POST", self.path, body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def do_GET(self):
        self._proxy_get("GET", self.path)

    def do_OPTIONS(self):
        self._proxy_get("OPTIONS", self.path)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Classifying proxy for llama.cpp")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--backend", default="http://127.0.0.1:8080")
    p.add_argument("--onnx_dir", default="gte_small_onnx")
    p.add_argument("--checkpoint", default="best_model.pt")
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    print("Loading classifier...")
    classifier = Classifier(args.onnx_dir, args.checkpoint, args.threshold)
    print(f"Classifier ready (threshold={args.threshold:.2f})")

    ProxyHandler.classifier = classifier
    ProxyHandler.backend = args.backend.rstrip("/")
    ProxyHandler.verbose = not args.quiet

    server = HTTPServer(("127.0.0.1", args.port), ProxyHandler)
    print(f"Proxy listening on http://127.0.0.1:{args.port}")
    print(f"Forwarding to {args.backend}")
    print(f"Point your web UI at http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
