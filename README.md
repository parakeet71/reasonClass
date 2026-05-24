# Reason-Class Router

Classify whether a prompt needs LLM reasoning. Routes to direct or reasoning model.

## Quick Start

```bash
pip install torch numpy optimum[onnxruntime] transformers requests python-dotenv

python router_service.py --prompt "Debug this race condition in my Go code"
```

### How it works

```
Prompt → gte-small ONNX embedding (50ms) → RouterMLP (1ms) → Score
                                                          ├── < 0.5  → Direct LLM
                                                          └── ≥ 0.5  → Reasoning LLM
```

## Files

`train/` contains retraining scripts and 30K labeled data. See `train/` for details.

| File | Purpose |
|------|---------|
| `proxy.py` | **Zero-config proxy** — sits in front of llama.cpp, auto-classifies every prompt |
| `router_service.py` | Inference service — classify + route to any OpenAI-compatible LLM |
| `best_model.pt` | RouterMLP checkpoint (368K params) |
| `gte_small_onnx/` | gte-small embedding model (ONNX, 128MB) |

## Usage

### Proxy mode (recommended for llama.cpp web UI)

```bash
# Terminal 1: start llama.cpp (default settings — proxy handles the system prompt)
llama-server -m your-model.gguf --port 8080 --host 127.0.0.1 --n-gpu-layers 99 --ctx-size 4096 --mlock

# Terminal 2: start the proxy
python proxy.py --port 8081 --backend http://127.0.0.1:8080

# Point your web UI at http://127.0.0.1:8081
```

The proxy intercepts every `/v1/chat/completions` request, classifies the prompt,
and injects the appropriate system prompt. Everything else (models list, tokenization,
etc.) passes through unchanged. Streaming works — tokens appear as they're generated.

```bash
python proxy.py --threshold 0.60       # stricter routing (prefer zero)
python proxy.py --quiet                # suppress classification logging
```

### CLI mode (single prompts)

```bash
python router_service.py --prompt "Your prompt here"

# Point to your LLM server
python router_service.py --server http://localhost:8080 --prompt "..."

# Use stricter threshold (fewer false positives)
python router_service.py --threshold prefer_zero --prompt "..."

# Run built-in test suite (20 prompts)
python router_service.py --test

# Custom model paths
python router_service.py --onnx_dir /path/to/gte_small_onnx --checkpoint /path/to/best_model.pt --prompt "..."
```

### Python API

```python
from router_service import Router, LLMClient, route_prompt

router = Router("gte_small_onnx", "best_model.pt")
llm = LLMClient("http://127.0.0.1:8080")

result = route_prompt(router, llm, "Debug this race condition")
print(result["score"])         # 0.535 — probability of needing reasoning
print(result["needs_reasoning"]) # True
print(result["response"])      # LLM output with reasoning prompt
```

## Architecture

```
RouterMLP (368K params):
  Input: 384-dim gte-small embedding
  ResBlock(384 → 256, dropout=0.35)
  ResBlock(256 → 128, dropout=0.25)
  ResBlock(128 → 64,  dropout=0.15)
  LayerNorm → Linear(64 → 1) → Sigmoid
```

## Performance

Trained on 24K DeepSeek-v4-flash-labeled prompts with gte-small embeddings.

| Threshold | Accuracy | F1 |
|-----------|:--------:|:---:|
| Balanced (0.50) | 96.7% | 0.708 |
| Prefer-zero (0.60) | 90.0% | 0.688 |

## Benchmarks

Measured on AMD Radeon RX 9070 XT (llama.cpp + ROCm).

### Classifier Latency (200 samples)

| Component | Avg | Median |
|-----------|----:|-------:|
| Embedding (gte-small ONNX) | 3.3ms | 2.5ms |
| RouterMLP forward pass | 0.4ms | 0.3ms |
| **Total classifier** | **3.7ms** | — |

The classifier overhead is constant regardless of LLM size.

### End-to-End: SmolLM2-360M

Small and fast. Good baseline for throughput.

| Path | Classifier | LLM | Total | Tokens |
|------|----------:|----:|------:|-------:|
| Direct | 3.7ms | 0.1s | **0.1s** | 16 |
| Reasoning | 3.7ms | 2.8s | **2.8s** | 479 |

- Classifier: 2.5% of total latency
- Saved vs always-reason: 2.6s per direct prompt

### End-to-End: Qwen3.5-9B

Larger model. Qwen is very verbose in reasoning, even when unnecessary.

| Path | Classifier | LLM | Total | Tokens |
|------|----------:|----:|------:|-------:|
| Direct | 3.7ms | 27.2s | **27.2s** | 262 |
| Reasoning | 3.7ms | 45.6s | **45.6s** | 789 |

- Classifier: **0.01%** of total latency (one ten-thousandth)
- Saved vs always-reason: **18.5s** per direct prompt
- Router accuracy on 10 test prompts: **10/10**

### Key Takeaway

For small models the router is nice. For large verbose models it's essential — it saves 18 seconds per prompt that doesn't need reasoning, and the overhead is only 0.01% of response time.

## Costs

Prompt labeling took about 3.8$ in deepseek-v4-flash API credits. 

## Limitations

- **Binary only.** This model outputs 0 or 1 — it cannot express "context-dependent" (class 2). The 3-class model (`train/train_router_3class.py`) addresses this.

- **No conversation history.** The classifier sees only the current prompt, not prior messages. Multi-turn cues ("no, change that to use a list") are judged on the prompt text alone. Words like "fix" or "explain" trigger reasoning regardless of context — usually what you want, but not always.

- **Embedding model quality.** The router is only as good as the embeddings. gte-small (384-dim) works well for English; multilingual prompts may degrade. Upgrading to a larger embedding model (gte-base, 768-dim) would improve accuracy at the cost of ~2-3x latency.

- **Single-region training data.** Trained on a subset of WildChat-4.8M, a dataset of chatbot conversations. Retrain with `train/deepseek_classify_prompt.py` on your own data. 

## Requirements

- Python ≥ 3.10
- PyTorch, NumPy
- optimum[onnxruntime], transformers
- requests, python-dotenv
