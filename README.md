# Miori

Metal + C++20 inference engine for `LiquidAI/LFM3-VL-2B-MLX-4bit` on Apple M6 / 16 GB.
No Python, no Objective-C, no MLX runtime: hand-written Metal kernels and a C++ host.

## Status (v0.1.0-core)

| Area | State |
|---|---|
| Text model (24 layers, 4-bit + 8-bit embed) | working |
| Tokenizer (128K BPE) + LFM3 chat template | working |
| Prefill (MPP tensor-core GEMM, bf16 weight cache) | working, ~2.7K tok/s @1K |
| Decode (4-bit GEMV) | working, ~97 tok/s |
| OpenAI-compatible HTTP + SSE | working, serialized |
| Paged KV / continuous batching | **not implemented** |
| Vision tower | **not implemented** |
| int8 KV / prefix cache / shared-prefix attention | **not implemented** |

Measured on the target M6/16GB (headless, 3-run median where applicable):

```
prefill 1002 tokens: 0.372 s = 2694 tok/s
decode 128 tokens:   1.311 s =  97.5 tok/s  (~118 GB/s effective)
server:              OpenAI /v1/chat/completions + SSE
```

## Build

Requires macOS 27.x, Xcode 27.1, Metal 4.1.

```bash
make -j
```

## Model

Place these files in `models/`:

```
model.safetensors        # 1.59 GB, from LiquidAI/LFM3-VL-2B-MLX-4bit
tokenizer.json
config.json
chat_template.jinja
```

The model is loaded with a C++ safetensors parser; no conversion step is required.
The engine builds a bf16 tensor-core cache (2.93 GB) at startup and keeps the
original 4-bit weights for decode.

## Run

```bash
./build/miori run --raw --prompt "The capital of France is" --max-tokens 32
./build/miori run --prompt "Explain RMSNorm." --max-tokens 128
./build/miori serve --port 8080
```

OpenAI API:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
```

## Tests

```bash
./build/miori test-gemv4      # 4-bit GEMV vs CPU dequant
./build/miori test-gemv8      # 8-bit GEMV vs CPU dequant
./build/miori test-gemm       # GEMM vs CPU
./build/miori test-attn       # decode attention vs CPU
./build/miori test-prefill-attn
./build/miori test-conv
./build/miori test-tokenizer
./build/miori bench --prefill 1024 --decode 128
```

## Known limitations

- Single sequence per process; the HTTP server serializes requests behind a mutex.
  The 64×2K / 64×4K concurrency targets from `docs/MILESTONES.md` are not met.
- No paged KV, no prefix cache, no int8 KV.
- Vision weights are loaded but the vision tower is not executed; image inputs
  are not yet supported by the API.
- Prefill chunks are 128 tokens; the bf16 cache costs 2.93 GB of the 12.7 GB
  working set.
- Greedy decoding is exact; temperature/top-k/top-p are implemented on the CPU.

## Docs

- [docs/README.md](docs/README.md) — frozen decisions and targets
- [docs/MILESTONES.md](docs/MILESTONES.md) — M0–M7 plan
- [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) — gates and measurement protocol
- [docs/checkpoints/](docs/checkpoints/) — actual checkpoint reports (C0–C7)

## Benchmark

```bash
./build/miori bench --prefill 1024 --decode 128
```

Reported on the target M6: prefill 2542 tok/s, decode 94.1 tok/s at 1K context
(97.5 tok/s at short context), ~114 GB/s effective.

4K context: prefill 1663.7 tok/s, decode 92.2 tok/s.

## Metrics

`GET /metrics` exposes `miori_requests_total` and `miori_output_tokens_total`.

## Perplexity

```bash
./build/miori ppl --file /tmp/en.txt --tokens 200
```
