# C7 — release

- Date: 2026-10-29
- Status: **PARTIAL** — v0.1.0-core, not feature-complete

## Delivered

- `build/miori` with `run`, `serve`, `info`, `test-*` commands.
- OpenAI-compatible HTTP + SSE.
- Full 24-layer LFM3 text model, tokenizer, chat template, sampling.
- 4-bit GEMV decode and MPP tensor-core prefill.
- Kernel unit tests and this checkpoint set.

## Not delivered

- Paged KV / continuous batching (C2).
- Vision (C4).
- int8 KV / prefix cache / shared-prefix attention / ICB (C5).
- 24h soak and full KL matrix (C6).

## Reproduce

```bash
make -j
./build/miori run --raw --prompt "The capital of France is" --max-tokens 32
./build/miori serve --port 8080
```
