# Miori benchmarks (v0.2.0)

Machine: Apple M6 Mac mini, 12 GPU cores, 16 GB unified memory, macOS 27.0.1,
Xcode 27.1. Model: `LiquidAI/LFM3-VL-2B-MLX-4bit` (1.591 GB).

All numbers are from this repository's binaries; commands are reproducible.

## Text, single sequence

| context | prefill | decode |
|---|---:|---:|
| 512 | 3118 tok/s | 90.1 tok/s |
| 1024 | ~2800 tok/s | 94.1 tok/s |
| 2048 | 2302 tok/s | 93.4 tok/s |
| 4096 | 1664 tok/s | 92.2 tok/s |
| short (2-token prompt) | — | **98.0 tok/s** |

Decode uses 4-bit `gemv4` with 16-way split-K attention for full layers at
`pos >= 512`. Prefill uses 4-bit weights pre-expanded to a 2.93 GB bf16
tile-major cache and MPP `matmul2d` (64-row tiles), 128-token chunks.

## Batch decode / server

- `BatchEngine` Bmax=32; packed multi-request prefill (`add_batch`), up to 32
  requests / 1024 tokens per pass. Packed prefill is bit-identical to the
  sequential path (first-token logits, KV, conv state).
- 64 short prompts (~23 tokens), 16 greedy tokens:

| server | wall |
|---|---:|
| sequential `add_sequence` | 2.94 s |
| packed prefill (default) | 1.98–2.11 s |

- `batch-prefill-test --n 64`: 2.095 s sequential → 0.664 s packed (3.15x,
  2234 tok/s prefill, 10.4 ms/request).
- B=32 bf16 MPP decode: **684 tok/s aggregate (46.8 ms/step)** after dynamic-M.
- B=4 exact batch: **19.6 ms/step** via the exact gemv path (0.29 s for 4
  concurrent HTTP, 16 tokens).
- B=4 exact 4-bit batch: 261 tok/s aggregate (15.3 ms/step, token-identical).
- Small queues (<4) are served by the single-sequence exact path.

## Vision

| resized | image tokens | cold | warm |
|---|---:|---:|---:|
| 256×224 | 56 | 14.4 ms | 8.5–13.6 ms |
| 512×480 | 240 | 42.8 ms | 33.4–36.1 ms |
| 1024×992 | 992 | 284.9 ms | 270.5–272.5 ms |
| 1600×1600 | 2500 | 1751.8 ms | 1728–1747 ms |

Tower + projector run bf16 MPP tensor-core GEMMs; CPU fp32 reference worst
per-token cosine 0.999706, max |GPU−CPU| 0.00462. Image splice is bit-identical
to the token path (`max |Δlogit| = 0`).

## int8 KV (opt-in, `MIORI_KV_INT8=1`)

Symmetric int8 K/V with one bf16 scale per `(kv_head, token)`; bf16 stays the
default. Measured 2026-10-29 (`docs/checkpoints/C5-int8-kv.md`).

| item | bf16 | int8 (per-32 group scales) |
|---|---:|---:|
| Engine KV / sequence | 92.27 MB | 49.02 MB |
| BatchEngine KV, Bmax=32 | 2952.8 MB | 1568.7 MB |
| attention KV payload / token @4K | 369.1 MB | 184.5 MB (+2.9 MB scales) |
| decode 4K (128 tok) | 87.5 tok/s | 87.6 tok/s |
| prefill 4K | 1630 tok/s | 1620 tok/s |
| decode 1K | 91.8 tok/s | 91.8 tok/s |
| PPL `docs/README.md` (1885 pos) | 19.7301 | 19.7364 (+0.032 %) |
| 4-domain 1K PPL / top-1 / mean KL | — | −0.067 % / 99.34 % / 2.2e-4 |

No decode speedup: single-stream decode at 4K is not KV-bandwidth-bound. The
gain is capacity (1.38 GB less batch KV). The strict 99.5 % top-1 project gate
is still not met, so the flag remains off by default; see
`docs/checkpoints/C5-int8-kv-groups.md`.

## Caches

| cache | cold | warm |
|---|---:|---:|
| system-prefix KV snapshot (647 tok) | 0.349 s | 0.042 s |
| image features, 992 tokens | 0.964 s | 0.534 s |

## Perplexity

| corpus | tokens | PPL |
|---|---:|---:|
| `The capital of France is Paris.` | 7 | 18.60 |
| 5-sentence English paragraph | 81 | 42.53 |
| `docs/README.md` (EN/ZH/Markdown) | 127 | 94.79 |

## Known gaps

- No paged KV; packed prefill blocks decode (no prefill/decode overlap).
- 64×2K / 64×4K aggregate target (2.5K output tok/s) is not met.
- Vision is one image at a time in the tower; multi-image requests encode
  sequentially. No ragged vision batch or vision queue.
- int8 KV is implemented (per-32 group scales) but opt-in and misses the
  99.5 % top-1 gate (99.34 % aggregate, PPL −0.07 %, mean KL 2.2e-4); radix
  prefix cache and shared-prefix attention are not implemented.
- Server batch path uses bf16 MPP; over 8 prompts x 32 tokens it agrees with
  the exact 4-bit path on 89.1% of tokens. `MIORI_EXACT=1` forces the exact
  single-sequence path.

## Streaming

- Batch greedy: one SSE delta per decode step (scheduler publishes new tokens).
- `MIORI_EXACT=1` and temperature sampling: one delta per sampled token
  (`Engine::generate_from_state_stream`).
- Image requests: SSE header after vision encode, then one delta per token
  (`Engine::decode_loop_stream`).
- 8 concurrent stream requests, 16 tokens: 0.78 s wall.

## Prefix cache

4 full-prefix snapshots (KV+conv, ~92 MB each) with LRU by system prompt.
Alternating two system prompts: A cold 0.118 s, B cold 0.084 s, A warm 0.043 s,
B warm 0.054 s (exact mode, 8 output tokens).
