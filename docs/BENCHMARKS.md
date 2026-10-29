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
- B=32 bf16 MPP decode: 544 tok/s aggregate (58.8 ms/step) in `batch-test`.
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
- int8 KV and radix prefix cache are not implemented.
- Server batch path uses bf16 MPP; over 8 prompts x 32 tokens it agrees with
  the exact 4-bit path on 89.1% of tokens. `MIORI_EXACT=1` forces the exact
  single-sequence path.
