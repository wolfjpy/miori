# C1 — single-request text

- Date: 2026-10-29
- Status: **GO with exceptions** (decode 97.5 vs target ≥100 tok/s; full CPU top-1 gate not run)

## Results

| Item | Target | Measured | Verdict |
|---|---:|---:|---|
| Decode | ≥100 tok/s | **97.5 tok/s** (128 tokens, median of 3) | marginal fail |
| Prefill | ≥2.5K tok/s | **2694 tok/s** (1002 tokens) | pass |
| Output sanity | factual | `The capital of France is` → `Paris` | pass |
| Arithmetic | coherent | `2,4,6,8,` → `10,12` | pass |
| Chat template | exact | `What is the capital of France?` → `Paris.` | pass |
| Tokenizer | round-trip | 10 MB mixed corpus not run; sample exact | partial |
| Top-1 vs CPU | ≥99.5% | not measured (no full CPU reference) | exception |
| Memory | ≤11 GB | ~4.7 GB (1.59 weights + 2.93 bf16 + 0.09 KV + activations) | pass |

## Notes

- Decode is 2.5% short of the 100 tok/s gate; effective bandwidth is ~118 GB/s of
  the ~125 GB/s sustained target. The remaining gap is dispatch/sampling overhead.
- Prefill is tensor-core bound (MPP `matmul2d`), 128-token chunks.
- No speculative decoding, no int8 KV, no paged KV.
