# C6 — quality & stability

- Date: 2026-10-29
- Status: **PARTIAL** — unit kernels validated; no soak / KL matrix

## Validated

| Kernel | Method | Result |
|---|---|---|
| 4-bit GEMV | CPU dequant, 2048 rows | max_abs 1.5e-7 |
| 8-bit GEMV | CPU dequant, 129 rows | max_abs 1.7e-7 |
| GEMM | CPU dot, strides | max_abs 0 |
| decode attention | CPU softmax/PV | max_abs 1.8e-7 |
| prefill attention | CPU QK/softmax/PV | max_abs 2e-3 |
| conv k=7 | CPU FIR + state | max_abs 0 |

Not done: 2048-position KL matrix, int8 KV quality, 32K validation, 24h soak,
fault injection.

## Perplexity harness (added)

`./build/miori ppl --file <corpus> --tokens N` runs teacher-forced NLL through the
decode path and reports perplexity. It prepends BOS (`<|startoftext|>`); without
BOS the same sentence scores PPL 771 for `The capital of France is Paris.`,
with BOS PPL 18.6 and `Paris` is rank 0 (NLL 0.004).

| Corpus | Tokens | PPL |
|---|---:|---:|
| `The capital of France is Paris.` | 7 | 18.60 |
| 5-sentence English paragraph | 81 | 42.53 |
| `docs/README.md` (mixed EN/ZH/Markdown) | 127 | 94.79 |

The English PPL is higher than expected for a 2B model; possible causes are the
approximate Unicode pre-tokenizer and the mixed-domain corpus. It is reported as
a regression baseline, not as a quality pass.

## Batch MPP vs exact single-stream (added)

`batch-test --kernels mpp`, 8 prompts, 32 greedy tokens:

```
batch decode: 31 steps, 248 tokens, 45.95 ms/step, 174 tok/s
tokens compared: 256, mismatches: 28 (89.1% agreement)
```

Packed prefill is bit-identical (first-token logits/KV/conv state), so the drift
comes from the bf16 MPP decode GEMM vs the 4-bit GEMV of the exact path.
`MIORI_EXACT=1` forces the server onto the exact single-sequence path; default
server uses MPP for bursts (aggregate throughput priority) and the exact path for
queues <4. This is a documented throughput/bit-exactness trade-off, not a
correctness bug in the packed prefill.
