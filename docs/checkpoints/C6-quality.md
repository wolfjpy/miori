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
