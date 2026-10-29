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

## Load soak (partial)

16 concurrent greedy requests per round, 8 output tokens, 120 rounds:

```
1920/1920 HTTP 200, 62 s wall (~31 req/s), no crash, no 429/500
```

This is a short soak, not the 24 h C6 gate. It does exercise packed prefill,
the 32-slot batch path, continuous admission and slot release repeatedly.

## 5-minute soak with resident memory (added)

`/metrics` now exposes `miori_resident_bytes` (mach task footprint).

16 concurrent greedy requests, 8 output tokens, 300 s:

```
requests: 9296, all HTTP 200, 74368 output tokens (~31 req/s)
resident: 10.22 GB -> 10.28 GB (+60 MB over the run)
```

No crash, no 429/500, no monotonic growth beyond allocator noise. The steady
footprint is ~10.3 GB, so the 16 GB machine has ~2.4 GB headroom; B=64 with
bf16 KV (5.9 GB) would not fit without mixed/int8 KV. Still not the 24 h C6 gate.

## Request latency histogram

`/metrics` now exposes `miori_request_duration_seconds` (cumulative buckets
10 ms .. 10 s + `+Inf`, count and sum), measured from the start of request
handling to the response/stream completion. Verified with a 4-token `2+2`
request: 1 request in the 100-250 ms bucket, sum 0.125 s.
