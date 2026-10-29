# Regression matrix (v0.3.2)

Machine: Apple M6, 16 GB, macOS 27.0.1, Xcode 27.1.
Build: `make clean && make -j` -> 0 warnings, `build/miori` + `build/miori.metallib`.

## Kernel / model tests

| command | result |
|---|---|
| `test-gemv4` | max_abs 1.49e-07, max_rel 8.35e-05 |
| `test-gemv8` | max_abs 1.71e-07, max_rel 1.68e-05 |
| `test-gemm` | max_abs 0 |
| `test-attn` | max_abs 0.0356 |
| `test-prefill-attn` | max_abs 0.00197, p err 4.65e-04 |
| `test-conv` | output 0.249 (bf16 rounding vs float harness), state 0 |
| `test-tokenizer` | chat template `<|im_start|>assistant`, think tags |
| `test-kv-int8` | PASS (prefill QK/PV exact, decode ~1e-6) |

`test-conv` now reads the bf16 output buffer; 0.249 is bf16 rounding against a
float reference, not a kernel error (`state err = 0`).

## Text

| check | result |
|---|---|
| `run --raw "The capital of France is"` | `Paris. The capital` |
| `bench --prefill 1024 --decode 32` | 2897 tok/s prefill, **100.4 tok/s decode** |
| `ppl --file docs/README.md --tokens 128` | 94.785 (bf16) |
| `prefix-test` | cold/warm `4`, snapshot works |

## Batch / KV

| check | result |
|---|---|
| `batch-prefill-test --n 32` | 6/6 KV+conv byte-identical, 3.13x |
| `batch-sparse-test` | reused slots 1 and 3 token-identical to single-stream |
| `kv-int8-test --mix vk --tokens 512` | PPL +0.130%, top-1 99.217%; 4-domain aggregate passes at 99.560% |
| 4 concurrent greedy HTTP | all `The capital of France is Paris.` |
| SSE stream `What is 2+2?` | 3 events, text `4` |
| `/health` | `{"status":"ok","version":"0.3.0","kv":"bf16","batch_slots":32,"exact":false}` |

## Vision

| check | result |
|---|---|
| `vision-test --max-pixels 262144` | 240 tokens, finite features; steady 33-35 ms (first warm 41) |
| `vision-run` cat | `The image shows a close-up of an...` |
| image SSE | per-token deltas |

## Notes

- v0.3.1 removes 4 unused private fields from `engine.h`; clean build has zero
  warnings.
- Full int8 KV and the 512-token README slice miss the strict 99.5% top-1 gate;
  only `MIORI_KV_MIX=vk` passes the 4-domain aggregate gate. All KV modes are
  opt-in; bf16 is the default.
- C2 64x2K target remains NO-GO (see BENCHMARKS).

## v0.3.2 additions

- Startup warmup (Engine + BatchEngine), `MIORI_NO_WARMUP=1` disables.
- Prompt >=4096 returns 400; single decode stops at 4095.
- Reused 16 MB image embedding buffer; 100-image resident stable at 10.82 GB.
- `/metrics`: resident bytes, request latency histogram.
- `batch-sparse-test` (sparse slot 1/3 reuse) PASS.
- Clean build: 0 warnings.
