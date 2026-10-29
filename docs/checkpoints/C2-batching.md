# C2 — fixed-slot batch decode (Bmax = 4)

- Date: 2026-10-29
- Status: **PARTIAL — batch decode works for up to 4 sequences; paged KV and
  continuous batching are still not implemented**
- Host: Apple M6 / 16 GB, macOS 27.0.1, Xcode 27.1, `-std=metal4.1`

## What was built

`src/batch.h` / `src/batch.cpp` add `miori::BatchEngine`, a second execution
path on top of the existing single-sequence `Engine`:

- **Shared weights.** The bf16 tensor-core cache (`Engine::wbf_`, 2.93 GB) and
  its per-tensor offsets are exposed through additive accessors
  (`Engine::bf16_buffer()`, `Engine::weight_offset()`, `bf16_bytes()`) and used
  by every slot. The 4-bit weights stay in the original safetensors buffer.
- **Per-slot state.** K and V for the 11 attention layers are allocated as
  `[4][4][4096][128]` bf16 (16 MB per layer per cache, 369 MB total); the 13
  conv layers get `[4][2][6*2048]` f32 (5.1 MB). Engine accessors
  (`kv_k`, `kv_v`, `conv_state`, `is_attention_layer`, `layer_kind`) expose the
  single-sequence buffers for copying without changing any existing API.
- **Prefill.** `add_sequence(prompt)` runs the prompt sequentially on the shared
  `Engine` (`reset`, `prefill_ids`, greedy `sample` for the first token), then
  copies the *used* KV positions per head (K head slabs, V row slabs) and the
  live conv state into a free slot. Prefill is therefore **not batched**; it is
  one prompt at a time on the tensor-core path.
- **Decode.** `decode_step()` feeds one token per running slot in a single
  command buffer: `embed8` (T = B), then 24 layers of `rmsnorm` + MPP
  `gemm_mpp` over the shared bf16 cache for q/k/v/o, conv in/out and FFN, with
  `gemm_naive` for the N = 16 attention gate (the cache stores that tensor
  row-major), then a final `rmsnorm` and a per-slot `gemv8` lm_head with byte
  offsets into a `[4][128000]` f32 logits buffer. Sampling is greedy on the CPU,
  one token per slot, stopping on EOS 124900 or `max_new_tokens`.
- **New kernels** (`shaders/batch.metal`): `attn_decode_b` (grid B×4, per-slot
  KV and positions, 256-token window for sliding layers), `conv7_b` (grid B×2048,
  per-slot double-buffered state), `kv_append_b`, `rope_batch`. Existing
  batched-row kernels (`gate_mul_b`, `rmsnorm`, `silu_mul`, `postnorm_add`,
  `conv_gate`, `embed8`, `to_f32`) are reused unchanged.
- **Test command.** `./build/miori batch-test --prompts "..." "..."` runs
  single-engine greedy per prompt as the reference, then the same prompts
  through the batch for `max_tokens` (default 16), compares ids and prints
  aggregate tok/s.

Activations are allocated with 64 rows (`[-1][2048]` padded) so the MPP 64-row
tile never reads or writes past a buffer; padding rows stay zero. `decode_step`
encodes all 4 rows because the MPP tile size makes the row count irrelevant;
only running slots are sampled.

### The batched 4-bit GEMV path (additive)

`gemm_mpp` recomputes a 64-row output tile per step, so its cost is nearly
constant in B. To also have a path that scales, `--kernels gemv4` switches the
same `BatchEngine` to `gemv4_b` (+ `gemv8_b` for lm_head): one 4-bit GEMV pass
per weight row with up to 4 accumulator rows, i.e. the 1.59 GB of 4-bit weights
is read once per step for all slots. Accumulation order per row matches the
single-sequence `gemv4`, so each slot's logits are bit-identical to the
single-stream engine. `batch-test` defaults to the specified bf16/MPP path;
`--kernels gemv4` selects the GEMV path.

## Measured

All numbers below are from the same process (bf16 cache built once, warm GPU),
target M6 / 16 GB. "Aggregate tok/s" counts tokens produced by `decode_step`
divided by total step time (includes GPU wait + CPU greedy sampling). The
single-stream reference in the same run is 97.7–98.6 tok/s.

Command (required test, 2 prompts, 16 tokens; median of 3):

```
./build/miori batch-test --prompts "The capital of France is" "List the first four primes."
```

| B | kernels | ms/step | aggregate tok/s | vs single-stream | ids vs single-stream |
|---:|---|---:|---:|---:|---:|
| 2 | `gemm_mpp(bf16)` (default) | 29.7 | **38.2** | 0.39× | 19/19 identical |
| 2 | `gemv4_b(4-bit)` | 15.3 | **74.2** | 0.76× | 19/19 identical |

Longer 32-token runs with paraphrased prompts (`--max-tokens 32`), B = 3 and 4
(2 runs each, spread < 1%):

| B | kernels | ms/step | aggregate tok/s | vs single-stream | ids vs single-stream |
|---:|---|---:|---:|---:|---:|
| 3 | `gemm_mpp(bf16)` | 33.7 | 89.0 | 0.91× | 89/96 |
| 3 | `gemv4_b(4-bit)` | 15.35 | 195.5 | 1.99× | 96/96 identical |
| 4 | `gemm_mpp(bf16)` | 35.9 | **111.3** | 1.14× | 121/128 |
| 4 | `gemv4_b(4-bit)` | 15.4 | **260.1** | 2.66× | 128/128 identical |

Reference single-stream rates for the same runs: 97.8 tok/s (4 sequences of 32
tokens, 1.27 s total). Prefill is sequential: 0.13 s for four 30-token prompts,
0.19 s for a 423-token prompt plus a 33-token prompt.

A 423-token prompt (4 prefill chunks) batched against a 33-token prompt gives
identical ids to single-stream on both paths, i.e. the per-head KV copy and the
per-slot sliding/full attention are correct past one prefill chunk.

Kernel breakdown of a 15.4 ms `gemv4_b` step at B = 4: ~10.5 ms streaming the
1.59 GB 4-bit weights (1.59 GB / ~150 GB/s), ~1.7 ms batched 8-bit lm_head
(262 MB), the rest elementwise/norm/attention/conv and dispatch. The
`gemm_mpp` step is ~23 ms of 64×N×K tensor-core work (fixed by the 64-row tile)
plus ~1.5 ms per slot of lm_head and ~4 ms of everything else.

## Failures and caveats

- **B = 2 on the default MPP path is a regression in aggregate throughput**
  (38 vs 98 tok/s). The 64-row MPP tile recomputes 64 rows per token no matter
  how many slots are active while streaming the 2.93 GB bf16 cache; batching
  only pays off when the same tile serves 4 tokens (1.14×). The same is true,
  but to a much smaller degree, for every MPP-based batch: per-step time is
  almost independent of B (29.7 → 35.9 ms for B = 2 → 4, the growth is lm_head).
- **The MPP path is not token-exact vs single-stream**: at B = 4, 7 of 128 ids
  differ (one sequence diverges at token 25) because q/k/v/o/FFN go through
  bf16 MPP GEMM instead of the 4-bit GEMV used by the reference. At B = 2 the
  19 ids tested matched. This is numerics, not a KV/state bug: the gemv4 path,
  which uses the same 4-bit kernels as the reference, matches 100%.
- **The batched 4-bit GEMV path is instruction bound**: each weight is
  multiplied by 4 activations in FP32 SIMT, so the step costs 15.3 ms for 2–4
  slots instead of 10.2 ms for one. Aggregate wins start at B = 3 (2.0×) and
  reach 2.66× at B = 4; at B = 2 it is still 0.76× of single-stream. A 4×4
  register tile or a bf16 tensor-core path with a 32-row tile is the next step.
- **Only 4 slots.** `Bmax` is a compile-time constant; there is no admission
  control and no way to grow beyond 4 without new buffers.
- **No paged KV, no prefix sharing, no continuous batching.** Each slot owns a
  contiguous 4096-position cache (16 MB/layer) and is filled only by the
  sequential prefill; there are no block tables, no COW, no eviction. The
  server (`src/server.cpp`) still serializes requests behind its mutex and does
  not use `BatchEngine`.
- **Prefill is not batched.** `add_sequence` resets and runs the shared Engine
  one prompt at a time, so time-to-first-token does not improve with batching.
- **Batch attention has no split-K.** `attn_decode_b` is one threadgroup per
  (slot, KV head) looping over all cached keys; at 4K context this is the
  latency-bound path the single-sequence engine avoided with 16-way split-K.
- Sampling is a 128K-element argmax per running slot on the CPU (~0.1 ms/slot);
  greedy only.

## What is missing for the original C2 gate

- 64-token paged KV, block tables, COW, prefix sharing, eviction.
- Ragged `[M, 2048]` execution and a continuous batching scheduler (admission,
  chunked prefill across requests, cancel).
- Server integration and the 64×2K / 64×4K aggregate ≥2.5K output tok/s target.
- Batch invariance / page-leak / cancel tests (none exist; only the greedy
  token-id comparison in `batch-test`).

## Files

- `src/batch.h`, `src/batch.cpp` — `BatchEngine` (new).
- `shaders/batch.metal` — `rope_batch`, `kv_append_b`, `attn_decode_b`,
  `conv7_b`, `gemv4_b`, `gemv8_b` (new).
- `src/ops.h`, `src/ops.cpp` — wrappers for the kernels above, `gemv8_off`,
  `gemv4_b`, `gemv8_b`.
- `src/engine.h`, `src/engine.cpp` — additive accessors and
  `is_attention_layer`/`layer_kind` (also used internally to replace the two
  duplicated full/sliding layer-id checks; no behaviour change).
- `src/main.cpp` — `batch-test` command.
- `README.md`, `docs/STATUS.md` — status text updated for C2.

## Server integration (added after the standalone test)

`BatchEngine` is now wired into the HTTP server for **greedy** requests:

- up to 4 requests are queued, prefilled sequentially under the Engine mutex,
  then decoded together (one token per slot per step);
- sampling requests (`temperature > 0.01`) still use the single-sequence path;
- 4 concurrent requests, 32 output tokens each: **0.682 s wall = 188 output
  tok/s aggregate**, versus ~98 tok/s for one stream (~1.9x aggregate).
- `batch-test` with 4 long prompts reports **261 tok/s aggregate** (15.3 ms/step,
  128/128 token-identical to single-stream greedy) on the 4-bit `gemv4_b` path.

Still missing for the original C2 gate: paged KV/block tables, continuous
admission, batched prefill, B > 4, split-K batched attention, 64×2K/4K targets.

### Negative result: B=8

Extending `gemv4_b` to 8 slots (8 accumulators, 8 input rows) raised the step
from 15.3 ms to 33.4 ms (239.6 tok/s aggregate at B=8) because of register
pressure; B=4 remains the sweet spot at 15.3 ms / 261 tok/s. The server keeps
Bmax=4. A 4x4 register-tiled batched GEMV or a 32-row MPP tile is the next step
for B ≥ 8.

## Continuous admission (added)

The scheduler no longer drains a fixed batch: when a slot finishes, its output is
collected, the slot is released, and the next queued request is prefilled into
that slot while the remaining slots keep decoding.

8 concurrent requests, 16 output tokens each:

```
0.792 s wall = 162 output tok/s aggregate
```

(vs ~2 × 0.45 s for two static batches of four). Prefill for a newly admitted
request still blocks the batch, because it runs on the shared Engine; batched
prefill is the next scheduler step.

## Throughput experiment: B=16/32 with bf16 MPP + bf16 lm_head

The per-slot 8-bit lm_head was the scaling bottleneck (B=16: 65 ms/step). Added a
524 MB bf16 embedding cache and one `gemm_naive` lm_head for all slots:

| B | kernels | ms/step | aggregate | vs single | id drift vs 4-bit ref |
|---|---:|---:|---:|---:|---:|
| 4 | gemv4_b | 15.3 | 261 | 2.66x | 0/128 |
| 16 | MPP + bf16 lm_head | 46.7 | 343 | 3.49x | 12/256 |
| 32 | MPP + bf16 lm_head | 58.8 | 544 | 5.56x | 0/512 (duplicate prompts) |

B=32 is a real throughput mode, but a single request through a 32-row MPP step
costs ~59 ms/token (~17 tok/s). A hybrid "small queue → single path, large queue →
batch" router made 64-concurrency worse (11.8 s vs 3.7 s) because arrivals kept
seeing a small backlog and serialized through the single path. It was reverted;
the server default remains **B=4, 4-bit gemv4_b, token-exact**.
