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

### Negative result: 32-row MPP tile

A `m=32, n=64` `matmul2d` variant (2 simdgroups) for B≤32 was measured at B=32:
68.6 ms/step / 466 tok/s, worse than the 64-row tile (58.8 ms / 544 tok/s).
The 64-row tile with 4 simdgroups is better even when M=32; the kernel is kept
but unused. Batch default remains B=4 gemv4_b.

## Server default: hybrid B=4-exact / B=32-MPP

`serve` now uses kBmax=32 with MPP bf16 for bursts and the single Engine for
small queues (<4 pending). 64 concurrent 16-token requests: 5.80 s (B=4 exact)
-> 3.49 s. The B=32 path is coherent but not token-identical to 4-bit GEMV; the
low-queue path is exact. Batched prefill remains the next bottleneck (~half the
64-concurrent wall time).

## Prefill profiling (B=32 server)

With `MIORI_BATCH_PROFILE=1`, `add_sequence` for a 23-token prompt on the M6:

```
prefill 32.6-33.1 ms, KV copy 1.2-2.4 ms, conv copy 0.0 ms
```

So 64-concurrent prefill-only wall time (~2.3 s) is almost entirely the
per-request 24-layer prefill graph: ~33 ms each, dominated by ~120 small MPP
GEMM launches (M=23 but a 64-row tile), not by KV copies. Clearing KV in
`Engine::reset` was removed (stale positions are never read) but measured
neutral. The next required change is **packed/batched prefill** (concatenate
requests, block-diagonal attention, per-request conv state), not more kernel
tuning; at 64 short prompts the current serial prefill is ~2.3 s of the 3.5 s
wall time.

## Packed multi-request prefill (added)

`BatchEngine::add_batch()` prefills N prompts in one pass. The prompts are
concatenated into a single `[T][2048]` activation buffer, each of the 24 layers
runs its GEMMs **once over all T rows** on the MPP tensor cores, and only the
request-boundary work stays per request:

- **Shared GEMMs.** `gemm_mpp` over the existing 2.93 GB bf16 cache for q/k/v/o,
  conv in/out and the FFN (`M = T`, padded to the 64-row tile); `gemm_naive` for
  the N = 16 attention gate and for the QK/PV per-request attention.
- **Request-local positions.** A `[T]` position buffer (`row - request_start`)
  drives RoPE via the existing `rope_batch` kernel and the new
  `kv_write_packed` kernel, which writes post-RoPE K (bf16) and V straight into
  each destination slot's KV cache — the per-head KV copies of `add_sequence`
  (1.2–2.4 ms/prompt) are gone.
- **Per-request attention.** For each request, its own token range is attended
  against its own slot KV only (block-diagonal by construction — a request can
  never see another's tokens). Chunks of 128 rows, relative `start_pos` and the
  256-token sliding mask reproduce `Engine::prefill_ids`'s score-matrix widths
  exactly.
- **Per-request conv.** `conv7` is called per request on slices of the packed
  `gated`/`bcx` buffers (new `conv7_off` wrapper, byte offsets), with a zero
  6-token window at the request start and the resulting state written directly
  into the slot's live double-buffer half.
- **First token.** Final RMSNorm over all T rows, the last row of every request
  gathered into a `[kBmax][2048]` buffer (`gather_rows`), then the same 8-bit
  `gemv8_b` kernel as the decode path (4 rows per dispatch, same accumulation
  order as the single-sequence `gemv8`) — 1 exact lm_head pass instead of 64.
- **Limits.** At most `kBmax` = 32 requests and `kPackRows` = 1024 concatenated
  tokens per pass; `add_batch` groups the input accordingly and falls back to
  `add_sequence` for any prompt longer than 1024 tokens (or longer than the
  4096-position slot). Packed activations cost 175 MB.

`add_sequence`, `decode_step`, the server's default path and the vision path are
unchanged; the packed prefill is opt-in in the server via
`MIORI_BATCH_PREFILL` (default on; `=0` disables).

### Exactness

The MPP GEMM result per element does not depend on the M tile shape:
`test-mpp-tile` runs the same 23-row A through `gemm_mpp32` (32-row tile, the
sequential path's kernel for T ≤ 32) and `gemm_mpp` (64-row tile):

```
test-mpp-tile: N=6144 K=2048, rows 0..22 of 64
gemm_mpp32(M=23) vs gemm_mpp(M=23)         0/141312 differ, max abs diff 0
gemm_mpp(M=23) vs gemm_mpp(M=64)           0/141312 differ, max abs diff 0
```

Everything else in the packed pass reuses the single-sequence kernels with
identical per-row math, so the packed path is bit-identical:

```
./build/miori batch-prefill-test --n 64 --repeat 5 --state-check 4 --long 423
batch-prefill-test: 64 prompts, 23..24 tokens (mean 23.2, total 1484), 2 wave(s) of <= 32
sequential Engine::prefill_ids : 2.095 s median (min 2.090 max 2.121) = 708 tok/s
packed BatchEngine::add_batch  : 0.669 s median (min 0.669 max 0.690) = 2218 tok/s
prefill speedup: 3.13x (32.73 -> 10.46 ms per request)
first tokens: 64/64 identical
first-token logits: 0/8192000 elements differ, max |diff| = 0
state check: 6/6 requests byte-identical (KV + conv state)
```

The state check compares, byte for byte, every used KV position of all 11
attention layers and the 6 x 2048 conv state of all 13 conv layers against the
state `Engine::prefill_ids` leaves behind for the same prompt (518 KB of KV per
23-token prompt, 639 KB of conv state per request).

Long prompt (several 128-token chunks, sliding layers past their 256-token
window), packed together with a short prompt in one `add_batch` call:

```
long prompt: 431 tokens + 23-token prompt in one add_batch call: 155.2 ms
  first tokens: 2/2 identical, logits 0/256000 differ (2 packed request(s)), max |diff| 0
  state check: 9,709,568 KV bytes and 638,976 conv bytes identical on both requests
```

A 1009-token prompt + a 23-token prompt (over the row budget, so it is split
into two packed passes) is also 2/2 first-token identical with 0 differing
logits; a 1213-token prompt exercises the `> kPackRows` fallback: first token
and KV/conv state still identical.

### Measured (packed prefill)

| Scenario | sequential `prefill_ids` | packed `add_batch` | speedup |
|---|---:|---:|---:|
| 64 x 23-token prompts (2 passes of 32) | 2.095 s / 708 tok/s | **0.669 s / 2218 tok/s** | **3.13x** |
| 423–431-token prompt + 23-token prompt | 0.42 s (0.40 + 0.02) | 0.155 s | 2.7x |
| one 1009-token prompt | ~0.40 s | 0.40 s | 1.0x (no packing gain) |
| one 23-token prompt | 32.7 ms | 10–12 ms | 2.8x |

64 concurrent HTTP requests (23-token prompts, greedy, `max_tokens` 16), same
binary, same request file, 4 runs each, median:

| server | wall clock | responses |
|---|---:|---|
| default | **2.94 s** (3.06 / 2.95 / 2.92 / 2.94) | — |
| `MIORI_BATCH_PREFILL=1` | **1.51 s** (1.60 / 1.50 / 1.51 / 1.49) | byte-identical (md5 match) |

The scheduler still uses Bmax = 32, so the 64 requests are two waves; each wave
now pays one packed prefill (~0.33 s for 32 x 23 tokens) instead of 32
sequential prefills (~1.05 s). What is left of the 1.5 s is the decode: two
waves x 10 output tokens x ~59 ms MPP step.

### What is missing

- **Bmax = 32** caps a packed pass at 32 requests; 64 concurrent prompts take
  two passes (the fixed per-pass overhead is small but the GEMM tile padding and
  the ~1.5 k extra launches of the per-request attention/conv loops are paid
  twice). A prefill-only staging buffer > kBmax slots would merge them.
- **kPackRows = 1024**: prompts longer than that still go through
  `add_sequence` (correct, not packed). Chunked/streaming packing would let a
  long prompt's 128-token chunks share the GEMM tiles with short prompts.
- **No mixed prefill/decode step.** A packed pass blocks the batch (the
  scheduler prefills, then decodes); continuous chunked prefill is still
  missing. This is the next latency win for a steady stream of arrivals.
- **Per-request dispatch loops.** Attention (3 launches per request chunk per
  attention layer) and conv (1 launch per request per conv layer) are not fused;
  at 32 short requests that is ~1.5 k extra launches, roughly 2% of the packed
  pass. Boundary-aware packed conv7 and per-request-offset attention kernels
  would remove them.
- **No packing gain for a single long prompt** (1009 tokens: 0.40 s either
  way); the per-request attention is the same work.
- **The MPP GEMM is now the cost**: the packed pass runs at ~6.6 TFLOPS
  effective (64x64 tiles). An M-first dispatch order (to share weight tiles in
  L2) was tried and measured neutral (0.669 -> 0.665 s, within noise) and
  reverted; a larger M tile per threadgroup is the next lever.
- `batch-test --kernels gemv4` remains broken since kBmax was raised to 32
  (`gemv4_b`/`gemv8_b` return early for B > 4). Pre-existing, unrelated to the
  packed prefill; the 4-bit batched path needs a B ≤ 4 mode to be usable again.

### New command

```
./build/miori batch-prefill-test [--n 64] [--repeat 3] [--state-check 4] \
    [--long 423] [--words]
```

Runs the 64-prompt and long-prompt comparisons above, prints first-token ids,
logits diff, KV/conv state diff and wall times. `--words` uses ~29-token
question prompts instead of the default ~23-token single-word turns.
`MIORI_BATCH_KEEP_LOGITS=1` (set by the test itself) keeps each request's
first-token logits across packed passes so they can be diffed.

### B=64 experiment

`kBmax=64`, MPP decode, 64 duplicate prompts x16 tokens in `batch-test`:

```
83.5 ms/step, 766 tok/s aggregate (7.9x single)
```

So batch decode scales to 64 slots. But on the 64-concurrent HTTP test the wall
time was 2.02 s, the same as B=32: packed prefill needs two passes at
`kPackRows=1024` (64x23 = 1484 tokens), and the one-wave decode only cancels
that. KV grows to 5.9 GB (32 slots = 2.95 GB). Reverted to kBmax=32; B=64 is
available as a compile-time experiment, not the server default.
