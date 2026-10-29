# C5 — mixed K/V precision (V int8) passes the gate

- Date: 2026-10-29
- Machine: Apple M6 Mac mini, 12 GPU cores, 16 GB, macOS 27.0.1 (26A434), Xcode 27.1
- Build: `make -j` (`metal4.1 -O3 -fno-fast-math`)
- Baseline: `ac3edb8` (32-group full-int8 KV); this change is committed with this report
- Status: **GO for the mixed mode `MIORI_KV_MIX=vk` (K bf16, V int8)**. It passes the
  strict 4-domain gate; bf16 remains the default and full int8 (K+V) remains opt-in
  but still below the top-1 gate.
- This supersedes the quality conclusion of
  [`C5-int8-kv-groups.md`](C5-int8-kv-groups.md) for the mixed configuration; the
  group-scale layout and 32-group int8 path are unchanged.

## Modes

| env | K | V | status |
|---|---|---|---|
| *(unset)* | bf16 | bf16 | default |
| `MIORI_KV_INT8=1` | int8 32-group | int8 32-group | opt-in, top-1 99.34 % (fails 99.5 % gate) |
| `MIORI_KV_MIX=kv` | int8 32-group | bf16 | available, top-1 99.24 % (fails) |
| `MIORI_KV_MIX=vk` | bf16 | int8 32-group | **opt-in, passes** |

The letter in `MIORI_KV_MIX` is the quantised side; `MIORI_KV_MIX` wins over
`MIORI_KV_INT8` when both are set. The int8 side keeps the per-32-group bf16
scale layout (`ks[h][pos][4]` / `vs[h][4][pos]`).

## Implementation

- The fused bf16 and int8 KV writers take a `mode` mask (1 = K, 2 = V,
  3 = both); mixed writes call one writer per side with the other side masked.
  The inactive destination is a small never-touched scratch buffer.
- New `attn_decode_mix`, `attn_decode_split_mix` and `attn_decode_b_mix` read
  K bf16 or int8 and V int8 or bf16 according to `k_int8`, keeping the K
  group-scale multiply before `simd_sum` on the int8 path.
- Prefill selects per side: int8 K uses `gemm_naive_qk_i8`, bf16 K the normal
  GEMM; V likewise for PV. Engine and BatchEngine allocations, snapshots,
  `add_sequence` copies and `batch-prefill-test` state checks are per side.
- `kv-int8-test --mix kv|vk` runs the candidate mode; `ppl --kv-int8` is
  unchanged. The bf16 default path is untouched (same contents and kernels).

## Kernel correctness — `test-kv-int8`

Log: `bench/results/C5-int8/mixed-test-kv-int8.txt`.

```
prefill (T=4, seq=4) QK raw score err = 0, PV max abs err = 0
decode (pos=600) max abs err = 1.19209e-06, split-K(16) = 1.16229e-06
mixed decode K int8/V bf16 err = 1.19209e-06, K bf16/V int8 err = 7.15256e-07
max quantisation error / 32-group max_abs: K 0.778%, V 0.776% (1/127 = 0.79%)
PASS
```

The mixed decode kernels are checked against the CPU reference using a bf16
cache for the unquantised side; both orientations are within the f32 reduction
order.

## Quality — `MIORI_KV_MIX=vk` (K bf16, V int8)

Four-domain, 1023 positions each (logs `mixed-quality-*-vk.txt`):

| Corpus (domain) | PPL bf16 | PPL vk | ΔPPL | top-1 | mean KL | p99 KL | max KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `docs/README.md` (EN) | 24.0640 | 24.0916 | +0.115 % | 99.413 % | 1.03e-4 | 7.45e-4 | 8.49e-4 |
| `docs/CHECKPOINTS.md` (ZH) | 44.2416 | 44.2977 | +0.127 % | 99.804 % | 2.71e-4 | 8.87e-4 | 2.01e-2 |
| `src/main.cpp` (code) | 9.0466 | 9.0379 | −0.096 % | 99.511 % | 1.36e-4 | 1.24e-3 | 1.99e-3 |
| `models/config.json` (JSON) | 6.9965 | 6.9916 | −0.070 % | 99.511 % | 7.34e-5 | 7.05e-4 | 7.29e-4 |
| **aggregate (4092 pos)** | **16.1117** | **16.1147** | **+0.019 %** | **99.560 %** | **1.46e-4** | **1.24e-3** (worst) | — |

Gate verdict:

| Gate | Threshold | Measured | Verdict |
|---|---:|---:|---|
| top-1 (4-domain aggregate) | ≥ 99.5 % | **99.560 %** (4074/4092) | **PASS** |
| mean KL | ≤ 1e-3 | **1.46e-4** | PASS |
| p99 KL | ≤ 5e-3 | **1.24e-3** (worst domain) | PASS |
| PPL regression | ≤ +1 % | **+0.019 %** (worst domain +0.127 %) | PASS |

`kv-int8-test --file docs/README.md --tokens 512 --mix vk` (511 positions) gives
top-1 **507/511 = 99.217 %**, PPL +0.130 %, mean KL 1.27e-4, p99 KL 8.45e-4;
that single 512-token slice is below 99.5 %, the aggregate above is the gate
the user specified. The K-int8/V-bf16 variant was also run and fails:
aggregate top-1 99.242 % (4061/4092), mean KL 1.73e-4, aggregate PPL −0.071 %
(logs `mixed-quality-*-kv.txt`).

## Memory

| item | bf16 | vk (K bf16 / V int8) |
|---|---:|---:|
| Engine KV / sequence | 92.27 MB | **70.65 MB (76.6 %)** |
| BatchEngine KV, Bmax=32 × 4K | 2952.8 MB | **2260.7 MB** |
| **64 concurrent × 2K KV (modelled)** | 2952.8 MB | **2260.7 MB** (saves 692 MB) |
| attention KV payload / token @4K | 369.1 MB | 276.8 MB |

64×2K is the same token count as 32×4K, hence the same batch bytes. For
reference, full int8 is 1568.7 MB at 64×2K but fails the quality gate.
The Engine figure includes both sides of the active mode; no unused side is
allocated in the server path.

## Performance

`bench --prefill 4096 --decode 128` (3-run median,
`bench/results/C5-int8/mixed-bench-vk.txt`):

| context | prefill bf16 | prefill vk | decode bf16 | decode vk |
|---|---:|---:|---:|---:|
| 1K | 2611 tok/s | 2591 tok/s | 91.1 tok/s | 89.9 tok/s |
| 4K | 1632 tok/s | 1621 tok/s | 87.6 tok/s | 85.2 tok/s |

Mixed V-int8 is ~1.3–2.7 % slower in decode than bf16 (mixed-format branch and
int8→float conversion), and ~0.7 % slower in prefill. The win is memory, not
speed. The harness's single-run 4K numbers vary with machine load (84–88 tok/s
bf16 in this session), so the 3-run medians above are the comparison.

## Batch correctness and server path

With `MIORI_KV_MIX=vk`:

- packed prefill is bit-identical to the sequential Engine path (first-token
  logits, KV+scales, conv state): 8 prompts, 4/4 state checks, 3.59×;
  64 prompts / 2 waves, 64/64 first tokens, 0/8,192,000 logits differ, 3.17×
  (`mixed-batch-prefill-vk.txt`, `mixed-batch-prefill-64-vk.txt`);
- `batch-test`: 19 tokens compared, 0 mismatches vs single-sequence greedy;
- `prefix-test`: int8 V snapshot/restore works, cold/warm output identical;
- bf16 regression: PPL 24.064, same attention-test values, packed prefill
  3.60×; full int8 path still works (PPL 24.061 on README 1023, unchanged).

## What is missing

- Full int8 K+V still fails the top-1 gate (99.34 %); only the mixed `vk` mode
  passes. `kv` mode is kept for comparison but is not recommended.
- No paged/radix/shared-prefix KV; the format flag is process-global and
  `BatchEngine` captures it at `init()`.
- No long-context B=32 throughput benchmark (only the modelled footprint), and
  no server metric exposing the KV mode.
- Mixed decode is slower than bf16; a future kernel could hoist the uniform
  mixed-format branch out of the key loop.

## Mixed KV under 64-concurrent load (added)

64 concurrent 23-token prompts, 16 greedy tokens, `MIORI_KV_MIX=vk`:

```
wall 2.04 s (bf16 default: 1.94-2.02 s)
all 64 responses in the <=2.5 s latency bucket
```

The mixed format saves ~0.7 GB of KV but is slightly slower at this context
(int8 V dequant); it remains opt-in. `MIORI_EXACT` / per-request `exact:true`
still use the selected KV mode.
