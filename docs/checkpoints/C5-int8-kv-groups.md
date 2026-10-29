# C5 — int8 KV with per-32 group scales (follow-up)

- Date: 2026-10-29
- Machine: Apple M6 Mac mini, 12 GPU cores, 16 GB, macOS 27.0.1 (26A434), Xcode 27.1
- Build: `make -j` (`metal4.1 -O3 -fno-fast-math`)
- Baseline: `2679822` (single-scale int8 KV); this change is committed with this report
- Status: **NO-GO for the strict C5 gate** — mean KL, p99 KL and PPL clear; top-1
  reaches 99.34 % aggregate vs the 99.5 % gate, so int8 KV remains opt-in and
  bf16 remains the default.

This report supersedes the scale layout and quality numbers in
[`C5-int8-kv.md`](C5-int8-kv.md); the kernel/structure notes there still apply.

## Change

The single bf16 scale per `(kv_head, token)` is replaced by one bf16 scale per
`(kv_head, token, 32-wide d group)` (K) and `(kv_head, 32-wide d group, token)`
(V). Still symmetric int8, no zero point, `scale = max_abs(group) / 127` stored
as bf16.

| | layout | scale |
|---|---|---|
| K | `[h][pos][128] int8` | `ks[h][pos][4] bf16`, g = d/32 |
| V | `[h][128][pos] int8` | `vs[h][4][pos] bf16`, g = d/32 |

The four 32-wide groups per 128-element head mean K's `[pos][4]` and V's
`[4][pos]` per-head scale blocks have the same byte size, so one byte offset
still selects a head in both.

Writer threadgroups are unchanged (128 threads per `(head, token)`); warp `w`
covers d-group `w`, so the per-group max is a single `simd_max` with no
cross-warp reduction. Decode readers use `g = lane/8` (a lane's four contiguous
dimensions always share a group). Prefill QK scales by `(key row, k/32 group)`
and PV by `(d/32 group, key column)` inside the existing `gemm_naive` tile.

One correctness detail found during bring-up: with per-group K scales the
partial dot product must be scaled **before** the `simd_sum`, not after (the
single-scale version could scale the reduced result). A first version scaled
after the reduction and showed ~0.05 absolute attention error; the kernel test
caught it.

Updated: `Engine` + `BatchEngine` allocations, all int8 writers, decode
attention (normal/split-K/batch), prefill QK/PV, snapshots/restore,
`add_sequence` copies, `batch-prefill-test` state comparison, `test-kv-int8` CPU
reference. The bf16 path is untouched (same PPL, same kernel-test outputs).

## Kernel correctness — `test-kv-int8`

`build/miori test-kv-int8` (log: `bench/results/C5-int8/test-kv-int8-groups.txt`):

```
prefill (T=4, seq=4) QK raw score err = 0, PV max abs err = 0
decode (pos=600) max abs err = 1.19209e-06, split-K(16) = 1.16229e-06
max quantisation error / 32-group max_abs: K 0.778%, V 0.776% (1/127 = 0.79%)
PASS
```

The CPU reference quantises per 32-wide group with the same bf16 scale rounding;
prefill QK/PV are bit-identical, decode differs only by f32 reduction order. The
worst-case step is still 1/127, but of each 32-value group's max instead of the
128-value token max, so the absolute error drops; the measured PPL/KL change
confirms it.

## Quality — `kv-int8-test`

Requested run, `--file docs/README.md --tokens 512` (511 positions), log
`quality-readme-512-groups.txt`:

```
PPL: bf16 42.2184 -> int8 42.2305 (delta +0.029%)
top-1 agreement: 507/511 = 99.217%
KL(ref||int8): mean 0.000205626, p99 0.00127703, max 0.00234008
4K bench: bf16 87.46 tok/s, int8 86.82 tok/s (0.993x)
```

Four-domain aggregate, 1023 positions each (`quality-*-groups.txt`):

| Corpus (domain) | PPL bf16 | PPL int8 | ΔPPL | top-1 | mean KL | p99 KL | max KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `docs/README.md` (EN) | 24.0640 | 24.0609 | −0.013 % | 99.413 % | 1.51e-4 | 1.20e-3 | 1.28e-3 |
| `docs/CHECKPOINTS.md` (ZH) | 44.2416 | 44.2332 | −0.019 % | 99.609 % | 3.35e-4 | 2.15e-3 | 1.76e-2 |
| `src/main.cpp` (code) | 9.0466 | 9.0249 | −0.240 % | 99.022 % | 2.46e-4 | 3.02e-3 | 8.03e-3 |
| `models/config.json` (JSON) | 6.9965 | 6.9969 | +0.006 % | 99.316 % | 1.34e-4 | 2.15e-3 | 2.24e-3 |
| **aggregate (4092 pos)** | **16.1117** | **16.1010** | **−0.067 %** | **99.340 %** | **2.17e-4** | **3.02e-3** (worst domain) | — |

Full `docs/README.md` (1885 positions): PPL 19.7301 → 19.7364 (+0.032 %),
top-1 1871/1885 = 99.257 %, mean KL 1.44e-4, p99 KL 5.82e-4.

Gate verdict:

| Gate | Threshold | Measured | Verdict |
|---|---:|---:|---|
| Task: PPL regression | ≤ +1 % | **−0.067 %** (worst domain −0.240 %) | PASS |
| CHECKPOINTS §3.4: mean KL | ≤ 1e-3 | **2.17e-4** | PASS |
| CHECKPOINTS §3.4: p99 KL | ≤ 5e-3 | **3.02e-3** (worst domain) | PASS |
| CHECKPOINTS §3.4: top-1 | ≥ 99.5 % | **99.340 %** | **FAIL** |
| Task: top-1 | ≥ 99 % | 99.340 % (worst domain 99.022 %) | PASS |

Group scales improved the aggregate versus the single scale (top-1 99.12 % →
99.34 %, mean KL 3.39e-4 → 2.17e-4, PPL +0.31 % → −0.07 %), but the remaining
gap to 99.5 % is 4072 − 4065 = **7 top-1 flips over 4092 positions**. The gate
was not weakened; per instruction the tuning stops here and the flag stays
opt-in/off by default.

## Memory and performance

| item | bf16 | int8 (group scales) |
|---|---:|---:|
| Engine KV / sequence | 92.27 MB | 49.02 MB (53.1 %) |
| BatchEngine KV, Bmax=32 | 2952.8 MB | 1568.7 MB (53.1 %) |
| attention KV payload / token @4K | 369.1 MB | 184.5 MB (+2.9 MB scales) |
| decode 4K (128 tok) | 87.5 tok/s | 87.6 tok/s |
| prefill 4K | 1630 tok/s | 1620 tok/s |
| decode 1K | 91.8 tok/s | 91.8 tok/s |

`./build/miori bench --prefill 4096 --decode 128` (3-run median,
`bench-groups.txt`): decode parity (87.5 → 87.6 tok/s), prefill −0.6 %. As with
the single-scale version, single-stream decode is not KV-bandwidth-bound; the
win is capacity (1.38 GB less batch KV).

## Batch correctness

With group scales, int8 packed prefill is still bit-identical to the sequential
int8 path (log `batch-prefill-int8-groups.txt`):

```
prefill speedup: 3.53x (32.33 -> 9.15 ms per request)
first tokens: 8/8 identical
first-token logits: 0/1024000 differ, max |diff| = 0
state check: 4/4 requests byte-identical (KV + conv state)
```

The state comparison includes the int8 K/V bytes and both group-scale arrays
(short prompt: 275264 bytes of KV+scales, 0 differ; 431-token prompt: 5158208
bytes, 0 differ). The 64-prompt / 2-wave run is also bit-identical: 64/64 first
tokens, 0/8192000 logits differ, 3.18x faster (log
`batch-prefill-int8-64-groups.txt`). Prefix snapshot/restore and
`batch-test` (0 token mismatches) pass; bf16 regression is unchanged (PPL
24.064, same attention-test values, packed prefill 3.62x).

## What would be needed for 99.5 % / why it is not done

Seven near-tie top-1 decisions remain. The remaining error is the symmetric
int8 quantisation of K/V itself (group scale ≈ 1/127 of each group's max); finer
32-wide groups have already removed the scale-sharing error. Crossing 99.5 %
would need one of:

- an asymmetric zero point per group (the task keeps symmetric int8, so not done);
- a higher-precision cache (bf16 V / int8 K mixed, or 10–12-bit packing), which
  gives back part of the memory win;
- 16-wide groups (layout/GEMM change) — expected to help less than the 32-wide
  step did and not evaluated.

Also missing: paged/radix/shared-prefix KV; the int8 flag remains process-global
and `BatchEngine` captures the format at `init()`; no long-context B=32
throughput benchmark (only footprint) and no server metric exposing int8 state.
