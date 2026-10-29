# C5 — optional int8 KV cache

- Date: 2026-10-29
- Machine: Apple M6 Mac mini, 12 GPU cores, 16 GB, macOS 27.0.1 (26A434), Xcode 27.1
- Build: `make -j` (`metal4.1 -O3 -fno-fast-math`)
- Baseline: `6feb0b5`; the int8 KV change is committed with this report
- Status: **PARTIAL / NO-GO for the strict C5 quality gate** — the int8 path is
  implemented, correct and opt-in, but `top-1` misses the 99.5 % project gate;
  it stays off by default (`MIORI_KV_INT8=1` enables it).

## What landed

A symmetric int8 KV cache behind `MIORI_KV_INT8=1`. bf16 remains the default and
its path is byte-identical when the flag is unset (verified: same PPL, same
`test-attn`/`test-prefill-attn` numbers, packed prefill still bit-identical).

| Item | bf16 (default) | int8 (opt-in) |
|---|---|---|
| K layout | `[h][pos][128]` bf16 | `[h][pos][128]` int8 + `ks[h][pos]` bf16 |
| V layout | `[h][128][pos]` bf16 | `[h][128][pos]` int8 + `vs[h][pos]` bf16 |
| Scale | — | `max_abs_over_128 / 127`, symmetric, no zero point |
| Engine KV / sequence | 92.27 MB | 46.86 MB (50.8 %) |
| Engine + 4 prefix snapshots | 461.4 MB | 234.3 MB (50.8 %) |
| BatchEngine Bmax=32 KV | 2952.8 MB | 1499.5 MB (50.8 %) |
| attention KV read / token @4K | 369.1 MB | 184.5 MB |

Implemented paths:

- writers: `kv_append_i8`, `kv_write_b_i8`, `kv_write_packed_i8`,
  `kv_append_b_i8` (one threadgroup of 128 threads per `(head, token)`, in-kernel
  max reduction → scale → quantise);
- readers: `attn_decode_i8`, `attn_decode_split_i8`, `attn_decode_b_i8`
  (dequantise K/V on load); `attn_combine` is unchanged because it only combines
  f32 partials;
- prefill: `gemm_naive_qk_i8` (int8 B, scale per key row) and
  `gemm_naive_pv_i8` (int8 B, scale per key column) replace the two
  `gemm_naive` calls in `Engine::p_attn` and in the packed prefill;
- snapshots/restore and `BatchEngine::add_sequence` copy the raw int8 bytes and
  the scale bytes;
- `Engine::set_kv_int8()` switches format at runtime for the comparison harness
  (allocation is lazy, data is not migrated);
- `kv-int8-test` harness + `ppl --kv-int8`, and `test-kv-int8` kernel test.

## Kernel correctness — `test-kv-int8`

`build/miori test-kv-int8` (log: `bench/results/C5-int8/test-kv-int8.txt`):

```
prefill (T=4, seq=4) QK raw score err = 0, PV max abs err = 0
decode (pos=600) max abs err = 1.01328e-06, split-K(16) = 1.07288e-06
max quantisation error / token max_abs: K 0.778%, V 0.754% (1/127 = 0.79%)
PASS
```

The CPU reference performs the same quantise/dequantise arithmetic (including
the bf16 scale rounding). The prefill QK and PV outputs are bit-identical; the
decode kernels differ only by the f32 reduction order (~1e-6). The split-K path
(16 splits, pos 600) is within 1.1e-6.

## Quality gate — `kv-int8-test`

`./build/miori kv-int8-test --file <corpus> --tokens 1024 [--no-bench]` runs the
same token sequence through the bf16 and int8 paths (teacher forcing, full 128K
vocabulary), reports PPL, top-1 agreement, and KL(ref‖int8) at 128 evenly spaced
positions. Logs: `bench/results/C5-int8/quality-*.txt`.

Per-domain, 1023 positions each:

| Corpus (domain) | PPL bf16 | PPL int8 | ΔPPL | top-1 | mean KL | p99 KL | max KL | mean \|Δlogit\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `docs/README.md` (EN/Markdown) | 24.0640 | 24.1423 | +0.325 % | 98.729 % | 4.19e-4 | 3.38e-3 | 6.32e-3 | 0.0249 |
| `docs/CHECKPOINTS.md` (ZH/EN) | 44.2416 | 44.7445 | +1.137 % | 99.120 % | 4.77e-4 | 2.67e-3 | 1.34e-2 | 0.0229 |
| `src/main.cpp` (code) | 9.0466 | 9.0210 | −0.282 % | 99.120 % | 3.11e-4 | 3.99e-3 | 4.82e-3 | 0.0205 |
| `models/config.json` (JSON) | 6.9965 | 7.0005 | +0.058 % | 99.511 % | 1.47e-4 | 1.02e-3 | 2.18e-3 | 0.0158 |
| **aggregate (4092 pos)** | **16.1117** | **16.1613** | **+0.308 %** | **99.120 %** | **3.39e-4** | **4.00e-3** (worst domain) | — | — |

The full `docs/README.md` (1885 positions) gives PPL 19.7301 → 19.7691
(+0.198 %), top-1 1867/1885 = 99.045 %, mean KL 3.21e-4, p99 KL 1.88e-3.

Gate verdict:

| Gate | Threshold | Measured | Verdict |
|---|---:|---:|---|
| Task target: PPL regression | ≤ +2 % | +0.31 % (aggregate; worst domain +1.14 %) | **PASS** |
| Task target: top-1 | ≥ 99 % | 99.12 % aggregate (worst 1K slice 98.73 %) | **PASS** (aggregate) |
| C5/CHECKPOINTS §3.4: top-1 | ≥ 99.5 % | 99.12 % | **FAIL** |
| C5/CHECKPOINTS §3.4: mean KL | ≤ 1e-3 | 3.39e-4 | PASS |
| C5/CHECKPOINTS §3.4: p99 KL | ≤ 5e-3 | 4.00e-3 (worst domain) | PASS |

So per the user's stated rule (≤2 % PPL regression, top-1 ≥ 99 %) the cache is
acceptable on the 4-domain 1K validation set, but it does **not** clear the
stricter 99.5 % project gate, and a 1K slice of one corpus (README) alone is
below 99 %. The flag is therefore opt-in and the default stays bf16.

## Performance — 4K decode

`MIORI_KV_INT8=1 ./build/miori bench --prefill 4096 --decode 128`
(log: `bench/results/C5-int8/bench.txt`, 3-run median):

| context | prefill bf16 | prefill int8 | decode bf16 | decode int8 |
|---|---:|---:|---:|---:|
| 1K | 2643 tok/s | 2618 tok/s | 92.1 tok/s | 92.2 tok/s |
| 4K | 1636 tok/s | 1622 tok/s | 88.3 tok/s | 87.9 tok/s |

**No decode speedup.** The 4K decode path is not KV-bandwidth-bound: halving the
369 MB/token attention read changes nothing measurable, while the int8 kernels
add per-element int→float conversion and a scale multiply. The win is capacity
(1.47 GB less KV at Bmax=32), not tok/s. Prefill is ~0.5–0.8 % slower in int8,
within run-to-run noise but consistently on the negative side.

## Correctness of the batch path

`MIORI_KV_INT8=1 ./build/miori batch-prefill-test --n 8 --repeat 1 --state-check 2 --long 423`
(log: `bench/results/C5-int8/batch-prefill-int8.txt`):

```
prefill speedup: 3.69x (33.42 -> 9.07 ms per request)
first tokens: 8/8 identical
first-token logits: 0/1024000 elements differ, max |diff| = 0
state check: 4/4 requests byte-identical (KV + conv state)
RESULT: packed prefill is bit-identical to the sequential path
```

The int8 state comparison covers int8 K/V bytes and both scale arrays. The bf16
path re-run is unchanged (0/518144 KV bytes and 0/9709568 long-prompt KV bytes
differ, same 3.62x speedup; log `batch-prefill-bf16.txt`).

A 64-prompt / 2-wave run (all 32 slots) also passes: 64/64 first tokens
identical, first-token logits 0/8,192,000 differ, packed prefill 0.654 s vs
2.067 s sequential = 3.16x (2270 tok/s).

Other checks:

- `MIORI_KV_INT8=1 prefix-test`: cold 0.271 s, snapshot 0.253 s, restore+gen
  0.053 s, cold/warm output identical (`prefix-int8.txt`);
- `MIORI_KV_INT8=1 batch-test`: 2 prompts, 19 tokens compared, 0 mismatches vs
  single-sequence greedy (`batch-test-int8.txt`);
- HTTP server with `MIORI_KV_INT8=1`: 6 concurrent exact requests, 2 sampling
  requests through the 4-snapshot prefix cache, all HTTP 200; `/metrics` ok.

## Not done / why the strict gate fails

- The scale is one bf16 value per `(kv_head, token)` as specified. The measured
  quantisation step is up to ~0.78 % of the token's max |K|/|V|, which is enough
  to flip 13/1023 near-tie top-1 decisions on the README slice.
- To reach the 99.5 % project gate the next levers are finer scales
  (per-32-element group, ~2x lower error, changes the layout and both GEMMs),
  an asymmetric zero point, or mixed K/V precision (the doc's suggested
  fallback). None of these are implemented.
- No paged KV, no per-page int8, no radix/shared-prefix cache; the int8 flag is
  process-global and `BatchEngine` captures the format at `init()`.
- BatchEngine int8 decode is implemented and correct, but no long-context
  (B=32 @4K) throughput benchmark was run; only the KV footprint is measured.
- The server has no metric exposing whether int8 is active (the startup log does).
