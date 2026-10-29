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

## Reproducible bench (added after the first report)

```
./build/miori bench --prefill 1024 --decode 128
prompt=1024 output=128
prefill: 0.403 s  2541.9 tok/s
decode:  1.360 s    94.1 tok/s  (114.0 GB/s effective)
```

At 1K context the decode rate drops from 97.5 (2-token context) to 94.1 because
the decode-attention kernel launches only 4 threadgroups per layer (one per KV
head) and is latency-bound. Split-K attention is the next step.

2K context (after 16-way split-K):

```
prompt=2048 output=64
prefill: 0.890 s  2302.4 tok/s
decode:  0.685 s    93.4 tok/s  (113.2 GB/s effective)
```

4K context:

```
prompt=4096 output=32
prefill: 2.462 s  1663.7 tok/s
decode:  0.347 s    92.2 tok/s  (111.8 GB/s effective)
```

## Short-context split-K tuning (negative, reverted)

Tried lowering the split-K threshold from 512 to 64 (16 splits) and 4 splits for
64..511: short-context decode moved only 85.6-87.8 -> 87.7-89.3 tok/s, within
noise. Reverted to `pos >= 512`, 16 splits.

After ~1 h of continuous benchmarking/soak the same binary measured 86-88 tok/s
at 1K (vs 100.4 tok/s on a fresh v0.3.2 run); this is device thermal/load
variance, not a code change. The C1 100 tok/s gate should be reported with the
measurement conditions (fresh boot/cooldown).

## Cooldown recheck

After a 5-minute idle cooldown the same v0.3.4 binary still measured 85.0-88.4
tok/s decode (1K and short context). The 100.4 tok/s observation during the
v0.3.2 regression was not reproducible; treat 85-88 tok/s as the sustained
number on this machine and 100.4 as a transient best-case. C1's 100 tok/s gate
is therefore **not consistently met**.
