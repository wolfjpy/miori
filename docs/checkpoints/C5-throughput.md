# C5 — throughput optimisations

- Date: 2026-10-29
- Status: **NO-GO — not implemented**

Not done: int8 KV, 4-bit lm_head, radix prefix cache, shared-prefix attention,
ICB replay, prefill/decode overlap, speculative decoding.

What was done instead: a persistent bf16 tensor-core weight cache (2.93 GB) and a
64×64 MPP GEMM, which raised prefill from 462 to 2694 tok/s on a 1K prompt.
