# C5 — throughput optimisations

- Date: 2026-10-29
- Status: **NO-GO — not implemented**

Not done: int8 KV, 4-bit lm_head, radix prefix cache, shared-prefix attention,
ICB replay, prefill/decode overlap, speculative decoding.

What was done instead: a persistent bf16 tensor-core weight cache (2.93 GB) and a
64×64 MPP GEMM, which raised prefill from 462 to 2694 tok/s on a 1K prompt.

## Single-prefix state cache (added)

`Engine::snapshot()` / `restore()` copy the KV (per head, up to the current
position) and conv state; `prefix-test` exercises a 647-token system prefix:

```
prefix=647 suffix=17
cold: 0.264 s   cached restore+gen: 0.052 s (snapshot 0.247 s)
cold out: 4    warm out: 4
```

A repeated long system prompt restores in ~5 ms of copy instead of re-prefilling
647 tokens; output is identical. This is a single-slot system-prefix cache, not a
radix tree, and it is now wired into the HTTP server (one system-prefix slot, invalidated when the system text changes).

HTTP end-to-end (647-token system prompt, `max_tokens=1`):

```
cold: 0.349 s
warm: 0.042 s   (restore + suffix prefill + 1 token)
```

~8.4x lower latency on repeated system prompts; output is unchanged.
