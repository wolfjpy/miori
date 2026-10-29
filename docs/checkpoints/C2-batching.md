# C2 — paged KV + continuous batching

- Date: 2026-10-29
- Status: **NO-GO — not implemented**

## What exists

- A single-sequence KV cache: 11 attention layers × 4 KV heads × 4096 positions,
  head-major, bf16 (92 MB).
- Sliding layers use the same cache with a 256-token window; conv layers carry a
  6-token state.
- The decode path is single-request; `Engine::generate` resets all state.

## What is missing

- 64-token paged KV, block tables, COW, prefix sharing.
- Ragged `[M, 2048]` execution and continuous batching.
- Admission control, chunked prefill scheduling across requests.
- Batch invariance, page-leak, cancel tests.

## Implication

The 高并发 targets (64×2K / 64×4K, aggregate ≥2.5K output tok/s) are **not met**.
The server serializes requests behind a mutex.
