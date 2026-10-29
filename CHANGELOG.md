# Changelog

## v0.3.4

- `/v1/models` reports `owned_by`, `context_length` and `modalities`.
- Clean build remains warning-free.

## v0.3.3

- Honor `top_p`, `top_k` and `seed` from chat requests; clamp `max_tokens`.
- Per-request `max_pixels` for image inputs (65536..4194304).
- 15-minute / 27,360-request soak with memory plateau (10.85 GB).
- C1 note: split-K threshold experiments were neutral and reverted; 100 tok/s
  depends on a fresh/cooldown device.
- Clean build: zero warnings.

## v0.3.2

- Startup warmup for Engine + BatchEngine (`MIORI_NO_WARMUP=1` to disable).
- 4K prompt/decode guards (400 for prompts >=4096, stop at position 4095).
- Reused 16 MB image embedding buffer; image-path resident high-water verified
  stable across 100 requests (10.82 GB).
- `/metrics`: resident bytes and request latency histogram.
- `batch-sparse-test` for sparse slot reuse.
- Docs refreshed to the v0.3.2 state; clean build has zero warnings.

## v0.3.1

- Remove unused private fields; clean build has zero warnings.
- Regression matrix: docs/REGRESSION.md.

## v0.3.0

- Mixed-precision KV: K bf16 + V int8 (`MIORI_KV_MIX=vk`) passes the strict
  quality gate (top-1 99.560%, mean KL 1.46e-4, PPL +0.019%), 76.6% KV memory.
- True SSE streaming for batch, exact/sampling and image requests.
- Per-request `exact: true`.
- `/health` reports version, KV mode and batch slots.
- 4-prefix LRU snapshot cache; dynamic-M batch decode.
- 64x2K C2 target documented as NO-GO.
