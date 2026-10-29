# Changelog

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
