# C6 — quality & stability

- Date: 2026-10-29
- Status: **PARTIAL** — unit kernels validated; no soak / KL matrix

## Validated

| Kernel | Method | Result |
|---|---|---|
| 4-bit GEMV | CPU dequant, 2048 rows | max_abs 1.5e-7 |
| 8-bit GEMV | CPU dequant, 129 rows | max_abs 1.7e-7 |
| GEMM | CPU dot, strides | max_abs 0 |
| decode attention | CPU softmax/PV | max_abs 1.8e-7 |
| prefill attention | CPU QK/softmax/PV | max_abs 2e-3 |
| conv k=7 | CPU FIR + state | max_abs 0 |

Not done: 2048-position KL matrix, int8 KV quality, 32K validation, 24h soak,
fault injection.
