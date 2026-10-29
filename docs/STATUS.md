# STATUS — 2026-10-29

## Delivered and measured

| Capability | State | Evidence |
|---|---|---|
| safetensors loader (732 tensors, 1.59 GB) | working | `miori info` |
| 4-bit GEMV / 8-bit GEMV | exact vs CPU | `C0`, `C6` |
| 24-layer text model + KV + conv state | working | `The capital of France is` → `Paris` |
| 128K BPE tokenizer + LFM3 chat template | working | `test-tokenizer` |
| MPP tensor-core prefill (bf16 cache) | 2694 tok/s @1K | `C1` |
| 4-bit decode | 97.5 tok/s | `C1` |
| OpenAI HTTP + SSE | working, serialized | `C3` |
| Vision / paged KV / int8 KV / prefix cache | **not implemented** | `C2`, `C4`, `C5` |

## Milestone verdicts

- M0 C0: GO with exceptions (ICB not attempted).
- M1 C1: GO with exceptions (97.5 vs 100 tok/s; full CPU top-1 matrix not run).
- M2 C2: NO-GO (no paged KV / continuous batching).
- M3 C3: PARTIAL (API works; 64-concurrency gate not met).
- M4 C4: NO-GO (vision tower not executed).
- M5 C5: NO-GO.
- M6 C6: PARTIAL (kernel-level validation only).
- M7 C7: PARTIAL (v0.1.0-core).

## Why

This session prioritised a correct, measured single-stream engine over breadth:
the 4-bit layout, attention, conv, prefill and serving path were debugged end to
end. The remaining milestones are substantial subsystems (paged KV + continuous
batching; vision tower) rather than tuning work, and were not started.

## Next (in order)

1. Paged KV + ragged continuous batching (C2) — unlocks the 高并发 targets.
2. Vision tower + projector + ImageIO (C4).
3. int8 KV, prefix cache, shared-prefix attention (C5).
4. ICB replay, 24h soak, KL matrix (C5/C6).
