# STATUS — 2026-10-29
- C2: PARTIAL — packed multi-request prefill landed (64 x 23-token prompts 2.10 s -> 0.67 s, bit-identical, server default); still no paged KV or continuous batching.

## Delivered and measured

| Capability | State | Evidence |
|---|---|---|
| safetensors loader (732 tensors, 1.59 GB) | working | `miori info` |
| 4-bit GEMV / 8-bit GEMV | exact vs CPU | `C0`, `C6` |
| 24-layer text model + KV + conv state | working | `The capital of France is` → `Paris` |
| 128K BPE tokenizer + LFM3 chat template | working | `test-tokenizer` |
| MPP tensor-core prefill (bf16 cache) | 2694 tok/s @1K | `C1` |
| 4-bit decode | 97.5 tok/s (short ctx), 94.1 tok/s (1K ctx, 16-way split-K) | `C1` |
| OpenAI HTTP + SSE | working, serialized | `C3` |
| Vision tower + projector + splice | working in CLI (`vision-run`); 512² = 32 ms (warm) | `C4` |
| Vision on MPP tensor cores | 8.2× vs SIMT: 261 → 32 ms (512²), 2028 → 271 ms (1024²) | `C4` |
| Vision in HTTP API | working (one base64 image per request) | `C4` |
| Multi-image batch / vision queue | **not implemented** | `C4` |
| Paged KV / int8 KV | **not implemented** | `C2`, `C5` |
| Single-prefix KV cache | working in CLI and HTTP (one system slot) | `C5` |
| Fixed-slot batch decode (Bmax=32) | working, server-integrated (hybrid 1 / B=32) | `C2` |
| Packed multi-request prefill | working: 64 x 23-token prompts 2.10 s -> 0.67 s (3.13x), first-token logits / KV / conv state bit-identical | `C2` |

## Milestone verdicts

- M0 C0: GO with exceptions (ICB not attempted).
- M1 C1: GO with exceptions (97.5 vs 100 tok/s; full CPU top-1 matrix not run).
- M2 C2: PARTIAL — fixed-slot batch decode (Bmax=32) is server-integrated and
  packed multi-request prefill now makes 64 short prompts 3.1x faster
  (2.10 s -> 0.67 s) with bit-identical first-token logits, KV and conv state;
  paged KV and continuous (chunked) batching are still missing.
- M3 C3: PARTIAL (API works; 64-concurrency gate not met).
- M4 C4: GO — captions end to end (CPU cosine 0.9997), 512² 32 ms warm vs the
  40 ms gate, 1024² 271 ms vs 600 ms. Not wired into the HTTP API and no
  multi-image batch; fused flash attention would remove the remaining N² traffic.
- M5 C5: PARTIAL (single-prefix snapshot cache; no radix/int8/shared-prefix).
- M6 C6: PARTIAL (kernel-level validation only).
- M7 C7: PARTIAL (v0.1.0-core).

## Why

This session prioritised a correct, measured single-stream engine over breadth:
the 4-bit layout, attention, conv, prefill and serving path were debugged end to
end. The remaining milestones are substantial subsystems (paged KV + continuous
batching; vision tower) rather than tuning work, and were not started.

## Next (in order)

1. Paged KV + ragged continuous batching on top of `BatchEngine` (C2): merge
   prefill and decode in one step (a packed prefill pass currently blocks the
   batch), pack more than 32 requests / 1024-token prompts per pass, and lift the
   4-bit batched GEMV path back to a usable B ≤ 4 mode.
2. Vision in the serving path: image_url/base64 input, vision queue, MPP bf16
   GEMM for the tower (C4 latency), multi-image ragged batch.
3. int8 KV, prefix cache, shared-prefix attention (C5).
4. ICB replay, 24h soak, KL matrix (C5/C6).
