# STATUS — 2026-10-29 (v0.3.3)

Miori is a Metal + C++20 LFM3-VL-2B-MLX-4bit engine for the Apple M6 / 16 GB
Mac mini. Text, batching, vision, mixed-precision KV, prefix snapshots and
streaming are implemented and measured. The original 64x2K/4K @ 2.5K output
tok/s target is **NO-GO** with the current fixed-slot / packed-prefill design.

## Capability snapshot

| Capability | State | Evidence |
|---|---|---|
| safetensors loader (732 tensors, 1.59 GB) | working | `miori info` |
| 4-bit / 8-bit GEMV, MPP prefill | exact vs CPU | `C0`, `REGRESSION` |
| 24-layer text model + KV + conv state | working | `Paris` |
| Tokenizer + LFM3 chat template | working | `test-tokenizer` |
| Single-stream decode | 98 tok/s short, 94.1 @1K, 92.2 @4K | `C1` |
| Single-stream prefill | 2.8K @1K, 1.66K @4K | `BENCHMARKS` |
| Batch decode | Bmax=32, dynamic M, exact GEMV for <=4 slots | `C2` |
| Packed multi-request prefill | 64x23 tok: 2.10 s -> 0.67 s, bit-identical | `C2` |
| HTTP server | OpenAI chat, SSE streaming, per-request `exact` | `C3` |
| 64 concurrent (16 output tok) | ~1.94 s wall | `C3` |
| Vision tower + projector + splice | 512² 33 ms warm, 1024² 271 ms | `C4` |
| Image API | base64 `image_url`, multi-image, feature cache | `C4` |
| KV snapshots | 4-prefix LRU (system prompt) | `C5` |
| int8 KV (full) | 53.1% KV, top-1 99.34% (strict gate missed) | `C5-int8-kv-groups` |
| mixed KV `MIORI_KV_MIX=vk` | **gate-passing**: top-1 99.560%, PPL +0.019%, KV 76.6% | `C5-int8-kv-mixed` |
| Paged / radix KV | **not implemented** | `C2`/`C5` |
| Multi-image ragged tower batch | **not implemented** | `C4` |
| 24 h soak / KL matrix | **not done** (8 min / 12k-request soak done) | `C6` |

## Milestone verdicts

- **M0 C0**: GO with exceptions (ICB replay not attempted).
- **M1 C1**: GO with exceptions — 98 vs 100 tok/s short-context; full CPU top-1
  matrix not run.
- **M2 C2**: PARTIAL / 2K target **NO-GO**. Packed prefill is bit-identical and
  3.1x faster; dynamic-M batch and exact GEMV for <=4 slots work. No paged KV,
  no chunked continuous prefill, no prefill/decode overlap. A 3781-token prompt
  takes 2.33 s prefill; 64x2K would be ~64 x 1.2 s of prefill alone.
- **M3 C3**: PARTIAL — all main paths stream SSE per token; per-request `exact`;
  `/health`, `/metrics` (resident, latency histogram, queue, image cache).
  64-concurrency aggregate target not met.
- **M4 C4**: GO — real captions, CPU fp32 cosine 0.9997, 512² 33 ms warm
  (40 ms gate), 1024² 271 ms (600 ms gate), multi-image base64 requests,
  feature cache, streaming. No ragged multi-image tower batch/queue.
- **M5 C5**: PARTIAL — 4-prefix LRU snapshots; full int8 misses 99.5% top-1,
  mixed `vk` passes all four gates (top-1 99.560%, PPL +0.019%, mean KL
  1.46e-4, p99 1.24e-3). Opt-in; bf16 default. No radix/shared pages.
- **M6 C6**: PARTIAL — kernel tests, PPL, KV modes, 12,409-request mixed soak,
  resident high-water verified stable across 100 image requests. No 24 h run or
  full KL matrix.
- **M7 C7**: **v0.3.3 released** (v0.3.0 features + regression matrix);
  binaries, benchmarks, checkpoints and changelog are present.

## Known gaps (in order of impact)

1. **C2 64x2K/4K target**: needs chunked continuous prefill + paged KV and a
   faster MPP path at M=32/64; slot count alone is not the limit.
2. **Paged/radix KV**: per-slot 4K KV is 92 MB (49 MB full int8, 71 MB mixed);
   no page sharing/copy-on-write.
3. **C6 24 h soak / KL matrix**: 12k-request and 100-image runs are stable;
   24 h and a full domain matrix remain.
4. **Vision ragged batch/queue**: multi-image requests encode sequentially.
5. **Full int8 KV**: 99.34% top-1; needs asymmetric zero point, mixed K/V
   (already passing as `vk`), or higher precision.

## Reproduce

```bash
make clean && make -j
./build/miori run --raw --prompt "The capital of France is" --max-tokens 4
./build/miori bench --prefill 1024 --decode 32
./build/miori batch-prefill-test --n 32
./build/miori batch-sparse-test
./build/miori vision-run --image cat.jpg --prompt "Describe this image." --max-tokens 16
./build/miori serve --port 8080
```

See `docs/BENCHMARKS.md`, `docs/REGRESSION.md`, and `docs/checkpoints/`.
