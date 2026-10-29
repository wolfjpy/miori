# Miori final report (v0.3.4)

## Scope

From-scratch Metal + C++20 inference engine for
`LiquidAI/LFM3-VL-2B-MLX-4bit` on the Apple M6 Mac mini (12 GPU cores, 16 GB,
macOS 27.0.1, Xcode 27.1). No Python, no Objective-C, no MLX runtime.

## What was built

- safetensors loader (732 tensors, 1.591 GB) into `MTL::Buffer`.
- 4-bit affine GEMV decode (16 warps/row) and 8-bit embedding GEMV; split-K
  attention for full layers.
- 24-layer language model: 13 gated conv (k=7), 5 sliding-256, 6 full attention,
  16Q/4KV, QK-norm, partial RoPE, sigmoid gate, sandwich norms.
- Prefill with a 2.93 GB bf16 tile-major weight cache and MPP `matmul2d`
  (64x64, 4 simdgroups), 128-token chunks.
- BatchEngine: 32 slots, packed multi-request prefill (`add_batch`), dynamic-M
  decode, exact 4-bit GEMV when <=4 low slots, continuous admission.
- Optional int8 KV: full int8 and mixed K/V (`MIORI_KV_MIX=vk`), 32-element
  group scales, snapshot/copy; bf16 default.
- 4-prefix KV snapshot cache (LRU by system prompt).
- Vision: ImageIO decode, smart_resize/bicubic, merge-block patchify, 14-layer
  tower (conv/attention), projector, image-token splice; bf16 MPP tile cache.
- OpenAI HTTP server: chat, SSE per-token streaming on batch/exact/sampling/image,
  multi-image base64, image feature cache, per-request `exact`, `max_pixels`,
  `top_p`/`top_k`/`seed`, `/health`, `/metrics`, `/v1/models`.
- CLI: `run`, `bench`, `ppl`, `prefix-test`, `batch-test`, `batch-prefill-test`,
  `batch-sparse-test`, `vision-run`, `vision-test`, `kv-int8-test`,
  `test-kv-int8`, kernel tests.

## Measured (fresh/cooldown where noted)

| Item | Result |
|---|---|
| decode short / 1K / 4K | best-case 98 / 94.1 / 92.2 tok/s; sustained 85-88 after cooldown |
| prefill 512 / 1K / 2K / 4K | 3118 / ~2800 / 2302 / 1664 tok/s |
| packed prefill 64x23 tok | 2.10 s -> 0.67 s (3.1x), bit-identical |
| batch B=4 exact / B=32 MPP | 19.6 ms/step / 46.8 ms/step (684 tok/s) |
| HTTP 4 / 16 / 64 concurrent (16 tok) | 0.29 / 1.00 / 1.94 s |
| vision 512² / 1024² | 33-35 ms / 271 ms |
| mixed KV `vk` | top-1 99.560%, PPL +0.019%, KV 76.6% |
| 15-min soak | 27,360 requests, all 200, resident plateaus 10.85 GB |

## Milestone verdicts

- M0/C0 GO (ICB not attempted).
- M1/C1 GO with exceptions: 100.4 was transient; sustained 85-88 tok/s
  after cooldown, so the 100 tok/s gate is not consistently met.
- M2/C2 PARTIAL: packed prefill + dynamic-M + fixed slots; **64x2K/4K @2.5K
  NO-GO** — 3781-token prompt takes 2.33 s prefill; no paged KV/chunked prefill.
- M3/C3 PARTIAL: full streaming/API works; 64-concurrency aggregate target not
  met.
- M4/C4 GO: captions, CPU cosine 0.9997, latency gates met, multi-image HTTP.
- M5/C5 PARTIAL: 4-prefix snapshots + mixed int8 KV passes the gate; no
  radix/paged sharing.
- M6/C6 PARTIAL: kernel/PPL/KL/27k soak; no 24 h soak or full KL matrix.
- M7/C7: v0.3.4 released with binaries, benchmarks, checkpoints, changelog.

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

## Known limits

Paged/radix KV, chunked continuous prefill, prefill/decode overlap, ragged
multi-image tower batch, 24 h soak and full KL matrix are not implemented.
Full int8 KV misses the strict 99.5% top-1 gate; mixed `vk` passes but is opt-in
and slightly slower at 4K. See `STATUS.md` and `BENCHMARKS.md` for details.
