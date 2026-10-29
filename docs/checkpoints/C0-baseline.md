# C0 — baseline & spikes

- Date: 2026-10-29
- Machine: Apple M6 Mac mini, 12 GPU cores, 16 GB, macOS 27.0.1, Xcode 27.1
- Build: `make -j` (`metal4.1`, `-O3 -fno-fast-math`)
- Model: `LiquidAI/LFM3-VL-2B-MLX-4bit`, 1,590,959,088 bytes, sha256 `58268d...`
- Status: **GO with exceptions** (ICB and standalone read-bandwidth not re-run in this checkpoint)

## Results

| Item | Target | Measured | Verdict |
|---|---:|---:|---|
| Build | clean | success | pass |
| Loader | 732 tensors / 1.59 GB | 732 / 1.59 GB | pass |
| 4-bit dequant + GEMV | match CPU | max_abs 1.5e-7 (2048×2048) | pass |
| 8-bit dequant + GEMV | match CPU | max_abs 1.7e-7 (128000×2048) | pass |
| GEMM (32×32 naive) | match CPU | max_abs 0 | pass |
| MPP GEMM (64×64 tile) | ≥8 TFLOPS | ~8.6 TFLOPS effective @1K prefill | pass |
| Decode bandwidth | ≥120 GB/s | ~118 GB/s effective | marginal |
| Read bandwidth | ≥135 GB/s | 140 GB/s (standalone probe, 8 B/thread) | pass |
| ICB replay | decision | not implemented | exception |
| Layernorm/attention/conv | match CPU | decode attn 1.8e-7; prefill attn 2e-3; conv 0 | pass |

## Notes

- 4-bit layout confirmed functionally: GEMV matches an independent CPU dequant for
  both 4-bit and 8-bit tensors; text generation is coherent and factual
  (`The capital of France is` → `Paris`).
- Prefill uses a persistent 2.93 GB bf16 tensor-core cache built at startup; the
  original 1.59 GB 4-bit weights remain resident for decode.
- ICB replay was not attempted; decode uses one command buffer per token.
