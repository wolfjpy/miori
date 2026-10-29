# C4 — vision (re-run)

- Date: 2026-10-29
- Status: **GO for functionality, NO-GO for latency** (caption path works end to end;
  512² encode is 254 ms vs the 40 ms gate)

## What was built

| Piece | Implementation |
|---|---|
| Image decode | ImageIO / CoreGraphics C API (`CGImageSourceCreateWithURL` + RGB8 bitmap context); JPEG and PNG |
| Preprocess | `smart_resize` (factor 32, min/max pixels from the processor defaults 65 536 / 16 777 216), antialiased bicubic (a = −0.75, PIL/torchvision weight normalisation), `(x/255 − 0.5)/0.5`, patchify channel-major with the temporal pair duplicated |
| Patchifier | bf16 GEMM 1536 → 1024 (`gemm_naive`) |
| Tower | 14 layers, `layer_types` from `vision_config`: 8 gated 3×3 depthwise-conv layers + 6 bidirectional GQA layers; pre-norm only; bf16 weights, fp32 accumulation |
| Conv layer | `in_proj` 1024 → 3072 split B,C,x; `gated = B·x`; merge-block → raster permute; zero-padded 3×3 depthwise (`spatial_conv` [1024,3,3,1]); `out = out_proj(C · conv)`; permute back |
| Attention | q 1024→1024, k/v 1024→512, out 1024→1024; 16 q heads / 8 kv heads / head_dim 64; per-head RMSNorm (eps 1e-5); axial 2D RoPE θ = 1e6 (16 freqs/axis, h/w concat + duplicate, rotating pairs (d, d+32)); scale 1/√64; bidirectional, materialised per query block of 128 so no N² allocation |
| MLP | w1/w3 1024→2560, SiLU(gate)·up, w2 2560→1024 |
| Projector | LayerNorm(1024, eps 1e-6) → 2×2 block reshape → 4096 → linear_1 + bias → exact GELU (A&S 7.1.26 erf, \|err\| ≤ 1.5e-7) → linear_2 4096→2048 + bias |
| Splice | `<\|vision_start\|>` 125009, N × `<\|image\|>` 125030, `<\|vision_end\|>` 125010; N = (H/32)·(W/32); 8-bit text embedding matrix built with `Engine::embed_tokens`, image rows replaced with projector output, prefilled by `Engine::prefill_embeds` |

New files: `src/vision.h`, `src/vision.cpp`, `shaders/vision.metal`.
Additive Engine API: `embed_tokens()`, `prefill_embeds()`; the existing
`prefill_chunk` / `forward` / `generate` path is untouched.
New CLI: `vision-run`, `vision-test` (with `--cpu-ref`, `--dump-ppm`, `--wstats`).

## Validation

| Item | Method | Result |
|---|---|---|
| Tower + projector layout | independent scalar fp32 C++ reference (`Vision::cpu_reference`), same weights, 4×4 synthetic grid | worst per-token cosine **0.999628**, max abs diff 0.00608, no non-finite values |
| Preprocessing | dump resized RGB (`--dump-ppm`) and inspect | correct orientation, correct aspect (1600×1598 → 1024×992) |
| Image token count | formula (H/32)·(W/32) vs spliced rows | 992 tokens for 1024×992; 240 for 512×480; 2500 for 1600×1600 — all equal |
| Captions | 3 unrelated public-domain photos | all correct (below) |
| Text path | `test-gemv4`, `test-gemm`, `test-attn`, `test-prefill-attn`, `test-conv`, `bench` | unchanged; `test-conv`'s 194.5 max-abs deviation is identical on a clean HEAD worktree |
| `prefill_embeds` vs `prefill_ids` | 423-token prompt (4 chunks), compare all 128K logits | **max abs diff 0**, same argmax 597 — the splice path is bit-identical to the token path |

## Commands and results

```
make -j                                  # clean, no new warnings
```

```
./build/miori vision-test --image cat.jpg --max-pixels 65536 --cpu-ref --text-check
  decoded 1600x1598 -> resized 256x224, grid 16x14 = 224 patches, 56 image tokens
  vision encode: 69.8 ms
  cpu-ref on synthetic 4x4 grid (16 patches): 1.1 s
  cpu-ref: worst per-token cosine = 0.999628, max |gpu-cpu| = 0.00608, non-finite=0
  text-check: 423 tokens (multi-chunk), max |logit(ids) - logit(embeds)| = 0, argmax 597/597
```

```
./build/miori vision-run --image cat.jpg --prompt "Describe this image." \
    --max-tokens 32 --max-pixels 1048576
  image: cat.jpg  1600x1598 -> 1024x992, grid 64x62, 3968 patches, 992 image tokens
  features: 992 tokens x 2048, L2=45.339 mean=0.0002 std=0.0318 min=-0.373 max=0.287
  prompt: 1020 tokens (992 image tokens spliced), vision 2030.8 ms
  prefill: 1020 tok 0.369s (2767 tok/s) | decode: 32 tok 0.335s (95.6 tok/s)
  --- output ---
  The image shows a close-up of an orange tabby cat with green eyes, looking
  directly at the camera. The background is blurred, highlighting the cat's ...
```

```
./build/miori vision-run --image gull.jpg --prompt "What is in this picture?" \
    --max-tokens 40 --max-pixels 1048576
  2272x1704 -> 1152x864, 3888 patches, 972 image tokens, vision 1987.8 ms
  The image shows a close-up profile of a seagull's head against a clear blue
  sky. The bird has white feathers, a yellow beak with a small red spot near the
  tip, and ...

./build/miori vision-run --image snowcat.jpg --prompt "Describe this photo in one sentence." \
    --max-tokens 32 --max-pixels 1048576
  3000x2000 -> 1248x832, 4056 patches, 1014 image tokens, vision 2129.3 ms
  A brown tabby cat is walking through the snow, looking back towards the
  camera.<|im_end|>

./build/miori vision-run --image gull.png ...        # PNG path
  The image shows a close-up profile of a white bird's head, likely a seagull or ...

./build/miori vision-run --image cat.jpg --max-pixels 262144
  1600x1598 -> 512x480, 960 patches, 240 image tokens, vision 261.1 ms
  (prompt "Describe.", 8 tokens) An orange tabby cat with green eyes
```

Latency (same M6):

| resized | patch grid | image tokens | vision encode |
|---|---:|---:|---:|
| 256 × 224 | 16×14 | 56 | 70 ms |
| 512 × 480 | 32×30 | 240 | **261 ms** |
| 1024 × 992 | 64×62 | 992 | 2 028 ms |
| 1152 × 864 | 72×54 | 972 | 1 984 ms |
| 1600 × 1600 (default max_pixels) | 100×100 | 2 500 | 11 009 ms |

Text path unchanged: `bench --prefill 1024 --decode 32` → 2 804.4 tok/s prefill,
96.9–97.1 tok/s decode (C1: 2 694 / 97.5).

Raw log of every run in this report: `bench/results/C4/vision.txt`.

## Verdict

- **Functionality: GO.** Real photos produce correct, specific captions with the
  image spliced into the language prompt; the text-only path is unchanged.
- **Latency: NO-GO.** 512² is 254 ms vs the 40 ms gate (6.4×) because the tower
  runs on `gemm_naive` (SIMT bf16, ~650 GFLOPS effective) instead of the MPP
  tensor-core GEMM. Fix: convert the bf16 vision weights once at load into the
  `[N/64][K][64]` tile layout used by `dequant_tile`, then dispatch `gemm_mpp`.
  Expected 5–8× on the GEMM-bound layers.
- **Scope:** one image per request, no ragged multi-image batch, no separate
  vision queue, no feature cache, not wired into the HTTP server. Video is out of
  scope.

## History

The first C4 report (same date, before this work) was **NO-GO — vision tower not
executed**: the 328 MB bf16 tower and 50 MB projector were loaded but unused, and
no image input existed. This report supersedes it; the earlier conclusion is kept
here rather than deleted.
