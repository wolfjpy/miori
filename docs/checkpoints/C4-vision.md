# C4 — vision (tensor-core re-run)

- Date: 2026-10-29
- Status: **GO** — captions work end to end and the latency gate is met
  (512×480 encode **31.6–34.5 ms** warm / 43.6 ms cold vs the 40 ms target;
  1024×992 **270.5–272.5 ms** vs the 600 ms target)

## What was built

| Piece | Implementation |
|---|---|
| Image decode | ImageIO / CoreGraphics C API (`CGImageSourceCreateWithURL` + RGB8 bitmap context); JPEG and PNG |
| Preprocess | `smart_resize` (factor 32, min/max pixels from the processor defaults 65 536 / 16 777 216), antialiased bicubic (a = −0.75, PIL/torchvision weight normalisation), `(x/255 − 0.5)/0.5`, patchify channel-major with the temporal pair duplicated |
| Patchifier | bf16 1536 → 1024 on tensor cores |
| Tower | 14 layers, `layer_types` from `vision_config`: 8 gated 3×3 depthwise-conv layers + 6 bidirectional GQA layers; pre-norm only; bf16 weights, fp32 accumulation |
| Conv layer | `in_proj` 1024 → 3072 split B,C,x; `gated = B·x`; merge-block → raster permute; zero-padded 3×3 depthwise (`spatial_conv` [1024,3,3,1]); `out = out_proj(C · conv)`; permute back |
| Attention | q 1024→1024, k/v 1024→512, out 1024→1024; 16 q heads / 8 kv heads / head_dim 64; per-head RMSNorm (eps 1e-5); axial 2D RoPE θ = 1e6 (16 freqs/axis, h/w concat + duplicate, pairs (d, d+32)); scale 1/√64; bidirectional, query blocks of 1024, keys padded to a multiple of 64 |
| MLP | w1/w3 1024→2560, SiLU(gate)·up, w2 2560→1024 |
| Projector | LayerNorm(1024, eps 1e-6) → 2×2 block reshape → 4096 → linear_1 + bias → exact GELU (A&S 7.1.26 erf, \|err\| ≤ 1.5e-7) → linear_2 4096→2048 + bias |
| Splice | `<\|vision_start\|>` 125009, N × `<\|image\|>` 125030, `<\|vision_end\|>` 125010; N = (H/32)·(W/32); 8-bit text embedding matrix built with `Engine::embed_tokens`, image rows replaced with projector output, prefilled by `Engine::prefill_embeds` |

New files: `src/vision.h`, `src/vision.cpp`, `shaders/vision.metal`.
Additive Engine API: `embed_tokens()`, `prefill_embeds()`; the existing
`prefill_chunk` / `forward` / `generate` path is untouched.
Additive ops: `gemm_mpp_off()` (byte offsets into A/B/C; `gemm_mpp` forwards to it).
New CLI: `vision-run`, `vision-test` (`--cpu-ref`, `--text-check`, `--dump-ppm`,
`--wstats`, `--repeat`).

## Tensor-core path (the latency fix)

`gemm_naive` was the whole cost of the first version (261 ms for 512×480). All
dense matmuls now run on `matmul2d`:

1. **Weight cache.** `tile_bf16` copies each bf16 `[N,K]` weight once at
   `Vision::init()` into the `[N/64][K][64]` tile-major layout that `gemm_mpp`
   consumes (the same layout `dequant_tile` produces for the 4-bit text
   weights). 40 tensors, **378 MB**, one `MTLBuffer` with per-tensor offsets
   (`woff_`). The original safetensors mapping is kept for the CPU reference and
   the non-GEMM kernels.
2. **Dense GEMMs.** Patchifier, q/k/v/o, FFN w1/w2/w3 and both projector
   matmuls go through `gemm_mpp` (N multiple of 64, contiguous A/C). The
   `gemm_naive` fallback stays in the dispatcher for layouts that do not fit
   (nothing currently takes it).
3. **Attention on tensor cores.** The old path used `gemm_naive` per
   (query block, head) with per-head byte offsets and a transposed V, which MPP
   cannot express. Instead the layer now keeps head-major views:
   - `vision_to_heads` → q `[16][N][64]`, k `[8][pseq][64]` (`pseq` = N padded to
     a multiple of 64, padding rows kept zero);
   - `tile_bf16` per kv head → k tiles `[8][pseq/64][64][64]`;
   - `vision_v_heads` → v `[8][pseq][64]` (its one-tile layout *is* the MPP
     layout, so no separate tiling);
   - QK `[1024×pseq×64]` and PV `[1024×64×pseq]` per head via `gemm_mpp_off`;
   - `vision_softmax` masks keys ≥ N and writes probability 0 for the padded
     columns, so the padded PV columns contribute nothing;
   - `vision_from_heads` merges the 16 head-major outputs back to `[N][1024]`
     for `out_proj` (MPP writes f32, so the head buffer is f32).
4. Activation buffers are pre-allocated for a 1024-patch image in `init()` so
   the first encode does not pay the first-touch cost.

## Validation

| Item | Method | Result |
|---|---|---|
| Tower + projector layout | independent scalar fp32 C++ reference (`Vision::cpu_reference`), same weights, 4×4 synthetic grid | worst per-token cosine **0.999706** (gate ≥0.9995), max abs diff **0.00462** (gate ≤0.01), no non-finite values |
| Preprocessing | dump resized RGB (`--dump-ppm`) and inspect | correct orientation, correct aspect (1600×1598 → 1024×992) |
| Image token count | formula (H/32)·(W/32) vs spliced rows | 992 tokens for 1024×992; 240 for 512×480; 2500 for 1600×1600 — all equal |
| Captions | 3 unrelated public-domain photos | all correct (below) |
| Text path | `test-gemv4`, `test-gemv8`, `test-gemm`, `test-attn`, `test-prefill-attn`, `test-conv`, `bench`, `run` | unchanged (numbers below) |
| `prefill_embeds` vs `prefill_ids` | 423-token prompt (4 chunks), compare all 128K logits | **max abs diff 0**, same argmax 597 — the splice path is bit-identical to the token path |

## Commands and results

```
make -j                                  # clean, no new warnings (4 pre-existing lm4_*)
```

```
./build/miori vision-test --image cat.jpg --max-pixels 65536 --cpu-ref --text-check
  decoded 1600x1598 -> resized 256x224, grid 16x14 = 224 patches, 56 image tokens
  cpu-ref: worst per-token cosine = 0.999706, max |gpu-cpu| = 0.00462, non-finite=0
  text-check: 423 tokens (multi-chunk), max |logit(ids) - logit(embeds)| = 0, argmax 597/597
```

```
./build/miori vision-run --image cat.jpg --prompt "Describe this image." \
    --max-tokens 32 --max-pixels 1048576
  image: cat.jpg  1600x1598 -> 1024x992, grid 64x62, 3968 patches, 992 image tokens
  features: 992 tokens x 2048, L2=45.334 mean=0.0002 std=0.0318 min=-0.373 max=0.289
  prompt: 1020 tokens (992 image tokens spliced), vision 289.6 ms
  prefill: 1020 tok 0.368s (2772 tok/s) | decode: 32 tok 0.334s (95.8 tok/s)
  --- output ---
  The image shows a close-up of an orange tabby cat with green eyes, looking
  directly at the camera. The background is blurred, highlighting the cat's ...
```

```
./build/miori vision-run --image gull.jpg --prompt "Describe this image." --max-tokens 32 --max-pixels 1048576
  2272x1704 -> 1152x864, 3888 patches, 972 image tokens
  The image shows a close-up profile of a seagull's head against a clear blue
  sky. The bird has white feathers, a yellow beak with a ...

./build/miori vision-run --image snowcat.jpg --prompt "Describe this image." --max-tokens 32 --max-pixels 1048576
  3000x2000 -> 1248x832, 4056 patches, 1014 image tokens
  The image shows a tabby cat walking through snow. The cat has brown and black
  stripes, green eyes, and is looking directly at the camera.<|im_end|>

./build/miori vision-run --image cat.jpg --prompt "Describe this image." --max-tokens 32
  (default max_pixels) 1600x1598 -> 1600x1600, 10000 patches, 2500 image tokens,
  vision 1752.6 ms
  The image shows a close-up of an orange tabby cat with green eyes, ...
```

Latency (same M6, `vision-test --repeat`; cold = first encode in the process):

| resized | patch grid | image tokens | before (gemm_naive) | cold | warm |
|---|---:|---:|---:|---:|---:|
| 256 × 224 | 16×14 | 56 | 70 ms | 14.4 ms | 8.5–13.6 ms |
| 512 × 480 | 32×30 | 240 | 261 ms | 43.6 ms | **31.6–34.5 ms** |
| 1024 × 992 | 64×62 | 992 | 2 028 ms | 284.9 ms | **270.5–272.5 ms** |
| 1600 × 1600 | 100×100 | 2 500 | 10 953 ms | 1 751.8 ms | 1 728–1 747 ms |

Speedup: 8.2× at 512×480, 7.5× at 1024×992, 6.3× at 2500 tokens. Text path
unchanged: `bench --prefill 1024 --decode 64` → 2 803.6 tok/s prefill, 95.1 tok/s
decode; `test-conv` 0.249 max-abs (the harness bf16 fix; before the fix the
reported 194.5 was a test artifact).

Profile of the 1024×992 encode (GPU stage time, `MIORI_VISION_PROFILE=1`):
attention **198 ms**, FFN 86 ms, conv layers 34 ms, projector 8 ms. The
remaining attention cost is memory traffic: materialising the 16 × N² score
matrices costs ~1 GB of writes + 1 GB of reads per layer at N=3968, so QK/PV run
at ~2 TFLOPS (K=64 gives low arithmetic intensity) while the FFN reaches
~10 TFLOPS.

## Verdict

- **Functionality: GO.** Real photos produce correct, specific captions with the
  image spliced into the language prompt; the text path is unchanged and the
  splice path is bit-identical to the token path.
- **Latency: GO.** 512×480 = 31.6–34.5 ms warm (43.6 ms cold) vs the 40 ms
  target; 1024×992 = 271 ms vs 600 ms; the processor-default 2500-token image is
  1.75 s (was 11 s).
- **Remaining work (not blocking):**
  - fused flash-style vision attention: the materialised score matrix is now the
    dominant cost above ~1 K patches; a fused QK/softmax/PV kernel would remove
    ~1.5 GB of traffic per layer and should bring 1024² to ~100 ms.
  - one image per request; no ragged multi-image batch, no vision queue, no
    feature cache, not wired into the HTTP server, video out of scope.
  - bicubic matches PIL/torchvision weight normalisation and uint8 rounding order
    but is not bit-exact vs torchvision.

## History

1. First C4 report (same date): **NO-GO — vision tower not executed**; the bf16
   tower and projector were loaded but unused and no image input existed.
2. Second run: tower + projector + ImageIO + splice implemented on
   `gemm_naive`; **GO for functionality, NO-GO for latency** (512² = 254 ms).
3. This report: all dense GEMMs and the attention moved to MPP tensor cores;
   **GO for functionality and latency**.

## HTTP image input (added)

`serve` now accepts OpenAI-style `image_url` data URLs (base64 JPEG/PNG). The
request path decodes to a temp file, preprocesses at max_pixels=1,048,576,
encodes the tower, splices the features into `<|image|>` tokens, and runs
`prefill_embeds` + decode under the Engine mutex. Text-only requests still use the
batch scheduler.

Measured:

```
POST /v1/chat/completions  content=[text, image_url(data:image/jpeg;base64,...)]
-> "The image shows a close-up of an orange tabby cat with green eyes, ..."
```

Multiple image_url parts are supported and compared in one prompt (each image is
encoded sequentially). Measured: cat + gull -> "The first image features a
close-up of an orange tabby cat with green eyes, while the second image shows a
seagull's head against a blue sky" (prompt_tokens 1998).

Limitations: images are encoded sequentially (no ragged tower batch, no vision
queue or feature cache), fixed max_pixels in the server, video out of scope.

## Image feature cache (added)

Server-side LRU (32 entries, keyed by 64-bit hash of the decoded bytes + size)
stores projected `[n_tokens][2048]` features. Repeated images skip preprocessing
and the vision tower entirely.

Cat image, `max_pixels=1048576` (992 image tokens), 16 output tokens:

```
first request:  0.964 s
second request: 0.534 s  (vision encode skipped)
```

Both return the same caption. Cache is in-memory only, capped at 32 images
(~128 MB at 992 tokens), and does not persist across server restarts.

`/metrics` now exposes `miori_image_cache_entries` (currently 32-entry LRU).
