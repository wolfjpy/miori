# Miori

Metal + C++20 inference engine for `LiquidAI/LFM3-VL-2B-MLX-4bit` on Apple M6 / 16 GB.
No Python, no Objective-C, no MLX runtime: hand-written Metal kernels and a C++ host.

## Status (v0.1.0-core)

| Area | State |
|---|---|
| Text model (24 layers, 4-bit + 8-bit embed) | working |
| Tokenizer (128K BPE) + LFM3 chat template | working |
| Prefill (MPP tensor-core GEMM, bf16 weight cache) | working, ~2.7K tok/s @1K |
| Decode (4-bit GEMV) | working, ~97 tok/s |
| OpenAI-compatible HTTP + SSE | working, serialized |
| Batch decode (fixed slots) | working, Bmax=32, server-integrated |
| Packed multi-request prefill (`add_batch`) | working: 64 × 23-token prompts 2.10 s → 0.67 s (3.1×), bit-identical |
| Vision tower (patchifier, 14 layers, projector) | working: `vision-run` captions photos |
| Vision on MPP tensor cores (bf16 tile-major cache) | working: 8.2× faster than the SIMT path |
| Vision in the HTTP API | working, one base64 image per request |
| Multi-image batch / vision queue | **not implemented** |
| Paged KV / continuous batching | **not implemented** |
| int8 KV (opt-in) | working behind `MIORI_KV_INT8=1`: Engine + BatchEngine, decode/prefill/snapshot; KV bytes halved (2.95 GB → 1.50 GB at Bmax=32); PPL +0.31 %, top-1 99.12 % (misses the 99.5 % strict gate, so bf16 is default) |
| Shared-prefix attention / radix cache | **not implemented** |

Measured on the target M6/16GB (headless, 3-run median where applicable):

```
prefill 1002 tokens: 0.372 s = 2694 tok/s
decode 128 tokens:   1.311 s =  97.5 tok/s  (~118 GB/s effective)
server:              OpenAI /v1/chat/completions + SSE
vision 512x480 image: 32 ms    (960 patches -> 240 image tokens, warm)
vision 1024x992 image: 271 ms  (3968 patches -> 992 image tokens, warm)
```

## Build

Requires macOS 27.x, Xcode 27.1, Metal 4.1.

```bash
make -j
```

## Model

Place these files in `models/`:

```
model.safetensors        # 1.59 GB, from LiquidAI/LFM3-VL-2B-MLX-4bit
tokenizer.json
config.json
chat_template.jinja
```

The model is loaded with a C++ safetensors parser; no conversion step is required.
The engine builds a bf16 tensor-core cache (2.93 GB) at startup and keeps the
original 4-bit weights for decode.

## Run

```bash
./build/miori run --raw --prompt "The capital of France is" --max-tokens 32
./build/miori run --prompt "Explain RMSNorm." --max-tokens 128
./build/miori serve --port 8080

# Vision: image tokens are spliced into the language prompt
./build/miori vision-run --image cat.jpg --prompt "Describe this image." \
    --max-tokens 32 --max-pixels 1048576
./build/miori vision-test --image cat.jpg --cpu-ref    # feature stats + CPU reference
```

`--max-pixels` bounds the smart-resized area (the processor default is
16 777 216, i.e. up to 2 500 image tokens for a 1600×1600 image and ~11 s of
vision compute; 1 048 576 gives 992 tokens and ~2 s). The image token count is
always `(H/32)*(W/32)` after resize and is printed at runtime.

OpenAI API:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
```

## Tests

```bash
./build/miori test-gemv4      # 4-bit GEMV vs CPU dequant
./build/miori test-gemv8      # 8-bit GEMV vs CPU dequant
./build/miori test-gemm       # GEMM vs CPU
./build/miori test-attn       # decode attention vs CPU
./build/miori test-prefill-attn
./build/miori test-conv
./build/miori test-tokenizer
./build/miori vision-test --image cat.jpg --cpu-ref       # vision vs fp32 CPU reference
./build/miori vision-test --image cat.jpg --text-check    # embed+prefill == token prefill (0 diff)
./build/miori vision-test --image cat.jpg --repeat 3     # cold/warm encode latency
./build/miori vision-test --image cat.jpg --dump-ppm /tmp/resized.ppm  # check preprocessing
./build/miori bench --prefill 1024 --decode 128
./build/miori batch-test --prompts "The capital of France is" "List the first four primes."
./build/miori batch-test --kernels gemv4 --prompts "..." "..."   # batched 4-bit GEMV path
./build/miori test-mpp-tile          # 32- vs 64-row MPP tile numerics (bit-identical)
./build/miori batch-prefill-test     # 64 packed prompts vs sequential prefill: timing + logits/KV diff
./build/miori batch-prefill-test --n 64 --repeat 3 --state-check 4 --long 423 [--words]
./build/miori test-kv-int8          # int8 KV writers/attention/prefill GEMM vs CPU reference
./build/miori kv-int8-test --file docs/README.md --tokens 1024  # bf16 vs int8 quality gate
```

## Known limitations

- The HTTP server serializes requests behind a mutex; the 64×2K / 64×4K
  concurrency targets from `docs/MILESTONES.md` are not met.
- Fixed-slot batch (Bmax=32) with contiguous per-slot KV; no paged KV and no
  continuous batching. Packed prefill covers up to 32 requests / 1024 tokens per
  pass (`MIORI_BATCH_PREFILL` in the server, on by default; set `=0` to disable); longer prompts
  fall back to the sequential prefill. See `docs/checkpoints/C2-batching.md`.
- No paged KV and no shared-prefix/radix cache (only 4 full-prefix snapshots).
  The optional int8 KV (`MIORI_KV_INT8=1`) halves KV memory but misses the
  project's 99.5 % top-1 gate (99.12 % aggregate, PPL +0.31 %), so bf16 is the
  default; see `docs/checkpoints/C5-int8-kv.md`.
- Vision works from the CLI only (`vision-run`, `vision-test`); the HTTP API has
  no image input yet, there is no multi-image batch and no vision feature cache.
- Vision attention still materialises the 16 × N² score matrices (about 2 GB of
  traffic per layer at 1024²); above ~1K patches that is the dominant cost
  (198 ms of the 271 ms). A fused flash-style QK/softmax/PV kernel is the next
  step; the dense GEMMs and QK/PV themselves run on tensor cores.
- The vision tower needs a 378 MB bf16 tile-major weight cache in addition to the
  1.59 GB safetensors mapping and the 2.93 GB text bf16 cache.
- Prefill chunks are 128 tokens; the bf16 cache costs 2.93 GB of the 12.7 GB
  working set, and the packed-prefill activations another 175 MB.
- Greedy decoding is exact; temperature/top-k/top-p are implemented on the CPU.

## Docs

- [docs/README.md](docs/README.md) — frozen decisions and targets
- [docs/MILESTONES.md](docs/MILESTONES.md) — M0–M7 plan
- [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) — gates and measurement protocol
- [docs/checkpoints/](docs/checkpoints/) — actual checkpoint reports (C0–C7)

## Benchmark

```bash
./build/miori bench --prefill 1024 --decode 128
./build/miori batch-test --prompts "The capital of France is" "List the first four primes."
./build/miori batch-test --kernels gemv4 --prompts "..." "..."   # batched 4-bit GEMV path
./build/miori batch-prefill-test --n 64 --repeat 5 --state-check 4 --long 423
```

Reported on the target M6: prefill 2542 tok/s, decode 94.1 tok/s at 1K context
(97.5 tok/s at short context), ~114 GB/s effective.

4K context: prefill 1663.7 tok/s, decode 92.2 tok/s.

Packed multi-request prefill, 64 x 23-token prompts (2 passes of 32, median of 5):

```
sequential Engine::prefill_ids : 2.095 s =  708 tok/s (32.7 ms/request)
packed BatchEngine::add_batch  : 0.669 s = 2218 tok/s (10.5 ms/request)   3.13x
first tokens 64/64 identical, first-token logits 0/8192000 differ, KV+conv state byte-identical
```

64 concurrent HTTP requests (23-token prompts, greedy, 16 max_tokens):
2.94 s -> 1.51 s with `MIORI_BATCH_PREFILL=1`, identical response bodies.

## Metrics

`GET /metrics` exposes `miori_requests_total` and `miori_output_tokens_total`.

## Perplexity

```bash
./build/miori ppl --file /tmp/en.txt --tokens 200
./build/miori ppl --file docs/README.md --tokens 1024 --kv-int8   # int8 KV PPL
./build/miori kv-int8-test --file docs/README.md --tokens 1024  # bf16 vs int8: PPL/top-1/KL + 4K decode
./build/miori test-kv-int8                                      # int8 KV kernels vs CPU reference
```
