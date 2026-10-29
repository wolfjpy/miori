# C3 — serving

- Date: 2026-10-29
- Status: **PARTIAL GO** — API works; concurrency gate not met

## Results

| Item | Target | Measured | Verdict |
|---|---:|---:|---|
| HTTP endpoint | OpenAI schema | `/v1/chat/completions` returns valid completion | pass |
| SSE | chunked stream + `[DONE]` | works (`Hello.` streamed) | pass |
| Prompt | LFM3 chat template | `Paris.` | pass |
| 64 concurrent | p99 TTFT/ITL | not tested; requests are serialized | fail |
| Backpressure | 429 | not implemented | fail |
| Memory | 1h stable | not soaked | partial |

## Notes

- `miori serve --port 8080` starts a thread-per-connection server; a global mutex
  serializes generation, so this is an API shim over the single-request engine,
  not a batching server.
- Streaming is simulated by chunking the already-generated text; TTFT equals full
  generation time.

## 4-slot batch server (added)

Greedy requests are batched up to 4 by a scheduler thread; prefill is sequential,
decode is shared. 4 concurrent 32-token requests complete in 0.682 s
(~188 output tok/s aggregate, ~1.9x single-stream). SSE is still emitted after
generation completes. 64-concurrency remains unmet.

## Continuous admission

The 4-slot server releases a finished slot and admits the next queued request
without draining the whole batch. 8 concurrent 16-token requests finish in
0.792 s (~162 output tok/s aggregate).

## 64-concurrent load + backpressure

64 concurrent requests, 16 greedy output tokens each:

```
5.80 s wall, 64/64 responses, ~176 output tok/s aggregate
```

All requests are admitted (thread-per-connection), but only 4 run per batch, so
this is 16 waves of 4 — aggregate is roughly the same as 4 concurrent, and p99
latency is ~5.8 s. The 64-concurrency gate is therefore still unmet.

Added: `miori_queue_depth` in `/metrics`, and a 429 response when the scheduler
queue exceeds 256 requests.

## Image input

`image_url` base64 (JPEG/PNG) is accepted and routed through the vision tower to
a captioned response. Text-only requests still use the 4-slot batch scheduler;
image requests take the single-sequence path under the Engine mutex.

## Hybrid routing (added)

The scheduler now picks a path per arrival pattern:

- queue < 4 and no active batch: serve one request through the single Engine
  (4-bit GEMV, exact, low latency);
- queue >= 4: fill up to 32 slots and decode with bf16 MPP (`gemm_mpp`) +
  bf16 lm_head.

Measured (16 greedy tokens per request):

```
single:        0.271 s
16 concurrent: 1.551 s   (~165 tok/s aggregate)
64 concurrent: 3.492 s   (~293 tok/s aggregate; was 5.80 s on B=4)
```

Outputs on the batch path are coherent (`The capital of France is` -> `Paris.`).
The batch path uses bf16 MPP and therefore is not token-identical to the 4-bit
single-stream path; the single path remains exact. Prefill is still sequential
and is now roughly half of the 64-concurrent wall time.

## Packed prefill on by default

`MIORI_BATCH_PREFILL` now defaults on. 64 concurrent 23-token prompts,
16 greedy tokens:

```
default (add_sequence): 2.94 s
packed prefill:         1.98-2.11 s  (our 16-token prompt; 1.49-1.60 s for
                                     shorter/EOS-early prompts)
```

Responses are byte-identical; the packed prefill pass is bit-identical to the
sequential path (first-token logits, KV, conv state).

## Exact mode

`MIORI_EXACT=1` disables the batch/MPP path and serves every greedy request
through the single Engine (4-bit GEMV + prefix cache). This is token-exact but
serial; default remains hybrid for aggregate throughput.

## Dynamic-M batch decode

Batch decode uses M = highest active slot + 1 and falls back to exact 4-bit GEMV
when ≤4 low slots are active. 16 greedy tokens:

```
4 concurrent:  0.291 s (exact 4-bit)
16 concurrent: 0.996 s
64 concurrent: 1.94 s
```

## True token streaming for the batch path (added)

For greedy requests served by the batch scheduler, the scheduler now appends each
newly sampled token to `job->out` and notifies the handler after every
`decode_step`; the handler sends the SSE header before generation and emits a
delta as soon as tokens arrive. Non-stream responses are unchanged.

Verified with 4 concurrent stream requests: the concatenated `delta.content`
reconstructs the full answer exactly, and 8 concurrent stream requests complete
in 0.78 s with 16 tokens each.

All main paths now stream per token: batch (per decode step), single
(`MIORI_EXACT=1`, temperature sampling) and image requests (`decode_loop_stream`
after `prefill_embeds`). The image stream test produced 10 events for 8 tokens
and the partial text `The image shows a close-up of an`.

### Single-path streaming (added)

`Engine::generate_from_state_stream(suffix, cfg, cb)` invokes the callback after
every sampled token. The server uses it for `MIORI_EXACT=1` and temperature>0
requests: SSE header is sent before prefill, then one delta per token. Image
requests still emit SSE after generation.

Verified: `MIORI_EXACT=1`, "What is 2+2?" -> 3 SSE events, text `4`;
temperature 0.7, 8 tokens -> 10 SSE events, text
"The sun is setting behind the mountains."

## Per-request exact mode

`{"exact": true}` on a chat request routes that request through the single
Engine (4-bit GEMV + prefix cache) even when the process default is the batch
path; it composes with `stream:true`. Verified: `What is 2+2?` with
`exact:true` -> `4`.

## Health endpoint

`GET /health` now reports runtime mode, e.g.:

```
{"status":"ok","version":"0.3.0","kv":"bf16","batch_slots":32,"exact":false}
```

`kv` reflects `MIORI_KV_MIX`/`MIORI_KV_INT8`; `exact` reflects the process-level
`MIORI_EXACT`.

## Startup warmup

The server now runs a one-token text warmup through both the single Engine and
the BatchEngine before accepting requests (touches all 24 layers, the bf16
weight cache, lm_head and the batch buffers). `MIORI_NO_WARMUP=1` disables it.

First `2+2` request after startup, 20 s after launch:

```
no warmup: 0.179 s, second 0.055 s
warmup:    0.148 s, second 0.048 s
```
