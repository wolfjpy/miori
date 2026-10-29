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
