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
