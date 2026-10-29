weights: 732 tensors, 1.59 GB
tokenizer: vocab=124893 merges=293320 specials=138
engine: bf16 weight cache 2.93 GB
engine: buffers ready, KV 92.3 MB (bf16)

kv-int8-test: docs/README.md, 511 positions (128 KL sample(s)), candidate K bf16 / V int8
KV bytes: Engine 92.27 MB -> 70.65 MB (76.6%); modelled BatchEngine Bmax=32 2952.8 MB -> 2260.7 MB (76.6%)
PPL: bf16 42.2184 -> K bf16 / V int8 42.2731 (delta +0.130%)
top-1 agreement: 507/511 = 99.217%
KL(ref||candidate) over 128 positions: mean 0.000126866, p99 0.000844636, max 0.00105721; mean |dlogit| 0.0158255
modelled attention KV read/token @4K: 369.1 MB bf16 -> 276.8 MB
quality gate (docs/CHECKPOINTS.md 3.4: top-1 >= 99.5%, mean KL <= 1e-3, p99 KL <= 5e-3): FAIL
success criterion (PPL delta <= +1% on top of the doc gate): FAIL

4K bench (prefill 4096 tokens, decode 32 tokens, greedy):
  bf16: prefill 2.490 s = 1644.9 tok/s, decode 0.368 s = 84.28 tok/s
  K bf16 / V int8: prefill 2.497 s = 1640.2 tok/s, decode 0.365 s = 84.88 tok/s
  decode speedup: 1.007x
