weights: 732 tensors, 1.59 GB
tokenizer: vocab=124893 merges=293320 specials=138
engine: bf16 weight cache 2.93 GB
engine: buffers ready, KV 92.3 MB (bf16)

kv-int8-test: docs/README.md, 511 positions (128 KL sample(s))
KV bytes: Engine 92.27 MB -> 49.02 MB (53.1%); modelled BatchEngine Bmax=32 2952.8 MB -> 1568.7 MB (53.1%)
PPL: bf16 42.2184 -> int8 42.2305 (delta +0.029%)
top-1 agreement: 507/511 = 99.217%
KL(ref||int8) over 128 positions: mean 0.000205626, p99 0.00127703, max 0.00234008; mean |dlogit| 0.0178697
modelled attention KV read/token @4K: 369.1 MB bf16 -> 184.5 MB int8
quality gate (docs/CHECKPOINTS.md 3.4: top-1 >= 99.5%, mean KL <= 1e-3, p99 KL <= 5e-3): FAIL
task target (PPL delta <= +2%, top-1 >= 99%): PASS

4K bench (prefill 4096 tokens, decode 32 tokens, greedy):
  bf16: prefill 2.489 s = 1645.7 tok/s, decode 0.354 s = 87.46 tok/s
  int8: prefill 2.501 s = 1637.4 tok/s, decode 0.357 s = 86.82 tok/s
  decode speedup: 0.993x
