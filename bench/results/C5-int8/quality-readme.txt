weights: 732 tensors, 1.59 GB
tokenizer: vocab=124893 merges=293320 specials=138
engine: bf16 weight cache 2.93 GB
engine: buffers ready, KV 92.3 MB (bf16)

kv-int8-test: docs/README.md, 1023 positions (128 KL sample(s))
KV bytes: Engine 92.27 MB -> 46.86 MB (50.8%); modelled BatchEngine Bmax=32 2952.8 MB -> 1499.5 MB (50.8%)
PPL: bf16 24.0640 -> int8 24.1423 (delta +0.325%)
top-1 agreement: 1010/1023 = 98.729%
KL(ref||int8) over 128 positions: mean 0.000418821, p99 0.00337689, max 0.00632085; mean |dlogit| 0.0248973
modelled attention KV read/token @4K: 369.1 MB bf16 -> 184.5 MB int8
quality gate (docs/CHECKPOINTS.md 3.4: top-1 >= 99.5%, mean KL <= 1e-3, p99 KL <= 5e-3): FAIL
task target (PPL delta <= +2%, top-1 >= 99%): FAIL

4K bench (prefill 4096 tokens, decode 32 tokens, greedy):
  bf16: prefill 2.481 s = 1650.7 tok/s, decode 0.355 s = 87.29 tok/s
  int8: prefill 2.491 s = 1644.1 tok/s, decode 0.357 s = 86.93 tok/s
  decode speedup: 0.996x
