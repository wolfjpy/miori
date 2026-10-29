weights: 732 tensors, 1.59 GB
tokenizer: vocab=124893 merges=293320 specials=138
engine: bf16 weight cache 2.93 GB
engine: buffers ready, KV 92.3 MB (bf16)

kv-int8-test: docs/README.md, 1023 positions (128 KL sample(s)), candidate K int8 / V bf16
KV bytes: Engine 92.27 MB -> 70.65 MB (76.6%); modelled BatchEngine Bmax=32 2952.8 MB -> 2260.7 MB (76.6%)
PPL: bf16 24.0640 -> K int8 / V bf16 24.0774 (delta +0.056%)
top-1 agreement: 1015/1023 = 99.218%
KL(ref||candidate) over 128 positions: mean 0.000147397, p99 0.000782251, max 0.00127938; mean |dlogit| 0.0186489
modelled attention KV read/token @4K: 369.1 MB bf16 -> 276.8 MB
quality gate (docs/CHECKPOINTS.md 3.4: top-1 >= 99.5%, mean KL <= 1e-3, p99 KL <= 5e-3): FAIL
success criterion (PPL delta <= +1% on top of the doc gate): FAIL
