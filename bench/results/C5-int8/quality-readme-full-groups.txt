weights: 732 tensors, 1.59 GB
tokenizer: vocab=124893 merges=293320 specials=138
engine: bf16 weight cache 2.93 GB
engine: buffers ready, KV 92.3 MB (bf16)

kv-int8-test: docs/README.md, 1885 positions (128 KL sample(s))
KV bytes: Engine 92.27 MB -> 49.02 MB (53.1%); modelled BatchEngine Bmax=32 2952.8 MB -> 1568.7 MB (53.1%)
PPL: bf16 19.7301 -> int8 19.7364 (delta +0.032%)
top-1 agreement: 1871/1885 = 99.257%
KL(ref||int8) over 128 positions: mean 0.00014421, p99 0.000581835, max 0.00109216; mean |dlogit| 0.0165933
modelled attention KV read/token @4K: 369.1 MB bf16 -> 184.5 MB int8
quality gate (docs/CHECKPOINTS.md 3.4: top-1 >= 99.5%, mean KL <= 1e-3, p99 KL <= 5e-3): FAIL
task target (PPL delta <= +2%, top-1 >= 99%): PASS
