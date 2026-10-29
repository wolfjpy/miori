# Contributing

## Build

```bash
make clean && make -j
```

Requires macOS 27.x, Xcode 27.1 and Metal 4.1. Model weights are intentionally
not committed; copy `model.safetensors`, `tokenizer.json`, `config.json` and
`chat_template.jinja` into `models/` before running the binary.

## Before opening a PR

Run the relevant checks from `docs/REGRESSION.md`:

```bash
./build/miori run --raw --prompt "The capital of France is" --max-tokens 4
./build/miori bench --prefill 1024 --decode 32
./build/miori batch-prefill-test --n 32
./build/miori batch-sparse-test
./build/miori vision-test --image cat.jpg --max-pixels 262144 --cpu-ref
```

Keep `make -j` warning-free. Do not commit model weights, images, build outputs,
logs, tokens or credentials. Report security issues via `SECURITY.md`, not a
public issue.

## Style

C++20, Metal 4.1, no Python/Objective-C/Swift in the engine. Keep kernels and
host wrappers paired, document layout/offset assumptions, and add a checkpoint
note under `docs/checkpoints/` for performance or correctness changes.
