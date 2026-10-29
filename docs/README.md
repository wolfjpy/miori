# Miori — LFM3-VL-2B 的 Metal 推理引擎

在 **Apple M6 / 16 GB** 上，为 [`LiquidAI/LFM3-VL-2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM3-VL-2B-MLX-4bit) 从零实现一个只使用 **Metal Shading Language + C++20** 的推理引擎。目标不是"能跑"，而是：

- **单流 decode 100–115 tok/s**，prefill 2.5–3.5K tok/s；
- **聚合吞吐优先**：agentic 64 并发（8K session、95% prefix hit）**1.6–2.0K output tok/s**；chat 32 并发 **0.8–1.0K output tok/s**；batch 64+，int8 KV，shared-prefix attention；
- 在 16 GB 统一内存内稳定运行，峰值占用 ≤11 GB；
- 文本 + 图像（不做视频/音频），OpenAI 兼容 SSE API。

---

## 1. 已冻结的决策

| ID | 决策 | 结论 |
|---|---|---|
| D1 | 目标硬件 | **仅 Apple M6，12 GPU 核，16 GB**。编译期常量；运行时检测到其他芯片直接报错，不做降级 |
| D2 | 模型 | LFM3-VL-2B-MLX-4bit，**权重数值不变**；只做无损重排（拼接 QKV/FFN、split planes、对齐 tile） |
| D3 | 模态范围 | **文本 + 图像**；视频/音频不做 |
| D4 | 有损优化 | int8 KV、4-bit lm_head **必须通过 KL gate** 才启用，否则回退 |
| D5 | API | OpenAI-compatible `/v1/chat/completions`，SSE 流式，base64 图像；JPEG/PNG 用 **ImageIO（C API）** 解码 |
| D6 | 计算栈 | MSL + C++20（metal-cpp）；**无 Python / ObjC / Swift / MPS / MPSGraph / BNNS / Accelerate** |
| D7 | 权重预处理 | 允许无损 repack，工具 `miori-pack`；原始 safetensors 仍可直接加载 |
| D8 | 优化目标 | **聚合吞吐优先**：batch 64+、int8 KV、shared-prefix attention；单流延迟为次要目标 |
| D9 | 并发口径 | 主调优点 **agentic 64 会话（8K session、95% hit、512 output）/ chat 32 并发（700 prompt、256 output）**；常驻容量 64×4K（int8 KV）；burst 128×2K；上限 128 交互 / 256 offline、32K ctx、8 GB KV |
| D10 | 验证 | **全部 Metal + C++**；不依赖外部 Python/PyTorch reference。质量基线用引擎 exact path + C++ CPU fp32 reference |

---

## 2. 目标工作负载

| 场景 | prompt | 输出 / turn | 并发 | 上下文 | 目标 output tok/s | 延迟 |
|---|---|---:|---:|---:|---:|---|
| Chat | 700 平均 / 1K p95 | 256 | 32 | ≤2K | 0.8–1.0K | ITL p99 ≤40 ms，TTFT p50 ≤0.5 s |
| Agentic | 8K session，95% prefix hit，200–400 new | 512 | 64 | 8K | **1.6–2.0K** | ITL p99 ≤30 ms，TTFT(hit) ≤150 ms |
| Mixed | 70% agentic + 30% chat | — | 64 | 4K | 1.4–1.7K（总 ≥2.8K） | ITL p99 ≤30 ms |
| Burst | 2K，80% hit | 512 | 128 | 2K | ≥1.5K | ITL p99 ≤50 ms |
| Long | 32K，80% hit | 128 | 8 | 32K | 50–60（cold 10–15） | 正确性优先 |
| Offline | 2K，80% hit | 512 | 256 | 2K | 1.5–1.8K | ITL ≤80 ms |

> **output tok/s 只统计输出 token**，total tok/s（prompt+output）另报。两者关系约为 `output ≈ total × O/(P+O)`：chat 的 700 prompt / 256 output 只有 ~27%，agentic 的 200 new / 512 output 有 ~72%。因此 agentic 才是吞吐型负载；冷启动的长 prompt chat 在 64 并发下会被 prefill 总算力拖到几百 output tok/s，必须靠 admission 和 prefix cache 控制。

**为什么常驻容量定在 64×4K：**

- 64 × 4K = 262K tokens；int8 KV ≈ 1.6 GB，fp16 KV ≈ 3.2 GB；
- 权重 1.59 GB + KV 1.6 GB + 状态 0.2 GB + 激活 0.3 GB ≈ 3.7 GB，留出 3–4 GB 给 prefix cache 和 32K 长请求；
- 纯 decode step 约 24 ms（125 GB/s、10 TFLOPS 有效值），64 / 0.024 ≈ 2.6K output tok/s；把一个 turn 的 prefill 算进去，output 约 1.4–1.8K，ITL ≈24 ms；
- 128 并发进入 compute-bound（约 41 ms/step），适合 burst/offline，不适合交互式 chat 的 p99；
- 32×8K 如果不共享 prefix，KV 读约 1.6 GB/step，output 只有 ~1.1K；shared-prefix attention 是 1.6K+ 的前提。

---

## 3. 非目标

- 视频 / 音频输入；
- 多机、多卡、跨进程张量并行；
- 其他 Apple Silicon 的兼容/降级；
- 训练、微调、量化算法研究；
- Python bindings、ObjC/Swift API、MPSGraph；
- 单流 >120 tok/s（除非 speculative decoding 生效）。

---

## 4. 文档

| 文档 | 内容 |
|---|---|
| [MILESTONES.md](MILESTONES.md) | M0–M7 里程碑、工作包、交付物、退出标准、排期、风险 |
| [CHECKPOINTS.md](CHECKPOINTS.md) | C0–C7 检查点门禁、测量协议、质量门、报告模板 |
| `docs/checkpoints/` | 每个检查点完成后生成的报告（`C0-baseline.md` … `C7-release.md`） |

---

## 5. 命名与单位约定

- 引擎名：**Miori**；二进制 `build/miori`，权重工具 `build/miori-pack`，基准工具 `build/bench-*`。
- 容量单位用十进制 GB（1 GB = 10^9 B），与 `MTLDevice.recommendedMaxWorkingSetSize` 一致。
- 里程碑编号 `M0…M7`；检查点编号 `C0…C7`，一个里程碑对应一个检查点。
- "Gate" 指必须量化通过、否则不得进入下一阶段的指标；"KL gate" 见 `CHECKPOINTS.md` §3。
