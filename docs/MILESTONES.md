# MILESTONES — Miori

> 目标硬件：Apple M6 / 12 GPU 核 / 16 GB / macOS 27.x / Metal 4.1
> 目标模型：`LiquidAI/LFM3-VL-2B-MLX-4bit`（1.591 GB，4-bit affine g32，embed 8-bit，vision bf16）
> 目标：聚合吞吐优先的文本 + 图像推理引擎，Metal + C++20，无 Python/ObjC。

---

## 1. 架构基线（一句话版）

**以 ragged continuous batching 为核心**：每个 step 把所有活跃请求的 token 拼成一个 `[M, 2048]` 激活矩阵，GEMM 共享权重；full-attention 层用 64-token page 的 paged KV（int8，COW 共享），sliding 层用 256-token ring，conv 层用 6-token 状态；prefill 按 128-token chunk 与 decode 混批；decode 用固定 batch bucket 的 ICB 回放压低 CPU 提交开销；vision 在独立队列编码后 splice 到 embedding。

关键数字（125 GB/s 持续读、10 TFLOPS 有效算力）：

- 单流 decode：1.212 GB/token → **9.7 ms → ~103 tok/s**（理论上限 115）；
- prefill：3.2 GFLOP/token → **3.1K tok/s**（区间 2.5–3.5K）；
- 64 并发 × 4K、int8 KV：约 24 ms/step；纯 decode 约 **2.6K output tok/s**，整 turn 约 **1.4–1.8K output tok/s**，ITL ~24 ms；
- agentic 64 并发（8K session、95% hit、200 new / 512 output）：**1.6–2.0K output tok/s**；
- KV：fp16 12.3 KB/token，int8 6.24 KB/token；每 seq 固定状态约 3.0 MB。

> output tok/s 只统计输出 token；total tok/s（prompt+output）另报。两者的比例约为 `O/(P+O)`，所以 chat 长 prompt 的 output 吞吐天然低于 agentic；这不是引擎缺陷，是 prefill 算力预算。

---

## 2. 里程碑总表

| 里程碑 | 周期 | 主题 | 出口检查点 | 一句话退出标准 |
|---|---:|---|---|---|
| **M0** | 第 1–2 周 | 基线与 spike | C0 | 构建、加载、布局、GEMV/GEMM、ICB 五个 spike 有明确结论 |
| **M1** | 第 3–5 周 | 单请求文本引擎 | C1 | 单流 ≥100 tok/s，prefill ≥2.5K，输出与 CPU reference 一致 |
| **M2** | 第 6–9 周 | 分页 KV + 连续批处理 | C2 | 64×2K 正确，总 token ≥2.2K，batch invariance 通过 |
| **M3** | 第 10–12 周 | 服务化 | C3 | OpenAI SSE 64 并发，p99 TTFT/ITL 达标，无泄漏 |
| **M4** | 第 13–15 周 | 视觉 | C4 | 512² 图像 ≤40 ms，caption 可用，文本无回归 |
| **M5** | 第 16–18 周 | 吞吐优化 | C5 | int8 KV/4-bit lm_head 过 KL gate，64 mixed ≥1.6K output，prefix 生效 |
| **M6** | 第 19–20 周 | 质量与稳定性 | C6 | 24h soak、32K 上下文、错误注入、最终 KL 报告 |
| **M7** | 第 21 周 | 发布 | C7 | v0.1.0 可复现构建 + 基准报告 + 已知限制 |

> 单人或 1.0 FTE 的估算；2–3 人可将 M3/M4 并行、M5 提前，压缩到 16–18 周。M0 的结论可能改变 M1/M2 的 kernel 方案，因此 M0 不允许被压缩成"只搭骨架"。

---

## 3. 详细里程碑

### M0 — 基线与 spike（第 1–2 周）

**目标**：在写引擎之前，把五个会改变架构的问题测掉；建立可复现的构建、加载、基准与文档骨架。

**工作包**

1. **构建系统**：`Makefile`；`xcrun metal -std=metal4.1 -O3 -fno-fast-math`；clang++ `-std=c++20 -O2`；metal-cpp 固定版本；`config/m6.h` 作为 host/MSL 共享常量（GPU 核数、tg 内存、page 大小、模型几何）。
2. **硬件门禁**：启动时校验 `MTLDevice.name == "Apple M6"`、12 GPU 核、`recommendedMaxWorkingSetSize ≥ 12 GB`；不匹配直接报错退出（D1）。
3. **权重加载器**：C++ 解析 safetensors header；校验 732 个 tensor 的 dtype/shape/offset；`mmap`/`pread` 到 1.59 GB `MTLBuffer`；256 B 对齐；记录每个 tensor 的 descriptor。
4. **MLX 4-bit 布局探针**：确认 nibble 顺序（8 个/uint32，低 nibble 在前）、group=32、scale/bias 的 bf16 解释、embed 8-bit 的 4/uint32 布局。用 base bf16 checkpoint 抽样 128 行做 cosine（≥0.95；QAD 会使权重整体偏移）验证；不通过则先修 loader，禁止继续。
5. **CPU fp32 reference**：C++ 慢速路径（dequant + fp32 矩阵乘 + 精确 RMSNorm/RoPE/softmax），先支持单层，再支持整模型（仅用于 256 个 position 的抽查）。
6. **微基准**：
   - 1.2 GB 流式读（目标 ≥135 GB/s）；
   - 2048×2048 4-bit GEMV（目标 ≥120 GB/s）；
   - MPP `matmul2d` + **threadgroup B tile**（dequant 到 8 KB tg，double buffer），M=8/16/64，目标 ≥8 TFLOPS @ M=512；
   - paged attention 骨架（2K/8K，B=1/32）；
   - 175 次 dispatch 的 CPU 编码开销；ICB 回放可行性。
7. **文档**：`docs/checkpoints/C0-baseline.md`；把每个 spike 的原始日志、命令、结论写清楚。

**交付物**：`build/miori`（仅 `--version`/`--probe`）、`build/bench-*`、`config/m6.h`、`docs/checkpoints/C0-baseline.md`。

**退出标准（C0）**：见 `CHECKPOINTS.md` C0；任一 spike 失败必须给出替代方案和重新估算，不能"先写着看"。

**风险/回退**：
- MPP 不接受 threadgroup B → 回退到 per-layer 全局 fp16 scratch（+10–15% prefill），M1 仍可继续；
- MLX 布局不符 → 只影响 loader，1–2 天修正；
- ICB 不可用 → Metal 4 低开销 encoder + 更激进 fusion，CPU 预算从 0.5 ms 收紧到 0.3 ms。

---

### M1 — 单请求文本引擎（第 3–5 周）

**目标**：一条请求、greedy、端到端跑通 24 层，数值可信，速度达到单流目标。

**工作包**

1. **Tokenizer（C++）**：128K vocab、293K merges；Unicode pre-tokenizer（ICU C API 或生成表）；added/special tokens 用 trie 优先匹配；byte-level 编解码；增量 detokenizer（UTF-8 边界）。
2. **Chat template（C++）**：LFM3 的 `<|im_start|>`、`<think>`、tools、`<|vision_start|><|image|><|vision_end|>`；20 个 golden case。
3. **Kernel v1**：
   - 4-bit GEMV（split planes，16 B q + 4 B sb，warp-per-row / 2-row / 4-row 三版选优）；
   - RMSNorm（fp32）、RoPE（full 0.5 partial / sliding full）、sigmoid gate；
   - full attention（连续 KV 即可，M1 不做 paging）、sliding attention（256 ring）、width-7 depthwise conv；
   - 8-bit lm_head GEMV；GPU sampling（temperature/top-k/top-p/repetition penalty）。
4. **24 层图**：QKV+gate 拼接、w1/w3 拼接、SiLU×up epilogue、post-norm+residual+next-norm 融合；每层约 7 个 dispatch。
5. **CLI**：`miori run --model ... --prompt ... --max-tokens ...`；`miori tokenize`；`miori logits`。
6. **数值验证**：逐层 vs CPU reference（max abs / cosine）；整模型 256 position top-1 ≥99.5%；NaN/Inf 检查。

**交付物**：`build/miori` 可单请求生成；`build/bench-gemv`、`bench-prefill`；`C1` 报告。

**退出标准（C1）**：单流 decode ≥100 tok/s（512 ctx），prefill ≥2.5K tok/s（1K prompt），top-1 ≥99.5% vs CPU fp32（256 pos，允许 ≤1 次翻转），tokenizer 10 MB 语料 100% round-trip。

**风险/回退**：
- Tokenizer regex 不一致 → ICU 精确实现，最多 1 周；
- 单流 <100 tok/s → 先查 GEMV 带宽（目标 ≥120 GB/s），再查 dispatch 数；不允许用"降精度"掩盖。

---

### M2 — 分页 KV + 连续批处理（第 6–9 周）

**目标**：从"一条请求"变成"64 条并发、ragged、chunked prefill"的引擎；这是 高并发 的地基。

**工作包**

1. **Page pool**：page=64 token；一个 page 覆盖 6 个 full 层（768 KB/page）；物理 page 池 + free list + refcount；block table per request。
2. **Ragged executor**：`[M,2048]` 激活；`cu_seqlens`、position、page table、state 指针；所有 GEMM 按 M 走；lm_head 只算需要的行。
3. **KV 写入/读取**：prefill chunk 写入 page；decode append；COW（refcount>1 时复制尾页）；sliding ring 与 conv state 的 per-request 更新。
4. **Scheduler v1**：token budget 2048、max 128 交互 / 256 offline；decode-first；chunked prefill 128（有 decode 时）/1024（空闲时）；admission（KV page + max context）；取消/超时；简单 fairness（round-robin + aging）。
5. **批量 attention**：varlen causal + sliding mask；GQA 4:1；decode split-K（>2K）与 prefill tile 两条路径。
6. **测试**：batch invariance（batch 64 greedy == batch 1，20 条 prompt，token ID 完全一致）；8K prompt 分 chunk vs 不分 chunk 一致；100 次随机 cancel 无 page 泄漏。

**交付物**：`miori serve` 雏形（暂不含 HTTP）；`bench-batch`；`C2` 报告。

**退出标准（C2）**：64×2K 冷 prompt 的 **总 token 吞吐 ≥2.2K/s**（output 单独报告，约 300–400/s），ITL p99 ≤30 ms，batch invariance 通过，1h 无泄漏。

**风险/回退**：
- Page 粒度 64 导致尾页浪费 → 先测 32/64/128，按 4K 上下文的内部碎片选；
- varlen attention 出现数值/掩码 bug → 保留"每请求单独 kernel"的慢路径作为对照，但不得作为发布路径。

---

### M3 — 服务化（第 10–12 周）

**目标**：把引擎变成可被真实客户端使用的 OpenAI 兼容服务；吞吐不能因 HTTP/序列化掉下来。

**工作包**

1. **HTTP/1.1 + SSE**：kqueue、keep-alive、chunked、背压；acceptor + worker 线程；无阻塞热路径。
2. **OpenAI schema**：`/v1/chat/completions`、`/v1/models`、`/health`、`/metrics`；`stream`、`max_tokens`、`temperature`、`top_p`、`top_k`、`stop`、`seed`、`tools`、`chat_template_kwargs`（`enable_thinking`）。
3. **JSON**：手写 parser/serializer；限制嵌套/长度；错误码；请求体大小上限。
4. **图像入口**：base64 data URL 解析（真正的视觉在 M4）；请求校验。
5. **流式 detokenizer**：按 token 增量输出；多字节 UTF-8 不截断；`finish_reason`、usage。
6. **可观测性**：TTFT/ITL 直方图、batch size、KV 使用、queue depth、prefix hit（M5 起）、每 kernel GPU 时间（抽样）。
7. **压测工具**：C++ `load-test`，N 个并发 SSE 客户端，开环/闭环两种；输出 p50/p99。

**交付物**：`build/miori serve`；`build/load-test`；`C3` 报告。

**退出标准（C3）**：64 并发 SSE 下 chat p99 TTFT ≤0.8 s（1K cold）、agentic p99 TTFT ≤150 ms（95% hit）、p99 ITL ≤30 ms；`mixed-agentic` output ≥1.4K tok/s、total ≥2.6K tok/s；429/取消/断连行为正确；1h 内存稳定。

**风险/回退**：
- SSE 每 token flush 的系统调用开销 → 合并 2–4 token 或 10 ms 定时 flush；
- JSON 解析成为热点 → 固定 schema 快路径 + 长度上限。

---

### M4 — 视觉（第 13–15 周）

**目标**：文本 + 图像端到端；视觉不阻塞文本 decode；图像 token 有上限，延迟可预测。

**工作包**

1. **图像预处理**：ImageIO 解码 JPEG/PNG；`smart_resize`（factor 32，min/max pixels）；bicubic（与参考实现对齐，允许 ≤1e-3 的像素差异）；normalize (x/255−0.5)/0.5；merge-block 顺序 patchify；temporal=2 复制。
2. **Vision tower**：patchifier 1536→1024；14 层（8 个 gated 3×3 depthwise conv + 6 个 bidirectional attention）；Q/K RMSNorm；axial RoPE；FFN；全部 bf16 + fp32 accumulate。
3. **Flash bidirectional attention**：按 image 的 `cu_seqlens`，不物化 N×N；支持多图 ragged batch。
4. **Projector**：LayerNorm(1024) → reshape 2×2 → 4096 → GELU → 2048；bias。
5. **Splice**：把 `<|image|>` 展开为 N 个 image token；embedding 替换；chunked prefill 时图像特征预先就绪。
6. **调度**：vision 独立队列；多请求图像合批；默认上限 1024 text tokens/图（可配）；图像特征按内容 hash 缓存。
7. **测试**：16 张图的 caption 人工 4/5；图像 token 数与 processor 公式一致；vision feature vs CPU reference cosine ≥0.999；图文混批不回归文本。

**交付物**：`miori serve` 支持 image_url(base64)；`bench-vision`；`C4` 报告。

**退出标准（C4）**：512² 图像端到端 ≤40 ms；1024² ≤120 ms；文本 C2/C3 指标不回归；多图 batch 正确。

**风险/回退**：
- 图像 resize 与参考实现有差异 → 记录 tolerance，不追求 bit-exact；质量用 caption/feature cosine 兜底；
- 视觉 compute 抢占 decode → vision 低优先级队列 + 限制并发图数。

---

### M5 — 吞吐优化（第 16–18 周）

**目标**：把"能并发"变成"高吞吐"：int8 KV、4-bit lm_head、prefix cache、shared-prefix attention、ICB 回放、prefill/decode overlap。

**工作包**

1. **int8 KV**：per (token, KV head) scale；写入量化、读出反量化；sliding ring 同样支持；KL gate。
2. **4-bit lm_head**：独立于 8-bit embedding 的 4-bit 副本（保持输入 embedding 8-bit）；KL gate；不通过则保留 8-bit。
3. **Prefix cache**：64-token page 的 radix tree；refcount + COW；LRU；prefix 边界的 **state bundle**（共享 full-KV pages + ~2.9 MB conv/sliding/hidden 拷贝）；fork 成本 ≤25 µs。
4. **Shared-prefix attention**：同一 step 内共享 prefix page 的多个请求，按 page 分组读 K/V，query 并行；目标 KV 读字节 −30%，agentic trace 聚合 +15%。
5. **ICB 回放**：按 batch bucket（1/2/4/8/16/32/64）录制 decode 图；稳定绑定；只改 token/position/block table 内容；CPU 提交 <0.1 ms/step；失败则回退低开销 encoder。
6. **Prefill/decode overlap**：双队列或时间片；chunk 64–128；测量 ITL 与聚合的折中；admission 限制冷 4K prompt 的并发。
7. **Speculative decoding（可选）**：prompt-lookup / n-gram，k=4，greedy 验证；只在 batch ≤8 且 acceptance 高时启用；不达标不发布。

**交付物**：`C5` 报告；默认配置（int8 KV 是否默认由 gate 决定）；`bench-prefix`、`bench-agentic`。

**退出标准（C5）**：
- int8 KV：top-1 ≥99.5%、mean KL ≤1e-3；
- 4-bit lm_head：top-1 ≥99.7%、mean KL ≤5e-4；
- 64 mixed-agentic：**output ≥1.6K tok/s**（stretch 1.8K），total ≥2.8K，ITL p99 ≤30 ms；
- 128×2K burst（80% hit）：output ≥1.5K tok/s；32×8K（95% hit，shared prefix）：output ≥1.6K tok/s；
- 8×32K（80% hit）：output ≥50 tok/s（cold 单独报告，约 10–15）；
- prefix hit ≥60%，hit p50 TTFT ≤150 ms；shared-prefix KV 读 −30%；ICB 回放正确。

**风险/回退**：
- int8 KV KL 不达标 → per-head scale / 混合精度（K int8、V fp16）/ 回退 fp16，并发目标下调；
- 4-bit lm_head 不达标 → 保留 8-bit，decode 约 −10%；
- shared-prefix attention 复杂度高 → 先交付 prefix cache（省 prefill），shared-prefix 作为 C5 的加分项而非阻塞项；
- ICB 不可用 → 接受 0.3–0.5 ms 提交开销，靠更大 batch 摊薄。

---

### M6 — 质量与稳定性（第 19–20 周）

**目标**：把"跑得快"变成"敢上线"。

**工作包**

1. **质量矩阵**：exact / int8 KV / 4-bit lm_head / 组合；4 域语料（EN chat、ZH chat、code、tool JSON）；2048 position；top-1、mean KL、p99 KL；per-domain 报告。
2. **长上下文**：32K 正确性；sliding/full 掩码边界；KV ~205 MB/seq（int8）；8×32K 压测。
3. **Soak**：24h 混合 trace；内存/VRAM/句柄/page 泄漏；p99 漂移；热状态记录。
4. **错误注入**：OOM、超长 prompt、非法图片、客户端断连、GPU 命令错误、取消风暴；全部 graceful。
5. **性能回归**：每次提交跑 C0/C1 微基准；C5 指标不得回退 >3%。
6. **文档**：README、模型/权重说明、已知限制、license 说明。

**交付物**：`C6` 报告；质量矩阵；soak 日志。

**退出标准（C6）**：24h 无泄漏、内存波动 ≤2%、p99 稳定；32K 正确；错误注入无 crash；所有 KL gate 有结论。

**风险/回退**：
- 热降频导致 p99 漂移 → 基准协议固定 warmup/median；产品侧加 admission 降载；
- 32K 下 KV 碎片 → page 32 备选；或限制单请求 16K。

---

### M7 — 发布（第 21 周）

**工作包**：v0.1.0 tag；可复现构建（Xcode/macOS/commit 固定）；SHA256；基准报告；README/quickstart；已知限制；license 合规（LFM1.0）；发布二进制 + 权重下载说明（`curl` 命令，不引入脚本语言）。

**退出标准（C7）**：干净机器按文档 30 分钟内跑通；基准数字可复现（±5%）；仓库无 Python/ObjC/Swift 代码；无未记录的有损默认项。

---

## 4. 排期与资源

| 周 | M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 |
|---:|---|---|---|---|---|---|---|---|
| 1–2 | ● | | | | | | | |
| 3–5 | | ● | | | | | | |
| 6–9 | | | ● | | | | | |
| 10–12 | | | | ● | | | | |
| 13–15 | | | | | ● | | | |
| 16–18 | | | | | | ● | | |
| 19–20 | | | | | | | ● | |
| 21 | | | | | | | | ● |

- **1 人**：21 周 + 2 周 buffer = **23 周**；M0/M1 不可并行，M3/M4 可部分并行。
- **2 人**：A 负责 kernels/engine（M0–M2、M5），B 负责 tokenizer/serving/vision（M3–M4）；预计 **16–18 周**。
- **3 人**：再加一人做质量/基准/发布；M5 的 prefix/shared-prefix 与 ICB 可并行；预计 **14–16 周**。
- 机器：同一台 M6/16GB 用于开发与基准；不做 CI GPU；基准时 headless（SSH、显示器休眠、无浏览器/IDE）。

---

## 5. 风险登记

| ID | 风险 | 概率 | 影响 | 缓解 / 触发 |
|---|---|---|---|---|
| R1 | MPP 不接受 threadgroup B tile | 中 | 高 | C0 spike；回退 per-layer 全局 fp16 scratch，prefill −10–15% |
| R2 | MLX 4-bit 布局/顺序判断错误 | 中 | 中 | C0 用 bf16 base 抽样 cosine；错则只改 loader |
| R3 | CPU 提交 175 dispatch >0.5 ms | 中 | 高 | ICB 回放；fusion；bucket 化；C0 先测 |
| R4 | Tokenizer regex 与参考不一致 | 高 | 中 | ICU 精确实现；10 MB 语料 round-trip；golden token IDs |
| R5 | 图像 resize 与参考有差异 | 中 | 低 | 允许 tolerance；caption/feature cosine 兜底 |
| R6 | int8 KV KL 不达标 | 中 | 高 | per-head scale / K-V 混合精度 / 回退 fp16；并发目标下调 |
| R7 | 4-bit lm_head KL 不达标 | 中 | 中 | 保留 8-bit；decode −10% |
| R8 | shared-prefix attention 复杂/didn't pay | 高 | 中 | 先交付普通 prefix cache；shared-prefix 作为 C5 加分项 |
| R9 | Hybrid prefix state 组合爆炸 | 高 | 中 | 只在 prefix 边界存 bundle；不做每 page snapshot |
| R10 | 16 GB 内存压力 / jetsam | 中 | 高 | KV 预算 8 GB；prefix LRU；admission；监控 |
| R11 | 热降频导致基准漂移 | 低 | 中 | 固定 warmup/median；记录 `pmset -g therm` |
| R12 | 无 Python reference，数值争议 | 低 | 中 | C++ CPU fp32 reference + bf16 base 权重对比；逐层 dump |

---

## 6. 变更管理

1. **门禁制**：没有 `Cx` 报告，不进入 `M(x+1)`；报告必须包含原始数据路径。
2. **默认值变更**：任何影响质量/性能的默认值（int8 KV、4-bit lm_head、spec、图像上限）必须带 KL 或吞吐数据，并在 `README` 记录。
3. **范围变更**：视频/音频、其他芯片、多机，一律进 backlog，不插队。
4. **性能回归**：C0/C1 微基准 + C2/C5 场景基准；回退 >3% 必须解释或回滚。
5. **文档同步**：`MILESTONES.md` 只改计划和门禁；实际结果写 `docs/checkpoints/Cx-*.md`，不篡改历史。

---

## 7. 完成定义（DoD）

一个里程碑只有同时满足以下条件才算完成：

- 代码可复现构建，无新增 warning；
- 该里程碑的 kernel/模块有 CPU reference 或 golden test；
- `docs/checkpoints/Cx-*.md` 已写，含环境、commit、命令、原始数据、结论；
- 性能/质量门全部达标，或有明确的、被记录的例外与回退；
- 内存/page/句柄无泄漏；
- README/接口文档同步；
- 不引入 Python/ObjC/Swift/MPS/MPSGraph/BNNS/Accelerate 依赖。
