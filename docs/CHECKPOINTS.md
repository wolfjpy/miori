# CHECKPOINTS — Miori

> 检查点（Checkpoint）是**门禁**，不是"进度汇报"。每个 `Cx` 必须给出量化数据、原始日志路径和 go/no-go 结论；没有通过 `Cx`，不进入 `M(x+1)`。
>
> 检查点报告写到 `docs/checkpoints/Cx-<name>.md`，模板见 §6。`docs/checkpoints/` 下的报告一旦提交不再修改，修正写在下一份报告里。

---

## 1. 测量协议（所有检查点共用）

### 1.1 环境

| 项 | 要求 |
|---|---|
| 机器 | Apple M6 Mac mini，12 GPU 核，16 GB；仅此一台 |
| 系统 | macOS 27.0.1（Build 26A434）或记录实际版本 |
| 工具链 | Xcode 27.1，Metal compiler 32023.921，`metal4.1` |
| 显示 | **headless**：SSH 登录，显示器休眠，不接 4K 桌面合成负载 |
| 电源 | AC，禁用睡眠；关闭浏览器/IDE/其他 GPU 应用 |
| 热状态 | 记录 `pmset -g therm`；必要时 `powermetrics --samplers gpu_power -n 1` |
| 构建 | Release：`-O3 -fno-fast-math`；记录 git commit、metallib hash、模型 sha256 |
| 预热 | 先跑 10 分钟同类负载，再开始计时 |
| 重复 | 每个场景 3 次，报告 **median**，同时给 min/max；不得只报最好一次 |

### 1.2 指标定义

| 指标 | 定义 |
|---|---|
| TTFT | 从请求进入到第一个流式 token 写出 |
| ITL | 相邻两个 token 的间隔；报 p50/p99，按请求内所有间隔统计 |
| 聚合吞吐 | 分两种：**output tok/s** = 输出 token / 墙钟；**total tok/s** = (prompt+output) / 墙钟。报告必须注明是哪一种 |
| prefill tok/s | prompt token 数 / prefill 墙钟时间（单请求、无并发干扰） |
| decode tok/s | 单流生成 token 数 / decode 墙钟时间 |
| step time | 一个调度 step 的 GPU 时间（含 prefill/decode 混批） |
| KV 读字节 | 每 step attention 实际读取的 K/V 字节（用计数器或解析模型估算，报告方法） |

### 1.3 负载形态

- **闭环**：固定 N 个客户端，每个发完一条再发下一条；用于容量/延迟曲线。
- **开环**：Poisson 到达，用于过载/排队/公平性；报告到达率与丢弃率。
- 检查点默认用闭环；C6 soak 用开环混合。

---

## 2. 基准集

全部由 C++ 工具 `build/gen-trace` 从公有领域文本/代码生成，固定 seed；不引入 Python。

| 名称 | 内容 | 规模 | 用途 |
|---|---|---|---|
| `chat-1k` | 700 平均、1K p95；256 output；cold | 512 条 | Chat 延迟/吞吐 |
| `chat-cold-2k` | 2K 固定 prompt；256 output；cold | 256 条 | 分页/批处理压力 |
| `agentic-8k` | 8K session；20 轮 tool call；95% prefix 复用（首轮 0）；512 output | 64 会话 | Agentic 吞吐、prefix |
| `code-2k` | 代码补全/编辑；2K prompt；128 output | 256 条 | Spec decode、结构化输出 |
| `burst-2k` | 2K；80% hit；512 output | 128 并发 | Burst 吞吐 |
| `offline-2k` | 2K；80% hit；512 output | 256 并发 | Offline 吞吐 |
| `long-32k` | 32K；80% hit；128 output | 8 并发 | 长上下文正确性/内存 |
| `mixed` | 70% agentic（8K 上限，平均 ~5K）+ 30% chat（1K）；约 4K 平均常驻 | 64 并发 | **主调优场景** |
| `images-512` | 512×512，1–4 张/请求 | 16 张 | 视觉延迟/质量 |
| `images-1024` | 1024×1024，1 张/请求 | 8 张 | 视觉上限 |
| `golden-20` | 20 条固定 prompt + exact path greedy token IDs | 20 条 | 回归/一致性 |

> `mixed` 是 C3/C5 的主场景；C2 用 `chat-cold-2k` 做分页/批处理压力。所有 "output tok/s" 只统计输出 token；"total tok/s" 统计 prompt+output。除非另注，聚合数字默认指 `mixed`。

---

## 3. 质量门协议（KL gate）

### 3.1 基线

- **GPU exact path**：fp16 KV、8-bit lm_head、bf16 激活、fp32 accumulate、无 spec、无 prefix 复用。
- **CPU fp32 reference**：C++ 慢速实现，dequant + fp32；用于 256 个 position 的抽查，防止"GPU exact path 自己错"。
- **权重基线**：base bf16 checkpoint 抽样 128 行，验证 4-bit 反量化 cosine ≥0.95（QAD 会使权重整体偏移；布局错误时相关性会接近 0）；只做一次，记录在 C0。

### 3.2 语料

2048 个 position，四域各 512：

1. EN chat；2. ZH chat；3. code；4. tool/JSON（agentic）。
固定 token IDs，固定 position；不截断、不采样。

### 3.3 流程

1. exact path 跑完 2048 position，logits 以 fp32 二进制落盘（`ref.logits`）。
2. candidate 跑同一输入，落盘。
3. C++ `kl-eval` 计算：
   - top-1 agreement；
   - mean KL（以 reference 为 p，candidate 为 q，逐 position 平均）；
   - p99 KL；
   - 可选：max |Δlogit|、per-domain 分解。
4. 报告必须给 per-domain 表，不能只给总体。

### 3.4 阈值

| 对比 | top-1 | mean KL | p99 KL | 失败动作 |
|---|---:|---:|---:|---|
| GPU exact vs CPU fp32（抽查 256 pos） | ≥99.5%（≤1 次翻转） | ≤1e-4 | ≤1e-3 | 修 kernel/图，不得继续 |
| int8 KV vs exact | ≥99.5% | ≤1e-3 | ≤5e-3 | per-head scale → K/V 混合精度 → 回退 fp16 |
| 4-bit lm_head vs exact | ≥99.7% | ≤5e-4 | ≤2e-3 | 保留 8-bit；或改 6-bit |
| int8 KV + 4-bit lm_head 组合 | ≥99.3% | ≤1.5e-3 | ≤8e-3 | 只启用通过单项的那个 |
| speculative decoding | 与 exact **token 级完全一致** | — | — | 任何不一致视为 bug |

> KL 在 128K 词表上计算；reference logits 保留 fp32，不做 top-k 截断。若某 position 的 reference logits 量级异常（>1e4），单独记录，不参与 mean KL（沿用 kata 的处理方式），但 top-1 仍计入。

### 3.5 其它质量门

- **Tokenizer**：10 MB 混合语料（EN/ZH/代码/emoji）encode→decode 字节完全一致；special/added token 优先匹配；图像占位符展开数量 = `grid_h*grid_w/4`；20 个 chat template golden 全通过。
- **Vision**：16 张图，vision tower 输出 vs CPU fp32 reference，cosine ≥0.999、max abs ≤1e-2；caption 人工 4/5；图像 token 数精确。
- **长上下文**：32K 下 sliding 窗口边界（第 255/256/257 个 key）与 full attention 掩码的逐 position 对比；不得出现越界/NaN。
- **数值卫生**：全程 NaN/Inf 计数为 0；KV page 未初始化区域必须被 mask，不得参与 softmax。

---

## 4. 检查点

### C0 — 基线与 spike（M0）

**前置**：仓库可构建；模型文件校验通过。

| 检查项 | 阈值 | 方法 |
|---|---:|---|
| 构建 | 成功，无新增 warning | `make clean && make`；记录 Xcode/metal 版本 |
| 加载 | 732 tensors，总字节 1,590,866,944，shape/dtype 全匹配 | loader 自检 + index.json 对比 |
| MLX 布局 | 抽样 128 行 vs bf16 base cosine ≥0.95（QAD 会使权重整体偏移，阈值不取 0.99；布局错误时相关性会接近 0） | `tools/check-dequant` |
| 流式读带宽 | ≥135 GB/s | 1.2 GB buffer，8 B/thread，3 次 median |
| 4-bit GEMV | ≥120 GB/s（2048×2048） | `bench-gemv`，记录 rows/warp 版本 |
| MPP GEMM | ≥8 TFLOPS @ M=512, N=2048, K=2048 | `bench-gemm`，threadgroup B tile |
| 提交开销 | 175 dispatch <0.5 ms/step | `bench-encode`；同时测 ICB 回放 |
| ICB | 明确 supported / unsupported；若 supported，回放正确且 <0.1 ms | `bench-icb` |
| 数值 | 单层 vs CPU fp32 max abs ≤1e-2（bf16） | `dump-logits` |

**失败动作**：MPP threadgroup B 失败 → 记录回退方案（全局 fp16 scratch）并重估 M1；布局失败 → 只修 loader；ICB 失败 → 接受低开销 encoder，M2 前重新评估 CPU 预算。

**产出**：`docs/checkpoints/C0-baseline.md` + 原始日志。

---

### C1 — 单请求文本（M1）

**前置**：C0 通过。

| 检查项 | 阈值 | 方法 |
|---|---:|---|
| 单流 decode | ≥100 tok/s @512 ctx | `miori run`，256 output，3 次 median |
| 单流 prefill | ≥2.5K tok/s @1K | 单请求，无并发 |
| 整模型 top-1 | ≥99.5% vs CPU fp32（256 pos，允许 ≤1 次翻转） | `kl-eval` |
| 逐层误差 | max abs ≤1e-2（bf16 容差） | 逐层 dump 对比 |
| Tokenizer | 10 MB 语料 100% round-trip | `miori tokenize --roundtrip` |
| Chat template | 20/20 golden | `tests/chat_template` |
| 采样 | greedy == argmax；top-k/top-p 分布 sanity | `tests/sampler` |
| 内存 | 单请求 ≤3 GB | `MTLDevice.currentAllocatedSize` |
| 数值卫生 | NaN/Inf = 0 | 全流程断言 |

**失败动作**：<100 tok/s 先查 GEMV 带宽与 dispatch 数；top-1 不达标查 RoPE 的 partial/zero-tail、sandwich norm 顺序、conv 的 7-tap 因果对齐。

**产出**：`C1-text.md`。

---

### C2 — 分页 KV + 连续批处理（M2）

**前置**：C1 通过。

| 检查项 | 阈值 | 方法 |
|---|---:|---|
| 64×2K 冷 prompt 总 token | ≥2.2K tok/s（output 单独报告，约 300–400/s） | `chat-cold-2k`，int8 关闭，闭环 64 |
| ITL | p99 ≤30 ms | 同上 |
| TTFT | p50 ≤0.5 s（1K cold，32 并发） | 同上 |
| Batch invariance | 20/20 条 greedy token ID 与 batch=1 完全一致 | `tests/batch_invariance` |
| Chunked prefill | 8K prompt，128 chunk == 不分 chunk | 同上 |
| Page 泄漏 | 100 次随机 cancel 后 free list 恢复 | `tests/page_leak` |
| COW | 共享前缀 + 尾页写入后，原请求内容不变 | `tests/cow` |
| 稳定性 | 1h 无泄漏、无 crash | soak 1h |

**失败动作**：ITL 超标 → 减小 prefill chunk / 限制冷 prompt 并发；batch invariance 失败 → 先修 attention 掩码与 position，不得用"近似"绕过。

**产出**：`C2-batching.md`。

---

### C3 — 服务化（M3）

**前置**：C2 通过。

| 检查项 | 阈值 | 方法 |
|---|---:|---|
| SSE 并发 | 64 条长连接稳定 | `load-test --clients 64 --stream` |
| TTFT | chat p99 ≤0.8 s（1K cold）；agentic p99 ≤150 ms（95% hit） | 同上 |
| ITL | p99 ≤30 ms | 同上 |
| 聚合 | `mixed-agentic` output ≥1.4K tok/s，total ≥2.6K tok/s | 同上 |
| OpenAI schema | 请求/响应/流式 chunk/usage/finish_reason 全通过 | `tests/openai_schema` |
| 背压 | 队列满返回 429；不 crash、不 OOM | 开环过载 |
| 断连 | 客户端中断后 1 s 内释放 KV | `tests/disconnect` |
| 内存 | 1h 稳定，波动 ≤2% | 同上 |

**失败动作**：SSE 每 token flush 导致吞吐下降 → 合并 2–4 token 或 10 ms flush；JSON 解析热点 → 固定 schema 快路径。

**产出**：`C3-serving.md`。

---

### C4 — 视觉（M4）

**前置**：C3 通过。

| 检查项 | 阈值 | 方法 |
|---|---:|---|
| 512² 端到端 | ≤40 ms | `bench-vision`，单图，3 次 median |
| 1024² 端到端 | ≤120 ms | 同上 |
| Vision 数值 | cosine ≥0.999，max abs ≤1e-2 vs CPU fp32 | 16 张图 |
| Caption | 人工 4/5 | 16 张图，固定 prompt |
| 图像 token 数 | 与 `grid_h*grid_w/4` 完全一致 | `tests/image_tokens` |
| 图文混批 | 文本 C2 指标回退 ≤3% | `mixed` + 图像 |
| 多图 batch | 2–4 张图 ragged 正确 | `tests/vision_batch` |
| 预处理 | resize/normalize 与参考差异 ≤1e-3 | 固定图片，逐像素 |

**失败动作**：resize 差异 → 记录 tolerance，不追求 bit-exact；视觉抢占 decode → 降 vision 队列优先级/限制并发图数。

**产出**：`C4-vision.md`。

---

### C5 — 吞吐优化（M5）

**前置**：C4 通过；C0 的 KL 工具可用。

| 检查项 | 阈值 | 方法 |
|---|---:|---|
| int8 KV 质量 | top-1 ≥99.5%，mean KL ≤1e-3，p99 ≤5e-3 | §3；4 域分解 |
| 4-bit lm_head 质量 | top-1 ≥99.7%，mean KL ≤5e-4，p99 ≤2e-3 | §3 |
| 64 mixed-agentic output | ≥1.6K tok/s（stretch 1.8K）；total ≥2.8K | int8 KV，闭环 64，70/30 |
| 64 mixed-agentic ITL | p99 ≤30 ms | 同上 |
| 128×2K burst output | ≥1.5K tok/s | 80% prefix hit，512 output |
| 32×8K agentic output | ≥1.6K tok/s | 95% prefix hit，shared-prefix attention |
| 8×32K output | ≥50 tok/s（80% hit；cold 约 10–15，单独报告） | `long-32k` |
| Prefix hit | ≥60%；hit TTFT p50 ≤150 ms | `agentic-8k` |
| Shared-prefix | KV 读字节 −30%；agentic 聚合 +15% | 计数器/解析模型 |
| ICB | 回放正确；提交 <0.1 ms/step | `bench-icb` |
| Spec（可选） | 与 exact token 级一致；code 单流 ≥150 tok/s | `code-2k` |

**失败动作**：int8 KV 不过 → per-head scale / K int8 + V fp16 / 回退 fp16，并下调并发目标；4-bit lm_head 不过 → 保留 8-bit；shared-prefix 不过 → 保留普通 prefix cache，作为已知限制；ICB 不过 → 接受 0.3–0.5 ms 提交，靠 batch 摊薄。

**产出**：`C5-throughput.md` + 最终默认配置表。

---

### C6 — 质量与稳定性（M6）

**前置**：C5 通过。

| 检查项 | 阈值 | 方法 |
|---|---:|---|
| 24h soak | 无 crash；内存波动 ≤2%；p99 漂移 ≤10% | 开环 `mixed`，记录每 10 min |
| 32K 上下文 | 正确；无 NaN/OOB；KV ~205 MB/seq（int8） | `long-32k` |
| 32K 性能 | 8×32K output ≥50 tok/s（80% hit；cold 10–15 单独报告） | 同上 |
| 错误注入 | OOM/超长/坏图/断连/取消风暴 0 crash | `tests/fault_injection` |
| 质量矩阵 | exact / int8 / 4-bit / 组合 全部有结论 | §3 |
| 回归 | C0/C1 微基准回退 ≤3% | 每次提交 |
| 文档 | README/已知限制/license 完整 | 人工检查 |

**失败动作**：热降频导致漂移 → 固定 warmup/median，产品侧 admission 降载；32K 碎片 → page 32 备选或限制 16K。

**产出**：`C6-quality.md`。

---

### C7 — 发布（M7）

**前置**：C6 通过。

| 检查项 | 阈值 | 方法 |
|---|---:|---|
| 干净构建 | 按文档 30 min 内跑通 | 新 clone + 新 build 目录 |
| 可复现 | 基准 ±5% | 同一机器，3 次 median |
| 依赖审计 | 无 Python/ObjC/Swift/MPS/MPSGraph/BNNS/Accelerate | `grep`/otool |
| 权重校验 | sha256 与发布说明一致 | `shasum -a 256` |
| 文档 | quickstart/API/已知限制/benchmark 完整 | 人工 |
| 版本 | v0.1.0 tag，二进制 + metallib | git tag |

**产出**：`C7-release.md` + release notes。

---

## 5. 报告之间的依赖

```
C0 ── C1 ── C2 ── C3 ── C4 ── C5 ── C6 ── C7
        │          │
        └── 质量工具 ─┘（C1 建立，C5 复用）
```

- C0 的布局/GEMM/ICB 结论会回写到 `MILESTONES.md` 的风险表；
- C1 建立的 `kl-eval`、逐层 dump、golden-20 在 C5/C6 复用；
- C2 的 batch invariance 是后续所有优化的回归基线；
- C5 的默认配置表是 C6 soak 和 C7 发布的输入。

---

## 6. 报告模板

```markdown
# Cx — <名称>

- 日期：
- Git commit：
- 模型 sha256：
- 机器/OS/Xcode/Metal：
- 构建参数：
- 热状态：

## 1. 目标与前置
（本检查点要回答的问题；前置 Cx-1 是否通过）

## 2. 变更
（相对上一检查点的代码/权重布局/默认值变化；没有就写 none）

## 3. 结果
| 指标 | 目标 | 实测（median / min / max） | 判定 |
|---|---:|---:|---|

原始数据：`bench/results/Cx/<name>.json`、日志路径。

## 4. 质量
（KL/top-1/per-domain；tokenizer；vision；NaN 计数）

## 5. 失败与回退
（哪些门没过；根因；回退方案；对后续里程碑的影响）

## 6. 结论
go / no-go；例外项（必须写明谁批准、何时复测）。

## 7. 下一步
（进入哪个里程碑；需要先做的 spike/修复）
```

---

## 7. 附录

### 7.1 关键公式

- 文本权重：917 MB；embed + lm_head：295 MB；**decode 每 token 读 1.212 GB**；4-bit lm_head 后约 1.081 GB。
- KV：full 层 6 × 2(K,V) × 4 heads × 128 × 2 B = **12,288 B/token**（fp16）；int8 约 **6,240 B/token**（含 scale）。
- 每 seq 固定状态：sliding 5 × 256 × 4 × 128 × 2 × 2 B = 2.62 MB；conv 13 × 2048 × 6 × 2 B = 0.32 MB；合计 **2.94 MB**。
- step 字节 ≈ `1.212 GB + B×L×KV + B×2.94 MB`（纯 decode；prefill chunk 另加）。
- step 时间 ≈ `max(step 字节 / 125 GB/s, B × 3.2 GFLOP / 10 TFLOPS)`。
- prefill tok/s ≈ `10–13 TFLOPS / 3.2 GFLOP ≈ 3.1–4.0K`。
- 显存 ≈ `1.59 GB + B×L×KV + B×2.94 MB + 0.5 GB`。
- **output tok/s ≈ total tok/s × O/(P+O)**；O 为输出 token，P 为本 turn 未命中 prefix、需要真正 prefill 的 token。chat 的 P 大，agentic 的 P 小，所以同样的 total 预算下 output 差很多。
- 32K int8 KV ≈ `32,768 × 6,240 B ≈ 205 MB/seq`。

### 7.2 术语

| 术语 | 含义 |
|---|---|
| prefill | 处理 prompt、写 KV 的阶段，compute-bound |
| decode | 逐 token 生成，weight-bandwidth-bound |
| ragged | 一个 batch 内多条请求、长度不一，用 `cu_seqlens` 拼接 |
| page | KV 分配单位，64 token；一个 page 覆盖 6 个 full 层 |
| COW | copy-on-write；共享前缀页在写入时复制 |
| bundle | prefix 边界的混合状态快照（conv + sliding + hidden） |
| ICB | indirect command buffer；录制 decode 图后回放，降低 CPU 提交 |
| MPP | MetalPerformancePrimitives；`matmul2d` 走 tensor core |
| ITL | inter-token latency，相邻输出 token 间隔 |
| TTFT | time to first token |
