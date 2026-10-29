# Checkpoint 报告

本目录存放 Miori 的 C0–C7 检查点报告，文件名 `Cx-<name>.md`（例如 `C0-baseline.md`、`C5-throughput.md`）。

规则：

- **未完成的检查点不预写报告**；报告只在门禁实际运行后生成。
- 模板见 [`../CHECKPOINTS.md`](../CHECKPOINTS.md) §6；报告必须包含环境、commit、命令、原始数据路径、go/no-go。
- 报告提交后不再修改；修正写在下一份报告里。
- 原始日志放在 `bench/results/Cx/`，报告中引用相对路径。
