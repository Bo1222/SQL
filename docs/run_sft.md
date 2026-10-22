# SFT 冷启动：执行记录

日期：2026-10-22 00:59–01:31（Asia/Shanghai）
模型：Qwen2.5-3B-Instruct + LoRA（r=16，alpha=32，7 个线性层）
硬件：910B4 单卡（32 GB）

## 一键执行

```bash
ssh verl_rl
bash ~/work/sqlrl/scripts/run_sft.sh
```

`run_sft.sh` 依次执行：构建 `val_200` / `rl_100` 子集 → LoRA SFT 1 epoch → val_200 评测 → rl_100 pass@8。

## 数据规模

| 项 | 数值 |
| --- | --- |
| SFT 样本 | 4,859（120 个库，题目级切分） |
| prompt token | mean 486 / p95 1,123 / max 2,930 |
| gold token | mean 30 / max 137 |
| 总 token（估算） | 约 2.51 M |
| max_len=2,048 后保留 | 4,794（丢 65 条，1.3%） |
| 有效 batch | 4 × grad_accum 4 = 16 |
| 优化步数 | 299 / epoch |

## 训练配置

| 项 | 值 |
| --- | --- |
| LoRA | r=16，alpha=32，dropout=0.05，q/k/v/o/gate/up/down |
| 可训练参数 | 29.9 M（0.96%） |
| lr | 1e-4，5% warmup + cosine |
| max_len | 2,048 |
| 梯度检查点 | 开 |
| 峰值显存 | 21.6 GB |
| 耗时 | 1,880 s（31.3 min） |
| loss | 1.86 → 0.14 |

> 首次用 max_len=3,072 时在 step 2 的 backward OOM（尝试再分配 6.72 GiB）；把 max_len 降到 2,048 并设置 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 后稳定跑完。

## 结果

**val_200（SFT 后）**

| 指标 | 数值 |
| --- | --- |
| EX | 70.0%（140/200） |
| 可执行 | 93.0% |
| 提取 | 100% |
| 硬作弊 | 0 |
| 错误 | no_such_column 12、misuse 1、other 1 |

**pass@8（rl_100，T=1.0）**

| 指标 | 数值 |
| --- | --- |
| pass@8 | 87.0% |
| 平均正确 / 8 | 4.78 |
| 混合组（0<正确<8） | 57.0% |
| 全 0 组 | 13.0% |
| 全对组 | 30.0% |
| 可执行率 | 84.9% |
| 提取率 | 100% |

分布：`0:13, 1:11, 2:6, 3:6, 4:8, 5:4, 6:9, 7:13, 8:30`。

## 门禁判断

按文档 7.1/7.2 的 RL 前提：

- 混合组 57% ≥ 40% ✅
- 平均正确 4.78，落在 2–6 区间 ✅
- 可执行率 84.9%，略低于 90% ⚠️（训练时监控）

结论：**可以进入 v1 GRPO**；如果训练中可执行率继续下降，先回查格式奖励/数据。

## 路径

| 内容 | 路径 |
| --- | --- |
| 入口脚本 | `~/work/sqlrl/scripts/run_sft.sh` |
| 训练脚本 | `~/work/sqlrl/train/sft_lora.py` |
| 训练日志 | `~/work/sqlrl/logs/sft_run.log` |
| 控制台日志 | `~/work/sqlrl/logs/sft_console.log` |
| SFT checkpoint | `~/work/sqlrl/checkpoints/sft_lora` |
| 训练摘要 | `~/work/sqlrl/checkpoints/sft_lora/train_summary.json` |
| val 200 | `~/work/sqlrl/reports/sft_val200.json` / `.jsonl` |
| pass@8 | `~/work/sqlrl/reports/pass8_rl100.json` / `.jsonl` |
| val 子集 | `~/work/sqlrl/data/processed/spider/val_200.jsonl` |
| rl 子集 | `~/work/sqlrl/data/processed/spider/rl_100.jsonl` |
