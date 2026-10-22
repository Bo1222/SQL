# sqlrl：Text-to-SQL 与多轮数据库探索

## BIRD Agent 新入口

已实现字段描述检索、取值探索、SQL 试执行、显式提交，以及 observation-masked 工具 SFT / 多轮 GRPO。BIRD 数据已准备为 SFT 4,205、RL 1,801、val 590；BIRD dev 1,534 条保留作最终评测。新入口与已有单轮实验分开，不自动启动完整训练。

使用方法、对照与边界见 [多轮 pipeline](docs/agent_pipeline.md)。开发检查：`bash scripts/run_bird_agent.sh smoke`；正式训练需显式选择阶段并通过训练侧 pilot。下方保留单轮阶段的实验记录。

按 `SQL_EXECUTION_REWARD_PLAN.md` 第 6 节实现，先做 v1 单轮，跑通后再做 v2 多轮。

> 完整执行命令、日志路径、常见问题见 **[EXECUTION.md](EXECUTION.md)**；项目进度见 **[docs/PROGRESS.md](docs/PROGRESS.md)**。

## 当前状态（2026-10-22）

| 阶段 | 状态 |
| --- | --- |
| 环境 / 数据切分 | ✅ |
| Spider base 200 | ✅ EX 51.5%、可执行 86.0% |
| SFT 冷启动 | ✅ val_200 EX 70.0%、pass@8 87.0% |
| GRPO | ✅ 200 步完成 | test EX 65.5%、可执行 95.0%；SFT test 69.0%（p=0.21）；val 74.0%；结果见 `docs/grpo_results.md` |
| v2 多轮 | ⚠️ 暂缓 | Spider 执行错误仅 5%，G1 的 ≥25% 不满足 |
| v2 多轮 | ⬜ 11/9 gate |

## 现状

- **主数据集：Spider 1.0**，已解压并切分：`sft 4,859 / rl 1,262 / val 866 / test 1,034`，库数 `120/120/20/20`，重叠检查全部为 0。
- **跨数据集测试：BIRD dev**（1,534 题 / 11 库），已跑 200 条零样本基线（EX 25.5%）。
- 模型：`/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct`（已通过 sha256 校验）。

## 目录

```
sqlrl/
├── env/sqlite_exec.py        # 只读执行、progress handler 超时
├── reward/exec_reward.py     # <answer> 提取、结果比对、防作弊标记、错误分类
├── data/schema.py            # CREATE TABLE DDL（Spider/BIRD 共用）
├── data/build_spider.py      # Spider -> test/val/sft/rl JSONL（去重、gold 校验、按库/按题切分）
├── data/build_bird.py        # BIRD dev -> JSONL（跨数据集测试用）
├── data/functional_check.py  # gold 可执行、schema、prompt 长度检查
├── rollout/hf_rollout.py     # HF generate（NPU，左 padding，stop_strings）
├── eval/eval_base.py         # 零样本评测：EX / 有效执行率 / 审计 / 错误类型
├── eval/recompute.py         # 不重新生成，从 raw output 重算指标
└── scripts/run_base_eval.sh  # BIRD 200 条检查 + base 评测
```

## 跑 Spider 切分

```bash
ssh verl_rl
source ~/work/verl_quickstart/scripts/env.sh
cd ~/work/sqlrl
python3 data/build_spider.py \
  --root data/raw/spider_extract/spider_data \
  --out data/processed/spider \
  --tokenizer /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct
```

产物：`data/processed/spider/{test,val,sft,rl}.jsonl`、`split_report.json`。

切分结果（2026-10-21）：

| 指标 | 数值 |
| --- | --- |
| sft / rl / val / test | 4,859 / 1,262 / 866 / 1,034 |
| 库数 | 120 / 120 / 20 / 20 |
| 重叠（val-train / test-train / sft-rl） | 0 / 0 / 0 |
| train 去重 / gold 失败 | 10 / 3 |
| prompt token（p50/p95/max） | 344 / 1,123 / 2,930 |

## Spider dev 200 条分层基线（2026-10-22）

| 指标 | 数值 |
| --- | --- |
| 提取成功 | 99.5% |
| 可执行 | 86.0% |
| EX | **51.5%**（103/200） |
| 硬作弊 | 0 |
| 单样本耗时 | 2.07 s |

错误：`no_such_column` 25、`no_such_function` 1、`no_such_table` 1、`extract_missing` 1。初版误判的 7 条 `multi_statement` 是“裸 SQL + 多余 `</answer>`”，修复提取后为 0 条。

## SFT 冷启动（2026-10-22）

| 项 | 值 |
| --- | --- |
| 数据 | 4,859 条（120 库）；max_len=2,048 后保留 4,794（丢 65 条） |
| 配置 | LoRA r=16，batch 4 × accum 4，lr 1e-4，1 epoch，299 步 |
| 耗时 / 显存 | 1,880 s（31.3 min）/ 峰值 21.6 GB |
| loss | 1.86 → 0.14 |

**val_200：** EX 70.0%、可执行 93.0%、提取 100%、硬作弊 0。
**pass@8（rl_100）：** 87.0%、平均正确 4.78/8、混合组 57%、全对 30%、全 0 13%、可执行 84.9%。

结论：混合组 57% ≥ 40%、平均 4.78 在 2–6 区间，**满足进入 v1 GRPO 的前提**；可执行率 84.9% 需在训练中监控。

## 跑 BIRD 跨数据集基线

```bash
bash ~/work/sqlrl/scripts/run_base_eval.sh
```

| 指标 | BIRD dev 200 |
| --- | --- |
| 提取成功 | 100% |
| 可执行 | 60.0% |
| EX | **25.5%**（simple 33.0% / moderate 18.2% / challenging 5.3%） |
| 硬作弊 | 0 |

## 评测口径

- **EX**：候选 SQL 与 gold SQL 的执行结果一致；gold 含 `ORDER BY` 时顺序敏感，否则按多重集比较。
- **有效执行率**：候选 SQL 能执行的比例，与 EX 分开报告。
- **审计**：硬作弊只看 `constant_no_from`（无 FROM 而 gold 有）；`empty_both`（空对空）与 `single_row_both`（单行对单行）只作 info——Spider/BIRD 都有合法的空结果和标量答案，不能一票否决。
- **错误类型**：语法、表名、列名、超时等分开计数。

## 下一步

1. 跑 Spider dev 200 条零样本基线。
2. SFT 冷启动（LoRA，只对模型 token 算 loss）。
3. v1 GRPO，再按 gate 决定是否做 v2 多轮。
