# 项目进度（2026-10-22）

## 新增：BIRD 数据库探索 pipeline

- 已复用本地 train.zip，固定官方过滤标注版本，清洗后 SFT 4,205 / RL 1,801 / val 590；val 的 7 个数据库与训练隔离，dev 1,534 条保留评测。
- 已实现字段检索、取值探索、只读 SQL 工具、多轮状态机、action-only SFT/GRPO、训练侧 pilot、验证选模及配对评测。
- 四卡小模型冒烟通过 2 个有效更新，包含空闲 rank、参考冻结、验证选模、adapter 保存和 W&B offline；正式 3B 的 6,144-token 单次合成反传通过。
- **未执行完整 Agent SFT/GRPO，不声称已经获得多轮准确率收益。** 新方案以 [agent_pipeline.md](agent_pipeline.md) 为准；下方 v2 gate 是原报错修复方案的历史记录。

## 一、状态总览

| 阶段 | 状态 | 关键结果 |
| --- | --- | --- |
| 昇腾环境 | ✅ 完成 | CANN 8.5.1 / torch_npu 2.9 / 910B4 单卡；verl 单步 GRPO 与 3B LoRA 已验证 |
| 数据下载 | ✅ 完成 | Spider 1.0 已解压（1.8 GB，166 库）；BIRD dev/train 已下载（跨数据集备用） |
| 数据切分 | ✅ 完成 | sft 4,859 / rl 1,262 / val 866 / test 1,034；库 120/120/20/20；重叠 0 |
| 基线评测 | ✅ 完成 | Spider dev 200：EX 51.5%、可执行 86.0%；BIRD dev 200：EX 25.5% |
| SFT 冷启动 | ✅ 完成 | 299 步 / 31.3 min / 21.6 GB；val_200 EX 70.0%、可执行 93.0% |
| GRPO | ✅ 完成（200 步） | test 65.5%、可执行 95.0%；val 74.0%；相对 SFT test -3.5（p=0.21）、val +4（p=0.057）；96.5 min、峰值 11.32 GB |
| v2 多轮 | ⚠️ 暂缓 | Spider 执行错误仅 5%，G1 的 ≥25% 不满足；需改 BIRD 或放弃执行反馈方向 |

## 二、关键指标

### Spider dev 200（base，零样本）

| 指标 | 数值 |
| --- | --- |
| EX | 51.5%（103/200） |
| 可执行 | 86.0% |
| 提取 | 99.5% |
| 硬作弊 | 0 |
| 错误 | no_such_column 25、no_such_function 1、no_such_table 1、extract_missing 1 |

### SFT（val_200）

| 指标 | 数值 |
| --- | --- |
| EX | 70.0%（140/200） |
| 可执行 | 93.0% |
| 提取 | 100% |
| 硬作弊 | 0 |
| 错误 | no_such_column 12、misuse 1、other 1 |

### SFT 后 pass@8（rl_100）

| 指标 | 数值 |
| --- | --- |
| pass@8 | 87.0% |
| 平均正确 / 8 | 4.78 |
| 混合组 | 57.0% |
| 全对组 | 30.0% |
| 全 0 组 | 13.0% |
| 可执行率 | 84.9% |

门禁判断：混合组 57% ≥ 40%、平均 4.78 在 2–6 → **满足进入 GRPO**；可执行率 84.9% 略低于 90%，训练时监控。

## 三、产物路径

| 内容 | 路径 |
| --- | --- |
| 切分数据 | `data/processed/spider/{test,val,sft,rl}.jsonl` + `split_report.json` |
| Spider base | `reports/spider_base200.json` / `.jsonl` / `recomputed.json` |
| BIRD base | `reports/base_eval_200.json` |
| SFT checkpoint | `checkpoints/sft_lora/`（含 `train_summary.json`） |
| SFT 评测 | `reports/sft_val200.json` / `.jsonl` |
| pass@8 | `reports/pass8_rl100.json` / `.jsonl` |
| GRPO 脚本 | `train/grpo.py`、`train/grpo_ddp.py`、`scripts/run_grpo_ddp.sh`、`train/grpo_tracking.py` |
| GRPO DDP 冒烟 | `reports/grpo_ddp_fix_wandb_20261022/real_out/`（短配置） |
| GRPO 默认冒烟 | `reports/grpo_default_smoke_20261022/out/`（修复前 2 步） |
| GRPO 修复验证 | `reports/grpo_default_smoke_fix_20261022/`（默认 2 步、最长 prompt 2/10 步） |
| GRPO 正式结果 | `reports/grpo_ddp_{val200,test200}.json`、`reports/grpo_ddp_test200_recomputed.json` |
| SFT test 基线 | `reports/sft_test200.json` |
| GRPO step100 | `reports/grpo_ddp_step100_test200.json` |
| 结果分析 | `docs/grpo_results.md` |
| W&B 配置 | `.env.example`、`train/grpo_tracking.py`、`requirements-config.txt` |
| 执行手册 | `EXECUTION.md`、`docs/run_sft.md`、`docs/run_spider_baseline.md`、`docs/grpo_memory_analysis.md`、`docs/grpo_ddp_debug.md` |

## 四、时间线

- **10/21**：昇腾环境 Gate；BIRD dev 基线；Spider 包下载与解压。
- **10/22**：Spider 切分；Spider dev 200 基线；SFT 冷启动；val/pass@8；GRPO 脚本与执行手册。
- **10/22 13:35**：修复 DDP 的因果 logprob、EOS 截断、全局 loss/KL 日志、torchrun launcher；接入 W&B（offline）；4 卡 2 步冒烟通过（短配置 1024/96，峰值 9.11 GB）。
- **10/22 13:55**：GRPO 两个脚本加梯度检查点；默认 3072/200 的 4 卡 2 步冒烟通过（峰值 10.93 GB），W&B online 同步（run `fe8hhjgu`）。
- **10/22 14:24**：正式跑时 rank2 OOM（28.36 GB）。定位为 gradient checkpointing 只在 `train()` 生效，而脚本一直 `eval()`，检查点实际未启用。
- **10/22 16:16–16:21**：修复模式切换 + dropout 置 0；默认 2 步 10.93 GB（`uvaksvk4`）；最长 8 条 prompt 2 步 11.32 GB（`qh5lthyg`）、10 步稳定 11.33 GB。
- **10/22 18:51**：GRPO 200 步完成（96.5 min，11.32 GB，W&B `endck7zy`）；val EX 74.0%、test EX 65.5%、test 可执行 95.0%。
- **10/22 19:01**：补 SFT 的 test_200 基线：EX 69.0%、可执行 91.5%。
- **10/22 19:13**：GRPO step_100 test EX 65.0%，与 step_200 的 65.5% 基本一致。
- **下一步**：正式 200 步 GRPO（DDP + 梯度检查点）→ val/test 评测 → 审计；11/9 v2 gate。

## 五、下一步（按顺序）

1. **先决定 v2 方向**：Spider 上 GRPO 后只剩 5% 执行错误，G1 的 ≥25% 不满足。可选：
   - v2 改用 BIRD dev（base 可执行 60%，约 40% 执行错误）；
   - 或放弃“执行反馈修复”，改做自一致性/验证器；
   - 或不做 v2，把时间用于把 v1 结论做扎实。

2. **如需更强统计**：把 test 从 200 扩到完整 1,034 条，SFT 和 GRPO 各约 45 min；当前 200 条的 95% 区间约 ±6.5 点，3–4 点差异不显著。

3. **汇报口径**：
   - SFT 相对 base：test 51.5% → 69.0%，+17.5 点，这是最稳的增益；
   - GRPO：可执行 91.5% → 95.0%，EX 65.5% vs 69.0%（p=0.21，不显著）；val 70.0% → 74.0%（p=0.057）；
   - 不写“GRPO 提升 EX”，写“执行奖励提升可执行率，EX 增益未在 200 条上证实”。

4. **v2 gate（11/9）**：按上面的新证据重新评估；原 G1 在 Spider 上大概率不通过。

## 六、已知风险

| 风险 | 说明 | 应对 |
| --- | --- | --- |
| GRPO EX 增益不显著 | test 65.5% vs SFT 69.0%（p=0.21）；val +4（p=0.057） | 不宣称 EX 提升；用可执行率与 val 叙述；必要时扩到 1,034 条 |
| v2 样本不足 | Spider GRPO 执行错误仅 5%，G1 要求 ≥25% | v2 改 BIRD 或放弃执行反馈方向 |
| 零方差组偏多 | 平均 2.26/4 组零方差，24/200 步被跳过 | 动态采样已丢弃；可换更难 prompt 或降低 SFT 强度 |
| 长序列 OOM | 已修复 | 200 步峰值 11.32 GB，稳定 |
| 多卡显存预期偏差 | DDP 复制完整模型，不按卡数除显存 | 4 卡约 8–9 GB 固定 + 单序列峰值；加卡只提吞吐 |
| BIRD train 未用 | 8.3 GB 已下载未解压 | 保留作第二套对照，不占训练主流程 |
