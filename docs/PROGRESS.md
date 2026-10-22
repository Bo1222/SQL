# 项目进度（2026-10-22）

## 一、状态总览

| 阶段 | 状态 | 关键结果 |
| --- | --- | --- |
| 昇腾环境 | ✅ 完成 | CANN 8.5.1 / torch_npu 2.9 / 910B4 单卡；verl 单步 GRPO 与 3B LoRA 已验证 |
| 数据下载 | ✅ 完成 | Spider 1.0 已解压（1.8 GB，166 库）；BIRD dev/train 已下载（跨数据集备用） |
| 数据切分 | ✅ 完成 | sft 4,859 / rl 1,262 / val 866 / test 1,034；库 120/120/20/20；重叠 0 |
| 基线评测 | ✅ 完成 | Spider dev 200：EX 51.5%、可执行 86.0%；BIRD dev 200：EX 25.5% |
| SFT 冷启动 | ✅ 完成 | 299 步 / 31.3 min / 21.6 GB；val_200 EX 70.0%、可执行 93.0% |
| GRPO 准备 | 🟡 脚本就绪，未启动 | 单卡在 step 37 OOM；已补 4 卡 DDP 脚本 `run_grpo_ddp.sh`，参数不变，未运行 |
| v2 多轮 | ⬜ 未开始 | 按 11/9 的 G0/G1/G2 判定 |

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
| GRPO 脚本 | `train/grpo.py`、`scripts/run_grpo.sh` |
| 执行手册 | `EXECUTION.md`、`docs/run_sft.md`、`docs/run_spider_baseline.md` |

## 四、时间线

- **10/21**：昇腾环境 Gate；BIRD dev 基线；Spider 包下载与解压。
- **10/22**：Spider 切分；Spider dev 200 基线；SFT 冷启动；val/pass@8；GRPO 脚本与执行手册。
- **下一步**：GRPO 冒烟 → 200 步 → val/test 评测 → 审计；11/9 v2 gate。

## 五、下一步（按顺序）

1. **GRPO 冒烟**（约 2–5 min）：
   ```bash
   python3 train/grpo.py --data data/processed/spider/rl.jsonl \
     --sft-adapter checkpoints/sft_lora --steps 2 --prompts-per-step 1 --n 2 \
     --out checkpoints/grpo_smoke
   ```
   看单步耗时、显存、reward/valid，确认 `train_log.jsonl` 正常。

2. **正式 GRPO**：
   ```bash
   bash scripts/run_grpo.sh
   ```
   200 步，n=8，4 prompts/step；预计 2–4 h，20–24 GB。

3. **评测与对照**：val_200 / test_200 的 EX、可执行率、McNemar、错误迁移；与 SFT 的 B 组对比。

4. **审计**：硬作弊、空对空、单行命中；确认 `ex_clean`。

5. **v2 gate（11/9）**：G0/G1/G2 + 200 条提示式修复 pilot。

## 六、已知风险

| 风险 | 说明 | 应对 |
| --- | --- | --- |
| GRPO 单步过慢 | 逐序列 forward/backward，32 条采样/步 | 先跑 2 步测时；必要时降 prompts/step 到 2 |
| 可执行率 84.9% | 低于 90% 门禁 | 训练中监控；掉太多先查格式奖励/数据 |
| 零方差组 | 全对/全错组无梯度 | 依赖动态采样丢弃；若比例过高降 SFT epoch |
| OOM | SFT 在 max_len 3,072 时 OOM | 已固定 max_len 2,048 + expandable_segments |
| BIRD train 未用 | 8.3 GB 已下载未解压 | 保留作第二套对照，不占训练主流程 |
