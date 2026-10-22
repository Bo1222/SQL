# sqlrl 执行手册（Spider v1 单轮）

> BIRD 数据库探索、多轮 SFT/GRPO 的新命令见 [docs/agent_pipeline.md](docs/agent_pipeline.md)。统一入口为 `bash scripts/run_bird_agent.sh <阶段>`，数据准备、轨迹构造、pilot、训练和评测均单独选择。本页后续命令保留为 Spider 单轮实验手册。

本手册只描述“怎么跑、看哪里”，不自动启动实验。所有命令在 `verl_rl` 上执行。

## 0. 环境

```bash
ssh verl_rl
source ~/work/verl_quickstart/scripts/env.sh
cd ~/work/sqlrl
```

已知可用栈：CANN 8.5.1、torch 2.9 + torch_npu 2.9、transformers 4.57.6、peft 0.18.1、910B4 单卡 32 GB。
模型：`~/work/env_verify/models/Qwen2.5-3B-Instruct`（sha256 已验证）。

### 0.1 当前进度（2026-10-22）

| 阶段 | 状态 |
| --- | --- |
| 环境 / 数据 | ✅ Spider 切分完成（sft 4,859 / rl 1,262 / val 866 / test 1,034） |
| 基线 | ✅ Spider dev 200 EX 51.5%；BIRD dev 200 EX 25.5% |
| SFT | ✅ 299 步 / 31.3 min；val EX 70.0%、pass@8 87.0% |
| GRPO | ✅ 200 步完成（96.5 min、11.32 GB）：val 74.0%、test 65.5%、test 可执行 95.0% |
| v2 | ⚠️ 暂缓：Spider 执行错误仅 5%，G1 的 ≥25% 不满足 |

详细进度与产物路径见 `docs/PROGRESS.md`。

## 1. 数据构建

Spider 包已解压到 `data/raw/spider_extract/spider_data/`。

```bash
python3 -u data/build_spider.py \
  --root data/raw/spider_extract/spider_data \
  --out data/processed/spider \
  --tokenizer /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct
```

输出：

- `data/processed/spider/{test,val,sft,rl}.jsonl`
- `data/processed/spider/split_report.json`

验收：`val_vs_train_dbs=0`、`test_vs_train_dbs=0`、`sft_vs_rl_questions=0`；gold 失败率 <5%。

## 2. 基线评测

### 2.1 Spider dev 200 条

```bash
bash scripts/run_spider_baseline.sh
```

产物：`reports/spider_base200.json`、`reports/spider_base200.jsonl`、`reports/spider_test200_manifest.json`。

### 2.2 BIRD dev 200 条（跨数据集）

```bash
bash scripts/run_base_eval.sh
```

产物：`reports/base_eval_200.json`、`reports/functional_check_200.json`。

## 3. SFT 冷启动

```bash
bash scripts/run_sft.sh
```

脚本依次做：构建 `val_200` / `rl_100` 子集 → LoRA SFT 1 epoch → val_200 评测 → pass@8。

关键配置：

| 项 | 值 |
| --- | --- |
| 数据 | `sft.jsonl`，4,859 条（max_len=2,048 后 4,794） |
| LoRA | r=16，alpha=32，7 个线性层 |
| batch | 4 × grad_accum 4 = 16 |
| lr | 1e-4，5% warmup + cosine |
| max_len | 2,048 |
| 步数 | 299 |
| 实测 | 31.3 min，峰值 21.6 GB，loss 1.86→0.14 |

产物：

- `checkpoints/sft_lora/`（adapter + `train_summary.json`）
- `reports/sft_val200.json` / `.jsonl`
- `reports/pass8_rl100.json` / `.jsonl`

门禁（实测）：val EX 70.0%、可执行 93.0%；pass@8 87.0%、混合组 57.0%、平均 4.78/8 → 满足 GRPO 前提。

## 4. GRPO

```bash
bash scripts/run_grpo.sh
```

脚本依次做：GRPO 200 步 → val_200 评测 → test_200 评测 → 审计重算。

初始化：`checkpoints/sft_lora` 作为 policy adapter，同时加载一份冻结副本作为 **reference**（KL 对齐 SFT，而不是 base）。

关键配置：

| 项 | 值 |
| --- | --- |
| 数据 | `rl.jsonl` 1,262 条 |
| 每步 prompt | 4 |
| 采样 | n=8，T=1.0，top_p=0.95，max_new=200 |
| 奖励 | 执行结果一致=1；常量答案等硬作弊=0 |
| 优势 | 组内减均值除标准差；零方差组丢弃 |
| lr | 1e-5，5% warmup + cosine |
| KL | 0.001（k3，对 SFT adapter） |
| clip | 0.2 |
| 步数 | 200 |

产物：

- `checkpoints/grpo/`（policy adapter + `train_log.jsonl` + `train_summary.json`）
- `reports/grpo_val200.json` / `.jsonl`
- `reports/grpo_test200.json` / `.jsonl`
- `reports/grpo_*_recomputed.json`

预估：单步 30–90 s（含 32 条采样与逐条反传），200 步约 2–4 h；显存约 20–24 GB。

### 4.1 多卡 DDP（单卡 GRPO OOM 时）

单卡 32 GB 在约 37 步时 OOM（峰值 28.7 GB）。不改任何超参，用 4 卡把每步的 4 个 prompt 分给 4 个 rank（每 rank 1 个 prompt、8 条采样），梯度 all-reduce：

```bash
CARDS=0,1,2,3 NPROC=4 bash scripts/run_grpo_ddp.sh
```

| 项 | 说明 |
| --- | --- |
| 超参 | 与单卡完全一致（steps=200、prompts=4、n=8、lr=1e-5、kl=0.001） |
| 并行方式 | `torchrun` + HCCL；每个 rank 独立加载 3B + policy/ref adapter |
| 每卡显存 | **默认 3072/200：2 步 10.93 GB；最长 8 条 prompt 连跑 10 步稳定 11.33 GB**。DDP 是模型复制，不按卡数除显存，详见 `docs/grpo_memory_analysis.md` |
| W&B | 默认 offline，rank 0 写 JSONL + W&B；见 4.2 |
| NPROC | 建议 4；>4 时多余 rank 空转（每步只有 4 个 prompt） |
| 日志 | `logs/grpo_ddp_run.log` |
| 产物 | `checkpoints/grpo_ddp`、`reports/grpo_ddp_{val200,test200}*.json` |

实现要点：每步用相同 Python RNG 选 4 个 prompt，各 rank 取 `prompts[rank::world_size]`；本地算奖励/优势；梯度 all-reduce 后所有 rank 用相同梯度做 AdamW，因此参数与优化器状态保持一致。零方差组仍按本地组丢弃。

**显存结论：** 默认 3072/200 的 4 卡 2 步为 **10.93 GB**；用最长 8 条 prompt 做压力测试，2 步 **11.32 GB**、连跑 10 步稳定 **11.33 GB**（10/10 次更新）。DDP 分的是 prompt，不是模型或单条序列，**不降低每卡峰值**；降显存来自梯度检查点。

> 注意：HuggingFace 的 gradient checkpointing 只在 `model.train()` 下生效。此前脚本一直 `eval()`，所以 14:24 的 OOM 是“检查点名义开启、实际未生效”。现在脚本在生成/参考阶段用 `eval()`，策略反传前切 `train()`，并把 dropout 置 0。详见 `docs/grpo_memory_analysis.md`。

### 4.2 W&B 记录

`train/grpo_tracking.py` 负责 W&B，只有 rank 0 初始化；JSONL 与 W&B 共用同一条未舍入记录。

```bash
# 默认 offline，不需要登录
bash scripts/run_grpo_ddp.sh

# 指定模式/项目/名称
WANDB_MODE=online WANDB_PROJECT=sqlrl WANDB_NAME=grpo-4card bash scripts/run_grpo_ddp.sh
```

已验证 online：2026-10-22 修复后的默认 2 步冒烟 `https://wandb.ai/ge1222-beijing-university-of-posts-and-telecommunications-/sqlrl-smoke/runs/uvaksvk4`；最长 prompt 2 步 `https://wandb.ai/ge1222-beijing-university-of-posts-and-telecommunications-/sqlrl-smoke/runs/qh5lthyg`。本地元数据见 `reports/grpo_default_smoke_fix_20261022/out*/wandb_run.json`。

配置来源：

- 根目录 `.env`（模板 `.env.example`）：`WANDB_API_KEY`、`WANDB_MODE`、`WANDB_PROJECT`、`WANDB_ENTITY`、`WANDB_NAME`、`WANDB_DIR`；
- 同名环境变量与 CLI（`--wandb-mode/--wandb-project/--wandb-entity/--wandb-name/--wandb-dir`）优先；
- 默认 offline，联网模式需先完成登录。

离线记录同步：

```bash
wandb sync checkpoints/grpo_ddp/wandb/offline-run-*/
```

产物：`checkpoints/grpo_ddp/{train_log.jsonl,wandb_run.json,wandb/offline-run-*}`。
本次 2 步冒烟的 W&B 离线校验见 `reports/grpo_ddp_fix_wandb_20261022/real_out/wandb_verified.json`。

### 4.3 运行结果（2026-10-22）

| 模型 | val_200 EX | test_200 EX | test 可执行 |
| --- | --- | --- | --- |
| base | — | 51.5% | 86.0% |
| SFT | 70.0% | **69.0%** | 91.5% |
| GRPO 200 步 | **74.0%** | 65.5% | **95.0%** |

- GRPO vs SFT：test -3.5 点（McNemar p=0.21，不显著）、val +4 点（p=0.057）；
- GRPO vs base：test +14 点，明确；
- 可执行率 91.5% → 95.0%，但“可执行但答案错”45 → 59；
- 完整分析见 `docs/grpo_results.md`；W&B：`https://wandb.ai/ge1222-beijing-university-of-posts-and-telecommunications-/sqlrl/runs/endck7zy`。

v2 注意：Spider 上 GRPO 后只剩 10/200 执行错误（5%），G1 要求 ≥25%，原 v2 计划需改 BIRD 或放弃。

## 5. 评测与审计

单独看一次 val：

```bash
python3 -u eval/eval_base.py \
  --data data/processed/spider/val_200.jsonl \
  --model /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct \
  --adapter checkpoints/grpo \
  --limit 200 --batch-size 4 --max-new-tokens 260 --log-every 20 \
  --out reports/grpo_val200.json
```

重算审计：

```bash
python3 -u eval/recompute.py \
  --records reports/grpo_val200.jsonl \
  --out reports/grpo_val200_recomputed.json
```

pass@8（可选，用于看训练后组内分布）：

```bash
python3 -u eval/pass_at_k.py \
  --data data/processed/spider/rl_100.jsonl \
  --model /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct \
  --adapter checkpoints/grpo \
  --n 8 --temperature 1.0 --limit 100 \
  --out reports/grpo_pass8_rl100.json
```

## 6. 日志速查

| 实验 | 日志 | 实时查看 |
| --- | --- | --- |
| Spider base 200 | `logs/spider_base200_<ts>.log` | `tail -f logs/spider_base200_*.log` |
| SFT | `logs/sft_run.log` | `tail -f logs/sft_run.log` |
| SFT 只看 step | 同上 | `tail -f logs/sft_run.log \| grep --line-buffered '^\[step\]'` |
| GRPO | `logs/grpo_run.log` | `tail -f logs/grpo_run.log` |
| GRPO DDP | `logs/grpo_ddp_run.log` | `tail -f logs/grpo_ddp_run.log` |
| GRPO 只看 step | 同上 | `tail -f logs/grpo_run.log \| grep --line-buffered '^\[step\]'` |
| 控制台 stderr | `logs/*_console.log` | `tail -f logs/sft_console.log` |
| 卡状态 | — | `npu-smi info \| grep -E '910B4\|HBM'` |

不刷屏地看当前步：

```bash
watch -n 5 "grep '^\[step\]' logs/grpo_run.log | tail -1"
```

## 7. 预期基线

| 阶段 | 指标 | 数值 |
| --- | --- | --- |
| base（Spider dev 200） | EX / 可执行 | 51.5% / 86.0% |
| SFT（val 200） | EX / 可执行 | 70.0% / 93.0% |
| SFT（rl_100 pass@8） | pass@8 / 混合组 | 87.0% / 57.0% |
| GRPO（val 200） | EX | 待跑，目标 >70% |

## 8. 常见问题

| 现象 | 处理 |
| --- | --- |
| NPU OOM | 降 `--max-len` 到 2,048、`--batch-size` 到 2；设置 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`；GRPO 单卡 OOM 用 `scripts/run_grpo_ddp.sh`（4 卡，参数不变） |
| 下载 503 / 限速 | 用 `http://proxy.modelarts.com:80`，不要用 `127.0.0.1:9999` 的默认代理 |
| 提取失败 | 检查是否写了 `(answer)` 或未闭合；提取器只认 `<answer>`/`<sql>`/fenced/裸 SQL |
| 训练奖励长期 0 | 先看 `valid_rate` 与提取率；再判断是否需要更多 SFT/课程学习，不要先调 GRPO |
| 零方差组过多 | 说明当前 checkpoint 在该 prompt 上全对/全错；可降 SFT epoch 或换更难子集 |
| adapter 加载失败 | 确认 `checkpoints/grpo/adapter_config.json` 存在（GRPO 保存已自动 flatten 多 adapter 的 `policy/` 子目录） |

## 9. 目录

```text
sqlrl/
├── data/        build_spider.py, sample_stratified.py, build_bird.py
├── env/         sqlite_exec.py
├── reward/      exec_reward.py
├── rollout/     hf_rollout.py
├── train/       sft_lora.py, grpo.py
├── eval/        eval_base.py, recompute.py, pass_at_k.py
├── scripts/     run_spider_baseline.sh, run_sft.sh, run_grpo.sh, run_base_eval.sh
├── docs/        run_spider_baseline.md, run_sft.md, grpo_ddp_debug.md, grpo_memory_analysis.md, SQL_EXECUTION_REWARD_PLAN.md
├── .env.example  W&B 配置模板
├── logs/        *.log
├── reports/     *.json / *.jsonl
└── checkpoints/ sft_lora/, grpo/
```
