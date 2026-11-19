# BIRD 多轮 SQL Agent

基于 [BIRD](https://bird-bench.github.io/) 的检索增强多轮 SQL Agent：模型通过 `schema_lookup`、`sql_execute`、`answer` 三个工具自主检索 schema、执行只读 SQL 并提交答案。仓库包含数据构建、SFT / Agentic RL 训练、离线评测与统计分析的完整代码。

## 目录结构

```
agent/         多轮 Agent 循环、SQL episode 运行时与协议
retrieval/     catalog 解析、BM25 schema 检索、join 关系与 E1.1/E1.2 Observation
tools/         只读 SQL worker、sql_execute、schema_lookup
reward/        BIRD-compatible EX、typed RL reward、过程信号
eval/          Agent / 检索评测、gold 抽取、配对统计
data/          数据集构建与校验（metadata、SFT、RL、split 泄漏检查）
train/         GRPO / DAPO 训练入口、Dynamic Sampling、padding
e1_alignment/  E1 对齐数据构建、Qwen3 SFT 训练与 provenance
configs/       agent loop、数据、模型、检索、reward、训练配置
scripts/       数据、训练、评测运行脚本
tests/         契约与单元测试
```

生成目录（`data/raw/`、`artifacts/`、`outputs/`、`checkpoints/`、`logs/`）不进入版本库。

## 环境要求

- Ascend 910B4 (32 GB HBM)，CANN 8.5.1，vLLM-Ascend
- Python 3.11，PyTorch 2.9 + torch_npu 2.9
- veRL、vLLM / vLLM-Ascend
- 运行 `source scripts/env.sh` 配置 CANN/ATB/HCCL 与 vLLM spawn 模式

`requirements-rebuild.txt` 记录额外的 SQL 解析依赖。

## 快速开始

```bash
source scripts/env.sh

# 数据与检索索引
python3 data/build_metadata.py
python3 data/build_rl_dataset.py --out artifacts/datasets/rl_full.parquet
python3 data/build_sft_dataset.py
python3 data/validate_splits.py
python3 data/validate_dataset.py --sample 100

# 测试
python3 -m pytest tests -q

# 检索 / Agent 离线评测
python3 eval/evaluate_retrieval.py --limit 120 --out outputs/eval/retrieval.json
python3 eval/evaluate_agent.py --limit 100 --backend unified --engine vllm \
  --model artifacts/checkpoints/step_600_merged --out outputs/eval/agent_val100.json

# 训练
bash scripts/run_m6_08m_sft_3ep.sh   # Qwen2.5-3B 全参数 SFT
bash scripts/run_m6_08m_rl.sh        # GRPO / DAPO Agentic RL
```

## 实验结果

BIRD 上的主要结果（EX，执行准确率）：

| 模型 | dev200 | test1534 |
|---|---|---|
| Qwen2.5-3B base + E1.1 检索 | 49.5% (99/200) | 30.70% (471/1534) |
| + E1.1 SFT（lr 5e-6，step 788） | **58.0%** (116/200) | **32.66%** (501/1534) |

- dev200：+8.5pp（McNemar p=0.0076）。
- test1534：+1.96pp（McNemar p=0.0675，DB bootstrap 95% CI [−2.84, +6.62]），属于弱正信号，不宣称稳健提升。
- 检索增强消融：Table Recall 0.894 → 0.996，Join Recall 0.623 → 0.817，prompt token −37%。
- Agentic RL（R1 GRPO 20 步、R3 DAPO Dynamic Sampling）在 dev200 上为中性（58.5% vs 58.0%，p=1.0），尚未观察到任务收益。

## 训练与评测约束

- **单卡 32 GB**：3B 使用全参数 SFT / FSDP2；更大模型需 LoRA 或多卡。
- **必须清空 `PYTORCH_NPU_ALLOC_CONF`**，否则与 vLLM-Ascend 的 NPU 内存池冲突。
- 训练 rollout 必须使用 `bird_sql_agent`（注册于 `configs/agent/agent_loops.yaml`）。
- SFT 与评测保持一致：`+bird_agent.use_chat_template=false`（裸文本 prompt）。
- dev200 仅用于 checkpoint 选择；test1534 为独立最终测试集。
