# BIRD 数据库探索与多轮 GRPO

本实现对应本轮确认的 BIRD 主训练方案。根目录 `Plan.md` 记录的是更早的 Spider 报错修复方案；它不作为新入口的默认配置。现有单轮训练、checkpoint 和评测结果保持独立。

## 逻辑与边界

模型初始看到问题、原始 evidence 和表名目录，自主选择检索字段描述、查看取值、试执行或提交。动作协议：

```text
<schema>loan status meaning</schema>
<values>{"table":"loan","column":"status","keyword":""}</values>
<sql>SELECT ...</sql>
<answer>SELECT ...</answer>
```

一次回复只能有一个完整动作。环境插入 `<observation>...</observation>`；不自动补齐截断动作，不把最后一条可执行 SQL 当成隐式提交。最多 4 次工具动作，再给一次提交机会，连续两次非法动作终止。无最终答案奖励为 0。

```text
序列       prompt | action | observation | action | observation | answer
loss_mask     0   |   1    |      0      |   1    |      0      |   1
attention     1   |   1    |      1      |   1    |      1      |   1
```

策略梯度、KL、SFT 的预测目标只包含模型 token；观测参与后续动作的条件计算，不作为待预测目标。边界来自运行时追加 token 的来源，不通过正则识别“观测”，模型伪造 observation 标签也不能绕过 loss。保存原始 token IDs，更新时不重新 tokenize 整段历史，不重新执行工具。轨迹记录 tokenizer 指纹，训练时拒绝不匹配的词表。

`masked_logprob` 用动作目标位置的前一个 hidden state 计算 logits，避免对长观测构建完整词表 logits。训练前向使用 `train()` 和梯度检查点；采样、old/ref 用 `eval()`；dropout 置 0。采样 temperature=1、top_p=1、top_k=0、repetition_penalty=1，与 old-policy 分布定义一致。

## 数据与评分

准备命令复用 `data/raw/train.zip`，先解出其中的 `train/train_databases.zip`，再展开数据库。只下载官方过滤版标注，不重复下载数据库。标注 commit、文件 SHA256、本地归档 SHA256、元数据来源和排除记录写入 manifest。过滤版数据源：<https://huggingface.co/datasets/birdsql/bird23-train-filtered>。

按固定 seed=0 留出 10% 数据库作 val；其余库按同库、相同 gold SQL 分组后以 70%/30% 分到 SFT/RL，防止完全相同 SQL 的改写问题跨侧。训练问题去重并只读执行校验 gold，失败记录原因。BIRD dev 保持 test，只用于冻结后的最终对照，不参与模板构造、课程筛选或选模。原始数据缺少 train difficulty 标签，不伪造难度等级。

实际准备结果为 SFT 4,205 / RL 1,801 / val 590；原始 6,601 条中 2 条问题重复、3 条 gold 在 30 秒预算内超时。Agent 的初始 prompt 全部低于 1,536 token。SQL SFT 的全 schema 输入有 275 条超过 6,144 token，这些样本会改用问题驱动的静态元数据检索，并在 `data_preparation.json` 计数，不根据 gold 选择表，也不静默截断问题。

CSV 描述保留字段说明、取值说明、未命名列与编码来源。`search_schema` 用本地 BM25；`inspect_column` 返回前 10,000 行源样本内的有限取值/范围，并标记样本不完整，不能把没有搜索命中当作不存在。

工具执行只允许单条 SELECT/WITH，使用 SQLite authorizer、只读 immutable URI、进度超时，以及独立子进程的硬超时/内存限制。模型不能传数据库路径。结果预览最多 5 行，使用 `has_more` 区分预览与精确行数。

**最终评分与工具预览分离。** 按 BIRD 公开实现的 `set(predicted_rows) == set(gold_rows)` 比较完整结果，不使用前 1,000 行截断比较；保留列顺序，忽略行顺序与重复行，不对数字做额外字符串归一化。评分版本为 `bird-public-set-v1-full-results`。参考：<https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py>。

主奖励是终局 EX 的 0/1，不给工具调用、可执行性或格式单独加分。`constant_no_from` 保留为审计指标，同时报告 EX 与 ex_clean，不把自定义审计分数混称为官方 EX。数据库丢失、gold 失败等基础设施问题单列；RL 丢弃整个受影响问题组，不把它误标为错误答案。测试集不静默剔除问题，出现基础设施问题时需要排查后重评。

## 分阶段执行

所有入口都要求明确阶段，不会自动串联完整训练。已有 `.env` 的 W&B 配置可复用；建议开发时 `WANDB_MODE=offline`。

```bash
# P0：本地数据准备与 gold 校验；不训练
bash scripts/run_bird_agent.sh prepare

# P1：BIRD 域内 SQL SFT，从已有 Spider SFT adapter 初始化
bash scripts/run_bird_agent.sh sql-sft

# P2：本地规则工具轨迹，仅使用 split=sft
bash scripts/run_bird_agent.sh demonstrations

# 用模板先教会协议；每种数据来源都有独立记录
bash scripts/run_bird_agent.sh tool-sft \
  --data data/processed/bird_agent/synthetic.jsonl --out checkpoints/agent_bootstrap

# 本地 bootstrap 模型采样，保留真实终局成功轨迹
bash scripts/run_bird_agent.sh demonstrations --source on-policy \
  --adapter checkpoints/agent_bootstrap --out data/processed/bird_agent/on_policy.jsonl
bash scripts/run_bird_agent.sh combine

# 最终工具 SFT，从 bird_sql_sft 开始，平衡两类监督
bash scripts/run_bird_agent.sh tool-sft

# P3：先在 val 比较，limit=0 表示完整集合
bash scripts/run_bird_agent.sh eval --mode agent --limit 200
bash scripts/run_bird_agent.sh eval --mode single --adapter checkpoints/bird_sql_sft \
  --out reports/agent_static_val.json --limit 200

# 仅用 RL 训练题做组奖励诊断；不是 test 评测。
# 合法动作率 >=95%、存在混合组、无基础设施失败，才通过默认启动门槛。
bash scripts/run_bird_agent.sh pilot

# P4：正式训练入口，只有明确运行此命令才会启动。
# 默认 100 次混合奖励有效更新，最多 300 次采样尝试。
CARDS=0,1,2,3 NPROC=4 SEED=0 WANDB_MODE=offline bash scripts/run_bird_agent.sh grpo

# 每 25 个有效更新做 val、保存 checkpoint，提升 >=0.5pp 保存 best。
# 连续 3 次无改进提前停止。val 只接受 DB 隔离的 split=val。
bash scripts/run_bird_agent.sh eval --adapter checkpoints/agent_grpo_seed0/best \
  --mode agent --limit 0 --out reports/agent_grpo_val.json
```

SFT/RL 入口拒绝错误 split。模板轨迹只示范直接提交、真实字段检索、取值查看和真实列名错误，监督目标可使用 gold，工具观测不会收到 gold。on-policy 轨迹不通过标准答案提示模型。两类轨迹按相同数量混合、去重，数量与来源写入 summary；如果本地模型没有成功轨迹，应先处理冷启动，不能用空文件假装完成第二阶段。

首版默认生成预算 2,048、上下文 6,144、初始 prompt 1,536、单动作 512、单观测 384 token。超长 prompt 明确失败而不是静默截掉问题；历史从不在更新前被改变。每 rank 顺序执行轨迹以控制峰值显存。四卡复制完整模型，不能按卡数除单轨迹显存。

## 对照、消融与统计

| 组 | 命令差异 | 用途 |
|---|---|---|
| A | `eval --mode single --adapter checkpoints/bird_sql_sft` | 固定检索一次，再单轮回答 |
| B | `eval --mode agent --adapter checkpoints/agent_sft` | 无 RL 的工具 SFT Agent，主控制组 |
| C | `grpo --max-tools 0 --static-retrieval --out ...` | 同源单轮 RL 对照 |
| D | `grpo`，再 `eval --mode agent --adapter .../best` | 自主多轮 RL，主实验 |
| E | `eval --mode vote --temperature 1` | 固定检索 + 候选 SQL 执行结果投票，无 gold 选优 |
| 固定流程 | `eval --mode fixed` | 固定检索、候选 SQL、试执行、最终修改 |

默认工具预算 4，vote 会扣除一次固定检索，最多执行 3 条候选；生成总预算在候选间平分。候选一致性按完整执行结果签名，不使用正确性 oracle。不同模式使用相同最大预算，实际 input/output token、工具次数、候选数及耗时全部报告，不能只根据最大预算声称实际成本相同。

```bash
# 消融：三个工具名分别是 search_schema / inspect_column / execute_sql
bash scripts/run_bird_agent.sh eval --disable-tool inspect_column --out reports/agent_no_values.json
bash scripts/run_bird_agent.sh eval --error-only --out reports/agent_error_only.json
bash scripts/run_bird_agent.sh eval --max-tools 2 --out reports/agent_two_tools.json

# 先冻结配置再对完整 BIRD dev 运行；不要用这个集合反复选模
bash scripts/run_bird_agent.sh eval --data data/processed/bird_agent/test.jsonl \
  --adapter checkpoints/agent_grpo_seed0/best --limit 0 --out reports/agent_grpo_test.json

# 两个 records 必须具有完全相同的唯一 ID、问题、数据库和 gold
bash scripts/run_bird_agent.sh compare --a reports/agent_sft_test.jsonl \
  --b reports/agent_grpo_test.jsonl --out reports/agent_paired.json
```

比较器提供逐题迁移、精确 McNemar、按 DB 聚类 bootstrap 95% 区间；核心 B/D 用 seed 0/1/2，同一 seed 内共享冷启动 checkpoint。单次 200 条 pilot 不能作为稳定 RL 增益证据，调用次数增多也不能代替最终 EX。

GRPO 对所有有效轨迹计算 action-only KL，全零/全一组优势为零但仍可有 KL 更新；`effective_updates` 只计有混合组的更新，`optimizer_steps` 包括 KL-only 更新。达到采样尝试上限会在 summary 标记 `requested_updates_reached=false`，不能当成完成请求的有效步数。

## 验证与当前交付

单元测试使用独立小模型 CE/梯度验证 mask、因果对齐和 checkpoint 重算；SQL 测试覆盖多语句、可写 CTE、扩展加载、硬超时、NULL、列顺序、重复行以及第 1,000 行之后的差异。

```bash
bash scripts/run_bird_agent.sh smoke
python3 -B -m pytest tests -q
python3 -B -m tests.agent_smoke prepare reports/bird_agent_smoke_20261022
```

`tests.agent_smoke worker` 是有界集成夹具：仅控制模型生成的动作，真实执行 SQL、log-prob、反传、HCCL、优化器、ref 冻结、保存和 W&B；它验证工程正确性，不衡量模型能力。正式模型的探索成功率、准确率增益和多 seed 结果需要后续实验，不在本次实现中伪造。实际验收结果见 `reports/bird_agent_smoke_20261022/verification.json`。

本次验收：全部 41 项测试通过；四卡/两个空闲 rank 完成 2 次有效更新且 ref 不变；SFT 与 GRPO 的实际 `.wandb` 二进制记录均与 JSONL 一致。正式 3B 的 6,144-token 上下文、2,048 个计入损失的动作 token 压力测试通过，峰值 11.26 GB。它是一次合成 CE 反传，不能代替完整 RL 运行的显存与效果验证。

四卡冒烟可使用新目录重跑，避免覆盖已有 adapter：

```bash
AGENT_SMOKE_ROOT=reports/bird_agent_smoke_local
python3 -B -m tests.agent_smoke prepare "$AGENT_SMOKE_ROOT"
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun \
  --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29537 \
  --module -- tests.agent_smoke worker "$AGENT_SMOKE_ROOT" \
  --data "$AGENT_SMOKE_ROOT/rl.jsonl" --model "$AGENT_SMOKE_ROOT/model" \
  --adapter "$AGENT_SMOKE_ROOT/initial_adapter" --out "$AGENT_SMOKE_ROOT/grpo_npu" \
  --device npu --steps 2 --max-attempts 2 --prompts-per-step 2 --n 2 \
  --save-every 1 --val-data "$AGENT_SMOKE_ROOT/val.jsonl" --val-every 1 --val-limit 2 \
  --wandb-mode offline --wandb-project sqlrl-agent-smoke

# 可选的正式 3B 单次压力测试，仅一个合成反传，不启动完整训练
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  python3 -B -m tests.agent_smoke long "$AGENT_SMOKE_ROOT"
```
