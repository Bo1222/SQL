# GRPO 多卡显存分析（2026-10-22）

## 结论（更新）

**DDP 本身不会把每卡显存按卡数除下去。** 每张卡各放一份完整 3B 模型，只把“每步 4 个 prompt”分给不同 rank；峰值由**单条最长序列**决定，不由每卡处理多少 prompt 决定。

**14:24 的 OOM 根因不是“梯度检查点不够”，而是检查点当时没有生效**：脚本一直把模型放在 `eval()`，而 HuggingFace 的 gradient checkpointing 只在 `training=True` 时启用。13:55 的 10.93 GB 冒烟峰值其实是 2 步内没遇到长序列，不能证明检查点起作用。

修复后（生成/参考用 `eval()`，策略反传前 `train()`，并把 dropout 置 0）：

| 场景 | 配置 | 峰值 | 结果 |
| --- | --- | --- | --- |
| 单卡 GRPO（修复前） | 3072/200 | **28.68 GB OOM** | 第 37 步 |
| 4 卡默认 2 步（修复前，检查点未生效） | 3072/200 | 10.93 GB | 2 步未遇长序列 |
| 4 卡默认 2 步（修复后） | 3072/200 | 10.93 GB | 峰值在生成阶段，W&B `uvaksvk4` |
| **4 卡 + 最长 8 条 prompt，2 步** | 3072/200 | **11.32 GB** | W&B `qh5lthyg` |
| **4 卡 + 最长 8 条 prompt，10 步** | 3072/200 | **11.33 GB** | 10/10 次更新，341 s，显存稳定 |
| **4 卡正式 200 步** | 3072/200 | **11.32 GB** | 96.5 min，176 次更新 |

结论：修复后默认 3072/200 在 4×32 GB 上稳定在 **~11.3 GB**，可以跑正式训练；DDP 仍然是“提吞吐”，降显存的是**真正生效的梯度检查点**。

## 显存构成

| 部分 | 估算 | 随卡数下降？ |
| --- | --- | --- |
| 3B bf16 权重（每卡一份） | ~6.2 GB | ❌ |
| policy + ref LoRA + 梯度/Adam | ~0.4 GB | ❌ |
| NPU 运行时 / allocator | ~1.5–2 GB | ❌ |
| 生成 KV（每 rank 同时 8 条） | 3072+200 约 0.9 GB | ❌ |
| 训练激活（检查点生效） | ~2 GB | ❌（同一序列） |

固定部分 8–9 GB，加长序列激活约 2–3 GB，合计 **11–12 GB**，与实测一致。

## 为什么 DDP 不降单卡峰值

1. 每个 rank 都 `model.to(device)` 一份完整模型；
2. 生成逐 prompt 做 n=8，与单卡同一 prompt 相同；
3. 训练逐序列反传，与单卡同一序列相同；
4. 分卡只改变“哪张卡遇到长 prompt”。

所以 4 卡收益是**吞吐约 4 倍**，不是单卡显存。

## 正确的梯度检查点写法

只在 `model.eval()` 时调用 `gradient_checkpointing_enable()` 不会生效。需要：

```python
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
for module in model.modules():
    if isinstance(module, torch.nn.Dropout):
        module.p = 0.0

# 生成 / old_logp / ref_logp
model.eval()
# 策略 logp + backward
model.train()
# 优化器 step 后切回
model.eval()
```

已在 `train/grpo.py`、`train/grpo_ddp.py` 中实现；dropout 置 0 是为了让 old/policy/ref 的 logprob 一致。

## 建议

1. 默认 3072/200 已经稳定 11.3 GB，**可以直接跑 200 步**；
2. 若 200 步中仍出现 OOM（碎片或极端样本），再降 `--max-prompt-tokens 2560` 或 `--max-new-tokens 150`；
3. FSDP/ZeRO 对 6.2 GB 模型收益有限，序列并行工程量大，暂不做；
4. 8 卡在 `prompts-per-step=4` 下后 4 个 rank 空转，不降显存。

## 复现

```bash
# 默认 2 步
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29513 -- \
  train/grpo_ddp.py --steps 2 --prompts-per-step 4 --n 8 \
  --max-prompt-tokens 3072 --max-new-tokens 200 \
  --out reports/grpo_default_smoke_fix_20261022/out --save-every 0 --wandb-mode online

# 最长 prompt 压力测试（8 条数据、10 步）
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29515 -- \
  train/grpo_ddp.py --data reports/grpo_default_smoke_fix_20261022/long_prompts.jsonl \
  --steps 10 --prompts-per-step 4 --n 8 \
  --max-prompt-tokens 3072 --max-new-tokens 200 \
  --out reports/grpo_default_smoke_fix_20261022/out_long10 --save-every 0 --wandb-mode offline
```

产物：`reports/grpo_default_smoke_fix_20261022/`（`run.log`、`run_long.log`、`run_long10.log`、`out/`、`out_long/`、`out_long10/`）。
