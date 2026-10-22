GRPO DDP 修复说明（2026-10-22）

本次修复涉及 `train/grpo_ddp.py`、`scripts/run_grpo_ddp.sh`，并将共享的生成与评分逻辑放在 `train/grpo_utils.py`，供单卡 `train/grpo.py` 复用。W&B 记录位于 `train/grpo_tracking.py`。本次验证使用独立输出和有限步数，没有执行正式训练脚本及其 val/test 评测。

**1. 因果 logprob：为什么旧代码错了一位**

修改前：

```python
keep = int(ids.numel() - start)
output = model(input_ids=ids.unsqueeze(0), logits_to_keep=keep, ...)
logits = output.logits[0]
targets = ids[start:]
logp = logits.float().log_softmax(-1).gather(1, targets[:, None])
```

设 `ids=[P0,P1,A0,A1]`，prompt 长度 `start=2`，回答长度 `keep=2`。Qwen2 的 `logits_to_keep=2` 仅截取最后两个输入位置，没有替调用方移位。

| 回答标签 | 旧代码所用 logits 的输入位置 | 该位置已经看过 | 应使用的位置 |
| --- | --- | --- | --- |
| A0 | 2 | P0,P1,A0 | 1：只看过 P0,P1 |
| A1 | 3 | P0,P1,A0,A1 | 2：只看过 P0,P1,A0 |

因此旧代码求的是已经见过目标 token 后的概率，而目标应为 `p(A0|P0,P1)` 与 `p(A1|P0,P1,A0)`。`old_logp/ref_logp/logp` 都调用该函数，所以策略梯度和 KL 一起受影响。

修复：保留 `keep+1` 个 logits，包含最后一个 prompt 位置，再丢弃最后一个输入位置：

```python
output = model(..., logits_to_keep=keep + 1)
logits = output.logits[0, :-1]
targets = ids[start:]
```

回归测试对比模型内置的 `labels` 因果交叉熵，并对比参数梯度；包含只有一个回答 token 的边界情况。

**2. 提前结束与 padding：为什么文本看起来正常，训练仍有问题**

修改前：

```python
text = tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
samples.append({"ids": row, "start": prompt_len, "text": text})
# scoring:
attention_mask = torch.ones_like(ids)
# optimization:
pg_loss = (...).mean()
kl_loss = (...).mean()
```

同组短回答可能是 `[A0,EOS,PAD,PAD]`。解码仅从显示文本中隐藏特殊 token；`ids` 仍保存整行。随后全 1 mask 和对所有响应位置求均值，让补齐 token 进入策略损失与 KL，也改变真实回答 token 的权重。

修复使用 `CompletionLengths` 停止条件，在每条序列第一次触发 EOS 或 `</answer>` 时记录绝对长度，然后截断保存的 `ids`。真实 EOS 或完整停止字符串被保留，之后的 padding 不进入评分。达到 max_new_tokens 但未提前结束的序列保留全部响应。

不能简单去掉所有尾部 pad ID：本机 Qwen2.5 的 `pad_token_id=151643`，该 ID 同时属于 generation_config 的 EOS 列表。第一次真正生成的终止 token 和停止后补齐的 token 可能具有相同 ID。按实际停止时刻记录长度才能正确区分。此前“43/64 条 padding”的精确口径是 43 条尾部有 pad ID；39 条含至少两个连续 pad ID，另外 4 条单个 ID 需要结合停止位置判定。

**3. 全局日志：为什么值会被低估**

修改前，各 rank 只累加自己的变量：

```python
loss_sum += float(loss.detach())
kl_sum += float(kl.mean().detach())
if rank == 0:
    record["loss"] = loss_sum / global_kept
    record["kl"] = kl_sum / global_kept
```

例如 4 张卡各保留 8 个样本、每卡损失和为 8，则正确均值为 `(8+8+8+8)/32=1`，旧日志写成 rank 0 的 `8/32=0.25`。当 rank 0 没有保留样本时，还会在其他 rank 有训练的情况下记录为 0。

修复先由全部 rank 对 `[loss_sum,kl_sum]` 执行 SUM，再由 rank 0 除以全局样本数写日志。空闲 rank 也参加同样顺序的 collective。梯度本身原来采用 `loss/global_kept` 后 SUM all-reduce，此处是正确的，不再额外除以 world_size。

现在 JSONL 和 W&B 共用同一条未舍入的记录，包含 reward、valid_rate、loss、KL、梯度范数、实际本步 lr、next_lr、有效样本数、零方差组数、全局峰值显存、实际更新计数、skipped、step_s 与 elapsed_s。全零组不会静默消失，summary 区分尝试步数、更新步数与跳过步数。

**4. launcher：为什么训练 Python 尚未执行就失败**

修改前：

```bash
torchrun --standalone --nproc_per_node="$NPROC" ... \
  train/grpo_ddp.py ... --n 8
```

本机 PyTorch 2.9.0 的 torchrun 参数解析器把 `--n` 视为 launcher 参数的缩写，同时匹配 `--nnodes`、`--nproc-per-node` 等选项，报 `ambiguous option: --n` 并返回 2。这发生在启动 worker 前。

修复在训练入口前添加 `--`，明确结束 launcher 参数解析。同时使用单节点静态 rendezvous，让 MASTER_PORT 真正生效：

```bash
torchrun --nnodes=1 --nproc_per_node="$NPROC" \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29511}" -- \
  train/grpo_ddp.py ... --n 8
```

**W&B 的实际记录流程**

仅 rank 0 创建一个 run；其余 rank 不初始化 W&B。全部 rank 先完成指标归约，rank 0 再同时写 JSONL 和 `run.log(record,step=step)`。退出上下文时写成功/失败状态并调用 `run.finish(exit_code=...)`。分布式进程组在 finally 清理。

根目录 `.env` 可配置 `WANDB_API_KEY` 及以下 W&B 配置；Python 入口和启动脚本自动加载，现有环境变量与 CLI 优先。模板见 `.env.example`，依赖见 `requirements-config.txt`。

新增 CLI：`--wandb-mode {disabled,offline,online}`、`--wandb-project`、`--wandb-entity`、`--wandb-name`、`--wandb-dir`；也接受同名 W&B 环境配置。默认 Python 入口尊重 WANDB_MODE，未设置时 disabled；启动脚本默认 offline，并在 source 公共 env.sh 前保存调用方指定的模式，防止其强制 disabled 覆盖用户意图。

prompt 抽样使用独立的 `random.Random(seed)`。这是接入 rank 0 专属 SDK 时必须处理的细节：如果继续用全局 Python random，任何只在 rank 0 消耗随机数的操作都可能导致各 rank 后续抽取不同的全局 prompt 列表。

本次不依赖登录验证离线记录。日志记录训练配置和标量指标，W&B SDK 另行记录其默认运行环境信息。线上模式需要在运行环境中完成 W&B 登录。

可复核路径：`<out>/train_log.jsonl`、`<out>/train_summary.json`、`<out>/wandb_run.json`、`<out>/wandb/offline-run-*/run-*.wandb`。本次测试通过解析最后一个二进制文件，逐项对比 JSONL，而非仅检查 SDK 控制台提示。

专用两步冒烟命令为：

```bash
bash reports/grpo_ddp_fix_wandb_20261022/run_real_smoke.sh
```

该命令不调用正式启动脚本，输出位于报告目录的 `real_out/`。完成登录后，可用 `wandb sync <out>/wandb/offline-run-...` 同步已经生成的离线记录，无需重新训练；本次未验证云端同步。

W&B 接口依据：[分布式记录文档](https://docs.wandb.ai/models/track/log/distributed-training)、[Run API](https://docs.wandb.ai/models/ref/python/experiments/run)；实际验证 SDK 版本为 0.25.1。

**显存与测试边界**

多卡仍是完整模型复制，按 prompt 分摊工作。原单卡实现已逐 prompt 生成 n=8、逐条反传；分卡不降低单次生成 batch 或单条反传序列的长度，因此不能保证解决长序列 OOM。本次真实模型仅验证 2 步、4 个短 RL prompt、n=8、max_new_tokens=96、max_prompt_tokens=1024。默认 3072+200 的最长输入路径仍需单独验证。
