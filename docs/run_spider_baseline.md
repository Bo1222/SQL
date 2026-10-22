# Spider dev 200 分层基线：执行记录

日期：2026-10-22 00:27–00:34（Asia/Shanghai）
主机：`notebook-cedc4280...`（910B4 单卡）
模型：`/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct`

## 一键执行

```bash
ssh verl_rl
bash ~/work/sqlrl/scripts/run_spider_baseline.sh
```

脚本里 `source env.sh` 后依次跑三步，所有输出同时写入 `logs/spider_base200_<timestamp>.log`。

## 步骤 1：分层抽样（`data/sample_stratified.py`）

- 输入：`data/processed/spider/test.jsonl`（1,034 题 / 20 库）
- 逻辑：
  1. 按 `db_id` 分组；
  2. 按各库题量做比例分配（每个库至少 1 条）；
  3. 用 `seed=0` 在库内洗牌后取前 N 条；
  4. 按 id 排序输出，写 manifest（每个库选了多少、选了哪些 id）。
- 输出：
  - `data/processed/spider/test_200.jsonl`
  - `reports/spider_test200_manifest.json`

日志里每个库一行 `[manifest] db=... available=... selected=...`，最后 `[manifest] wrote ...`。

## 步骤 2：零样本评测（`eval/eval_base.py`）

- 加载 3B bf16 到 NPU（实测 6.0 s）；
- 按 batch=4 生成，`max_new_tokens=260`，`stop_strings=["</answer>"]`，greedy；
- 每生成完 20 条打一行 `[gen] done/total elapsed`；
- 然后逐条评分：
  1. 从输出提取 SQL（`<answer>` → `<sql>` → 未闭合 → fenced → 裸 SQL）；
  2. 执行 gold，拿参考结果；
  3. 执行候选；
  4. 比较结果（gold 含 `ORDER BY` 时顺序敏感，否则多重集）；
  5. 审计：`constant_no_from` 为硬作弊；`empty_both`、`single_row_both` 只记 info；
  6. 每评完一条就追加写入 `reports/spider_base200.partial.jsonl` 并 flush；
  7. 每 20 条打一行 `[score] done/total ex=... valid=... elapsed=... last=...`。
- 输出：
  - `reports/spider_base200.json`（汇总）
  - `reports/spider_base200.jsonl`（逐样本完整记录）

## 步骤 3：重算审计（`eval/recompute.py`）

不重新生成，只读取 `reports/spider_base200.jsonl` 里的 `raw_output`，用当前提取/比对/审计规则重算一遍，输出 `reports/spider_base200_recomputed.json`。用于改动审计规则后快速复核。

## 路径清单

| 内容 | 路径 |
| --- | --- |
| 入口脚本 | `~/work/sqlrl/scripts/run_spider_baseline.sh` |
| 分层抽样 | `~/work/sqlrl/data/sample_stratified.py` |
| 评测脚本 | `~/work/sqlrl/eval/eval_base.py` |
| 重算脚本 | `~/work/sqlrl/eval/recompute.py` |
| 200 条子集 | `~/work/sqlrl/data/processed/spider/test_200.jsonl` |
| 抽样清单 | `~/work/sqlrl/reports/spider_test200_manifest.json` |
| 运行日志 | `~/work/sqlrl/logs/spider_base200_20261022_002701.log` |
| 汇总结果 | `~/work/sqlrl/reports/spider_base200.json` |
| 逐样本结果 | `~/work/sqlrl/reports/spider_base200.jsonl` |
| 逐样本中间态 | `~/work/sqlrl/reports/spider_base200.partial.jsonl` |
| 重算结果 | `~/work/sqlrl/reports/spider_base200_recomputed.json` |

## 实测中间过程（日志摘录）

```text
[manifest] source=data/processed/spider/test.jsonl total=1034 dbs=20 limit=200 seed=0
[manifest] db=car_1 available=92 selected=18
[manifest] wrote data/processed/spider/test_200.jsonl and reports/spider_test200_manifest.json

[start] 2026-10-22 00:27:14 data=data/processed/spider/test_200.jsonl n=200 batch=4 max_new=260
[load] model loaded in 6.0s
[gen] 20/200 elapsed=49.9s
[gen] 100/200 elapsed=195.7s
[gen] 200/200 elapsed=400.7s
[gen] done 200 predictions in 406.6s

[score] 20/200 ex=12 (60.0%) valid=15 (75.0%) elapsed=407s last=spider_dev_000111
[score] 100/200 ex=52 (52.0%) valid=80 (80.0%) elapsed=413s last=spider_dev_000528
[score] 200/200 ex=97 (48.5%) valid=161 (80.5%) elapsed=413s last=spider_dev_001032
[done] wrote reports/spider_base200.json and reports/spider_base200.jsonl
```

## 结果

| 指标 | 数值 |
| --- | --- |
| n | 200 |
| gold 可执行 | 200（100%） |
| 提取成功 | 199（99.5%） |
| 可执行 | 172（86.0%） |
| 正确（EX） | 103（51.5%） |
| 硬作弊 | 0 |
| 空对空（info） | 9 |
| 单行对单行（info） | 56 |
| 单样本耗时 | 2.07 s |
| 总耗时 | 413 s |

错误类型：`no_such_column` 25、`no_such_function` 1、`no_such_table` 1、`extract_missing` 1（初版日志里的 97/48.5% 是修复提取器之前的评分，最终以 `spider_base200_recomputed.json` 为准）。

## 备注

- 唯一提取失败（`spider_dev_000584`）是模型写成 `(answer)` 且被截断，属于格式偏差，不是解析器漏判。
- 初版标记的 7 条 `multi_statement` 经检查是“裸 SQL + 末尾多余 `</answer>`”，不是真正的多语句；提取器已剥离游离标签，这些样本正常执行，真正多语句为 0 条。
- `empty_both` 9 条经检查是合法的空结果（gold 本身为空），不能当作弊；已降级为 info。
