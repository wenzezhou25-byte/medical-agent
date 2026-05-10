# RAG 评测工作流

## 1) 生成冻结盲测集

```bash
python build_eval_sets.py
```

输出在 `eval_sets/`：
- `tune_questions.json`：调参集
- `dev_questions.json`：开发验证集
- `blind_questions.json`：盲测集（建议冻结）
- `split_manifest.json`：拆分记录

建议在每次确认版本后冻结：

```bash
cp eval_sets/blind_questions.json eval_sets/blind_questions_YYYY-MM-DD.json
cp eval_sets/split_manifest.json eval_sets/split_manifest_YYYY-MM-DD.json
```

本仓库当前已冻结版本：
- `eval_sets/blind_questions_2026-05-10.json`
- `eval_sets/split_manifest_2026-05-10.json`

## 2) 跑关键词 + 文档级评测

```bash
python evaluate_rag.py --questions "eval_sets/blind_questions_2026-05-10.json" --report "eval_sets/blind_report_2026-05-10.json" --top-k 7
```

说明：
- 若样本包含 `gold_sources` 或 `gold_evidence`，报告会自动输出 `Recall@k / MRR@k`。
- 无标注时，仍输出关键词命中评测。

## 3) 跑线上回放覆盖评测

回放文件支持 JSON 数组 / JSONL，每条需有 `question` 或 `query` 字段：

```bash
python replay_eval.py --input "replay_queries.jsonl" --report "replay_eval_report.json" --top-k 7
```

该报告用于看检索覆盖率与来源分布，不替代人工标注准确率评测。
