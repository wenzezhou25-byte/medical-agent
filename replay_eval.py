import argparse
import json
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS

from config import VECTOR_STORE_PATH
from embedding_provider import get_embeddings
from rag_utils import classify_query_type, create_hybrid_retriever, retrieve_evidence_docs


def _read_replay_queries(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError("回放文件需为 JSON 数组或 JSONL。")


def _extract_query(row: dict[str, Any]) -> str:
    for key in ("question", "query", "user_query", "content"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def replay_evaluate(input_path: Path, report_path: Path, top_k: int) -> None:
    if not Path(VECTOR_STORE_PATH).exists():
        raise FileNotFoundError(f"未找到向量库目录：{VECTOR_STORE_PATH}")

    rows = _read_replay_queries(input_path)
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = create_hybrid_retriever(vectorstore, vector_k=8, bm25_k=10, vector_weight=0.65, bm25_weight=0.35)

    per_query = []
    category_stats: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows, 1):
        question = _extract_query(row)
        if not question:
            continue
        category = classify_query_type(question)
        docs = retrieve_evidence_docs(retriever, question, top_k=top_k)
        sources = [str((doc.metadata or {}).get("source_name", "")) for doc in docs]

        bucket = category_stats.setdefault(category, {"count": 0, "non_empty": 0, "avg_unique_sources": 0.0})
        bucket["count"] += 1
        if docs:
            bucket["non_empty"] += 1
        bucket["avg_unique_sources"] += len({src for src in sources if src})

        per_query.append(
            {
                "index": idx,
                "question": question,
                "category": category,
                "retrieved_count": len(docs),
                "top_sources": sources[:3],
            }
        )

    total = len(per_query)
    for stat in category_stats.values():
        c = stat["count"] or 1
        stat["coverage"] = f"{(stat['non_empty'] / c * 100):.1f}%"
        stat["avg_unique_sources"] = round(stat["avg_unique_sources"] / c, 2)
        del stat["non_empty"]

    report = {
        "input_file": str(input_path),
        "top_k": top_k,
        "query_count": total,
        "global_coverage": f"{(sum(1 for q in per_query if q['retrieved_count'] > 0) / total * 100) if total else 0.0:.1f}%",
        "category_stats": category_stats,
        "samples": per_query[:50],
        "notes": [
            "回放评测用于观察线上问题是否可被检索到，不替代标注评测。",
            "建议配合 blind 集合与文档级 Recall@k 一起看。",
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n回放评测已保存到: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于历史问题回放评估检索覆盖率。")
    parser.add_argument("--input", required=True, help="回放文件（JSON/JSONL）")
    parser.add_argument("--report", default="replay_eval_report.json", help="输出报告路径")
    parser.add_argument("--top-k", type=int, default=7, help="每条查询保留的证据条数")
    args = parser.parse_args()
    replay_evaluate(Path(args.input), Path(args.report), args.top_k)
