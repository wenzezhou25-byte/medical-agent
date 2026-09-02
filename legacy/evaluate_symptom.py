"""症状→推荐用药 候选药抽样查看。

说明：本脚本输出【非准确率指标】。推荐用药的对错不唯一（同类药有多种，
还依赖患者年龄/禁忌/严重度等 query 中缺失的信息），因此不能像"问药信息/查章节"
那样用准确率/召回/MRR 衡量。这里仅作"候选药抽样的可读性查看"，帮助人工判断
检索是否回到了语义相关的药类，不作为 RAG 检索性能依据。
"""

import json

from vector_store import VectorStore

from config import VECTOR_STORE_PATH
from embedding_provider import get_embeddings
from rag_utils import create_hybrid_retriever, retrieve_evidence_docs

QUESTIONS = "eval_sets/symptom_questions.json"
REPORT = "eval_sets/symptom_report.json"
TOP_K = 7


def hit_token(source_name, tokens):
    for t in tokens:
        if t in source_name:
            return t
    return None


def main():
    questions = json.load(open(QUESTIONS, encoding="utf-8"))
    vs = VectorStore.load_local(VECTOR_STORE_PATH, get_embeddings())
    retriever = create_hybrid_retriever(vs, vector_k=8, bm25_k=12, vector_weight=0.65, bm25_weight=0.35)

    details, hit_n = [], 0
    for q in questions:
        docs = retrieve_evidence_docs(retriever, q["question"], top_k=TOP_K)
        sources = [str((d.metadata or {}).get("source_name", "")) for d in docs]
        hit_tok, hit_idx = None, None
        for i, s in enumerate(sources):
            t = hit_token(s, q["expected_tokens"])
            if t:
                hit_tok, hit_idx = t, i
                break
        if hit_tok:
            hit_n += 1
        details.append({
            "question": q["question"],
            "note": q.get("note"),
            "expected_tokens": q["expected_tokens"],
            "命中药": hit_tok,
            "命中位置": hit_idx,
            "是否命中": bool(hit_tok),
            "Top证据": [{
                "source_name": s,
                "section_title": str((d.metadata or {}).get("section_title", "")),
                "rerank_score": (d.metadata or {}).get("rerank_score"),
            } for d, s in zip(docs[:5], sources[:5])],
        })

    report = {
        "指标性质": "非准确率指标，仅供候选药抽样查看；不作为 RAG 检索性能依据",
        "测试总数": len(questions),
        "命中数": hit_n,
        "候选药抽样·命中数(非指标)": f"{hit_n}/{len(questions)}",
        "详细结果": details,
    }
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("候选药抽样·命中数(非准确率指标) =", f"{hit_n}/{len(questions)}")
    for d in details:
        mark = "Y" if d["是否命中"] else "N"
        print(f"  [{mark}] {d['question']}  ->  命中药={d['命中药']} 位置={d['命中位置']}")
    print("报告已保存:", REPORT)


if __name__ == "__main__":
    main()