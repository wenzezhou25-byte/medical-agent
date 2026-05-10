import argparse
import json
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS

from config import VECTOR_STORE_PATH
from embedding_provider import get_embeddings
from rag_utils import classify_query_type, create_hybrid_retriever, retrieve_evidence_docs


TEST_QUESTIONS_PATH = Path("test_questions.json")
REPORT_PATH = Path("retrieval_accuracy_report_rerank.json")

SYNONYMS = {
    "低盐": ["低钠", "少盐", "减盐", "限盐", "减少钠盐", "盐摄入<5g", "每日食盐不超过5g"],
    "低脂": ["低脂肪", "少油", "减脂", "少油少脂", "控制脂肪摄入"],
    "控糖": ["控血糖", "血糖控制", "降糖"],
    "钾元素": ["钾", "补钾", "富钾"],
    "戒烟限酒": ["戒烟", "限酒", "禁烟", "少喝酒", "戒酒", "限制饮酒"],
    "运动": ["锻炼", "有氧运动", "规律运动", "适量运动", "每周150分钟", "中等强度运动"],
    "体重管理": ["控制体重", "减重", "减肥", "体重控制", "腰围管理", "体重达标"],
    "解热镇痛": ["退烧", "止痛", "解热", "镇痛"],
    "止咳药": ["止咳", "镇咳", "化痰", "祛痰", "缓解咳嗽"],
    "多喝水": ["多饮水", "补充水分", "充足水分", "大量饮水", "注意补水"],
    "就医指征": ["就医", "就诊", "看医生", "及时就医", "必要时就医"],
    "维A酸类药物": ["维A酸", "维甲酸", "异维A酸"],
    "抗氧化剂": ["抗氧化", "抗氧"],
    "抗菌药物": ["抗生素", "抗菌", "抗感染"],
    "剂量与疗程": ["剂量", "疗程", "用法用量"],
    "不良反应": ["副作用", "副反应", "不良事件"],
    "全身不适": ["乏力", "精神不振", "全身症状"],
    "皮肤苍白": ["面色苍白", "脸色苍白", "肤色发白"],
    "恶心呕吐": ["恶心", "呕吐", "反胃", "干呕"],
    "发热": ["发烧", "体温升高"],
    "惊厥": ["抽搐", "痉挛", "抽风"],
    "温开水": ["温水", "适量温开水", "开水"],
    "口服": ["内服"],
    "适应症": ["功能主治", "主治", "用于", "用于治疗", "适用于"],
    "贮藏": ["储存", "保存", "存放", "贮存"],
    "禁忌": ["禁用", "禁止", "不宜", "避免使用"],
    "慎用": ["谨慎使用", "应慎用"],
    "孕妇及哺乳期妇女用药": ["孕妇", "哺乳期", "妊娠", "备孕", "怀孕"],
    "减少钠盐": ["低盐", "限盐", "少盐", "减盐", "减少食盐摄入"],
    "增加钾盐摄入": ["补钾", "增加钾摄入", "富钾", "钾盐"],
    "规律运动": ["适量运动", "有氧运动", "每周运动"],
    "减轻精神压力": ["减压", "缓解压力", "心理调节"],
    "合并症": ["并发症", "合并疾病", "合并基础疾病"],
    "感冒": ["上呼吸道感染", "普通感冒"],
    "监测": ["监控", "定期检查", "复查"],
    "控制体重": ["体重管理", "减重", "控制腰围", "监测体重", "体重及腰围", "超重", "肥胖"],
    "保持健康睡眠": ["规律作息", "睡眠管理", "保证睡眠", "作息"],
    "痤疮丙酸杆菌": ["微生物", "细菌", "菌群", "痤疮相关微生物"],
}


def load_test_questions(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def match_keywords(retrieved_text: str, expected_keywords):
    matched = []
    for keyword in expected_keywords:
        candidates = [keyword] + SYNONYMS.get(keyword, [])
        if any(candidate in retrieved_text for candidate in candidates):
            matched.append(keyword)
    return matched


def extract_gold_sources(item: dict[str, Any]) -> list[str]:
    sources = []
    for source_name in item.get("gold_sources", []):
        if source_name:
            sources.append(str(source_name).strip())
    for evidence in item.get("gold_evidence", []):
        if isinstance(evidence, dict):
            source_name = evidence.get("source_name")
            if source_name:
                sources.append(str(source_name).strip())
    deduped = []
    seen = set()
    for src in sources:
        key = src.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(src)
    return deduped


def evaluate(
    questions_path: Path = TEST_QUESTIONS_PATH,
    report_path: Path = REPORT_PATH,
    top_k: int = 7,
    keyword_pass_threshold: float = 0.6,
):
    if not Path(VECTOR_STORE_PATH).exists():
        raise FileNotFoundError(f"未找到向量库目录：{VECTOR_STORE_PATH}")

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = create_hybrid_retriever(vectorstore, vector_k=8, bm25_k=10, vector_weight=0.65, bm25_weight=0.35)
    test_questions = load_test_questions(questions_path)

    results = []
    correct_count = 0
    category_stats = {}
    doc_label_count = 0
    hit_count = 0
    reciprocal_rank_sum = 0.0
    category_doc_stats = {}

    for item in test_questions:
        question = item["question"]
        expected_keywords = item["expected_keywords"]
        category = item.get("category") or classify_query_type(question)
        docs = retrieve_evidence_docs(retriever, question, top_k=top_k)
        retrieved_text = "\n".join(doc.page_content for doc in docs)
        matched_keywords = match_keywords(retrieved_text, expected_keywords)
        match_rate = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0.0
        is_correct = match_rate >= keyword_pass_threshold
        if is_correct:
            correct_count += 1
        bucket = category_stats.setdefault(category, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if is_correct:
            bucket["correct"] += 1

        evidence = []
        for doc in docs[:3]:
            metadata = doc.metadata or {}
            evidence.append(
                {
                    "source_name": metadata.get("source_name"),
                    "page": metadata.get("page"),
                    "section_title": metadata.get("section_title"),
                    "document_type": metadata.get("document_type"),
                    "rerank_score": metadata.get("rerank_score"),
                }
            )

        result_row = {
            "问题": question,
            "分类": category,
            "预期关键词": expected_keywords,
            "匹配关键词": matched_keywords,
            "匹配率": f"{match_rate * 100:.1f}%",
            "是否准确": is_correct,
            "Top证据": evidence,
        }

        gold_sources = extract_gold_sources(item)
        if gold_sources:
            doc_label_count += 1
            source_rank = {}
            for rank, doc in enumerate(docs, 1):
                source_name = str((doc.metadata or {}).get("source_name", "")).strip().lower()
                if source_name and source_name not in source_rank:
                    source_rank[source_name] = rank

            first_hit_rank = None
            for source_name in gold_sources:
                rank = source_rank.get(source_name.lower())
                if rank is not None:
                    first_hit_rank = rank if first_hit_rank is None else min(first_hit_rank, rank)

            is_hit = first_hit_rank is not None and first_hit_rank <= top_k
            if is_hit:
                hit_count += 1
                reciprocal_rank_sum += 1.0 / first_hit_rank

            doc_bucket = category_doc_stats.setdefault(
                category, {"labeled": 0, "hit": 0, "rr_sum": 0.0}
            )
            doc_bucket["labeled"] += 1
            if is_hit:
                doc_bucket["hit"] += 1
                doc_bucket["rr_sum"] += 1.0 / first_hit_rank

            result_row["gold_sources"] = gold_sources
            result_row[f"Recall@{top_k}_命中"] = bool(is_hit)
            result_row[f"MRR@{top_k}_贡献"] = round((1.0 / first_hit_rank) if is_hit else 0.0, 4)

        results.append(result_row)

    overall_acc = correct_count / len(test_questions) if test_questions else 0.0
    category_report = {
        category: {
            "测试数": stat["total"],
            "准确数": stat["correct"],
            "准确率": f"{(stat['correct'] / stat['total'] * 100) if stat['total'] else 0.0:.1f}%"
        }
        for category, stat in category_stats.items()
    }
    report = {
        "测试总数": len(test_questions),
        "准确数": correct_count,
        "整体检索准确率": f"{overall_acc * 100:.1f}%",
        "分类统计": category_report,
        "详细结果": results,
        "评测说明": [
            "采用混合检索(BM25 + 向量)后再进行轻量重排。",
            f"按 {questions_path.name} 中的 expected_keywords 与同义词表进行命中评估。",
            "优先区分药品说明书题(drug_insert)与泛医学题(disease_general)进行分类统计。",
            f"匹配率达到 {keyword_pass_threshold * 100:.0f}% 及以上视为本题检索准确。",
        ],
    }

    if doc_label_count:
        doc_category_report = {}
        for category, stat in category_doc_stats.items():
            labeled = stat["labeled"]
            doc_category_report[category] = {
                "标注题数": labeled,
                f"Recall@{top_k}": f"{(stat['hit'] / labeled * 100) if labeled else 0.0:.1f}%",
                f"MRR@{top_k}": round((stat["rr_sum"] / labeled) if labeled else 0.0, 4),
            }
        report["文档级指标"] = {
            "标注题总数": doc_label_count,
            f"Recall@{top_k}": f"{(hit_count / doc_label_count * 100) if doc_label_count else 0.0:.1f}%",
            f"MRR@{top_k}": round((reciprocal_rank_sum / doc_label_count) if doc_label_count else 0.0, 4),
            "分类统计": doc_category_report,
            "说明": "仅对包含 gold_sources 或 gold_evidence 的样本统计。",
        }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n评测结果已保存到: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 RAG 检索效果（关键词 + 可选文档级指标）。")
    parser.add_argument("--questions", default=str(TEST_QUESTIONS_PATH), help="评测集 JSON 文件路径")
    parser.add_argument("--report", default=str(REPORT_PATH), help="评测报告输出路径")
    parser.add_argument("--top-k", type=int, default=7, help="召回并重排后用于评测的 top-k")
    parser.add_argument(
        "--keyword-pass-threshold",
        type=float,
        default=0.6,
        help="关键词匹配率阈值（0~1），达到后计为准确",
    )
    args = parser.parse_args()
    evaluate(
        questions_path=Path(args.questions),
        report_path=Path(args.report),
        top_k=args.top_k,
        keyword_pass_threshold=args.keyword_pass_threshold,
    )
