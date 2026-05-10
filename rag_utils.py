import os
import re
from typing import Callable, List, NamedTuple, Sequence, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers.ensemble import EnsembleRetriever


SECTION_TITLES = [
    "药品名称",
    "成份",
    "性状",
    "适应症",
    "功能主治",
    "规格",
    "用法用量",
    "不良反应",
    "禁忌",
    "注意事项",
    "药物相互作用",
    "药理作用",
    "贮藏",
    "包装",
    "有效期",
    "执行标准",
    "批准文号",
    "生产企业",
    "儿童用药",
    "老年用药",
    "孕妇及哺乳期妇女用药",
    "临床表现",
    "症状",
    "诊断",
    "治疗",
    "饮食",
    "生活方式",
    "预防",
    "就医指征",
    "慎用",
    "警告",
]

SECTION_ALIASES = {
    "功能主治": "适应症",
    "主治功能": "适应症",
    "适用症": "适应症",
    "用量用法": "用法用量",
    "使用方法": "用法用量",
    "副作用": "不良反应",
    "副反应": "不良反应",
    "相互作用": "药物相互作用",
    "药物相互作用及配伍禁忌": "药物相互作用",
    "孕妇及哺乳期用药": "孕妇及哺乳期妇女用药",
    "孕妇、哺乳期妇女用药": "孕妇及哺乳期妇女用药",
    "儿童使用": "儿童用药",
    "老人用药": "老年用药",
    "临床症状": "临床表现",
    "症状表现": "临床表现",
    "临床表现和症状": "临床表现",
    "生活饮食": "饮食",
    "饮食注意": "饮食",
    "生活调理": "生活方式",
    "何时就医": "就医指征",
    "就诊指征": "就医指征",
    "贮存": "贮藏",
    "储存": "贮藏",
    "储藏": "贮藏",
}

SECTION_LABELS = SECTION_TITLES + list(SECTION_ALIASES.keys())

SECTION_PATTERN = re.compile(
    r"^\s*(?:第[一二三四五六七八九十0-9]+[章节]\s*)?"
    r"(?:[一二三四五六七八九十0-9]+[、.)）]\s*)?"
    r"(?:【)?(?P<title>" + "|".join(re.escape(title) for title in SECTION_LABELS) + r")(?:】)?"
    r"(?:[:：]\s*)?(?P<rest>.*)$"
)

INLINE_SECTION_PATTERN = re.compile(
    r"(?:【(?P<bracket_title>"
    + "|".join(re.escape(title) for title in SECTION_LABELS)
    + r")】|(?P<plain_title>"
    + "|".join(re.escape(title) for title in SECTION_LABELS)
    + r")\s*[:：])"
)

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=180,
    separators=["\n\n", "\n", "。", "；", "！", "？", "：", "，", " "],
    length_function=len,
)

STOPWORDS = {
    "什么", "哪些", "怎么", "如何", "请问", "可以", "是否", "一起", "需要", "日常", "患者", "问题", "一下",
    "这个", "那个", "我们", "你们", "他们", "她们", "自己", "目前", "还有", "已经", "今天", "现在",
}

SECTION_HINTS = {
    "适应症": ["作用", "适应症", "治疗", "主治", "用于"],
    "用法用量": ["用法", "用量", "剂量", "频次", "一天", "怎么吃", "多久", "服用"],
    "不良反应": ["副作用", "不良反应", "副反应", "不舒服", "不适"],
    "禁忌": ["禁忌", "不能", "禁用", "禁止", "慎用"],
    "注意事项": ["注意事项", "注意", "提醒", "警示"],
    "药物相互作用": ["一起吃", "联用", "相互作用", "冲突", "同服"],
    "儿童用药": ["儿童", "小孩", "小儿"],
    "老年用药": ["老人", "老年", "高龄"],
    "孕妇及哺乳期妇女用药": ["孕妇", "怀孕", "哺乳", "备孕"],
    "贮藏": ["保存", "贮藏", "冷藏", "存放"],
    "临床表现": ["症状", "表现", "临床表现", "体征"],
    "症状": ["症状", "表现", "体征"],
    "治疗": ["治疗", "疗法", "方案", "诊疗"],
    "饮食": ["饮食", "吃什么", "忌口", "营养"],
    "生活方式": ["生活方式", "作息", "锻炼", "运动"],
    "就医指征": ["何时就医", "就医", "就诊", "严重", "尽快就医"],
}

LOW_VALUE_SECTIONS = {
    "药品名称",
    "规格",
    "包装",
    "批准文号",
    "执行标准",
    "生产企业",
}

SECTION_INFERENCE_RULES = {
    "药物相互作用": ["相互作用", "联合用药", "联用", "合用", "同时服用"],
    "不良反应": ["不良反应", "副作用", "副反应"],
    "禁忌": ["禁忌", "禁用", "禁止使用"],
    "注意事项": ["注意事项", "特别注意", "警示语"],
    "适应症": ["适应症", "功能主治", "用于治疗"],
    "用法用量": ["用法用量", "一次", "一日", "口服"],
    "临床表现": ["临床表现", "常见症状", "症状表现", "表现为"],
    "症状": ["症状", "体征", "伴有"],
    "治疗": ["治疗", "诊疗", "治疗原则", "治疗方案"],
    "饮食": ["饮食", "忌口", "宜吃", "少吃", "多吃"],
    "生活方式": ["生活方式", "控制体重", "规律运动", "减轻精神压力", "健康睡眠", "戒烟", "戒酒"],
    "贮藏": ["贮藏", "贮存", "储存", "遮光", "密封"],
    "就医指征": ["及时就医", "尽快就医", "应立即就诊", "必要时就医"],
}

QUESTION_TYPE_HINTS = {
    "drug_insert": ["作用", "用法", "用量", "禁忌", "副作用", "说明书", "药品", "胶囊", "片", "颗粒", "散"],
    "disease_general": ["症状", "表现", "诊断", "高血压", "发热", "痤疮", "饮食", "治疗", "指南"],
}

DRUG_NAME_PATTERN = re.compile(
    r"([\u4e00-\u9fff]{2,20}(?:胶囊|片|颗粒|散|口服液|注射液|滴丸|滴剂|软膏|乳膏|栓|丸|糖浆))"
)


def infer_document_type(source_name: str, text: str) -> str:
    source = (source_name or "").lower()
    content_preview = (text or "")[:300]
    drug_markers = ["国药准字", "说明书", "胶囊", "片", "颗粒", "散", "口服液", "注射液"]
    disease_markers = ["指南", "诊疗", "共识", "临床表现", "症状", "治疗原则", "饮食"]

    if any(marker.lower() in source for marker in drug_markers):
        return "drug_insert"
    if any(marker in content_preview for marker in ["批准文号", "规格", "不良反应", "禁忌", "用法用量"]):
        return "drug_insert"
    if any(marker.lower() in source for marker in disease_markers):
        return "disease_general"
    if any(marker in content_preview for marker in ["临床表现", "诊断", "治疗", "饮食", "预防"]):
        return "disease_general"
    return "general"


def classify_query_type(query: str) -> str:
    query = query or ""
    if any(token in query for token in QUESTION_TYPE_HINTS["drug_insert"]):
        if any(token in query for token in ["症状", "表现", "诊断", "指南"]) and not any(
            token in query for token in ["说明书", "药品", "胶囊", "片", "颗粒", "散"]
        ):
            return "disease_general"
        return "drug_insert"
    if any(token in query for token in QUESTION_TYPE_HINTS["disease_general"]):
        return "disease_general"
    return "general"


def extract_drug_name_candidates(query: str) -> List[str]:
    query = query or ""
    candidates: List[str] = []

    for match in DRUG_NAME_PATTERN.findall(query):
        name = match.strip()
        if len(name) >= 3:
            candidates.append(name)

    # 常见口语化药名，例如“999感冒灵”
    normalized = query.replace("（", "(").replace("）", ")")
    if "999感冒灵" in normalized:
        candidates.extend(["999感冒灵", "感冒灵"])

    if "布洛芬" in normalized:
        candidates.append("布洛芬")
    if "头孢" in normalized:
        candidates.append("头孢")
    if "二甲双胍" in normalized:
        candidates.append("二甲双胍")
    if "蒙脱石" in normalized:
        candidates.append("蒙脱石")
    if "来那度胺" in normalized:
        candidates.append("来那度胺")

    return _unique_terms(candidates)


def normalize_source_name(source_name: str) -> str:
    if not source_name:
        return "unknown"
    return os.path.splitext(os.path.basename(str(source_name)))[0]


def split_medical_sections(text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    current_title = "未分类"
    buffer: List[str] = []

    for raw_line in text.splitlines():
        split_lines = _split_inline_section_lines(raw_line)
        for line in split_lines:
            line = line.strip()
            if not line:
                if buffer and buffer[-1] != "":
                    buffer.append("")
                continue

            matched = SECTION_PATTERN.match(line)
            if matched:
                raw_title = matched.group("title").strip()
                title = SECTION_ALIASES.get(raw_title, raw_title)
                remainder = matched.group("rest").strip()
                if buffer:
                    content = "\n".join(buffer).strip()
                    if content:
                        sections.append((current_title, content))
                current_title = title
                buffer = [remainder] if remainder else []
            else:
                buffer.append(line)

    if buffer:
        content = "\n".join(buffer).strip()
        if content:
            sections.append((current_title, content))

    return sections or [("未分类", text.strip())]


def infer_section_title(section_title: str, text: str) -> str:
    if section_title != "未分类":
        return section_title

    preview = (text or "")[:200]
    for inferred_title, hints in SECTION_INFERENCE_RULES.items():
        if any(hint in preview for hint in hints):
            return inferred_title
    return section_title


def _split_inline_section_lines(raw_line: str) -> List[str]:
    line = (raw_line or "").strip()
    if not line:
        return [""]

    matches = list(INLINE_SECTION_PATTERN.finditer(line))
    if not matches:
        return [line]

    pieces: List[str] = []
    cursor = 0
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
        if start > cursor:
            prefix = line[cursor:start].strip()
            if prefix:
                pieces.append(prefix)
        segment = line[start:end].strip()
        if segment:
            pieces.append(segment)
        cursor = end

    if cursor < len(line):
        suffix = line[cursor:].strip()
        if suffix:
            pieces.append(suffix)
    return pieces or [line]


def build_structured_documents(
    documents: Sequence[Document],
    cleaner: Callable[[str], str],
) -> List[Document]:
    structured_docs: List[Document] = []
    source_sorted_docs = sorted(
        documents,
        key=lambda doc: (
            str((doc.metadata or {}).get("source_name") or normalize_source_name((doc.metadata or {}).get("source", ""))),
            _safe_page_number((doc.metadata or {}).get("page")),
        ),
    )

    for source_doc in source_sorted_docs:
        cleaned_text = cleaner(source_doc.page_content or "")
        if len(cleaned_text.strip()) <= 20:
            continue

        base_metadata = dict(source_doc.metadata or {})
        source_name = base_metadata.get("source_name") or normalize_source_name(base_metadata.get("source", ""))
        page = str(base_metadata.get("page", ""))
        document_type = infer_document_type(source_name, cleaned_text)
        section_docs = split_medical_sections(cleaned_text)

        for section_index, (section_title, section_text) in enumerate(section_docs):
            if len(section_text.strip()) <= 20:
                continue
            section_title = infer_section_title(section_title, section_text)

            section_metadata = {
                **base_metadata,
                "source_name": source_name,
                "page": page,
                "section_title": section_title,
                "section_index": section_index,
                "document_type": document_type,
            }
            chunk_docs = TEXT_SPLITTER.split_documents([Document(page_content=section_text, metadata=section_metadata)])
            for chunk_index, chunk_doc in enumerate(chunk_docs):
                chunk_text = cleaner(chunk_doc.page_content or "")
                if len(chunk_text.strip()) <= 20:
                    continue

                metadata = dict(chunk_doc.metadata or {})
                metadata["chunk_index"] = chunk_index
                metadata["chunk_id"] = f"{source_name}-{page or '0'}-{section_index}-{chunk_index}"
                structured_docs.append(Document(page_content=chunk_text, metadata=metadata))

    return structured_docs


def get_vectorstore_documents(vectorstore) -> List[Document]:
    docstore = getattr(vectorstore, "docstore", None)
    raw_docs = getattr(docstore, "_dict", {}) if docstore is not None else {}
    documents = [doc for doc in raw_docs.values() if isinstance(doc, Document)]

    deduped: List[Document] = []
    seen = set()
    for doc in documents:
        key = (
            doc.page_content,
            doc.metadata.get("source_name"),
            doc.metadata.get("page"),
            doc.metadata.get("section_title"),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    return deduped


def _unique_terms(terms: Sequence[str]) -> List[str]:
    seen = set()
    results = []
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            results.append(term)
    return results


def extract_query_terms(query: str) -> List[str]:
    terms: List[str] = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9.+-]*", query):
        item = part.strip().lower()
        if not item or item in STOPWORDS:
            continue
        terms.append(item)
        if re.fullmatch(r"[\u4e00-\u9fff]+", item):
            if 2 <= len(item) <= 4:
                terms.append(item)
            elif len(item) > 4:
                for n in (2, 3, 4):
                    for i in range(0, len(item) - n + 1):
                        gram = item[i:i + n]
                        if gram not in STOPWORDS:
                            terms.append(gram)
    return _unique_terms(terms)


def dedupe_documents(docs: Sequence[Document]) -> List[Document]:
    deduped: List[Document] = []
    seen = set()
    for doc in docs:
        metadata = doc.metadata or {}
        key = (
            metadata.get("chunk_id"),
            metadata.get("source_name"),
            metadata.get("page"),
            metadata.get("section_title"),
            doc.page_content,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def rerank_documents(query: str, docs: Sequence[Document], top_k: int = 5) -> List[Document]:
    query_lower = query.lower()
    query_terms = extract_query_terms(query)
    drug_name_candidates = extract_drug_name_candidates(query)
    query_type = classify_query_type(query)
    query_intents = {
        "drug_info": any(token in query for token in ["作用", "适应症", "主治", "用于"]),
        "dosage": any(token in query for token in ["怎么吃", "用法", "用量", "口服", "服用", "一日", "一次", "冲服"]),
        "storage": any(token in query for token in ["贮藏", "贮存", "储存", "保存", "存放", "遮光", "密封"]),
        "symptom": any(token in query for token in ["症状", "表现", "体征"]),
        "diet": any(token in query for token in ["饮食", "忌口", "吃什么", "营养"]),
        "lifestyle_bp": ("高血压" in query) and any(token in query for token in ["饮食", "生活方式", "调整", "日常"]),
        "etiology": any(token in query for token in ["病因", "发病机制", "原因", "机制"]),
        "interaction": any(token in query for token in ["一起吃", "联用", "相互作用", "冲突", "同服"]),
    }
    reranked = []

    for doc in dedupe_documents(docs):
        metadata = dict(doc.metadata or {})
        section_title = str(metadata.get("section_title", ""))
        source_name = str(metadata.get("source_name", ""))
        document_type = str(metadata.get("document_type", "general"))
        text = doc.page_content or ""
        text_lower = text.lower()
        combined = f"{section_title} {source_name} {text_lower}"

        overlap_score = sum(1.2 for term in query_terms if term in combined)
        exact_query_bonus = 4.0 if query_lower and query_lower in combined else 0.0
        section_boost = 0.0
        section_penalty = 0.0
        for target_section, hints in SECTION_HINTS.items():
            if section_title == target_section and any(hint in query for hint in hints):
                section_boost += 2.5
        if source_name and source_name.lower() in query_lower:
            section_boost += 3.0
        if drug_name_candidates and any(name in source_name for name in drug_name_candidates):
            section_boost += 4.0
        if section_title and section_title in query:
            section_boost += 2.0
        if query_intents["drug_info"] and section_title in {"适应症", "功能主治", "药理作用"}:
            section_boost += 2.5
        if query_intents["dosage"] and section_title in {"用法用量", "注意事项", "儿童用药", "老年用药"}:
            section_boost += 4.0
        if query_intents["dosage"] and section_title in {"药理作用", "成份", "性状"}:
            section_penalty += 1.8
        if query_intents["storage"] and section_title in {"贮藏", "注意事项"}:
            section_boost += 4.0
        if query_intents["storage"] and any(token in combined for token in ["贮藏", "贮存", "储存", "密封", "遮光"]):
            section_boost += 3.2
        if query_intents["lifestyle_bp"]:
            lifestyle_terms = ["减少钠盐", "增加钾盐", "控制体重", "规律运动", "精神压力", "健康睡眠", "戒烟", "戒酒"]
            lifestyle_hits = sum(1 for token in lifestyle_terms if token in combined)
            if lifestyle_hits:
                section_boost += 2.0 + lifestyle_hits * 1.2
        if query_intents["etiology"]:
            etiology_terms = ["病因", "发病机制", "雄激素", "皮脂", "痤疮丙酸杆菌", "微生物增殖"]
            etiology_hits = sum(1 for token in etiology_terms if token in combined)
            if etiology_hits:
                section_boost += 2.0 + etiology_hits * 1.3
        if query_intents["symptom"] and section_title in {"症状", "临床表现"}:
            section_boost += 3.0
        if query_intents["diet"] and section_title in {"饮食", "注意事项", "生活方式"}:
            section_boost += 3.0
        if query_intents["interaction"] and section_title in {"药物相互作用", "禁忌", "注意事项"}:
            section_boost += 3.0
        if section_title in LOW_VALUE_SECTIONS and not any(key in query for key in [section_title, "名称", "规格", "厂家"]):
            section_penalty += 1.5
        if section_title == "未分类":
            section_penalty += 0.6
        if query_type == "drug_insert" and document_type == "drug_insert":
            section_boost += 2.0
        elif query_type == "disease_general" and document_type == "disease_general":
            section_boost += 2.0
        elif query_type != "general":
            section_penalty += 1.0

        score = overlap_score + exact_query_bonus + section_boost - section_penalty
        metadata["rerank_score"] = round(score, 3)
        reranked.append(Document(page_content=doc.page_content, metadata=metadata))

    reranked.sort(
        key=lambda item: (
            item.metadata.get("rerank_score", 0.0),
            len(item.page_content or ""),
        ),
        reverse=True,
    )
    return reranked[:top_k]


def _safe_page_number(value) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 10**9


class RetrieverBundle(NamedTuple):
    primary: object
    fallback: object | None = None
    all_documents: tuple[Document, ...] = ()


def create_hybrid_retriever(
    vectorstore,
    *,
    vector_k: int = 8,
    bm25_k: int = 10,
    vector_weight: float = 0.65,
    bm25_weight: float = 0.35,
):
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": vector_k})
    documents = get_vectorstore_documents(vectorstore)
    if not documents:
        return RetrieverBundle(primary=vector_retriever, fallback=None, all_documents=())

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = bm25_k

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )
    return RetrieverBundle(primary=retriever, fallback=bm25_retriever, all_documents=tuple(documents))


def retrieve_evidence_docs(retriever_bundle, query: str, top_k: int = 5) -> List[Document]:
    primary = getattr(retriever_bundle, "primary", retriever_bundle)
    fallback = getattr(retriever_bundle, "fallback", None)
    all_documents = getattr(retriever_bundle, "all_documents", ())
    try:
        docs = primary.invoke(query)
    except Exception:
        if fallback is None:
            raise
        docs = fallback.invoke(query)
    query_type = classify_query_type(query)
    if query_type != "general":
        typed_docs = [doc for doc in docs if (doc.metadata or {}).get("document_type") == query_type]
        if typed_docs:
            docs = typed_docs
    drug_name_candidates = extract_drug_name_candidates(query)
    if drug_name_candidates:
        name_filtered_docs = []
        for doc in docs:
            metadata = doc.metadata or {}
            source_name = str(metadata.get("source_name", ""))
            if any(name in source_name for name in drug_name_candidates):
                name_filtered_docs.append(doc)
        # 若首轮召回里没有命中药名文件，则从全量文档按 source_name 硬匹配补召回
        if not name_filtered_docs and all_documents:
            for doc in all_documents:
                metadata = doc.metadata or {}
                source_name = str(metadata.get("source_name", ""))
                if any(name in source_name for name in drug_name_candidates):
                    name_filtered_docs.append(doc)
        if name_filtered_docs:
            docs = name_filtered_docs
    return rerank_documents(query, docs, top_k=top_k)


def format_docs_for_prompt(docs: Sequence[Document]) -> str:
    if not docs:
        return "无相关本地文档信息。"

    blocks = []
    for idx, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        source_name = metadata.get("source_name") or normalize_source_name(metadata.get("source", ""))
        page = metadata.get("page") or "?"
        section_title = metadata.get("section_title") or "未分类"
        rerank_score = metadata.get("rerank_score")
        score_text = f" | 相关度={rerank_score}" if rerank_score is not None else ""
        header = f"[证据 {idx}] 文件={source_name} | 页码={page} | 章节={section_title}{score_text}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n--- [文档片段] ---\n\n".join(blocks)
