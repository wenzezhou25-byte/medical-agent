from __future__ import annotations

import os
import re
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

from retrieval_core import Chunk, JiebaBM25, split_text_recursive
from config import LOW_QUALITY_CROSS_THRESHOLD, LOW_QUALITY_RULE_THRESHOLD

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


def _strip_leading_noise(match: str, known_names=None) -> str:
    """剥离正则贪心吞入的前缀噪声（如「我之前吃」），返回真正的药名。

    优先用知识库已知标准药名做最长子串匹配。无已知药名时回退原始匹配（退化路径，
    此时正则可能误吞前缀，生产环境会在检索链路基于向量库注入 known_names）。
    """
    name = str(match).strip()
    if known_names:
        for known in sorted(known_names, key=len, reverse=True):
            if known in name:
                return known
    return name


def extract_drug_name_candidates(
    query: str, known_names=None
) -> List[str]:
    query = query or ""
    candidates: List[str] = []

    # 已知药名最长子串匹配（有知识库标准药名时最可靠）：
    # 能精准命中「布洛芬缓释胶囊」这类含剂型标准名，避免正则把「我之前吃」等
    # 动词/代词前缀一并吞入。名称越完整越优先。
    if known_names:
        for name in sorted(known_names, key=len, reverse=True):
            if name in query:
                candidates.append(name)

    # 正则兜底：匹配「中文 + 剂型后缀」，并借助已知药名剥离前缀噪声。
    for match in DRUG_NAME_PATTERN.findall(query):
        name = _strip_leading_noise(match, known_names)
        if name not in candidates and len(name) >= 3:
            candidates.append(name)

    # 常见口语化药名/商品名 → 库内标准药名（含剂型）。
    # 用户常省略剂型或用商品名提问，仅靠“感冒灵”这类子串会同时命中
    # 复方感冒灵/乐信感冒灵等不同药，导致目标药被同类挤掉。这里展开成
    # 库内真实存在的标准名，配合 rerank 的精确/子串加分能锁定目标药。
    normalized = query.replace("（", "(").replace("）", ")")
    if "999感冒灵" in normalized:
        candidates.extend(["999感冒灵", "感冒灵胶囊", "感冒灵颗粒", "感冒灵片"])

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

    # 泛名核心成分子串匹配：用户用裸泛名（无剂型）提问时也能被识别进候选。
    # 对知识库已注册标准名剥剂型/盐前缀得核心成分（「阿莫西林胶囊」→「阿莫西林」），
    # 若核心成分出现在 query 中，补入该核心名，使「阿莫西林」这类泛名可被缓存与检索命中。
    if known_names:
        _seen_cores = set()
        for name in sorted(known_names, key=len):
            core = _strip_dosage_and_salt(name)
            if len(core) < 2 or core in _seen_cores or core in candidates:
                continue
            _seen_cores.add(core)
            if core in query:
                candidates.append(core)

    # 泛名→库内单方标准名扩展：当 query 完全未带剂型（如“二甲双胍肾功能不全时能用吗？”）
    # 时，直接以泛名子串匹配会让复方制剂（二甲双胍格列本脲等）反超单方目标药。若已知
    # 向量库药名，把泛名展开成库内“单成分”标准名（核心成分==泛名，如“盐酸二甲双胍片”），
    # 使 rerank 能触发含剂型精确命中(typed_match)把单方药锁定在顶部。
    if known_names and not any(_name_has_dosage(n) for n in candidates):
        generic_stems = [n for n in candidates if not _name_has_dosage(n)]
        for stem in generic_stems:
            for name in sorted(known_names, key=len):
                if name in candidates:
                    continue
                if stem in name and _strip_dosage_and_salt(name) == stem:
                    candidates.append(name)

    return _unique_terms(candidates)


def strip_drug_core(name: str) -> str:
    """返回药物核心成分名（剥剂型+盐前缀），用于把同一药物的不同写法归并为一种。"""
    return _strip_dosage_and_salt(name)


def upsert_drug_cache(cache: List[str], name: str) -> None:
    """按「核心成分」去重的 MRU 插入：同一药物只保留一条，重复提及刷新到末尾。

    - 与缓存中核心成分相同的条目视为同一药物（如「布洛芬缓释胶囊」vs「布洛芬」），
      先移除旧条目，避免同一味药被当成“两个药”参与双药指代；
    - 合并时保留更具体的写法（含剂型优先，同具体度保留后提及的）；
    - 末尾 = 最近提及，供指代消解取「最近焦点」。
    """
    if not name:
        return
    core = _strip_dosage_and_salt(name)
    match_idx = None
    for i, existing in enumerate(cache):
        if existing and _strip_dosage_and_salt(existing) == core:
            match_idx = i
            break
    if match_idx is not None:
        existing = cache.pop(match_idx)
        if _name_has_dosage(existing) and not _name_has_dosage(name):
            name = existing
    cache.append(name)


DOSAGE_SUFFIXES = (
    "胶囊", "软胶囊", "片", "颗粒", "散", "口服液", "口服溶液", "口服混悬液",
    "滴丸", "滴剂", "滴眼液", "软膏", "乳膏", "膏", "栓", "丸", "糖浆",
    "冲剂", "咀嚼片", "泡腾片", "缓释片", "缓释胶囊", "肠溶片", "肠溶胶囊",
    "干混悬剂", "混悬液", "注射液", "注射用", "贴", "气雾剂", "喷雾剂",
    "凝胶", "酊剂", "片剂", "胶囊剂",
)


def _name_has_dosage(name: str) -> bool:
    """判断候选名是否含剂型后缀（用于与库内标准名精确匹配）。"""
    for suffix in DOSAGE_SUFFIXES:
        if name.endswith(suffix) or suffix in name:
            return True
    return False


# 常见成盐前缀（如“盐酸二甲双胍片”→“二甲双胍”），用于识别单成分药。
SALT_PREFIXES = (
    "氢溴酸", "盐酸", "硫酸", "磷酸", "硝酸", "苯磺酸", "甲磺酸", "马来酸",
    "枸橼酸", "柠檬酸", "富马酸", "琥珀酸", "醋酸", "乙酸", "乳酸", "碳酸",
    "鞣酸", "酒石酸", "水杨酸", "烟酸", "山梨酸",
)


def _strip_dosage_and_salt(name: str) -> str:
    """去掉剂型后缀与成盐前缀，返回药物核心成分名。

    用于泛名扩展时区分“单方药”与“复方药”：单方药核心成分==泛名，
    而复方药（如“二甲双胍格列本脲片”）核心成分是多个成分拼接，不等于泛名。
    """
    core = name.strip()
    for suffix in sorted(DOSAGE_SUFFIXES, key=len, reverse=True):
        if core.endswith(suffix):
            core = core[: -len(suffix)]
            break
    for prefix in SALT_PREFIXES:
        if core.startswith(prefix):
            core = core[len(prefix):]
            break
    return core


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
    documents: Sequence[Chunk],
    cleaner: Callable[[str], str],
) -> List[Chunk]:
    structured_docs: List[Chunk] = []
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
            chunk_texts = split_text_recursive(section_text)
            for chunk_index, raw_chunk_text in enumerate(chunk_texts):
                chunk_text = cleaner(raw_chunk_text or "")
                if len(chunk_text.strip()) <= 20:
                    continue

                metadata = dict(section_metadata)
                metadata["chunk_index"] = chunk_index
                metadata["chunk_id"] = f"{source_name}-{page or '0'}-{section_index}-{chunk_index}"
                structured_docs.append(Chunk(page_content=chunk_text, metadata=metadata))

    return structured_docs


def get_vectorstore_documents(vectorstore) -> List[Chunk]:
    docstore = getattr(vectorstore, "docstore", None)
    raw_docs = getattr(docstore, "_dict", {}) if docstore is not None else {}
    # P3 原生化之前 docstore 里可能是 langchain Document；这里不做类型过滤，
    # 只需对象带 page_content/metadata（Chunk 与 Document 均满足）。
    documents = [doc for doc in raw_docs.values() if hasattr(doc, "page_content")]

    deduped: List[Chunk] = []
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


def dedupe_documents(docs: Sequence[Chunk]) -> List[Chunk]:
    deduped: List[Chunk] = []
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


_reranker = None
_reranker_failed = False


def get_reranker():
    """懒加载 cross-encoder 重排模型；禁用或加载失败时返回 None 回退规则重排。"""
    global _reranker, _reranker_failed
    if _reranker_failed:
        return None
    if _reranker is not None:
        return _reranker
    try:
        from config import (
            HF_ENDPOINT,
            RERANK_CACHE_DIR,
            RERANK_ENABLED,
            RERANK_MAX_LENGTH,
            RERANK_MODEL,
        )
        if not RERANK_ENABLED:
            _reranker_failed = True
            return None
        if HF_ENDPOINT:
            os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
        # 若本地缓存快照已完整存在，则强制离线加载，避免联网 HEAD 检查(如
        # huggingface.co 不可达时长时间重试卡住首次回答)。
        local_dir = os.path.join(
            RERANK_CACHE_DIR, "models--" + RERANK_MODEL.replace("/", "--")
        )
        if os.path.isdir(local_dir):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(
            RERANK_MODEL,
            max_length=RERANK_MAX_LENGTH,
            cache_folder=RERANK_CACHE_DIR,
        )
        print(f"[rerank] 已加载 cross-encoder: {RERANK_MODEL}")
    except Exception as exc:
        _reranker_failed = True
        print(f"[rerank] cross-encoder 加载失败，回退规则重排: {exc}")
    return _reranker


def rerank_documents(
    query: str,
    docs: Sequence[Chunk],
    top_k: int = 5,
    known_names=None,
) -> List[Chunk]:
    query_lower = query.lower()
    query_terms = extract_query_terms(query)
    drug_name_candidates = extract_drug_name_candidates(query, known_names)
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
        has_typed_match = False
        for target_section, hints in SECTION_HINTS.items():
            if section_title == target_section and any(hint in query for hint in hints):
                section_boost += 2.5
        if source_name and source_name.lower() in query_lower:
            section_boost += 3.0
        if drug_name_candidates:
            # 含剂型的完整标准药名命中（source_name 相等或以此为前缀）优先于泛名子串命中，
            # 避免“感冒灵胶囊”被“复方感冒灵/乐信感冒灵”、“盐酸二甲双胍片”被复方制剂挤掉。
            typed = [n for n in drug_name_candidates if _name_has_dosage(n)]
            has_typed_match = any(source_name == n or source_name.startswith(n) for n in typed)
            if has_typed_match:
                section_boost += 6.0
                metadata["_typed_match"] = True
            elif any(n in source_name for n in drug_name_candidates):
                section_boost += 2.5
        # 复方/联合制剂降权：当 query 用泛名（如“二甲双胍”“头孢氨苄”）提问而未指定剂型时，
        # 目标通常是成分相对单一的单方药；复方制剂（格列/甲氧苄啶/罗格列酮/氯苯那敏等复配成分）
        # 在相似度上更接近完整药名，会反超单方，此处主动压低以避免同类药混淆。
        if not has_typed_match and (
            source_name.startswith("复方")
            or any(comp in source_name for comp in ["甲氧苄啶", "格列", "罗格列酮", "双氯", "马来酸"] if comp in source_name)
        ):
            section_penalty += 2.0
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
        metadata["rerank_source"] = "rule"  # 规则粗排分，判定低质量时按规则分数尺度选择阈值
        reranked.append(Chunk(page_content=doc.page_content, metadata=metadata))

    reranked.sort(
        key=lambda item: (
            item.metadata.get("rerank_score", 0.0),
            len(item.page_content or ""),
        ),
        reverse=True,
    )

    # cross-encoder 精排：先用规则粗排取候选，再用 bge-reranker 精排
    # 规则已精确定位到目标药（_typed_match）的文档保底置顶，模型只在其余候选中重排，
    # 避免 bge-reranker 把“复方/相似药”错误抬升到精确目标药之前。
    if len(reranked) > top_k:
        model = get_reranker()
        if model is not None:
            typed_kept = [d for d in reranked if d.metadata.get("_typed_match")]
            rest = [d for d in reranked if not d.metadata.get("_typed_match")]
            candidate_k = min(len(rest), max(top_k * 2, 8))
            candidates = rest[:candidate_k]
            try:
                if candidates:
                    pairs = [(query, doc.page_content or "") for doc in candidates]
                    scores = model.predict(pairs)
                    for doc, score in zip(candidates, scores):
                        doc.metadata["rerank_source"] = "cross"  # cross-encoder 分，判定低质量时按 [0,1] 尺度设阈值
                        doc.metadata["rerank_score"] = round(float(score), 4)
                    candidates.sort(
                        key=lambda item: item.metadata.get("rerank_score", 0.0),
                        reverse=True,
                    )
                merged = typed_kept + candidates
                # P2-11：merged 或补位用的 reranked 可能含同一文档（reranked 已含
                # typed_kept），按 chunk_id 去重补齐，避免重复返回同一证据。
                seen: set = set()
                result: List[Chunk] = []
                for doc in merged + reranked:
                    key = doc.metadata.get("chunk_id") or (
                        doc.metadata.get("source_name")
                        + "|" + str(doc.metadata.get("section_title") or "")
                        + "|" + (doc.page_content or "")
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append(doc)
                    if len(result) >= top_k:
                        break
                return result
            except Exception as exc:
                print(f"[rerank] cross-encoder 精排失败，回退规则分: {exc}")

    return reranked[:top_k]


def _safe_page_number(value) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 10**9


def _is_low_quality_docs(docs) -> bool:
    """判断 RAG 检索结果是否为空或低质量（用于触发自动联网兜底 / 证据不足提示）。

    通过 rerank_documents 写入的 rerank_source 区分分数尺度选取阈值：
      - "cross"：cross-encoder 分，低于 config.LOW_QUALITY_CROSS_THRESHOLD 视为弱命中；
      - 其他（"rule" 或缺失）：规则粗排分，低于 config.LOW_QUALITY_RULE_THRESHOLD 视为弱命中。
    空结果、或所有文档都无有效 rerank_score 时一律视为低质量。
    阈值集中在 config.py，便于调参和单测。
    """
    if not docs:
        return True
    scored = [d for d in docs if isinstance(d.metadata.get("rerank_score"), (int, float))]
    if not scored:
        return True
    best_doc = max(scored, key=lambda d: d.metadata["rerank_score"])
    best = best_doc.metadata["rerank_score"]
    if best_doc.metadata.get("rerank_source") == "cross":
        return best < LOW_QUALITY_CROSS_THRESHOLD
    return best < LOW_QUALITY_RULE_THRESHOLD


class _Bm25Retriever:
    """包装 JiebaBM25，提供与 langchain BM25Retriever 一致的 invoke(query)/k 接口。"""

    def __init__(self, bm25: JiebaBM25, k: int = 10) -> None:
        self.bm25 = bm25
        self.k = k

    def invoke(self, query: str) -> List[Chunk]:
        return self.bm25.invoke(query, k=self.k)


class RetrieverBundle(NamedTuple):
    primary: object
    fallback: object | None = None
    all_documents: tuple[Chunk, ...] = ()
    # 知识库已知药名（去重源文件名）。随 bundle 一起构建/重建，始终反映当前向量库，
    # 空库时为空元组；由检索链路注入 extract_drug_name_candidates / rerank_documents。
    known_names: tuple[str, ...] = ()


class WeightedHybridRetriever:
    """Simple weighted rank fusion without langchain EnsembleRetriever."""

    def __init__(
        self,
        bm25_retriever,
        vector_retriever,
        *,
        bm25_weight: float,
        vector_weight: float,
    ):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def invoke(self, query: str) -> List[Chunk]:
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)

        merged: dict[tuple, tuple[float, Chunk]] = {}

        def _ingest(docs: Sequence[Chunk], weight: float):
            for rank, doc in enumerate(docs, 1):
                metadata = doc.metadata or {}
                key = (
                    metadata.get("chunk_id"),
                    metadata.get("source_name"),
                    metadata.get("page"),
                    metadata.get("section_title"),
                    doc.page_content,
                )
                score = weight / rank
                current = merged.get(key)
                if current is None or score > current[0]:
                    merged[key] = (score, doc)

        _ingest(bm25_docs, self.bm25_weight)
        _ingest(vector_docs, self.vector_weight)

        ordered = sorted(merged.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ordered]


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
        return RetrieverBundle(
            primary=vector_retriever, fallback=None, all_documents=(), known_names=()
        )
    known_names = tuple(
        sorted(
            {
                str((doc.metadata or {}).get("source_name", "")).strip()
                for doc in documents
                if (doc.metadata or {}).get("source_name")
            }
        )
    )

    bm25_retriever = _Bm25Retriever(JiebaBM25.from_chunks(documents), k=bm25_k)

    retriever = WeightedHybridRetriever(
        bm25_retriever,
        vector_retriever,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
    )
    return RetrieverBundle(
        primary=retriever,
        fallback=bm25_retriever,
        all_documents=tuple(documents),
        known_names=known_names,
    )


def _retrieve_evidence_docs_with_breakdown(retriever_bundle, query: str, top_k: int = 5):
    """检索证据文档，返回 (docs, breakdown_ms)。

    breakdown 包含各阶段耗时，用于定位检索链路瓶颈。
    """
    import time

    breakdown = {}
    primary = getattr(retriever_bundle, "primary", retriever_bundle)
    fallback = getattr(retriever_bundle, "fallback", None)
    all_documents = getattr(retriever_bundle, "all_documents", ())
    # 知识库已知药名直接取自 bundle（构建时推导，始终反映当前向量库）；
    # 兼容旧 bundle 未知该字段时回退到现场推导。
    known_names = getattr(retriever_bundle, "known_names", ())
    if not known_names and all_documents:
        known_names = tuple(
            sorted(
                {
                    str((doc.metadata or {}).get("source_name", "")).strip()
                    for doc in all_documents
                    if (doc.metadata or {}).get("source_name")
                }
            )
        )

    t0 = time.perf_counter()
    try:
        docs = primary.invoke(query)
    except Exception:
        if fallback is None:
            raise
        docs = fallback.invoke(query)
    breakdown["hybrid_search_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    query_type = classify_query_type(query)
    if query_type != "general":
        typed_docs = [doc for doc in docs if (doc.metadata or {}).get("document_type") == query_type]
        if typed_docs:
            docs = typed_docs
    drug_name_candidates = extract_drug_name_candidates(query, known_names)
    # 兜底：用向量库中已有的真实药名对命中做最长匹配识别。
    # 覆盖正则白名单覆盖不到的名称（药膏/药酒/疫苗/含片/贴等后缀，字母开头的 B族/DHA-EPA/L-门冬…，
    # 引号包裹，以及无后缀药名），避免「相似药混淆」。名称越完整越优先。
    if known_names:
        for name in sorted(known_names, key=len, reverse=True):
            if not name:
                continue
            _name, _query = name, query
            for ch in "“”‘’「」『』":
                _name = _name.replace(ch, "")
                _query = _query.replace(ch, "")
            if (_name in query) or (_name in _query):
                if name not in drug_name_candidates:
                    drug_name_candidates.append(name)
    if drug_name_candidates:
        name_filtered_docs = []
        seen_keys = set()
        # 合并首轮召回与全量文档命中药名的片段，确保拿到该药全部章节供后续章节筛选使用；
        # 避免「首轮已命中该药某一章节」时跳过补召回，导致候选只剩错误章节。
        pool = list(docs)
        if all_documents:
            pool.extend(all_documents)
        for doc in pool:
            metadata = doc.metadata or {}
            source_name = str(metadata.get("source_name", ""))
            if not any(name in source_name for name in drug_name_candidates):
                continue
            key = (source_name, metadata.get("section_title"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            name_filtered_docs.append(doc)
        if name_filtered_docs:
            docs = name_filtered_docs
    # 显式章节筛选：问题里点名了正式章节时（如「X的用法用量/注意事项/不良反应/禁忌/适应症…」），
    # 优先只保留该章节的候选，避免重排把内容稠密的「成分」等错误章节排在前面；没有则回退全量。
    explicit_section_docs = select_explicit_section_docs(query, docs)
    if explicit_section_docs:
        docs = explicit_section_docs
    breakdown["filter_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    docs = rerank_documents(query, docs, top_k=top_k, known_names=known_names)
    breakdown["rerank_ms"] = (time.perf_counter() - t0) * 1000

    return docs, breakdown


def retrieve_evidence_docs(retriever_bundle, query: str, top_k: int = 5) -> List[Chunk]:
    """检索证据文档（兼容旧接口，不返回耗时分解）。"""
    docs, _ = _retrieve_evidence_docs_with_breakdown(retriever_bundle, query, top_k=top_k)
    return docs


SECTION_QUERY_TOKENS = {
    "副作用": "不良反应",
    "不良反应": "不良反应",
    "反应用": "不良反应",
    "用法用量": "用法用量",
    "用药方法": "用法用量",
    "适应症": "适应症",
    "适应症状": "适应症",
    "主治": "适应症",
    "禁忌": "禁忌",
    "注意事项": "注意事项",
    "注意": "注意事项",
    "主要成分": "成分",
    "成份": "成分",
    "成分": "成分",
    "贮藏": "贮藏",
    "贮存": "贮藏",
    "储存": "贮藏",
    "药理作用": "药理作用",
    "药物相互作用": "药物相互作用",
}


def select_explicit_section_docs(query: str, docs: Sequence[Chunk]):
    """若 query 显式点名某个正式章节，返回该章节的候选文档；否则返回 None。"""
    if not query or not docs:
        return None
    for token in sorted(SECTION_QUERY_TOKENS.keys(), key=len, reverse=True):
        if token not in query:
            continue
        wanted = SECTION_QUERY_TOKENS[token]
        matched = [
            doc
            for doc in docs
            if wanted in str((doc.metadata or {}).get("section_title", ""))
        ]
        if matched:
            return matched
    return None


# ===================== 内容净化（P0-14 / P1-10） =====================
# 统一防线：入库前（P1-10）与工具输出回灌前（P0-14 第三层）都调用。
# 注意：这是第一道过滤，不是 100% 拦截；真正的兜底是「工具输出不可信」的
# system prompt 约束 + 写工具隔离（save_user_medical_record 只收用户直接陈述）。
_INJECTION_PATTERNS = [
    r"忽略.{0,6}(之前|以上|上述|所有).{0,10}(指令|提示|规则|要求)",
    r"(you are now|system prompt|ignore (all|previous) instructions)",
    r"(请|请务必|你现在|立即)?调用\s*(save_user_medical_record|rag_search|web_search|conflict_checker|search_nearby_hospitals)",
    r"(现在|从此|从现在)开始.{0,6}(扮演|你是|你就是)",
    r"(忘记|无视).{0,4}(之前|所有)?.{0,6}(指令|对话|提示)",
    r"请输出\s*(JSON|json|json格式)",
]


def sanitize_untrusted_text(text: Optional[str]) -> str:
    """净化不可信文本：删除控制/隐形/双向翻转字符；命中注入模式时只剥离指令子串、
    保留其余正文（避免整段误伤）；若剥离后已无有效内容则返回空串，由调用方整段丢弃。

    注意：这是第一道软过滤，无法 100% 拦截；真正的兜底是「工具输出不可信」的
    system prompt 约束 + 写工具隔离（save_user_medical_record 只收用户直接陈述）。
    """
    if not text:
        return ""
    # 控制字符 + 隐形/双向控制符。\u200b-\u200f 零宽、\ufeff BOM、\u2028\u2029 行/段分隔、
    # \u202a-\u202e（LRE/RLE/PDF/LRO/RLO）与 \u2060-\u206f 为 PDF 隐藏反转注入的经典载体。
    text = re.sub(
        r"[\x00-\x1f\x7f\u200b-\u200f\ufeff\u2028\u2029\u202a-\u202e\u2060-\u206f]",
        "",
        text,
    )
    for pat in _INJECTION_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()


def format_docs_for_prompt(docs: Sequence[Chunk]) -> str:
    if not docs:
        return "无相关本地文档信息。"

    blocks = []
    for idx, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        source_name = metadata.get("source_name") or normalize_source_name(metadata.get("source", ""))
        page = metadata.get("page") or "?"
        section_title = metadata.get("section_title") or "未分类"
        document_type = metadata.get("document_type") or "未标注"
        rerank_score = metadata.get("rerank_score")
        score_text = f" | 相关度={rerank_score}" if rerank_score is not None else ""
        # header 中显式列出 document_type，便于模型在四分类判定时区分
        # 药品说明书(drug_insert)与泛医学指南(disease_general)
        header = (
            f"[证据 {idx}] 文件={source_name} | 页码={page} | "
            f"章节={section_title} | 文档类型={document_type}{score_text}"
        )
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n--- [文档片段] ---\n\n".join(blocks)
