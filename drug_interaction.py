# -*- coding: utf-8 -*-
"""药物相互作用检测。

从 app.py 拆出，并在原「检索 + 风险关键词」启发式基础上增强为：
  1. 知识库混合检索召回候选证据（保留原召回层）
  2. 用 LLM 对「是否构成临床意义的相互作用」做二分类
  3. 输出风险等级（轻微/中度/严重）与用药建议

关键词启发式仅作为 LLM 不可用时的回退（fallback），不再作为主判定。
不依赖 Streamlit。
"""
import json
import re

import dashscope
from dashscope import Generation

from config import LLM_MODEL, get_required_env
from rag_utils import (
    format_docs_for_prompt,
    retrieve_evidence_docs,
)

# 旧关键词启发式（仅在 LLM 分类失败时兜底使用）
RISK_KEYWORDS = ["禁忌", "禁止", "不宜", "避免", "冲突", "严重", "出血", "中毒", "不良反应", "拮抗", "禁用"]


def _build_interaction_prompt(drug_a, drug_b, evidence_text):
    """构造 LLM 二分类 prompt，要求严格输出 JSON。"""
    evidence = (evidence_text or "").strip() or "（知识库未检索到相关说明）"
    return f"""你是一名具备20年经验的临床药师，请判断【{drug_a}】和【{drug_b}】联合使用是否存在**有临床意义的**药物相互作用。

知识库检索到的相关说明如下：
---
{evidence}
---

请基于上述证据判断；若证据不足或与两药无关，判定为「无明确相互作用」，**不得臆断或编造**。

严格只输出一个 JSON 对象，不要输出任何解释文字、注释或 markdown 代码块围栏。JSON 字段如下：
{{
  "has_conflict": true 或 false,
  "risk_level": "严重" 或 "中度" 或 "轻微" 或 "无",
  "reason": "简要说明相互作用机制或依据（1-2句）",
  "suggestion": "给患者的服用建议（1-2句，无冲突时可为空字符串）"
}}

要求：has_conflict=false 时 risk_level 必须为「无」。"""


def _parse_interaction_json(content):
    """从模型输出中稳健提取并校验 JSON。"""
    text = (content or "").strip()
    # 去掉可能的 markdown 代码块围栏
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM 返回中未找到 JSON 对象")

    obj = json.loads(text[start:end + 1])
    has_conflict = bool(obj.get("has_conflict", False))
    risk_level = str(obj.get("risk_level", "无")).strip()
    if not has_conflict:
        risk_level = "无"
    return {
        "has_conflict": has_conflict,
        "risk_level": risk_level,
        "reason": str(obj.get("reason", "")).strip(),
        "suggestion": str(obj.get("suggestion", "")).strip(),
        "risk_keyword": "",
    }


def _classify_interaction_llm(drug_a, drug_b, evidence_text):
    """用 LLM 判断两药是否存在临床意义的相互作用。

    返回统一 verdict 结构；失败时抛异常，由调用方回退关键词启发式。
    """
    api_key = get_required_env("DASHSCOPE_API_KEY")
    dashscope.api_key = api_key

    prompt = _build_interaction_prompt(drug_a, drug_b, evidence_text)
    response = Generation.call(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        result_format="message",
        stream=False,
    )

    if response.status_code != 200:
        raise RuntimeError(f"LLM 调用失败: {response.message} (code={response.code})")

    output = response.output
    choices = output.get("choices", []) if output else []
    if not choices:
        raise RuntimeError("LLM 未返回有效内容")

    content = choices[0].get("message", {}).get("content", "")
    return _parse_interaction_json(content)


def _keyword_heuristic(drug_a, drug_b, docs):
    """旧关键词启发式 fallback，返回与 LLM 一致的 verdict 结构。"""
    for doc in docs:
        content = doc.page_content or ""
        for kw in RISK_KEYWORDS:
            if kw in content:
                return {
                    "has_conflict": True,
                    "risk_level": "中度",
                    "risk_keyword": kw,
                    "reason": f"命中风险关键词「{kw}」",
                    "suggestion": "请咨询医生或药师确认是否可以联用。",
                }
    return {"has_conflict": False, "risk_level": "无", "risk_keyword": "", "reason": "", "suggestion": ""}


def check_drug_interaction(new_drug_name, existing_drugs_list, retriever):
    """
    检查新药与现有药物列表是否存在冲突。

    校验：
      - 直接复用调用方传入的混合检索器（向量+BM25），不再内部重建，避免每次检测重建 BM25
      - 先检索召回候选证据，再交给 LLM 二分类；LLM 失败时回退关键词启发式
    返回: (has_conflict, conflict_details)
      conflict_details 每一项含 drug_pair / risk_level / risk_keyword / reason / suggestion / evidence
    """
    if not retriever or not new_drug_name.strip():
        return False, []

    clean_new_drug = new_drug_name.strip()
    conflicts = []

    for old_drug in existing_drugs_list:
        clean_old_drug = (old_drug or "").strip()
        if not clean_old_drug or clean_old_drug == clean_new_drug:
            continue

        query = f"{clean_new_drug} 和 {clean_old_drug} 一起服用有什么禁忌或相互作用？能同时吃吗？"
        try:
            docs = retrieve_evidence_docs(retriever, query, top_k=3)
        except Exception as e:
            print(f"[drug_interaction] 检测 {clean_new_drug} 和 {clean_old_drug} 时检索出错：{e}")
            continue

        evidence_text = format_docs_for_prompt(docs)

        # 优先 LLM 二分类；不可用时回退关键词启发式
        try:
            verdict = _classify_interaction_llm(clean_new_drug, clean_old_drug, evidence_text)
        except Exception as e:
            print(f"[drug_interaction] LLM 分类失败，回退关键词启发式：{e}")
            verdict = _keyword_heuristic(clean_new_drug, clean_old_drug, docs)

        if verdict.get("has_conflict"):
            conflicts.append({
                "drug_pair": f"{clean_new_drug} + {clean_old_drug}",
                "risk_level": verdict.get("risk_level", "未知"),
                "risk_keyword": verdict.get("risk_keyword", ""),
                "reason": verdict.get("reason", ""),
                "suggestion": verdict.get("suggestion", ""),
                "evidence": (docs[0].page_content[:200] + "..." if docs else ""),
            })

    return len(conflicts) > 0, conflicts