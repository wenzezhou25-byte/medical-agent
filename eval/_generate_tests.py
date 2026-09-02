"""临时脚本：基于当前向量库中的家庭用药数据生成一套测试题。

每道题从一个实际存在的药品+说明书章节生成，自带 gold_sources(药名) 与 section_title，
使评测既能算关键词准确率，也能算 Recall@k / MRR@k。
"""
import json
import sys
from pathlib import Path

# 基于脚本位置定位项目根，把项目根与 _local_libs 一并加入 sys.path
_BASE = Path(__file__).resolve().parent.parent
for _p in (str(_BASE), str(_BASE / "_local_libs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jieba  # noqa: E402

from vector_store import VectorStore  # noqa: E402

from config import VECTOR_STORE_PATH  # noqa: E402
from embedding_provider import get_embeddings  # noqa: E402

STOP = set(
    "的 了 吗 呢 我 你 他 她 它 和 与 及 或 是 也 就是 什么 有哪些 怎么 可以 用于 治疗 "
    "作用 使用 帮助 要 能 会 都 一 个 这 那 这些 那些 对 在 有 进行 主要 本品 用药 服用 "
    "应 需 者 注意 情况 时 后 前 中 等 之 该 其 每 各 次 日 片 粒 应予 应避免 用 药 的 和 "
    "或 及 须 请 禁 忌 慎 间 服 后".split()
)
# 预置医学主题词，命中即保留（提升关键词质量与可读性）
GOOD_KW = (
    "不良反应", "适应症", "禁忌", "注意事项", "用法用量", "药理作用", "儿童用药",
    "孕妇", "哺乳期", "肝功能", "肾功能", "过敏", "过敏史", "饮酒", "驾驶",
    "茶碱类", "退热", "止痛", "解热镇痛", "抗感染", "抗生素", "口服", "皮下注射",
    "肌肉注射", "静脉滴注", "血药浓度", "每日", "每次", "一次", "一日", "疗程",
)
SECTIONS = ["适应症", "用法用量", "不良反应", "禁忌", "注意事项", "贮藏", "成分"]
MAX_DRUGS = 40
MAX_SECTIONS_PER_DRUG = 2
OUT = Path(__file__).resolve().parent / "eval_sets" / "family_drug_questions.json"
DROP_KW = ("禁用", "慎用", "禁止", "服用", "使用", "用于", "治疗", "适应症")


def pick_keywords(text):
    """用 jieba 分词，优先保留医学主题词，其次取 2~6 字的实词。"""
    seen, out, reserve = set(), [], []
    for w in jieba.cut(text or ""):
        w = w.strip()
        if w in STOP or len(w) < 2 or len(w) > 6 or w in seen:
            continue
        if any(k in w for k in DROP_KW) and w not in GOOD_KW:
            continue
        if w in GOOD_KW:
            reserve.append(w)
            seen.add(w)
        else:
            out.append(w)
    # 先放主题词，不足再用普通词补齐
    combined = reserve + [w for w in out if w not in seen]
    return combined[:3]


def main():
    vs = VectorStore.load_local(VECTOR_STORE_PATH, get_embeddings())
    docstore = getattr(vs, "docstore", None)
    docs = list(getattr(docstore, "_dict", {}).values())

    by_drug = {}
    for doc in docs:
        m = doc.metadata or {}
        drug = (m.get("source_name") or "").strip()
        sec = (m.get("section_title") or "").strip()
        if drug and sec:
            by_drug.setdefault(drug, {})[sec] = doc.page_content

    drugs = sorted(by_drug.keys())[:MAX_DRUGS]
    items = []
    for drug in drugs:
        secs = [s for s in SECTIONS if by_drug[drug].get(s) and len(by_drug[drug][s].strip()) >= 50]
        if not secs:
            secs = [s for s, t in by_drug[drug].items() if t and len(t.strip()) >= 50]
        if not secs:
            continue
        secs = secs[:MAX_SECTIONS_PER_DRUG]
        for sec in secs:
            text = by_drug[drug][sec]
            kw = pick_keywords(text)
            if len(kw) < 2:
                continue
            q = f"{drug}的{sec}需要注意什么？" if sec == "注意事项" else f"{drug}的{sec}是什么？"
            items.append({
                "question": q,
                "expected_keywords": kw[:3],
                "gold_sources": [drug],
                "section_title": sec,
                "category": "drug_insert",
                "note": f"自动生成·{drug}/{sec}",
            })

    json.dump(items, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"生成 {len(items)} 道题，来自 {len(drugs)} 种药 → {OUT}")
    for it in items[:6]:
        print(" -", it["question"], "| kw:", it["expected_keywords"])


if __name__ == "__main__":
    sys.exit(main())