# -*- coding: utf-8 -*-
"""针对 rag_utils._is_low_quality_docs 的单元测试。

只测纯逻辑（无需向量库/联网/UI），参照 smoke_test.py 的脚本式断言风格。

运行：
    python test_rag_quality.py
"""
from rag_utils import _is_low_quality_docs
from retrieval_core import Chunk


PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def doc(score, source=None):
    """构造带 rerank_score / rerank_source（可省略）的检索结果文档。"""
    metadata = {}
    if score is not None:
        metadata["rerank_score"] = score
    if source is not None:
        metadata["rerank_source"] = source
    return Chunk(page_content="测试内容", metadata=metadata)


# 分支一：空结果
print("== 空结果 ==")
check("空列表判定为低质量", _is_low_quality_docs([]) is True)
check("None 判定为低质量", _is_low_quality_docs(None) is True)

# 分支二：无有效分数
print("== 无有效分数 ==")
check("缺 rerank_score 判定为低质量", _is_low_quality_docs([doc(None)]) is True)
check("全无分数（含有害字符串分数）判定为低质量",
      _is_low_quality_docs([doc("abc"), doc("0.5")]) is True)

# 分支三：规则分（rule / 缺失）按 <0 判定
print("== 规则分阈值（<0 视为低质量） ==")
check("rule 负分判定为低质量", _is_low_quality_docs([doc(-0.3, "rule")]) is True)
check("rule 零分判定为低质量（0 < 0 不成立）", _is_low_quality_docs([doc(0.0, "rule")]) is False)
check("rule 正分判定为高质量", _is_low_quality_docs([doc(0.5, "rule")]) is False)
check("无 rerank_source 时按规则阈值（负分低质量）", _is_low_quality_docs([doc(-0.1)]) is True)

# 分支四：cross-encoder 分按 <0.25 判定
print("== cross-encoder 阈值（<0.25 视为低质量） ==")
check("cross 低分（0.1）判定为低质量", _is_low_quality_docs([doc(0.1, "cross")]) is True)
check("cross 临界高分（0.3）判定为高质量", _is_low_quality_docs([doc(0.3, "cross")]) is False)
check("cross 满分（0.9）判定为高质量", _is_low_quality_docs([doc(0.9, "cross")]) is False)

# 分支五：混合来源时取最高分文档对应的阈值
print("== 混合来源（取最高分对应尺度） ==")
mixed = [doc(0.1, "rule"), doc(0.2, "cross")]
check("最高分 0.2 为 cross，<0.25 低质量", _is_low_quality_docs(mixed) is True)
mixed2 = [doc(0.1, "rule"), doc(0.3, "cross")]
check("最高分 0.3 为 cross，>=0.25 高质量", _is_low_quality_docs(mixed2) is False)


print()
print(f"结果：{PASS} 通过, {FAIL} 失败")
if FAIL:
    raise SystemExit(1)