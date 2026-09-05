# -*- coding: utf-8 -*-
"""retrieval_core.py 的 split_text_recursive / _safe_hard_cut 单元测试。

纯函数/纯正则，不联网、不加载模型。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval_core import CHUNK_SIZE, _safe_hard_cut, split_text_recursive  # noqa: E402


def test_short_text_returned_unchanged():
    text = "一段不含任何分隔符的短文本甲乙丙丁"
    assert _safe_hard_cut(text, CHUNK_SIZE) == [text]
    assert split_text_recursive(text) == [text]


def test_short_text_separator_is_stripped_not_kept():
    # 默认 keep_separator=False：分隔符「。」被消费，不保留在输出中
    chunks = split_text_recursive("短文本，不需要切分。")
    assert chunks == ["短文本，不需要切分"]


def test_hard_cut_long_no_separator_text_chunks_within_limit():
    # 无任何分隔符/剂量的长文本，硬切后每段长度不超过 chunk_size
    chunk_size = 100
    text = "甲" * 5000
    chunks = _safe_hard_cut(text, chunk_size)
    assert len(chunks) > 1
    assert all(len(c) <= chunk_size for c in chunks)


def test_split_recursive_long_no_separator_chunks_within_limit():
    # split_text_recursive 兜底路径同样保证每段不超过 chunk_size
    # （chunk_overlap 必须小于 chunk_size，否则合并后可能撑破单块上限）
    chunk_size, chunk_overlap = 100, 20
    text = "甲" * 5000  # 无默认分隔符，触发 _safe_hard_cut
    chunks = split_text_recursive(text, separators=[""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) > 1
    assert all(len(c) <= chunk_size for c in chunks)


def test_hard_cut_protects_dose_token_mg():
    # 剂量 token「500mg」应落在首个 chunk 尾部，不被一分为二
    text = "甲" * 92 + "500mg" + "乙" * 200
    chunks = _safe_hard_cut(text, 100)
    # 保护生效：首块以完整 500mg 收尾（否则会变成 …500m + 乙…）
    assert chunks[0].endswith("500mg")
    assert any("500mg" in c for c in chunks)


def test_hard_cut_protects_dose_token_frequency():
    # 「一日3次」的频率 token 同样不能被拦腰切断
    text = "甲" * 92 + "一日3次" + "乙" * 200
    chunks = _safe_hard_cut(text, 100)
    assert chunks[0].endswith("一日3次")
    assert any("一日3次" in c for c in chunks)


def test_split_recursive_keeps_dose_tokens_intact():
    # 经整体切分管道，输入中的所有剂量 token 仍完整存在于某一块中
    chunk_size, chunk_overlap = 100, 20
    text = ("甲" * 95 + "500mg" + "甲" * 95 + "乙" * 20 + "一日3次" + "丙" * 60)
    chunks = split_text_recursive(text, separators=[""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert all(len(c) <= chunk_size for c in chunks)
    assert any("500mg" in c for c in chunks)
    assert any("一日3次" in c for c in chunks)


def test_dose_token_not_bisected_across_any_boundary():
    # 通用性：token 完整出现于单一 chunk（若被切断，则没有任何 chunk 含完整 token）
    chunk_size, chunk_overlap = 100, 20
    text = ("甲" * 90 + "500mg" + "乙" * 90 + "丙" * 90 + "一日3次" + "丁" * 90)
    chunks = split_text_recursive(text, separators=[""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for token in ("500mg", "一日3次"):
        assert any(token in c for c in chunks), f"token {token} 被切断"