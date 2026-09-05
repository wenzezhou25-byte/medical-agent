# -*- coding: utf-8 -*-
"""
retrieval_core.py —— 检索原生基础层（替代 LangChain 三者）。

本模块目标是用「零框架」的自写实现替换掉 rag_utils 里的三件套依赖：
  * langchain_core.documents.Document               -> Chunk
  * langchain_text_splitters.RecursiveCharacterTextSplitter -> split_text_recursive
  * langchain_community.retrievers.BM25Retriever     -> JiebaBM25

约定：新的检索管道统一使用 Chunk；它保留了 page_content / metadata 两个
对外属性，因此依赖方（如 evaluate_rag.py / replay_eval.py 用
doc.page_content 和 doc.metadata.get(...) 读取结果）无需改动。

jieba 分词说明：项目在 _local_libs/jieba。调用方如需预装，可在 import 本
模块前执行：
    import sys; sys.path.insert(0, r"D:\\medical_agent\\_local_libs")
若 jieba 不可用，本模块自动降级为「汉字/英文正则分词」，保证不阻断运行。
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


def tokenize(text: str) -> List[str]:
    """对一段文本分词（jieba 优先，回退正则分词）。"""
    return list(_TOKENIZE(text or ""))


@dataclass
class Chunk:
    """检索文档的轻量载体，等价替换 langchain_core.documents.Document。"""

    page_content: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata 必须是 dict")
        if not self.metadata:
            self.metadata = {}

    def __repr__(self) -> str:  # 便于调试
        return (
            f"Chunk(src={self.metadata.get('source_name', '?')}, "
            f"sec={self.metadata.get('section_title', '?')}, "
            f"len={len(self.page_content or '')})"
        )


# ---------------------------------------------------------------------------
# 递归切分：与 RecursiveCharacterTextSplitter 保持近似等价的行为
# ---------------------------------------------------------------------------
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
_DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "：", "，", " ", ""]

# 剂量/次数 token：硬切长文本时绝不在这些 token 中间切断（P2-13）
_DOSE_TOKEN = re.compile(r"\d+(?:\.\d+)*\s*(?:mg|ml|g|ug|iu|单位|毫克|毫升|克|片|粒|次|天|日|袋|支|丸|滴)")


def _safe_hard_cut(text: str, chunk_size: int) -> List[str]:
    """在无分隔符的兜底路径上硬切超长段，但避免在剂量/次数 token 中间切断。

    在 [start, start+chunk_size) 窗口内找最后一个剂量 token 的结束位置，若它位于
    窗口的「中后段」且剩余较短，则选择在其后切断而非机械地从 900 处拦腰截断，
    从而保住“一日3次”“500mg”这类剂量表述不被拆成两半。

    下限约束：仅当剂量边界已推进超过窗口一半（best - start > chunk_size / 2）才
    采用剂量切点，避免窗口开头恰有剂量 token 时把 chunk 切得过短。
    """
    if len(text) <= chunk_size:
        return [text]
    out: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        limit = min(start + chunk_size, n)
        cut = limit
        best = start
        for m in _DOSE_TOKEN.finditer(text, start, limit):
            if m.end() > best:
                best = m.end()
        if best - start > chunk_size // 2 and limit - best < chunk_size:
            cut = best
        out.append(text[start:cut])
        start = cut
    return out or [text]


def _split_text_with_regex(text: str, separator: Optional[str], chunk_size: int) -> List[str]:
    """按单个分隔符把 text 切成多段，分隔符不保留（与默认 keep_separator=False 一致）。"""
    if not separator:  # 无有效分隔符 -> 先按句级标点切自然边界，仍超长的段再做保护剂量的硬切
        pieces = [s for s in re.split(r"(?<=[。！？；\n])", text) if s] or [text]
        out: List[str] = []
        for p in pieces:
            if len(p) <= chunk_size:
                out.append(p)
            else:
                out.extend(_safe_hard_cut(p, chunk_size))
        return out
    return re.split(re.escape(separator), text)


def _join_docs(docs: Sequence[str], separator: Optional[str]) -> Optional[str]:
    joined = (separator or "").join(docs).strip()
    return joined if joined else None


def _merge_splits(
    splits: Sequence[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """把碎块按 chunk_size 合并，超出部分按 chunk_overlap 移交给下一块前缀。"""
    docs: List[str] = []
    current_doc: List[str] = []
    total = 0
    for d in splits:
        _len = len(d)
        if current_doc and total + _len > chunk_size:
            doc = _join_docs(current_doc, separator)
            if doc is not None:
                docs.append(doc)
            while total > chunk_overlap:
                total -= len(current_doc[0])
                current_doc = current_doc[1:]
        current_doc.append(d)
        total += _len
    doc = _join_docs(current_doc, separator)
    if doc is not None:
        docs.append(doc)
    return docs


def split_text_recursive(
    text: str,
    separators: Optional[Sequence[str]] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """递归切分文本，等价逼近 RecursiveCharacterTextSplitter.split_text。"""
    seps = list(separators if separators is not None else _DEFAULT_SEPARATORS)

    final_chunks: List[str] = []
    separator = seps[-1]
    new_separators: List[str] = []
    for i, _s in enumerate(seps):
        if _s == "":
            separator = _s
            break
        if re.search(_s, text):
            separator = _s
            new_separators = seps[i + 1:]
            break

    _separator = separator if separator != "" else None
    splits = _split_text_with_regex(text, _separator, chunk_size)
    _good_splits = [s for s in splits if s] or [text]

    _all_splits: List[str] = []
    for _s in _good_splits:
        if len(_s) < chunk_size:
            _all_splits.append(_s)
        elif _separator is not None and len(_good_splits) > 1:
            # 本层确有细分出多段 -> 对超长段继续递归细分
            _all_splits.extend(
                split_text_recursive(_s, new_separators or seps, chunk_size, chunk_overlap)
            )
        else:
            # 无可细分 separator（本层只切出整段）-> 保护剂量的硬切兜底，终止递归
            _all_splits.extend(_safe_hard_cut(_s, chunk_size))

    final_chunks.extend(_merge_splits(_all_splits, _separator, chunk_size, chunk_overlap))
    return final_chunks


# ---------------------------------------------------------------------------
# BM25（Okapi BM25）原生实现，等价替代 BM25Retriever，采用 jieba 中文分词
# ---------------------------------------------------------------------------
class JiebaBM25:
    """基于 jieba 分词的 Okapi BM25。

    用法（对齐 langchain 的 BM25Retriever）：
        bm25 = JiebaBM25.from_chunks(chunks, k1=1.5, b=0.75)
        hits = bm25.invoke("布洛芬的用法用量", k=8)   # List[Chunk]
    """

    DEFAULT_K1 = 1.5
    DEFAULT_B = 0.75

    def __init__(
        self,
        corpus_terms: Sequence[List[str]],
        documents: Sequence[Chunk],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if not documents:
            raise ValueError("BM25 需要至少 1 篇文档")
        self.documents = list(documents)
        self.corpus_terms = [list(t) for t in corpus_terms]
        self.k1 = k1
        self.b = b
        self.doc_len = [len(t) for t in self.corpus_terms]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        self.idf = self._prepare_idf(self.corpus_terms)

    @staticmethod
    def _prepare_idf(corpus_terms: Sequence[List[str]]) -> dict:
        df: dict = {}
        for terms in corpus_terms:
            for w in set(terms):
                df[w] = df.get(w, 0) + 1
        n = len(corpus_terms)
        return {w: float((n - f + 0.5) / (f + 0.5)) + 1.0 for w, f in df.items()}

    @staticmethod
    def _term_freq(terms: List[str], term: str) -> int:
        return terms.count(term)

    def score(self, query_terms: Sequence[str], i: int) -> float:
        dl = self.doc_len[i]
        score = 0.0
        for t in query_terms:
            qf = self._term_freq(self.corpus_terms[i], t)
            if qf == 0 or t not in self.idf:
                continue
            tf_part = (qf * (self.k1 + 1)) / (qf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            score += self.idf[t] * tf_part
        return score

    def get_scores(self, query: str) -> List[float]:
        query_terms = tokenize(query)
        if not query_terms:
            return [0.0] * len(self.documents)
        return [self.score(query_terms, i) for i in range(len(self.documents))]

    def invoke(self, query: str, k: int = 8) -> List[Chunk]:
        scored = zip(self.get_scores(query), self.documents)
        top = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
        return [doc for _, doc in top]

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], **kwargs) -> "JiebaBM25":
        corpus = [tokenize(c.page_content) for c in chunks]
        return cls(corpus, list(chunks), **kwargs)


# ---------------------------------------------------------------------------
# jieba 加载（置于模块尾部，便于在类/函数定义后统一初始化）
# ---------------------------------------------------------------------------
def _load_tokenizer():
    global _TOKENIZE
    try:
        _libs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_local_libs")
        if _libs not in sys.path:
            sys.path.insert(0, _libs)
        import jieba  # noqa: F401

        def _cut(text):
            return [w for w in jieba.cut(text or "") if w.strip()]

        _TOKENIZE = _cut
    except Exception:  # pragma: no cover - 降级路径
        _WORD_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9.+-]*")

        def _cut(text):  # type: ignore[misc]
            return [m.group(0) for m in _WORD_RE.finditer(text or "")]

        _TOKENIZE = _cut  # type: ignore[assignment]


_TOKENIZE = None  # type: ignore[assignment]
_load_tokenizer()