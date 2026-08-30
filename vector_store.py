# -*- coding: utf-8 -*-
"""
vector_store.py —— 原生 FAISS 向量存储（替代 langchain_community.vectorstores.FAISS）。

对外接口对齐 langchain FAISS，便于调用方在仅改 import 的情况下切换：
  from_documents(documents, embedding_function)
  add_documents(documents) / add_texts(texts, metadatas)
  similarity_search(query, k) / similarity_search_with_score(query, k)
  as_retriever(search_kwargs={"k": ...})   # 返回带 invoke(query) 的对象
  save_local(folder_path) / load_local(folder_path, embedding_function, ...)
  .docstore._dict                          # 供 get_vectorstore_documents 读取

embedding_function 只需提供 embed_query(text) -> List[float]
和 embed_documents(texts) -> List[List[float]] 两个方法。
相似度采用「L2 归一化 + 内积」，对归一化后的向量与 L2 距离排序等价。

落盘格式（目录内两个文件）：
  index.faiss   faiss 二进制索引
  chunks.json   文档列表（docstore_id / page_content / metadata）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import faiss
import numpy as np

from retrieval_core import Chunk


class _VectorRetriever:
    """对齐 langchain BaseRetriever.invoke(query) 的最小编程接口。"""

    def __init__(self, store: "VectorStore", k: int = 4) -> None:
        self.store = store
        self.k = k

    def invoke(self, query: str) -> List[Chunk]:
        return self.store.similarity_search(query, k=self.k)


class VectorStore:
    """轻量原生 FAISS 向量存储，容器统一为 Chunk。"""

    def __init__(
        self,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        index: Optional[Any] = None,
        documents: Optional[dict] = None,
        index_to_id: Optional[list] = None,
    ) -> None:
        self.embedding_function = embedding_function
        self.index = index
        self._dim: Optional[int] = None
        self.documents: dict = documents if documents is not None else {}  # docstore_id -> Chunk
        self.index_to_id: list = index_to_id if index_to_id is not None else []  # faiss 位置 -> docstore_id

        class _DocStore:
            pass

        self.docstore = _DocStore()
        self.docstore._dict = self.documents

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _normalize(vector) -> np.ndarray:
        arr = np.asarray(vector, dtype="float32")
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 0 else arr

    # ------------------------------------------------------------ 写入
    @classmethod
    def from_documents(cls, documents: Sequence[Chunk], embedding_function, **kwargs) -> "VectorStore":
        store = cls(embedding_function=embedding_function)
        store.add_documents(documents)
        return store

    @classmethod
    def from_texts(cls, texts: Sequence[str], embedding, metadatas: Optional[Sequence[dict]] = None, **kwargs) -> "VectorStore":
        docs = [Chunk(page_content=t, metadata=m or {}) for t, m in zip(texts, metadatas or [])]
        return cls.from_documents(docs, embedding)

    def add_documents(self, documents: Sequence[Chunk]) -> List[str]:
        docs = list(documents)
        if not docs:
            return []
        texts = [doc.page_content or "" for doc in docs]
        vectors = self.embedding_function.embed_documents(texts)
        matrix = np.stack([self._normalize(vec) for vec in vectors]).astype("float32")
        if self.index is None:
            self._dim = matrix.shape[1]
            self.index = faiss.IndexFlatIP(self._dim)
        start = len(self.index_to_id)
        self.index.add(matrix)
        ids: List[str] = []
        for offset, doc in enumerate(docs):
            cid = f"doc-{start + offset}"
            self.documents[cid] = doc
            self.index_to_id.append(cid)
            ids.append(cid)
        return ids

    def add_texts(self, texts: Sequence[str], metadatas: Optional[Sequence[dict]] = None) -> List[str]:
        docs = [Chunk(page_content=t, metadata=m or {}) for t, m in zip(texts, metadatas or [])]
        return self.add_documents(docs)

    # ------------------------------------------------------------ 检索
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = self._normalize(self.embedding_function.embed_query(query)).astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query_vec, k)
        results: List[tuple] = []
        for score, idx in zip(scores[0], indices[0]):
            pos = int(idx)
            if pos < 0 or pos >= len(self.index_to_id):
                continue
            cid = self.index_to_id[pos]
            results.append((self.documents[cid], float(score)))
        return results

    def similarity_search(self, query: str, k: int = 4) -> List[Chunk]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        k = (search_kwargs or {}).get("k", 4)
        return _VectorRetriever(self, k)

    # ------------------------------------------------------------ 持久化
    def save_local(self, folder_path: str) -> None:
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            raise ValueError("没有可保存的索引。")
        faiss.write_index(self.index, str(folder / "index.faiss"))
        chunks: List[dict] = []
        for cid in self.index_to_id:
            doc = self.documents[cid]
            chunks.append(
                {
                    "docstore_id": cid,
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )
        with (folder / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=1, default=str)

    @classmethod
    def load_local(cls, folder_path: str, embedding_function, allow_dangerous_deserialization: bool = False):
        folder = Path(folder_path)
        index = faiss.read_index(str(folder / "index.faiss"))
        with (folder / "chunks.json").open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        documents: dict = {}
        index_to_id: list = []
        for item in chunks:
            cid = item["docstore_id"]
            documents[cid] = Chunk(page_content=item["page_content"], metadata=item["metadata"])
            index_to_id.append(cid)
        store = cls(
            embedding_function=embedding_function,
            index=index,
            documents=documents,
            index_to_id=index_to_id,
        )
        store._dim = index.d
        return store