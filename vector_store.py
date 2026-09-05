# -*- coding: utf-8 -*-
"""
vector_store.py —— 原生 FAISS 向量存储（替代 langchain_community.vectorstores.FAISS）。

对外接口对齐 langchain FAISS，便于调用方在仅改 import 的情况下切换：
  from_documents(documents, embedding_function)
  add_documents(documents) / add_texts(texts, metadatas)
  similarity_search(query, k) / similarity_search_with_score(query, k)
  as_retriever(search_kwargs={"k": ...})   # 返回带 invoke(query) 的对象
  save_local(folder_path) / load_local(folder_path, embedding_function)
  .docstore._dict                          # 供 get_vectorstore_documents 读取

落盘格式（目录内三个文件）：
  index.faiss   faiss 二进制索引
  chunks.json   文档列表（docstore_id / page_content / metadata）
  meta.json     元数据（version / embedding 标识 / 各文件 sha256），用于加载时
                版本兼容与完整性校验（防损坏、半截写入；不防恶意篡改）。

完整性说明：meta.json 与数据文件同目录保存，攻击者若能改数据文件就能同步改
meta.json，因此 hash 校验只作为健壮性/一致性防线，真正的安全边界是入库前的
内容净化 + 「工具输出当不可信数据」处理。

embedding_function 只需提供 embed_query(text) -> List[float]
和 embed_documents(texts) -> List[List[float]] 两个方法。
相似度采用「L2 归一化 + 内积」，对归一化后的向量与 L2 距离排序等价。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import faiss
import numpy as np

from retrieval_core import Chunk

META_VERSION = 1


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
        elif matrix.shape[1] != self.index.d:
            # 主/备 embedding 维度不一致会直接让 index.add 崩溃，这里提前给出明确报错
            raise RuntimeError(
                f"新增向量维度({matrix.shape[1]})与已有索引维度({self.index.d})不一致，"
                f"embedding 可能回退到了不同的向量空间，请检查 EMBEDDING_PROVIDER 配置或重建知识库。"
            )
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
        if query_vec.shape[1] != self.index.d:
            raise RuntimeError(
                f"查询向量维度({query_vec.shape[1]})与索引维度({self.index.d})不一致，"
                f"embedding 可能回退到了 hashing 或其他向量空间，请检查 EMBEDDING_PROVIDER 配置或重建知识库。"
            )
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
    @staticmethod
    def _file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    def _embedding_identity(self) -> Dict[str, Any]:
        """尽力提取 embedding 标识（provider/model/dim），拿不到就省略。

        dim 是维度校验的关键；provider/model 供诊断参考，非强校验。探测失败时
        静默降级，不让落盘/加载因可选的诊断信息而失败。
        """
        emb = self.embedding_function
        if emb is None:
            return {}
        info: Dict[str, Any] = {}
        try:
            if hasattr(emb, "get_dim"):
                info["dim"] = int(emb.get_dim())
        except Exception:
            pass
        model = getattr(emb, "model_name", None) or getattr(emb, "_model_name", None)
        if model:
            info["model"] = model
        primary = getattr(emb, "primary", None)
        if primary is not None:
            info["provider"] = type(primary).__name__
        elif model:
            info["provider"] = "fastembed"
        elif isinstance(emb, object) and type(emb).__name__ == "LocalHashEmbeddings":
            info["provider"] = "hash"
        return info

    @staticmethod
    def _read_meta(folder: Path) -> Optional[dict]:
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            return None
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_meta(self, folder: Path) -> None:
        meta = {
            "version": META_VERSION,
            "embeddings": self._embedding_identity(),
            "files": {
                "index.faiss": self._file_sha256(folder / "index.faiss"),
                "chunks.json": self._file_sha256(folder / "chunks.json"),
            },
        }
        tmp = folder / "meta.json.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        # 原子替换，避免进程中断留下半个 meta.json
        os.replace(str(tmp), str(folder / "meta.json"))

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
        # 数据写完后生成 meta.json（含各文件 sha256），供加载时做完整性校验
        self._write_meta(folder)

    @classmethod
    def _validate_meta(cls, folder: Path, index) -> None:
        """校验 meta.json：version 兼容 + 文件 hash 未被改动（防损坏/半截写入）。

        注意：meta 与数据同目录，仅挡损坏，不防有目录写权限的攻击者（其可同步
        改写 meta）。真正的安全边界在入库净化 + 工具输出当不可信数据。
        """
        meta = cls._read_meta(folder)
        if meta is None:
            print(f"[vector_store] 未找到 meta.json，跳过完整性校验（旧库或异常目录）: {folder}")
            return
        if int(meta.get("version", -1)) != META_VERSION:
            raise RuntimeError(
                f"知识库版本不支持: 磁盘为 v{meta.get('version')}, 本程序仅支持 v{META_VERSION}，请重建知识库。"
            )
        for name, expect in (meta.get("files") or {}).items():
            fpath = folder / name
            if not fpath.exists():
                raise RuntimeError(f"知识库缺少数据文件: {name}")
            if cls._file_sha256(fpath) != expect:
                raise RuntimeError(
                    f"知识库完整性校验失败: {name} 与 meta.json 记录不一致，"
                    f"文件可能被修改或损坏，请重建知识库。"
                )
        meta_emb = meta.get("embeddings") or {}
        meta_dim = meta_emb.get("dim")
        if meta_dim is not None and int(meta_dim) != index.d:
            raise RuntimeError(
                f"知识库索引维度({index.d})与 meta 记录的维度({meta_dim})不一致，请重建知识库。"
            )

    @classmethod
    def load_local(cls, folder_path: str, embedding_function):
        folder = Path(folder_path)
        index = faiss.read_index(str(folder / "index.faiss"))
        # 完整性校验放在读取内容之后、依赖文件被消除之前，保证读的是未被篡改的文件
        cls._validate_meta(folder, index)
        with (folder / "chunks.json").open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        documents: dict = {}
        index_to_id: list = []
        for item in chunks:
            cid = item["docstore_id"]
            documents[cid] = Chunk(page_content=item["page_content"], metadata=item["metadata"])
            index_to_id.append(cid)
        # 结构一致性：向量数与文档数必须对齐，防止手动改动 chunks.json 造成的漂移
        if index.ntotal != len(index_to_id) or len(index_to_id) != len(chunks):
            raise RuntimeError(
                f"知识库结构不一致: 索引向量数({index.ntotal})/chunks 数({len(chunks)})/"
                f"docstore 数({len(index_to_id)}) 不相等，请重建知识库。"
            )
        store = cls(
            embedding_function=embedding_function,
            index=index,
            documents=documents,
            index_to_id=index_to_id,
        )
        store._dim = index.d
        # 加载时即校验维度一致性，把错误提前到加载阶段，而不是等到查询时崩溃。
        # embedding_function.get_dim() 触发一次模型加载/探测；缺失时用 embed_query 探测。
        if embedding_function is not None:
            if hasattr(embedding_function, "get_dim"):
                cur_dim = int(embedding_function.get_dim())
            else:
                try:
                    cur_dim = len(embedding_function.embed_query("。"))
                except Exception:
                    cur_dim = None
            if cur_dim is not None and cur_dim != index.d:
                raise RuntimeError(
                    f"知识库索引维度({index.d})与当前 embedding 维度({cur_dim})不一致，"
                    f"可能由 embedding 回退到 hashing 或模型变更导致，请重建知识库。"
                )
        return store