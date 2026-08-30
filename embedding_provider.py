# -*- coding: utf-8 -*-
"""embedding_provider.py —— 原生向量化封装（不依赖 LangChain）。

对外提供 Embeddings 语义实现（无需继承任何 langchain 基类）：
  - LocalHashEmbeddings: 完全离线的哈希向量
  - FastEmbedEmbeddings: 基于 fastembed 库（onnxruntime）的语义向量
  - ResilientEmbeddings: primary 优先，失败自动回退 fallback

统一接口（与 vector_store 兼容）：
  embed_documents(texts) -> List[List[float]]
  embed_query(text)      -> List[float]
"""

import hashlib
import math
import os
import re
from typing import Iterable, List

from config import BASE_DIR, get_env

DEFAULT_EMBEDDING_PROVIDER = get_env("EMBEDDING_PROVIDER", "auto").lower()
DEFAULT_EMBEDDING_MODEL = get_env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
DEFAULT_EMBEDDING_CACHE_DIR = get_env(
    "EMBEDDING_CACHE_DIR",
    str(BASE_DIR / ".cache" / "fastembed"),
)
DEFAULT_HASH_DIM = int(get_env("EMBEDDING_DIM", "512"))
DEFAULT_HF_ENDPOINT = get_env("HF_ENDPOINT", "")


class LocalHashEmbeddings:
    """完全离线的哈希向量方案，无需外部 API 或模型下载。"""

    def __init__(self, dimension: int = 512):
        self.dimension = max(128, int(dimension))

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[\u4e00-\u9fff]{1,4}|[A-Za-z0-9][A-Za-z0-9.+-]*", (text or "").lower())

    def _hash_token(self, token: str) -> tuple:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % self.dimension
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        return index, sign

    def _embed_text(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        tokens = self._tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            index, sign = self._hash_token(token)
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


class FastEmbedEmbeddings:
    """基于 fastembed（onnxruntime）的原生语义向量，替代 langchain FastEmbedEmbeddings。"""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: str = DEFAULT_EMBEDDING_CACHE_DIR,
        providers: list = ("CPUExecutionProvider",),
        max_length: int = 512,
        **kwargs,
    ):
        if DEFAULT_HF_ENDPOINT:
            os.environ["HF_ENDPOINT"] = DEFAULT_HF_ENDPOINT
        # 模型实际在首次 embed 时懒加载，避免初始化阶段联网阻塞
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._providers = list(providers)
        self._max_length = max_length
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
                providers=self._providers,
                max_length=self._max_length,
            )
        return self._model

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        text_list = [str(t) for t in texts]
        if not text_list:
            return []
        model = self._get_model()
        vectors = model.embed(text_list)  # iterable of np.ndarray
        return [list(map(float, vec)) for vec in vectors]

    def embed_query(self, text: str) -> List[float]:
        model = self._get_model()
        vec = next(model.query_embed(str(text)))
        return list(map(float, vec))


class ResilientEmbeddings:
    """优先使用主 embedding，运行时失败则自动回退到备用 embedding。"""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        text_list = list(texts)
        try:
            return self.primary.embed_documents(text_list)
        except Exception as exc:
            print(f"[embedding_provider] 主 embedding 文档向量失败，自动回退: {exc}")
            return self.fallback.embed_documents(text_list)

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.primary.embed_query(text)
        except Exception as exc:
            print(f"[embedding_provider] 主 embedding 查询向量失败，自动回退: {exc}")
            return self.fallback.embed_query(text)


def _build_fastembed_embeddings():
    try:
        return FastEmbedEmbeddings(
            model_name=DEFAULT_EMBEDDING_MODEL,
            cache_dir=DEFAULT_EMBEDDING_CACHE_DIR,
            providers=["CPUExecutionProvider"],
        )
    except ImportError as exc:
        raise ImportError(
            "未安装 fastembed，请先执行 `pip install fastembed` 或 `pip install -r requirements.txt`。"
        ) from exc


def get_embeddings():
    provider = DEFAULT_EMBEDDING_PROVIDER
    fallback = LocalHashEmbeddings(dimension=DEFAULT_HASH_DIM)

    if provider in {"hash", "hashing", "local_hash"}:
        return fallback

    if provider in {"fastembed", "auto"}:
        try:
            primary = _build_fastembed_embeddings()
            return ResilientEmbeddings(primary=primary, fallback=fallback)
        except Exception as exc:
            print(f"[embedding_provider] FastEmbed 初始化失败，自动回退到 hashing: {exc}")
            return fallback

    raise ValueError(f"暂不支持的 embedding provider: {provider}")