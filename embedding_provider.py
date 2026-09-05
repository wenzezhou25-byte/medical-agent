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

from config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_DIM,
    HF_ENDPOINT,
)

# 所有默认值统一从 config 读取（避免两处解析 source-of-truth 漂移）
DEFAULT_EMBEDDING_PROVIDER = EMBEDDING_PROVIDER.lower()
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL
DEFAULT_EMBEDDING_CACHE_DIR = EMBEDDING_CACHE_DIR
# EMBEDDING_DIM 仅作为「纯 hash 回退」的兜底维度；fastembed 路径会由 get_embeddings()
# 把 fallback 维度对齐到主模型（bge 系列 384 维），故不应再被独立用于主模型。
DEFAULT_HASH_DIM = int(EMBEDDING_DIM)
DEFAULT_HF_ENDPOINT = HF_ENDPOINT


class LocalHashEmbeddings:
    """完全离线的哈希向量方案，无需外部 API 或模型下载。"""

    def __init__(self, dimension: int = 512):
        self.dimension = max(128, int(dimension))
        # 作为 FastEmbed 初始化失败的降级路径时置 True，供上层识别「非语义检索」
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """是否被用作主 embedding 失败后的降级实现（hashing 非语义）。"""
        return self._degraded

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

    def get_dim(self) -> int:
        return self.dimension


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

    def get_dim(self) -> int:
        """探测模型输出维度，首次调用会触发模型加载。

        用非空探针探测，避免空串对个别模型返回零向量/触发 StopIteration。
        """
        vec = self.embed_query("。")
        if not vec or all(float(x) == 0.0 for x in vec):
            raise RuntimeError("embedding 维度探测失败：模型返回空/零向量")
        return len(vec)


class ResilientEmbeddings:
    """优先使用主 embedding，运行时失败则自动回退到备用 embedding。

    只允许在**建库/文档向量**阶段回退（配合 `degraded` 标志由上层决定是否中止）；
    查询阶段不允许静默回退——主模型与 hash 是不同向量空间，回退给出的相似度无意义，
    会静默污染结果，故查询失败直接抛错，提示重建知识库。
    """

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """是否曾发生过主 embedding 回退（可能造成向量空间不一致）。"""
        return self._degraded

    def get_dim(self) -> int:
        return self.primary.get_dim()

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        text_list = list(texts)
        try:
            return self.primary.embed_documents(text_list)
        except Exception as exc:
            self._degraded = True
            print(f"[embedding_provider] 主 embedding 文档向量失败，已标记 degraded 并回退: {exc}")
            return self.fallback.embed_documents(text_list)

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.primary.embed_query(text)
        except Exception as exc:
            self._degraded = True
            raise RuntimeError(
                "主 embedding 查询失败（可能已发生向量空间不一致），"
                "无法用 hash 向量替代，请检查 EMBEDDING_PROVIDER/模型配置或重建知识库。"
            ) from exc


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
            # fallback 维度与 primary 对齐（例如 bge 系列输出 512 维），
            # 而非沿用 DEFAULT_HASH_DIM，避免主/备维度不一致导致 index.search 崩溃。
            fallback = LocalHashEmbeddings(dimension=primary.get_dim())
            return ResilientEmbeddings(primary=primary, fallback=fallback)
        except Exception as exc:
            # 显式标注「fastembed 路径实际未生效」并打上 degraded 标志，
            # 避免上层把 hash 回退误判为语义检索。
            print(f"[embedding_provider] FastEmbed 初始化/维度探测失败，回退到 hashing（非语义检索）: {exc}")
            fallback._degraded = True
            return fallback

    raise ValueError(f"暂不支持的 embedding provider: {provider}")