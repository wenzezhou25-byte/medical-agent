import hashlib
import math
import os
import re
from typing import Iterable, List

from langchain_core.embeddings import Embeddings

from config import BASE_DIR, get_env

DEFAULT_EMBEDDING_PROVIDER = get_env("EMBEDDING_PROVIDER", "fastembed").lower()
DEFAULT_EMBEDDING_MODEL = get_env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
DEFAULT_EMBEDDING_CACHE_DIR = get_env(
    "EMBEDDING_CACHE_DIR",
    str(BASE_DIR / ".cache" / "fastembed"),
)
DEFAULT_HASH_DIM = int(get_env("EMBEDDING_DIM", "512"))
DEFAULT_HF_ENDPOINT = get_env("HF_ENDPOINT", "")


class LocalHashEmbeddings(Embeddings):
    """完全离线的哈希向量方案，无需外部 API 或模型下载。"""

    def __init__(self, dimension: int = 512):
        self.dimension = max(128, int(dimension))

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[\u4e00-\u9fff]{1,4}|[A-Za-z0-9][A-Za-z0-9.+-]*", (text or "").lower())

    def _hash_token(self, token: str) -> tuple[int, float]:
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


class ResilientEmbeddings(Embeddings):
    """优先使用主 embedding，运行时失败则自动回退到备用 embedding。"""

    def __init__(self, primary: Embeddings, fallback: Embeddings):
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


def _build_fastembed_embeddings() -> Embeddings:
    try:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    except ImportError as exc:
        raise ImportError(
            "未安装 fastembed，请先执行 `pip install fastembed` 或 `pip install -r requirements.txt`。"
        ) from exc

    if DEFAULT_HF_ENDPOINT:
        os.environ["HF_ENDPOINT"] = DEFAULT_HF_ENDPOINT

    return FastEmbedEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL,
        cache_dir=DEFAULT_EMBEDDING_CACHE_DIR,
        providers=["CPUExecutionProvider"],
    )


def get_embeddings() -> Embeddings:
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
