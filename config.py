import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 依赖未安装时回退到系统环境变量
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent


def load_environment() -> None:
    """优先从项目根目录的 .env 加载本地配置。"""
    if load_dotenv is not None:
        load_dotenv(BASE_DIR / ".env")


def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


def get_required_env(name: str) -> str:
    value = get_env(name)
    if not value:
        raise ValueError(f"未找到环境变量 {name}，请先配置 .env 或系统环境变量。")
    return value


load_environment()

DASHSCOPE_API_KEY = get_env("DASHSCOPE_API_KEY")
TAVILY_API_KEY = get_env("TAVILY_API_KEY")
GAODE_MAP_KEY = get_env("GAODE_MAP_KEY")
EMBEDDING_PROVIDER = get_env("EMBEDDING_PROVIDER", "fastembed")
EMBEDDING_MODEL = get_env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_CACHE_DIR = get_env("EMBEDDING_CACHE_DIR", str(BASE_DIR / ".cache" / "fastembed"))
EMBEDDING_DIM = get_env("EMBEDDING_DIM", "512")
HF_ENDPOINT = get_env("HF_ENDPOINT", "https://hf-mirror.com")

LLM_MODEL = get_env("LLM_MODEL", "qwen3-max")

VECTOR_STORE_PATH = str(BASE_DIR / "vector_store")
BASE_DATA_PATH = str(BASE_DIR / "data")
STATS_SAVE_PATH = str(BASE_DIR / "knowledge_base_stats.json")

# 重排模型（cross-encoder）
RERANK_ENABLED = get_env("RERANK_ENABLED", "1") not in {"0", "false", "False", "no", "off"}
RERANK_MODEL = get_env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MAX_LENGTH = int(get_env("RERANK_MAX_LENGTH", "1024"))
RERANK_CACHE_DIR = get_env("RERANK_CACHE_DIR", str(BASE_DIR / ".cache" / "sentence_transformers"))

# RAG 检索结果低质量判定阈值（用于触发「本地检索不充分」的联网兜底 / 证据不足提示）。
# 依据 rag_utils.rerank_documents 写入的 rerank_source 区分分数尺度：
#   - rule  规则粗排分：< LOW_QUALITY_RULE_THRESHOLD(=0) 视为命中极弱/被降权
#   - cross cross-encoder 分：认为在 [0,1] 尺度，低于 LOW_QUALITY_CROSS_THRESHOLD 视为弱命中
# 具体可由主程序判定逻辑读取这些常量，便于统一调参与单测。
LOW_QUALITY_RULE_THRESHOLD = float(get_env("LOW_QUALITY_RULE_THRESHOLD", "0.0"))
LOW_QUALITY_CROSS_THRESHOLD = float(get_env("LOW_QUALITY_CROSS_THRESHOLD", "0.25"))
