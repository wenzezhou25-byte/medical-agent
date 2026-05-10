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

VECTOR_STORE_PATH = str(BASE_DIR / "vector_store")
BASE_DATA_PATH = str(BASE_DIR / "data")
STATS_SAVE_PATH = str(BASE_DIR / "knowledge_base_stats.json")
