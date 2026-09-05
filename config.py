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

# DASHSCOPE_API_KEY 不在此读为模块常量：消费方（agent_core/drug_interaction）
# 通过 get_required_env 在用时校验，避免产生"配了没生效"的死常量（P2-23/D）。
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

# 若重排模型已在本机缓存，强制离线加载。必须在本模块(全项目最先加载)设置：
# huggingface_hub 在 import 时就把 HF_HUB_OFFLINE 读成模块常量，之后再设不生效，
# 导致无法访问 huggingface.co 时要联网重试 5 次、阻塞首次回答数十秒。
if RERANK_ENABLED and os.path.isdir(
    os.path.join(RERANK_CACHE_DIR, "models--" + RERANK_MODEL.replace("/", "--"))
):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# RAG 检索结果低质量判定阈值（用于触发「本地检索不充分」的联网兜底 / 证据不足提示）。
# 依据 rag_utils.rerank_documents 写入的 rerank_source 区分分数尺度：
#   - rule  规则粗排分：< LOW_QUALITY_RULE_THRESHOLD(=0) 视为命中极弱/被降权
#   - cross cross-encoder 分：认为在 [0,1] 尺度，低于 LOW_QUALITY_CROSS_THRESHOLD 视为弱命中
# 具体可由主程序判定逻辑读取这些常量，便于统一调参与单测。
LOW_QUALITY_RULE_THRESHOLD = float(get_env("LOW_QUALITY_RULE_THRESHOLD", "0.0"))
LOW_QUALITY_CROSS_THRESHOLD = float(get_env("LOW_QUALITY_CROSS_THRESHOLD", "0.25"))


def validate_critical_env(required=("DASHSCOPE_API_KEY",), optional=("TAVILY_API_KEY", "GAODE_MAP_KEY", "LLM_MODEL")):
    """启动时一次性校验环境变量（P2-23/E）：

    - 关键密钥（缺省 DASHSCOPE_API_KEY）缺失或仍是占位符 → 直接抛错，fail-fast；
    - 可选密钥 / 模型名未配置或仍为占位符 → 仅打印告警，功能降级但不阻断启动。

    注意：`LLM_MODEL` 等项在代码里定义了默认值（见本模块常量），其默认值不会写入
    os.environ，因此完全未配置等价于走默认，不应视为缺失；只有**显式配置成占位符**
    时才告警。
    """
    # 这些项在模块里存在代码默认，未显式配置时不算缺失。
    keys_with_default = {"LLM_MODEL"}

    def _explicit_value(name):
        # 取 .env / 系统变量的显式配置；未配置返回 None（与 get_env 的代码默认区分开）
        return os.environ.get(name)

    def _is_placeholder(value):
        value = (value or "").strip()
        if not value:
            return True
        lower = value.lower()
        return lower.startswith(("your-", "xxx")) or "placeholder" in lower

    missing_required = [n for n in required if _is_placeholder(_explicit_value(n))]
    missing_optional = []
    for n in optional:
        value = _explicit_value(n)
        if value is None:
            if n not in keys_with_default:
                missing_optional.append(n)
        elif _is_placeholder(value):
            missing_optional.append(n)
    for name in missing_optional:
        print(f"[config] 可选环境变量未配置或仍为占位符，相关功能将降级: {name}")
    if missing_required:
        raise ValueError(
            "关键环境变量缺失或仍为占位符，无法启动智能医疗助手: "
            + ", ".join(missing_required)
            + "。请在 .env 或系统环境变量中配置后重试。"
        )
    return missing_required, missing_optional
