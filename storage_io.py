# -*- coding: utf-8 -*-
"""统一 JSON 原子落盘工具 + 落盘加密 + 访问审计。

P1-22：所有 JSON 持久化（user_data.py / auth.py）统一走本模块，
避免中途崩溃/断电留下半截损坏文件。要点：
- 先写同目录唯一临时文件（含 pid + 线程 + uuid，避免并发撞名）；
- fsync 刷盘后 os.replace 原子替换；
- 目录不存在时自动创建；
- 任一环节失败时清理临时文件并向上抛异常。

数据安全加固：
- 配置 DATA_ENC_KEY（Fernet 对称密钥）后，写入走 atomic_write_json_encrypted
  以密文落盘；load_json 优先按密文读，解密失败回退明文，兼容存量旧文件。
- 未配置 DATA_ENC_KEY 时自动降级为明文，行为与加固前完全一致（仅多一条告警）。
- 每次读/写通过 _audit 追加写入 data/audit.log，供访问审计。
"""
import json
import logging
import os
import threading
import uuid

# Fernet 为可选依赖：确保未安装 cryptography 时项目仍可正常以明文运行。
try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:  # pragma: no cover - 依赖未安装时降级明文
    Fernet = None
    InvalidToken = None

# 未配置 DATA_ENC_KEY 时仅告警一次，避免每次读写在终端刷屏。
_warned_no_key = False

# 进程内按目标文件路径加的锁，避免多会话并发写同一文件导致"丢更新"。
# Streamlit 单进程多线程，用 threading.Lock 即可。
_locks = {}
_locks_guard = threading.Lock()

# 审计日志：项目 data/ 目录下的 audit.log（追加写）。
_AUDIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "audit.log")
_audit_lock = threading.Lock()
_audit_logger = None


def _audit(action: str, path: str):
    """追加一行访问审计到 data/audit.log：时间 / action(READ|WRITE) / 路径 / 当前用户。

    当前用户从环境变量 CURRENT_USER 读取，读不到时记为 "unknown"。
    记录中不包含任何数据内容或密钥。日志追加写为进程内单例，加锁保证多线程安全。
    """
    global _audit_logger
    if _audit_logger is None:
        with _audit_lock:
            if _audit_logger is None:
                os.makedirs(os.path.dirname(_AUDIT_PATH), exist_ok=True)
                logger = logging.getLogger("storage_io.audit")
                logger.setLevel(logging.INFO)
                logger.propagate = False  # 避免向上冒泡到根 logger 重复输出
                handler = logging.FileHandler(_AUDIT_PATH, encoding="utf-8", mode="a")
                handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))
                logger.addHandler(handler)
                _audit_logger = logger
    user = os.environ.get("CURRENT_USER") or "unknown"
    _audit_logger.info("%s\t%s\t%s", action, path, user)


def _fernet():
    """从环境变量 DATA_ENC_KEY 构造 Fernet 实例。

    未配置时返回 None（降级明文）并打印一条告警；配置了但 cryptography 缺失或
    密钥非法时同样降级明文，保证任何情况下都不阻断功能。
    """
    global _warned_no_key
    key = os.environ.get("DATA_ENC_KEY") or ""
    if not key:
        if not _warned_no_key:
            _warned_no_key = True
            print("未配置 DATA_ENC_KEY，健康数据将明文存储")
        return None
    if Fernet is None:
        print("未安装 cryptography，无法加密健康数据，将明文存储")
        return None
    try:
        return Fernet(key.encode("utf-8"))
    except Exception:  # 非法密钥：降级明文，保证功能可用
        if not _warned_no_key:
            _warned_no_key = True
            print("DATA_ENC_KEY 非法，健康数据将明文存储")
        return None


def _lock_for(path: str):
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())


def atomic_write_json(path, data):
    """将 data 以 JSON 形式原子写入 path。

    成功则旧文件被全新文件替换；失败不修改原文件，并清理残留临时文件。
    """
    _audit("WRITE", path)
    with _lock_for(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # 唯一临时名：并发写同一文件时不会互相截断
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())  # 刷到磁盘，进一步降低断电丢数据
            os.replace(tmp, path)
        except Exception:
            # json.dump / os.replace 失败：清理临时文件后再抛出，避免遗留垃圾
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            raise


def atomic_write_json_encrypted(path, data):
    """将 data 写成 JSON；配置 DATA_ENC_KEY 时密文落盘，否则退回明文。

    有密钥：json.dumps 后经 Fernet 加密成 token（二进制），复用「tmp+fsync+os.replace」
    原子写；无密钥：直接退回 atomic_write_json 的明文写，保证功能不降级。
    """
    fernet = _fernet()
    if fernet is None:
        # 未配密钥：复用明文原子写（其内部已记审计），行为与加固前一致。
        atomic_write_json(path, data)
        return

    _audit("WRITE", path)
    with _lock_for(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        try:
            payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            token = fernet.encrypt(payload)
            with open(tmp, "wb") as f:
                f.write(token)
                f.flush()
                os.fsync(f.fileno())  # 刷到磁盘后才原子替换
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            raise


def load_json(path):
    """从 path 读取 JSON；文件不存在返回 None。

    有密钥：先按二进制读并经 Fernet.decrypt 解密后 json.loads；解密抛 InvalidToken
    说明是旧版明文文件，回退按明文 json.loads 读（向后兼容存量数据）。
    无密钥：直接按明文读取。
    """
    _audit("READ", path)
    if not os.path.exists(path):
        return None

    fernet = _fernet()
    if fernet is None:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 有密钥：先尝试密文，解密失败按旧版明文兼容读回。
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception:
        raise
    try:
        return json.loads(fernet.decrypt(raw).decode("utf-8"))
    except InvalidToken:
        # 容器内容不是有效 Fernet token → 极可能是加固前的明文文件，按明文读回。
        return json.loads(raw.decode("utf-8"))