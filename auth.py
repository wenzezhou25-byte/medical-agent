# -*- coding: utf-8 -*-
"""最小登录体系：密码哈希、账号注册与登录校验。

从 app.py 拆出。UI 渲染(render_login_gate)仍保留在 app.py，本模块只提供纯逻辑。
"""
import os
import re
import hmac
import hashlib
import traceback
from datetime import datetime, timedelta
from config import BASE_DATA_PATH
from storage_io import load_json, atomic_write_json_encrypted

# 密码哈希方案：
# - 新账号使用 PBKDF2-HMAC-SHA256（带随机 salt + 高迭代次数），格式：
#     pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
# - 老账号仍为 sha256(password)（64 位十六进制），登录成功后自动迁移到新格式。
PBKDF2_ITERATIONS = 120_000
PBKDF2_SALT_BYTES = 16
PBKDF2_HASH_BYTES = 32
LEGACY_HASH_PREFIX = "sha256$"

AUTH_USERS_PATH = os.path.join(BASE_DATA_PATH, "auth_users.json")

# 连续失败锁定策略（P1-21，本地应用，偏保守）
MAX_FAILED_ATTEMPTS = 5      # 连续失败次数达到该值即锁定
LOCKOUT_MINUTES = 15         # 锁定时长
LOGIN_SESSION_HOURS = 12     # 单次登录会话有效期


def _legacy_sha256_hash(raw_password: str) -> str:
    """旧版 sha256(password)，仅用于兼容已存在的 auth_users.json。"""
    return hashlib.sha256(raw_password.encode("utf-8")).hexdigest()


def _new_pbkdf2_hash(raw_password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    """新版带 salt 的 PBKDF2-HMAC-SHA256。"""
    salt = os.urandom(PBKDF2_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        raw_password.encode("utf-8"),
        salt,
        iterations,
        dklen=PBKDF2_HASH_BYTES,
    )
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def _verify_pbkdf2(stored_hash: str, raw_password: str) -> bool:
    try:
        algo, iter_str, salt_hex, hash_hex = stored_hash.split("$", 3)
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    try:
        iterations = int(iter_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        raw_password.encode("utf-8"),
        salt,
        iterations,
        dklen=len(expected),
    )
    # 使用常量时间比较，避免时序侧信道
    return hmac.compare_digest(digest, expected)


def hash_password(raw_password: str) -> str:
    """对外暴露的统一入口，新密码一律使用 PBKDF2。"""
    return _new_pbkdf2_hash(raw_password)


def _verify_password(stored_hash: str, raw_password: str) -> bool:
    """统一密码校验：支持新版 PBKDF2 与旧版 sha256。"""
    if not stored_hash:
        return False
    if stored_hash.startswith("pbkdf2_sha256$"):
        return _verify_pbkdf2(stored_hash, raw_password)
    # 旧版 sha256：64 位十六进制（可能带 sha256$ 前缀，也可能没有）
    legacy_value = stored_hash[len(LEGACY_HASH_PREFIX):] if stored_hash.startswith(LEGACY_HASH_PREFIX) else stored_hash
    if len(legacy_value) == 64:
        try:
            return hmac.compare_digest(legacy_value, _legacy_sha256_hash(raw_password))
        except Exception:
            print(traceback.format_exc())
            return False
    return False


def _is_legacy_hash(stored_hash: str) -> bool:
    if not stored_hash:
        return False
    if stored_hash.startswith("pbkdf2_sha256$"):
        return False
    legacy_value = stored_hash[len(LEGACY_HASH_PREFIX):] if stored_hash.startswith(LEGACY_HASH_PREFIX) else stored_hash
    return len(legacy_value) == 64


def load_auth_users():
    try:
        data = load_json(AUTH_USERS_PATH)
    except Exception:
        print(traceback.format_exc())
        print("[auth] 账号数据读取失败，请检查 auth_users.json")
        return {}
    return data or {}


def save_auth_users(users):
    # 原子写入（统一走 storage_io，与 user_data 落盘一致）；配置密钥时加密落盘。
    atomic_write_json_encrypted(AUTH_USERS_PATH, users)


def register_account(username: str, password: str):
    username = (username or "").strip()
    password = (password or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,20}", username):
        return False, "账号需为3-20位，仅支持字母、数字、下划线或中划线。"
    if len(password) < 6:
        return False, "密码至少6位。"
    users = load_auth_users()
    if username in users:
        return False, "账号已存在。"
    users[username] = {"password_hash": hash_password(password), "created_at": datetime.now().isoformat()}
    save_auth_users(users)
    return True, "注册成功，请登录。"


def login_session_is_valid(login_ts) -> bool:
    """判断一次登录时间戳是否仍在会话有效期内（P1-21）。"""
    if not login_ts:
        return False
    try:
        login_time = float(login_ts)
    except (TypeError, ValueError):
        return False
    return (datetime.now().timestamp() - login_time) < LOGIN_SESSION_HOURS * 3600


def authenticate_account(username: str, password: str):
    """登录校验（P1-21 加固）：

    - 连续失败锁定：达到 MAX_FAILED_ATTEMPTS 次即锁定 LOCKOUT_MINUTES 分钟
      （计数/锁定期存于 auth_users.json，key=username 的 user 记录）。
    - 校验成功清除失败计数与锁定。
    - 旧版 sha256 账号登录成功后自动迁移到 PBKDF2。

    返回 (bool, message)。
    """
    users = load_auth_users()
    username = (username or "").strip()
    now = datetime.now()
    user = users.get(username)

    # 账号不存在：不记失败数，避免通过提示差异暴露账号是否存在
    if not user:
        return False, "账号或密码错误"

    # 已锁定：直接拒绝
    locked_until = user.get("locked_until")
    if locked_until:
        try:
            if datetime.fromisoformat(locked_until) > now:
                return False, f"连续 {MAX_FAILED_ATTEMPTS} 次失败，账号已临时锁定，约 {LOCKOUT_MINUTES} 分钟内无法登录。"
        except ValueError:
            pass  # 非法时间戳：不阻断，继续正常校验

    stored_hash = user.get("password_hash") or ""
    raw_password = password or ""
    if not _verify_password(stored_hash, raw_password):
        attempts = int(user.get("failed_attempts") or 0) + 1
        user["failed_attempts"] = attempts
        if attempts >= MAX_FAILED_ATTEMPTS:
            user["locked_until"] = (now + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
            msg = f"连续 {MAX_FAILED_ATTEMPTS} 次失败，账号已锁定 {LOCKOUT_MINUTES} 分钟，请稍后再试。"
        else:
            msg = "账号或密码错误"
        try:
            save_auth_users(users)
        except Exception:
            print(traceback.format_exc())
        return False, msg

    # 登录成功：清除锁定与失败计数
    if user.get("failed_attempts") or user.get("locked_until"):
        user.pop("failed_attempts", None)
        user.pop("locked_until", None)
    # 老账号迁移到新哈希格式
    if _is_legacy_hash(stored_hash):
        try:
            user["password_hash"] = hash_password(raw_password)
            user["migrated_to_pbkdf2_at"] = datetime.now().isoformat()
        except Exception:
            print(traceback.format_exc())
    try:
        save_auth_users(users)
    except Exception:
        print(traceback.format_exc())
    return True, "登录成功"