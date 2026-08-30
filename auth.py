# -*- coding: utf-8 -*-
"""最小登录体系：密码哈希、账号注册与登录校验。

从 app.py 拆出。UI 渲染(render_login_gate)仍保留在 app.py，本模块只提供纯逻辑。
"""
import os
import re
import json
import hmac
import hashlib
import traceback
from datetime import datetime
from config import BASE_DATA_PATH

# 密码哈希方案：
# - 新账号使用 PBKDF2-HMAC-SHA256（带随机 salt + 高迭代次数），格式：
#     pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
# - 老账号仍为 sha256(password)（64 位十六进制），登录成功后自动迁移到新格式。
PBKDF2_ITERATIONS = 120_000
PBKDF2_SALT_BYTES = 16
PBKDF2_HASH_BYTES = 32
LEGACY_HASH_PREFIX = "sha256$"

AUTH_USERS_PATH = os.path.join(BASE_DATA_PATH, "auth_users.json")


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
    if os.path.exists(AUTH_USERS_PATH):
        try:
            with open(AUTH_USERS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            print(traceback.format_exc())
            print("[auth] 账号数据读取失败，请检查 auth_users.json")
    return {}


def save_auth_users(users):
    os.makedirs(BASE_DATA_PATH, exist_ok=True)
    # 先写临时文件再原子替换，避免写入中途崩溃损坏 auth_users.json
    tmp_path = AUTH_USERS_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, AUTH_USERS_PATH)
    except Exception:
        print(traceback.format_exc())
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


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


def authenticate_account(username: str, password: str):
    """登录校验；若为旧版 sha256 账号，登录成功后自动迁移到 PBKDF2。"""
    users = load_auth_users()
    username = (username or "").strip()
    user = users.get(username)
    if not user:
        return False
    stored_hash = user.get("password_hash") or ""
    raw_password = password or ""
    if not _verify_password(stored_hash, raw_password):
        return False
    # 老账号迁移到新哈希格式
    if _is_legacy_hash(stored_hash):
        try:
            user["password_hash"] = hash_password(raw_password)
            user["migrated_to_pbkdf2_at"] = datetime.now().isoformat()
            save_auth_users(users)
        except Exception:
            print(traceback.format_exc())
    return True