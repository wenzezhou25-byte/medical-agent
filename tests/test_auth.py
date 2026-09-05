# -*- coding: utf-8 -*-
"""auth.py 的 hash_password / _verify_password 单元测试。

纯函数测试，不触及磁盘、不依赖 .env 密钥、不联网。
"""
import os
import sys

# 保证 pytest 在本目录下运行也能导入项目根模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth


def test_hash_password_returns_pbkdf2_format():
    h = auth.hash_password("secret123")
    assert h.startswith("pbkdf2_sha256$")
    # 无前缀纯 64 位十六进制的旧版特征不应出现
    assert len(h) > 64
    # 至少包含 4 段：算法$迭代$salt$hash
    assert len(h.split("$")) == 4


def test_hash_password_salt_is_random():
    # 同名密码两次哈希结果应不同（随机 salt）
    h1 = auth.hash_password("secret123")
    h2 = auth.hash_password("secret123")
    assert h1 != h2


def test_verify_pbkdf2_correct_password_returns_true():
    h = auth.hash_password("secret123")
    assert auth._verify_password(h, "secret123") is True


def test_verify_pbkdf2_wrong_password_returns_false():
    h = auth.hash_password("secret123")
    assert auth._verify_password(h, "wrongpass") is False


def test_verify_empty_stored_hash_returns_false():
    assert auth._verify_password("", "secret123") is False
    assert auth._verify_password(None, "secret123") is False


def test_verify_legacy_sha256_hash(monkeypatch):
    """旧版 64 位十六进制哈希（无前缀）应能正确校验。"""
    legacy = auth._legacy_sha256_hash("oldpass")
    assert len(legacy) == 64
    assert all(c in "0123456789abcdef" for c in legacy)
    assert auth._verify_password(legacy, "oldpass") is True
    assert auth._verify_password(legacy, "badpass") is False


def test_verify_legacy_sha256_with_prefix():
    """旧版哈希带 sha256$ 前缀时也应兼容通过。"""
    legacy = auth._legacy_sha256_hash("oldpass")
    prefixed = auth.LEGACY_HASH_PREFIX + legacy
    assert auth._verify_password(prefixed, "oldpass") is True
    assert auth._verify_password(prefixed, "badpass") is False


def test_verify_unrecognized_format_returns_false():
    # 既不是 pbkdf2 也不是 64 位十六进制
    assert auth._verify_password("md5$abc123", "secret123") is False
    assert auth._verify_password("123", "secret123") is False