# -*- coding: utf-8 -*-
"""storage_io.py 的加密读写与向后兼容测试。

覆盖：
- 未配置 DATA_ENC_KEY 时明文读写且可读回；
- 配置 DATA_ENC_KEY 后新写入为密文（非明文 JSON），且能被 load_json 读回；
- 已存在的明文旧文件在配置密钥后仍能兼容读回（向后兼容存量数据）。
"""
import os
import sys
import json

# 保证 pytest 在本目录下运行也能导入项目根模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cryptography.fernet import Fernet

import storage_io


def _write_plain(tmp_path, name, value):
    """手工写入明文 JSON 文件，模拟加固前的存量数据。"""
    p = tmp_path / name
    p.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return str(p)


def test_load_json_missing_file_returns_none(tmp_path):
    assert storage_io.load_json(str(tmp_path / "nope.json")) is None


def test_load_legacy_plaintext_when_key_configured(tmp_path, monkeypatch):
    """配了密钥后，已存在的明文旧文件仍应能被正常读回（向后兼容）。"""
    monkeypatch.delenv("DATA_ENC_KEY", raising=False)
    path = _write_plain(tmp_path, "profile_爸爸.json", {"age": "60", "allergies": "青霉素"})
    # 切换到有密钥
    monkeypatch.setenv("DATA_ENC_KEY", Fernet.generate_key().decode())

    data = storage_io.load_json(path)
    assert data == {"age": "60", "allergies": "青霉素"}


def test_key_configured_writes_ciphertext_and_roundtrip(tmp_path, monkeypatch):
    """配密钥后：写入的是密文（非法 JSON 明文），且 load_json 能正确读回。"""
    monkeypatch.setenv("DATA_ENC_KEY", Fernet.generate_key().decode())
    path = str(tmp_path / "profile_x.json")
    storage_io.atomic_write_json_encrypted(path, {"age": "30", "gender": "男"})

    raw = tmp_path / "profile_x.json"
    assert raw.exists()
    assert "age" not in raw.read_text(encoding="utf-8")  # 非明文 JSON
    # 但用 JSON 明文解析会失败，进一步确认是密文
    with open(path, "rb") as f:
        raw_bytes = f.read()
    assert isinstance(Fernet(os.environ["DATA_ENC_KEY"].encode()).decrypt(raw_bytes), bytes)

    assert storage_io.load_json(path) == {"age": "30", "gender": "男"}


def test_no_key_plaintext_roundtrip(tmp_path, monkeypatch):
    """未配密钥时读写为明文，且能正常读回（与加固前行为一致）。"""
    monkeypatch.delenv("DATA_ENC_KEY", raising=False)
    path = tmp_path / "profile_y.json"
    storage_io.atomic_write_json_encrypted(str(path), {"age": "40"})

    assert path.read_text(encoding="utf-8").startswith("{")  # 明文 JSON
    assert storage_io.load_json(str(path)) == {"age": "40"}