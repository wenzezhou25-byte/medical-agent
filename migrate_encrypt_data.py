# -*- coding: utf-8 -*-
"""一次性迁移：把 data/ 下存量明文用户数据加密落盘。

只处理含个人健康信息的用户数据文件（auth_users / chat_history / profile / med_log），
不碰数据集源文件（chip-2025-raw.json / family_drugs.json）。
幂等：已加密的文件自动跳过。运行前自动备份明文到 backup_before_encrypt_<时间戳>/。

运行：
    D:\\ananconda3\\envs\\medical_agent\\python.exe migrate_encrypt_data.py
"""
import os
import sys
import json
import shutil
import glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BASE_DATA_PATH
from storage_io import load_json, atomic_write_json_encrypted, _fernet

# 只匹配用户数据文件；数据集源文件不在此列，不会被误加密。
PATTERNS = ["auth_users.json", "chat_history_*.json", "profile_*.json", "med_log_*.json"]


def _is_plaintext(path):
    """明文 JSON 能被 json.loads 解析，Fernet 密文不能——以此判断是否还需迁移。"""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        json.loads(raw.decode("utf-8"))
        return True
    except Exception:
        return False


def main():
    if _fernet() is None:
        print("❌ 未配置 DATA_ENC_KEY，加密未生效，中止。请先配置 .env 里的 DATA_ENC_KEY。")
        sys.exit(1)

    targets = sorted({p for pat in PATTERNS for p in glob.glob(os.path.join(BASE_DATA_PATH, pat))})
    if not targets:
        print("没有找到需要迁移的文件。")
        return

    backup_dir = os.path.join(
        BASE_DATA_PATH, f"backup_before_encrypt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(backup_dir, exist_ok=True)

    migrated = already = skipped = 0
    for path in targets:
        fn = os.path.basename(path)
        if not _is_plaintext(path):
            print(f"  [已是密文] {fn}")
            already += 1
            continue
        shutil.copy2(path, os.path.join(backup_dir, fn))  # 先备份明文
        data = load_json(path)
        if data is None:
            print(f"  [警告] {fn} 内容为空，跳过")
            skipped += 1
            continue
        atomic_write_json_encrypted(path, data)
        print(f"  [已加密] {fn}")
        migrated += 1

    print(f"\n✅ 完成：加密 {migrated} 个，已是密文 {already} 个，跳过 {skipped} 个。")
    print(f"明文备份在：{backup_dir}")
    print("确认一切正常后可删除该备份目录。")


if __name__ == "__main__":
    main()
