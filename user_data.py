# -*- coding: utf-8 -*-
"""多用户档案与用药记录持久化。

从 app.py 拆出：负责家庭成员档案(profile_*.json)与用药计划(med_log_*.json)的读写，
以及多用户列表枚举。不依赖 Streamlit。
"""
import os
import json
import traceback
from config import BASE_DATA_PATH


def get_safe_user_id(user_id):
    if not user_id:
        return "default"
    return "".join([c for c in str(user_id) if c.isalnum() or c in '-_']) or "default"


def get_user_profile_path(user_id):
    safe_id = get_safe_user_id(user_id)
    return os.path.join(BASE_DATA_PATH, f"profile_{safe_id}.json")


def get_user_med_log_path(user_id):
    safe_id = get_safe_user_id(user_id)
    return os.path.join(BASE_DATA_PATH, f"med_log_{safe_id}.json")


def get_chat_history_path(username):
    safe_name = get_safe_user_id(username)
    return os.path.join(BASE_DATA_PATH, f"chat_history_{safe_name}.json")


def _atomic_write_json(path, data):
    """原子写 JSON：先写同目录临时文件，再 os.replace 覆盖。

    改写了原来直接 open('w') 覆盖的方案：中途崩溃/断电不会留下半截损坏文件
    （P1-22），与 auth.py 的 tmp+replace 落盘方式保持一致。
    """
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())  # 刷到磁盘，进一步降低断电丢数据
    os.replace(tmp, path)


def load_user_profile(user_id="default"):
    path = get_user_profile_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            print(f"[load_user_profile] 读取失败 path={path}")
            print(traceback.format_exc())
    return {"age": "", "gender": "未知", "allergies": "", "chronic_diseases": "", "current_medications": ""}


def save_user_profile(profile, user_id="default"):
    path = get_user_profile_path(user_id)
    try:
        _atomic_write_json(path, profile)
    except Exception:
        print(f"[save_user_profile] 写入失败 path={path}")
        print(traceback.format_exc())
        raise


def load_medication_data(user_id="default"):
    path = get_user_med_log_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            print(f"[load_medication_data] 读取失败 path={path}")
            print(traceback.format_exc())
    return {"plans": [], "logs": {}}


def save_medication_data(data, user_id="default"):
    path = get_user_med_log_path(user_id)
    _atomic_write_json(path, data)


def load_chat_history(username, greeting=None, max_rounds=30):
    """按登录账号加载最近 max_rounds 轮对话历史（每轮含用户+助手两条）。"""
    path = get_chat_history_path(username)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            messages = data.get("messages", [])
            if messages:
                return messages[-(max_rounds * 2):]
        except Exception:
            print(f"[load_chat_history] 读取失败 path={path}")
            print(traceback.format_exc())
    return [{"role": "assistant", "content": greeting}] if greeting else []


def save_chat_history(messages, username, max_rounds=30, greeting=None):
    """按登录账号保存最近 max_rounds 轮对话历史，自动过滤开场白。"""
    path = get_chat_history_path(username)
    try:
        filtered = [
            msg for msg in messages
            if not (greeting and msg.get("role") == "assistant" and msg.get("content") == greeting)
        ]
        trimmed = filtered[-(max_rounds * 2):]
        _atomic_write_json(path, {"messages": trimmed})
    except Exception:
        print(f"[save_chat_history] 写入失败 path={path}")
        print(traceback.format_exc())


def clear_chat_history(username, greeting=None):
    """清空某账号的对话历史文件，并返回仅含开场白的消息列表。"""
    path = get_chat_history_path(username)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            print(f"[clear_chat_history] 删除失败 path={path}")
            print(traceback.format_exc())
    return [{"role": "assistant", "content": greeting}] if greeting else []


def get_all_users():
    users = ["default"]
    if not os.path.exists(BASE_DATA_PATH):
        return users
    for f in os.listdir(BASE_DATA_PATH):
        if f.startswith("profile_") and f.endswith(".json"):
            name = f[8:-5]
            if name not in users:
                users.append(name)
    return sorted(users)


def create_new_user(new_name):
    if not new_name or new_name.strip() == "":
        return False
    safe_name = get_safe_user_id(new_name)

    path = get_user_profile_path(safe_name)
    if not os.path.exists(path):
        save_user_profile({
            "age": "", "gender": "未知", "allergies": "",
            "chronic_diseases": "", "current_medications": ""
        }, safe_name)
        return True
    return False
