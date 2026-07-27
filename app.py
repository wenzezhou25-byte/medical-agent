import sys
import os
import locale
import requests
import json
import traceback
import tempfile
import shutil
import hashlib
import hmac
from pathlib import Path
import re
import urllib.parse
import time
from datetime import datetime, timedelta
from config import BASE_DATA_PATH, GAODE_MAP_KEY, VECTOR_STORE_PATH, get_required_env
from embedding_provider import get_embeddings
from rag_utils import (
    build_structured_documents,
    create_hybrid_retriever,
    format_docs_for_prompt,
    retrieve_evidence_docs,
)

# ================= 导入联网搜索工具 =================
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableLambda


# ================= 编码修复 =================
if sys.platform == "win32":
    if 'streamlit' not in sys.modules:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ["PYTHONIOENCODING"] = "utf-8"

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader

# 页面配置需在首个 Streamlit UI 调用前设置
st.set_page_config(page_title="🏥 智能医疗助手 (家庭版)", layout="wide", page_icon="🩺")
DEFAULT_GREETING = "您好，请问有需要帮助的吗？"
AUTH_USERS_PATH = os.path.join(BASE_DATA_PATH, "auth_users.json")

# ================= 自定义 CSS 样式 =================
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 1rem; text-align: left; }
    .sub-title { font-size: 1.1rem; color: #6B7280; margin-bottom: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ================= 🆕 多用户管理工具函数 =================

def get_safe_user_id(user_id):
    if not user_id: return "default"
    return "".join([c for c in str(user_id) if c.isalnum() or c in '-_']) or "default"


def get_user_profile_path(user_id):
    safe_id = get_safe_user_id(user_id)
    return os.path.join(BASE_DATA_PATH, f"profile_{safe_id}.json")


def get_user_med_log_path(user_id):
    safe_id = get_safe_user_id(user_id)
    return os.path.join(BASE_DATA_PATH, f"med_log_{safe_id}.json")


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
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
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
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


# ================= 🔐 最小登录体系 =================

# 密码哈希方案：
# - 新账号使用 PBKDF2-HMAC-SHA256（带随机 salt + 高迭代次数），格式：
#     pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
# - 老账号仍为 sha256(password)（64 位十六进制），登录成功后自动迁移到新格式。
PBKDF2_ITERATIONS = 120_000
PBKDF2_SALT_BYTES = 16
PBKDF2_HASH_BYTES = 32
LEGACY_HASH_PREFIX = "sha256$"


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
            st.error("❌ 账号数据读取失败，请联系管理员。")
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


def render_login_gate():
    st.markdown("## 🔐 账号登录")
    st.caption("为保护家庭健康档案，需登录后使用。")
    tab_login, tab_register = st.tabs(["登录", "注册"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("账号", placeholder="请输入账号")
            password = st.text_input("密码", type="password", placeholder="请输入密码")
            submitted = st.form_submit_button("登录", type="primary", use_container_width=True)
            if submitted:
                if authenticate_account(username, password):
                    st.session_state.is_authenticated = True
                    st.session_state.auth_username = username.strip()
                    st.success("✅ 登录成功")
                    st.rerun()
                else:
                    st.error("❌ 账号或密码错误")

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("新账号", placeholder="3-20位字母/数字/_/-")
            new_password = st.text_input("新密码", type="password", placeholder="至少6位")
            register_submit = st.form_submit_button("注册账号", use_container_width=True)
            if register_submit:
                ok, msg = register_account(new_username, new_password)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.warning(f"⚠️ {msg}")


# ================= 🛡️ 新增：药物冲突检测工具 =================

def check_drug_interaction(new_drug_name, existing_drugs_list, vectorstore):
    """
    检查新药与现有药物列表是否存在冲突
    返回: (has_conflict, conflict_details)
    """
    if not vectorstore or not new_drug_name.strip():
        return False, []

    clean_new_drug = new_drug_name.strip()
    conflicts = []

    # 高风险关键词
    risk_keywords = ["禁忌", "禁止", "不宜", "避免", "冲突", "严重", "出血", "中毒", "不良反应", "拮抗", "禁用"]

    # 遍历现有药物进行两两检查
    for old_drug in existing_drugs_list:
        clean_old_drug = old_drug.strip()
        if not clean_old_drug or clean_old_drug == clean_new_drug:
            continue

        query = f"{clean_new_drug} 和 {clean_old_drug} 一起服用有什么禁忌或相互作用？能同时吃吗？"

        try:
            retriever = create_hybrid_retriever(vectorstore, vector_k=5, bm25_k=6, vector_weight=0.6, bm25_weight=0.4)
            docs = retrieve_evidence_docs(retriever, query, top_k=3)

            # 分析检索结果
            for doc in docs:
                content_cn = doc.page_content

                # 简单关键词匹配逻辑
                found_risk = False
                matched_keyword = ""
                for kw in risk_keywords:
                    if kw in content_cn:
                        found_risk = True
                        matched_keyword = kw
                        break

                if found_risk:
                    conflicts.append({
                        "drug_pair": f"{clean_new_drug} + {clean_old_drug}",
                        "risk_keyword": matched_keyword,
                        "evidence": content_cn[:200] + "..."
                    })
                    break  # 找到一个证据就停止对该药对的检索，避免重复
        except Exception as e:
            print(f"检测 {clean_new_drug} 和 {clean_old_drug} 时出错：{e}")
            continue

    return len(conflicts) > 0, conflicts


# ================= 用户创建逻辑 =================

def create_new_user(new_name):
    if not new_name or new_name.strip() == "":
        return False
    safe_name = get_safe_user_id(new_name)
    if safe_name == "default":
        if os.path.exists(get_user_profile_path("default")):
            pass

    path = get_user_profile_path(safe_name)
    if not os.path.exists(path):
        save_user_profile({
            "age": "", "gender": "未知", "allergies": "",
            "chronic_diseases": "", "current_medications": ""
        }, safe_name)
        return True
    return False


# ================= 原有工具函数 (时间/地图/清洗) =================

def get_today_date_str():
    return datetime.now().strftime("%Y-%m-%d")


def get_current_time_str():
    return datetime.now().strftime("%H:%M")


def is_time_to_take(scheduled_time_str, window_minutes=30):
    now = datetime.now()
    try:
        scheduled = datetime.strptime(scheduled_time_str, "%H:%M").time()
        scheduled_dt = datetime.combine(now.date(), scheduled)
        start_window = scheduled_dt - timedelta(minutes=window_minutes)
        end_window = scheduled_dt + timedelta(minutes=window_minutes)
        return start_window <= now <= end_window
    except Exception:
        print(f"[is_time_to_take] 时间解析失败 value={scheduled_time_str!r}")
        print(traceback.format_exc())
        return False


def clean_text_content(text):
    if not text:
        return ""

    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("【", "[").replace("】", "]")
    text = re.sub(r"([\u4e00-\u9fa5])\1{2,}", r"\1", text)
    text = re.sub(r"([,\.!?;:，。！？；：])\1+", r"\1", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\b(\d+)\s+\1\b", r"\1", text)
    text = text.replace("注意事项注意事项", "注意事项")
    text = text.replace("禁忌禁忌", "禁忌")
    text = text.replace("用法用量用法用量", "用法用量")
    return text.strip()


def get_route_info(origin_lat, origin_lon, dest_lat, dest_lon, api_key):
    base_url = "https://restapi.amap.com/v3/direction"
    drive_res, walk_res = "🚗 --", "🚶 --"
    drive_distance_m, walk_distance_m = None, None
    try:
        d_params = {"origin": f"{origin_lon},{origin_lat}", "destination": f"{dest_lon},{dest_lat}", "key": api_key,
                    "extensions": "base", "output": "json", "strategy": 2}
        resp = requests.get(f"{base_url}/driving", params=d_params, timeout=8)
        data = resp.json()
        if data.get("status") == "1" and data.get("route", {}).get("paths"):
            path = data["route"]["paths"][0]
            drive_distance_m = int(path["distance"])
            drive_res = f"🚗 {round(int(path['duration']) / 60)}分 ({round(int(path['distance']) / 1000, 1)}km)"
        else:
            # 高德 API 返回失败时保留 info 以便排查（如 USERKEY_PLAT_NOMATCH / OUT_OF_SERVICE）
            info = data.get("info") or "未知错误"
            drive_res = f"🚗 不可达 ({info})"
            print(f"[get_route_info] driving 接口失败 status={data.get('status')} info={info}")
    except Exception:
        print("[get_route_info] driving 请求异常")
        print(traceback.format_exc())
    try:
        w_params = {"origin": f"{origin_lon},{origin_lat}", "destination": f"{dest_lon},{dest_lat}", "key": api_key,
                    "extensions": "base", "output": "json"}
        resp = requests.get(f"{base_url}/walking", params=w_params, timeout=8)
        data = resp.json()
        if data.get("status") == "1" and data.get("route", {}).get("paths"):
            path = data["route"]["paths"][0]
            walk_distance_m = int(path["distance"])
            walk_res = f"🚶 {round(int(path['duration']) / 60)}分 ({round(int(path['distance']) / 1000, 1)}km)"
        else:
            info = data.get("info") or "未知错误"
            walk_res = f"🚶 不可达 ({info})"
            print(f"[get_route_info] walking 接口失败 status={data.get('status')} info={info}")
    except Exception:
        print("[get_route_info] walking 请求异常")
        print(traceback.format_exc())
    # 为了与高德地图默认“驾车路线”一致，排序只采用驾车距离
    route_distance_m = drive_distance_m
    return drive_res, walk_res, route_distance_m


def geocode_address(address, api_key):
    if not address:
        return None, None
    address = str(address).strip()

    # 先按 POI 文本检索定位，避免“学校/园区”等关键词被 geocode 解析到偏移点
    try:
        poi_resp = requests.get(
            "https://restapi.amap.com/v3/place/text",
            params={"keywords": address, "key": api_key, "output": "json", "offset": 1},
            timeout=5,
        )
        poi_data = poi_resp.json()
        if poi_data.get("status") == "1" and poi_data.get("pois"):
            loc = poi_data["pois"][0].get("location")
            if loc:
                lon, lat = loc.split(",")
                return float(lat), float(lon)
        else:
            print(f"[geocode_address] POI 解析失败 address={address} status={poi_data.get('status')} info={poi_data.get('info')}")
    except Exception:
        print(f"[geocode_address] POI 请求异常 address={address}")
        print(traceback.format_exc())

    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {"address": address, "key": api_key, "output": "json"}
    try:
        resp = requests.get(url, params=params, timeout=3)
        data = resp.json()
        if data.get("status") == "1" and data.get("geocodes"):
            loc = data["geocodes"][0]["location"]
            lon, lat = loc.split(",")
            return float(lat), float(lon)
        else:
            print(f"[geocode_address] geocode 接口失败 address={address} status={data.get('status')} info={data.get('info')}")
    except Exception:
        print(f"[geocode_address] geocode 请求异常 address={address}")
        print(traceback.format_exc())
    return None, None


def search_poi_candidates(keyword, api_key, limit=8):
    if not keyword:
        return []
    try:
        resp = requests.get(
            "https://restapi.amap.com/v3/place/text",
            params={"keywords": str(keyword).strip(), "key": api_key, "output": "json", "offset": limit},
            timeout=5,
        )
        data = resp.json()
        if data.get("status") != "1" or not data.get("pois"):
            if data.get("status") != "1":
                print(f"[search_poi_candidates] 接口失败 keyword={keyword} status={data.get('status')} info={data.get('info')}")
            return []
        candidates = []
        for poi in data.get("pois", [])[:limit]:
            loc = poi.get("location")
            if not loc:
                continue
            candidates.append({
                "name": poi.get("name", ""),
                "address": poi.get("address", ""),
                "location": loc,
                "id": poi.get("id", ""),
            })
        return candidates
    except Exception:
        print(f"[search_poi_candidates] 请求异常 keyword={keyword}")
        print(traceback.format_exc())
        return []


# 医疗机构类型评分：数字越大优先级越高。
# 综合医院/专科医院 > 急救中心/疾控 > 卫生服务中心 > 卫生院 > 门诊部 > 诊所
# 放在 search_nearby_hospitals 之前，便于在召回阶段提前停止并预填 _type_score 字段。
MEDICAL_TYPE_SCORE_RULES = [
    (5, ["医院"]),
    (4, ["急救中心", "急救站", "疾控", "疾病预防"]),
    (3, ["社区卫生服务中心", "社区卫生服务站", "卫生服务中心", "卫生服务站"]),
    (2, ["卫生院", "乡镇卫生院"]),
    (1, ["门诊部", "门诊", "医务室", "护理院", "护理站", "诊所"]),
]


def score_medical_institution_type(name):
    """根据机构名称给出类型评分；未命中给 0（仍参与排序，不硬过滤）。"""
    if not name:
        return 0
    name_str = str(name)
    best = 0
    for score, keywords in MEDICAL_TYPE_SCORE_RULES:
        if any(kw in name_str for kw in keywords):
            best = max(best, score)
    return best


def search_nearby_hospitals(location_query, radius=5000):
    if not GAODE_MAP_KEY: return [
        {"name": "⚠️ 未配置地图 API", "address": "", "distance": "-", "tel": "-", "location": ""}]
    try:
        query_lat, query_lon = geocode_address(location_query, GAODE_MAP_KEY)
        if query_lat is None or query_lon is None:
            return [{"name": "❌ 地址解析失败", "address": "", "distance": "-", "tel": "-", "location": ""}]
        location = f"{query_lon},{query_lat}"

        # 扩大 POI 召回：分别查询多个关键词并合并结果。
        # 不再只查“医院”，避免卫生院/社区服务中心/门诊/诊所/急救/疾控等在 API 阶段就被漏掉。
        # 关键词列表控制在 8 个以内，避免请求过多拖慢 UI。
        keywords = [
            "医院",
            "卫生院",
            "社区卫生服务中心",
            "社区卫生服务站",
            "门诊部",
            "诊所",
            "急救中心",
            "疾控中心",
        ]

        blacklist = ["酒店", "宾馆", "餐厅", "超市", "学校", "公司"]
        # whitelist 覆盖各类正规医疗机构名称特征，不因名字不含“医院”就丢弃
        whitelist = ["医院", "卫生", "诊所", "门诊", "疾控", "急救", "医务", "护理", "社区卫生服务"]

        def _parse_distance(d_str):
            try:
                d_str = str(d_str).strip()
                if '公里' in d_str:
                    return float(d_str.replace('公里', '')) * 1000
                elif '米' in d_str:
                    return float(d_str.replace('米', ''))
                else:
                    return float(d_str)
            except Exception:
                return 999999

        seen = set()  # 去重 key 集合：优先 POI id，否则 name+location
        hospitals = []
        soft_candidates = []  # fallback 储备：未命中 whitelist 但通过黑名单的 POI
        any_keyword_succeeded = False  # 至少一个关键词请求成功
        # 轻量提前停止阈值：
        # - 候选总数达到 50 即停止后续关键词请求（与最终返回上限一致）；
        # - 已收集到 20 个“高质量”机构（类型评分 >= 3，即医院/急救疾控/社区卫生服务中心）
        #   也停止，避免在已经足够好时继续打 5 秒超时的网络请求。
        HARD_COUNT_LIMIT = 50
        HIGH_QUALITY_THRESHOLD = 20
        high_quality_count = 0

        for kw in keywords:
            try:
                search_resp = requests.get("https://restapi.amap.com/v3/place/around", params={
                    "location": location, "keywords": kw,
                    "radius": radius, "key": GAODE_MAP_KEY, "output": "json", "offset": 50
                }, timeout=5)
                search_data = search_resp.json()
                if search_data.get("status") != "1":
                    info = search_data.get("info") or "未知错误"
                    print(f"[search_nearby_hospitals] keyword={kw} 接口失败 status={search_data.get('status')} info={info}")
                    continue
                any_keyword_succeeded = True
                pois = search_data.get("pois") or []
                for poi in pois:
                    name = poi.get("name", "")
                    if not name:
                        continue
                    if any(word in name for word in blacklist):
                        continue
                    poi_location = poi.get("location", "")
                    if not poi_location:
                        continue
                    # 去重：有 id 用 id，否则用 name+location
                    poi_id = poi.get("id", "")
                    dedupe_key = f"id:{poi_id}" if poi_id else f"nl:{name}|{poi_location}"
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)

                    # 预计算类型评分，避免 tab2 重复计算；soft_candidates 也保留以备 fallback 后再评分
                    type_score = score_medical_institution_type(name)
                    item = {
                        "name": name,
                        "address": poi.get("address", ""),
                        "distance": poi.get("distance", ""),
                        "tel": poi.get("tel", ""),
                        "location": poi_location,
                        "_matched_keyword": kw,  # 仅用于调试，不影响 UI
                        "_type_score": type_score,
                    }
                    if any(word in name for word in whitelist):
                        hospitals.append(item)
                        if type_score >= 3:
                            high_quality_count += 1
                    else:
                        soft_candidates.append(item)
            except Exception:
                print(f"[search_nearby_hospitals] keyword={kw} 请求异常")
                print(traceback.format_exc())
                continue

            # 轻量提前停止：单关键词处理完毕后再判断，避免在关键词中途截断造成统计不一致
            if len(hospitals) >= HARD_COUNT_LIMIT or high_quality_count >= HIGH_QUALITY_THRESHOLD:
                print(
                    f"[search_nearby_hospitals] 提前停止 keyword={kw} "
                    f"collected={len(hospitals)} high_quality={high_quality_count}"
                )
                break

        # 所有关键词都失败时才返回错误提示；部分成功则返回成功结果
        if not any_keyword_succeeded:
            return [{"name": "❌ 高德接口全部请求失败", "address": "", "distance": "-", "tel": "-", "location": ""}]

        # fallback：strict whitelist 无结果时，回退到医疗专有词命中的软候选。
        # 不使用单字 "室/站/所/医" —— 会误匹配加油站(站)/派出所(所)/办公室(室)。
        # 真正的医疗"室/站/所"（卫生室/医务室/急救站）已包含 whitelist 中的
        # "卫生/医务/急救"，不会进 soft_candidates；这里补 whitelist 未覆盖的
        # 眼科/口腔/康复/体检/药房/疗养/妇产/心理/精神等医疗专有词。
        if not hospitals and soft_candidates:
            medical_fallback_kw = [
                "药房", "药店", "疗养", "康复", "体检",
                "口腔", "眼科", "妇产", "精神", "心理",
            ]
            for item in soft_candidates:
                nm = item["name"]
                if any(k in nm for k in medical_fallback_kw):
                    hospitals.append(item)
                if len(hospitals) >= 3:
                    break

        if not hospitals:
            return [{"name": "🔍 附近暂未找到正规医疗机构", "address": "", "distance": "-", "tel": "-", "location": ""}]

        # 限制最多 50 个候选：按 POI 距离取最近的 50 个，避免后续路线规划过多
        hospitals.sort(key=lambda x: _parse_distance(x["distance"]))
        return hospitals[:50]
    except Exception as e:
        print("[search_nearby_hospitals] 网络请求异常")
        print(traceback.format_exc())
        return [{"name": "❌ 网络请求错误", "address": str(e), "distance": "-", "tel": "-", "location": ""}]


def perform_web_search(query):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.startswith("tvly-dev-PLACEHOLDER"):
        return "⚠️ 联网搜索未配置有效 API Key。"
    try:
        search_tool = TavilySearchResults(max_results=3, search_depth="advanced", include_answer=True)
        results = search_tool.invoke(query)
        context_parts = []
        if isinstance(results, list):
            for i, res in enumerate(results):
                if isinstance(res, dict):
                    title = res.get('title', '无标题')
                    snippet = res.get('content', res.get('snippet', ''))
                    url = res.get('url', '')
                    context_parts.append(f"{i + 1}. 【{title}】: {snippet} (来源：{url})")
                else:
                    context_parts.append(str(res))
        elif isinstance(results, str):
            context_parts.append(results)
        return "【互联网最新资讯】:\n" + "\n".join(context_parts) + "\n"
    except Exception as e:
        return f"⚠️ 联网搜索出错：{str(e)}"


# ================= RAG 核心功能 =================
@st.cache_resource
def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH): return None
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception as e:
        st.error(f"❌ 加载向量库失败：{e}")
        return None


@st.cache_resource
def load_hybrid_retriever():
    vectorstore = load_vector_store()
    if not vectorstore:
        return None
    return create_hybrid_retriever(vectorstore, vector_k=8, bm25_k=10, vector_weight=0.65, bm25_weight=0.35)


def build_knowledge_base_from_upload(uploaded_files):
    """重建知识库：先写入临时目录并校验可加载，再原子替换旧 vector_store。

    任何阶段失败都会保留旧知识库；只有新索引构建并校验成功后才会替换。
    """
    if not uploaded_files: return False
    temp_dir = tempfile.mkdtemp(prefix="pdf_temp_")
    # 新索引落盘目录，构建成功后再原子替换 VECTOR_STORE_PATH；
    # 延后创建，避免函数提前返回时残留无用临时目录。
    staging_dir = None
    backup_dir = None
    try:
        with st.spinner("📂 正在保存文件..."):
            os.makedirs(BASE_DATA_PATH, exist_ok=True)
            uploaded_name_map = {}
            for idx, file in enumerate(uploaded_files):
                uploaded_name_map[idx] = file.name
                with open(os.path.join(temp_dir, f"{idx}.pdf"), "wb") as f:
                    f.write(file.getbuffer())

        with st.spinner("🔄 正在构建知识库..."):
            documents = []
            pdf_files = sorted(list(Path(temp_dir).glob("*.pdf")))
            for i, pdf_file in enumerate(pdf_files, 1):
                try:
                    loader = PyMuPDFLoader(str(pdf_file))
                    docs = loader.load()
                    for doc in docs:
                        doc.page_content = clean_text_content(doc.page_content)
                        original_name = uploaded_name_map.get(i - 1, pdf_file.name)
                        doc.metadata = {
                            "source": f"doc_{i}",
                            "source_name": original_name,
                            "page": str(doc.metadata.get("page", i)),
                        }
                    documents.extend(docs)
                except Exception as e:
                    st.warning(f"⚠️ 文件 {i} 处理失败：{e}")
                    print(f"[build_knowledge_base_from_upload] 文件 {i} 处理失败：{e}")
                    print(traceback.format_exc())

            if not documents:
                st.error("❌ 未提取到任何有效文本")
                return False

            clean_splits = build_structured_documents(documents, clean_text_content)

            if not clean_splits:
                st.error("❌ 没有有效的文本片段")
                return False

            embeddings = get_embeddings()
            with st.spinner("正在生成向量..."):
                vectorstore = FAISS.from_documents(clean_splits, embeddings)

            # 1) 先把新索引写入 staging 目录，避免污染当前 VECTOR_STORE_PATH
            staging_dir = tempfile.mkdtemp(prefix="vector_store_staging_")
            vectorstore.save_local(staging_dir)

            # 2) 校验 staging 索引可加载，避免写入损坏的索引
            with st.spinner("🧪 正在校验新知识库..."):
                try:
                    verify_embeddings = get_embeddings()
                    FAISS.load_local(
                        staging_dir,
                        verify_embeddings,
                        allow_dangerous_deserialization=True,
                    )
                except Exception:
                    st.error("❌ 新知识库校验失败，已保留旧知识库。")
                    print("[build_knowledge_base_from_upload] staging 索引校验失败：")
                    print(traceback.format_exc())
                    return False

            # 3) 备份旧 vector_store（如果存在），原子替换
            old_exists = os.path.exists(VECTOR_STORE_PATH)
            if old_exists:
                backup_dir = VECTOR_STORE_PATH + ".bak_tmp"
                # 清理可能残留的旧备份
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir, ignore_errors=True)
                try:
                    os.replace(VECTOR_STORE_PATH, backup_dir)
                except OSError:
                    # 跨盘或权限问题导致 os.replace 失败时回退到移动
                    shutil.move(VECTOR_STORE_PATH, backup_dir)

            try:
                os.replace(staging_dir, VECTOR_STORE_PATH)
                staging_dir = None  # 已成功替换，无需再清理
            except OSError:
                # 跨盘时 os.replace 可能失败，回退到移动
                shutil.move(staging_dir, VECTOR_STORE_PATH)
                staging_dir = None

            # 4) 替换成功后删除旧备份
            if backup_dir and os.path.exists(backup_dir):
                shutil.rmtree(backup_dir, ignore_errors=True)
                backup_dir = None

            st.success("✅ 知识库重建成功！")
            load_vector_store.clear()
            load_hybrid_retriever.clear()
            return True
    except Exception:
        # 任何意外异常都不应破坏现有 vector_store
        st.error("❌ 知识库构建过程出现异常，旧知识库已保留。")
        print("[build_knowledge_base_from_upload] 构建异常：")
        print(traceback.format_exc())
        # 还原旧备份：覆盖三种情况
        # - VECTOR_STORE_PATH 不存在：直接恢复
        # - VECTOR_STORE_PATH 存在但加载校验失败：删除残缺目录后恢复
        # - VECTOR_STORE_PATH 存在且可加载：新库已就绪，删除旧备份
        if backup_dir and os.path.exists(backup_dir):
            try:
                if not os.path.exists(VECTOR_STORE_PATH):
                    # 情况1：新库未落地，直接恢复旧库
                    try:
                        os.replace(backup_dir, VECTOR_STORE_PATH)
                    except OSError:
                        shutil.move(backup_dir, VECTOR_STORE_PATH)
                    backup_dir = None
                else:
                    # 校验新库是否完整可加载
                    new_loads_ok = False
                    try:
                        verify_embeddings = get_embeddings()
                        FAISS.load_local(
                            VECTOR_STORE_PATH,
                            verify_embeddings,
                            allow_dangerous_deserialization=True,
                        )
                        new_loads_ok = True
                    except Exception:
                        print("[build_knowledge_base_from_upload] 新库校验失败，回滚到旧库：")
                        print(traceback.format_exc())

                    if new_loads_ok:
                        # 情况3：新库可加载，删除旧备份
                        shutil.rmtree(backup_dir, ignore_errors=True)
                    else:
                        # 情况2：新库残缺/损坏，删除残缺目录后恢复旧库
                        shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)
                        try:
                            os.replace(backup_dir, VECTOR_STORE_PATH)
                        except OSError:
                            shutil.move(backup_dir, VECTOR_STORE_PATH)
                    backup_dir = None
            except Exception:
                # 恢复失败要给 Streamlit 错误提示并打印 traceback
                st.error("❌ 旧知识库恢复失败，请检查 vector_store 目录或联系管理员。")
                print("[build_knowledge_base_from_upload] 旧库恢复失败：")
                print(traceback.format_exc())
        return False
    finally:
        for d in (temp_dir, staging_dir):
            if d and os.path.exists(d):
                try:
                    shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    print(f"[build_knowledge_base_from_upload] 临时目录清理失败：{d}")
                    print(traceback.format_exc())


# 假设 perform_web_search 和其他导入已在文件顶部定义

def get_rag_chain(vectorstore, history="", enable_web_search=False, user_profile=None):
    # 初始化组件
    # 防御：如果知识库未加载，绝不允许调用 None.as_retriever()。
    hybrid_retriever = load_hybrid_retriever()
    if hybrid_retriever is None:
        if vectorstore is None:
            raise RuntimeError("knowledge_base_unavailable")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    else:
        retriever = hybrid_retriever
    llm = ChatTongyi(model="qwen-plus", dashscope_api_key=get_required_env("DASHSCOPE_API_KEY"))

    # --- 1. 预处理用户档案 (Python 层完成) ---
    profile_text = "无特定用户档案信息。"
    if user_profile:
        p_parts = []
        if user_profile.get('age'): p_parts.append(f"年龄：{user_profile['age']}岁")
        if user_profile.get('gender') and user_profile['gender'] != '未知':
            p_parts.append(f"性别：{user_profile['gender']}")
        if user_profile.get('allergies'):
            p_parts.append(f"⚠️ 过敏史：{user_profile['allergies']}")
        if user_profile.get('chronic_diseases'):
            p_parts.append(f"🏥 慢性病：{user_profile['chronic_diseases']}")
        if user_profile.get('current_medications'):
            p_parts.append(f"💊 正在服药：{user_profile['current_medications']}")

        if p_parts:
            # 将档案整合为一段明确的指令文本
            profile_text = (
                    "### 👤 用户个人档案 (必须优先参考)\n" +
                    "\n".join(p_parts) +
                    "\n\n⚠️ **重要约束**: 若药物与上述档案（如过敏、慢性病）冲突，必须在回答第一段发出🚨高危警示！"
            )

    # --- 2. 构建 Prompt 模板 ---
    # 注意：
    # 1. 外层使用 f""" 以便插入 {profile_text}
    # 2. 所有 LangChain 动态变量必须使用 {{variable}} (双花括号)
    # 3. 删除了原代码中错误的 {user_profile} 引用，因为已通过 profile_text 注入
    template = f"""你是一名拥有20年临床经验的**资深执业药师与全科医生**。你的任务是基于【用户档案】和【检索到的本地知识库证据】，为用户提供**明确、可追溯**的医疗与用药指导。

{profile_text}

### ⛔ 核心铁律 (违反即失败)
1. **本地证据优先**：所有结论必须基于下方【本地知识库片段】。本地知识库与【互联网最新资讯】冲突时，**一律以本地说明书/指南为准**，互联网内容仅作为补充背景，不得覆盖本地依据。
2. **强制四分类结论**：回答**第一行**必须是以下四者之一，且用方括号包裹，不得加任何前缀或解释：
   - 【可以按说明书使用】
   - 【不建议自行使用】
   - 【禁止/避免使用】
   - 【当前知识库无法判断】
3. **四分类判定规则**：
   - 【可以按说明书使用】：本地知识库明确支持该用法，且未检索到禁忌/冲突/特殊人群风险。
   - 【不建议自行使用】：涉及孕妇、哺乳期、儿童、老人、慢病、过敏、合并用药、剂量调整，或本地证据出现“慎用、医生指导、遵医嘱、注意事项”。
   - 【禁止/避免使用】：本地证据明确出现“禁用、禁忌、禁止、避免、不得、不宜”等强风险表达。
   - 【当前知识库无法判断】：本地知识库未检索到足够证据。此时必须在【依据】小节说明缺少哪类证据（如“未检索到儿童用药依据”“未检索到两药相互作用依据”）。
4. **证据充分时必须明确判断**：不得为了安全而回避。证据支持“可用”就明确说“可以按说明书使用”，证据支持“禁用”就明确说“禁止/避免使用”。
5. **禁止模糊回避**：
   - 禁止只回答“建议咨询医生”作为唯一结论。
   - 证据充分时禁止使用“可能、也许、大概、一般来说”等模糊词。
   - “建议咨询医生”只能出现在【下一步建议】中作为具体行动，不能作为结论本身。
6. **强制引用来源**：涉及**剂量、频次、给药途径、疗程、禁忌、慎用、孕妇/哺乳期/儿童/老人用药、药物相互作用、严重不良反应**等内容时，**必须**在相应条目后标注来源，格式为 `[来源：文件名 / 第X页 / 章节名]`。
7. **数字零容忍**：涉及剂量、时间、年龄时，**必须逐字摘录原文数字**，严禁模糊化（如“一次1片，一日3次”不得写成“按说明服用”）。
8. **免责声明位置**：医疗免责声明只能放在回答**最后**，不能放开头或结论中。
9. **针对性过滤**：只回答与用户问题强相关的内容。若用户未问及“贮藏”，则不要主动罗列“放在儿童不能接触的地方”等通用废话，除非该药有特殊贮藏要求（如冷藏）。

### 🧠 思考与合成流程
1. **阅读**：仔细阅读所有本地检索片段，识别哪些片段真正回答了用户问题。
2. **证据分级**：判断证据是否充分支持“可用/不建议/禁止”三分类之一；若不充分，明确缺什么。
3. **去重合并**：识别重复信息，合并为通顺段落。
4. **结论先行**：先给四分类结论，再给依据，最后给行动建议。

### 📝 输出格式（必须严格按此顺序，缺失或乱序即失败）
第一行：四分类结论之一（如【可以按说明书使用】），不要加任何前缀。

### 直接结论
1-2 句话直接回答用户问题，避免绕圈。与第一行结论保持一致。

### 依据
列出支持结论的本地证据，每条带 `[来源：文件名 / 第X页 / 章节名]`。若结论为【当前知识库无法判断】，必须在此说明缺少哪类证据、不能判断的具体原因。

### 风险点
只列与问题直接相关的禁忌、慎用、不良反应、特殊人群风险或相互作用。没有检索到则写“当前知识库未检索到相关风险依据”。

### 下一步建议
给具体行动，不要泛泛说“咨询医生”。例如：
- “按说明书剂量服用”
- “不要自行合用，带药盒咨询医生/药师”
- “若出现皮疹、呼吸困难等症状立即就医”
- “补充年龄、孕哺状态、正在服用药物后再判断”

### 证据来源
列出本次回答主要依据的 1-3 条证据（文件名 / 页码 / 章节）。

### 医疗免责声明
本回答仅基于当前本地知识库和用户提供信息，不构成正式诊断；具体用药请遵医嘱或咨询执业药师。

---
### 📥 动态输入数据
【对话历史】: {{history}}
【本地知识库片段】 (请综合归纳以下内容，不要照抄):
{{context}}
【互联网最新资讯】 (仅作补充，不得覆盖本地依据):
{{web_context}}
【用户最新问题】: {{question}}

---
### 🗣️ 请开始综合回答 (第一行必须是四分类结论):
"""

    # 创建 Prompt 对象
    prompt = PromptTemplate.from_template(template)

    # --- 3. 定义辅助函数 ---
    def create_chain_inputs(inputs):
        question = inputs["question"]
        history = inputs.get("history", "")

        # 执行检索
        retrieval_start = time.perf_counter()
        docs = retrieve_evidence_docs(retriever, question, top_k=7)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        local_context = format_docs_for_prompt(docs)

        # 执行联网搜索 (如果开启)
        web_context = "无互联网搜索内容。"
        web_ms = 0.0
        if inputs.get("enable_web_search", False):
            web_start = time.perf_counter()
            try:
                web_context = perform_web_search(question)
            except Exception as e:
                web_context = f"联网搜索失败：{str(e)}"
                print(f"[get_rag_chain] 联网搜索失败：{e}")
                print(traceback.format_exc())
            web_ms = (time.perf_counter() - web_start) * 1000

        st.session_state["_last_rag_metrics"] = {
            "retrieval_ms": retrieval_ms,
            "web_ms": web_ms,
            "retrieved_docs_count": len(docs),
            "context_chars": len(local_context),
        }

        # 返回字典，键名必须与 Prompt 中的 {{key}} 对应
        return {
            "context": local_context,
            "web_context": web_context,
            "history": history,
            "question": question
        }

    # --- 4. 组装链 ---
    pre_process = RunnableLambda(create_chain_inputs)

    # 链式调用：预处理 -> Prompt填充 -> LLM生成 -> 解析字符串
    return pre_process | prompt | llm | StrOutputParser()


# ================= 界面布局 =================
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
if "auth_username" not in st.session_state:
    st.session_state.auth_username = ""
if not st.session_state.is_authenticated:
    render_login_gate()
    st.stop()


col_title, col_logo = st.columns([4, 1])
with col_title:
    st.markdown('<h1 class="main-title">🩺 智能医疗知识库 + 家庭健康档案</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.caption(f"已登录账号：`{st.session_state.auth_username}`")
    if st.button("🚪 退出登录", use_container_width=True):
        st.session_state.is_authenticated = False
        st.session_state.auth_username = ""
        st.session_state.messages = [{"role": "assistant", "content": DEFAULT_GREETING}]
        st.rerun()
    st.divider()

    # ================= 👨‍👩‍👧‍👦 多用户切换模块 =================
    with st.expander("👥 家庭成员管理", expanded=True):
        all_users = get_all_users()

        if 'current_user' not in st.session_state:
            st.session_state.current_user = "default"

        selected_user = st.selectbox(
            "当前查看/编辑的用户:",
            options=all_users,
            index=all_users.index(st.session_state.current_user) if st.session_state.current_user in all_users else 0,
            key="user_selector"
        )

        if selected_user != st.session_state.current_user:
            st.session_state.current_user = selected_user
            st.session_state.messages = [{"role": "assistant", "content": DEFAULT_GREETING}]
            keys_to_clear = ['temp_med_name', 'temp_med_dosage', 'temp_med_freq', 'temp_med_times']
            for k in keys_to_clear:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

        with st.expander("➕ 添加新成员"):
            new_user_name = st.text_input("新成员称呼 (如：爸爸、妈妈)", key="new_user_input")
            if st.button("创建档案", key="create_user_btn"):
                if create_new_user(new_user_name):
                    st.success(f"✅ 已创建成员：{new_user_name}")
                    st.session_state.current_user = new_user_name
                    st.rerun()
                else:
                    st.error("❌ 名称无效或已存在")

    st.divider()
    # =======================================================

    # ================= 📚 知识管理 (共享) =================
    with st.expander("📚 共享知识库", expanded=False):
        uploaded_files = st.file_uploader("拖拽 PDF 到此处", type=["pdf"], accept_multiple_files=True,
                                          help="所有成员共享此知识库")
        if st.button("🚀 上传并重建索引", type="primary", use_container_width=True):
            if uploaded_files:
                if build_knowledge_base_from_upload(uploaded_files):
                    st.balloons()
                    st.rerun()
            else:
                st.warning("⚠️ 请先选择文件")

    st.divider()

    # ================= 👤 个人健康档案 (隔离) =================
    with st.expander("👤 个人健康档案", expanded=False):
        current_profile = load_user_profile(st.session_state.current_user)

        with st.form("profile_form"):
            p_age = st.text_input("年龄", value=current_profile.get("age", ""), placeholder="例如：35")
            p_gender = st.selectbox("性别", ["未知", "男", "女"], index=["未知", "男", "女"].index(
                current_profile.get("gender", "未知")) if current_profile.get("gender") in ["未知", "男", "女"] else 0)
            p_allergies = st.text_area("⚠️ 过敏史 (重要)", value=current_profile.get("allergies", ""),
                                       placeholder="例如：青霉素...")
            p_chronic = st.text_area("🏥 慢性病史", value=current_profile.get("chronic_diseases", ""),
                                     placeholder="例如：高血压...")
            p_meds = st.text_area("💊 正在服用的药物", value=current_profile.get("current_medications", ""),
                                  placeholder="例如：阿司匹林...")

            submitted = st.form_submit_button("💾 保存档案", use_container_width=True, type="primary")
            if submitted:
                save_user_profile(
                    {"age": p_age, "gender": p_gender, "allergies": p_allergies, "chronic_diseases": p_chronic,
                     "current_medications": p_meds},
                    st.session_state.current_user
                )
                st.success("✅ 档案已保存！")
                st.rerun()

        if current_profile.get('allergies') or current_profile.get('chronic_diseases'):
            st.markdown("---")
            st.caption(f"📋 **{st.session_state.current_user}** 的生效档案:")
            info_text = []
            if current_profile.get('age'): info_text.append(f"🎂 {current_profile['age']}岁")
            if current_profile.get('gender') != '未知': info_text.append(f"🚻 {current_profile['gender']}")
            if current_profile.get('allergies'): info_text.append(f"⚠️ 过敏：{current_profile['allergies']}")
            if current_profile.get('chronic_diseases'): info_text.append(f"🏥 病史：{current_profile['chronic_diseases']}")
            st.info("\n".join(info_text))

    st.divider()

    # ================= 💊 用药提醒管理 (隔离 + 冲突检测) =================
    st.markdown("### 💊 用药提醒管理")
    med_data = load_medication_data(st.session_state.current_user)

    with st.expander("➕ 添加新用药计划", expanded=False):
        if 'temp_med_name' not in st.session_state: st.session_state.temp_med_name = ""
        if 'temp_med_dosage' not in st.session_state: st.session_state.temp_med_dosage = ""
        if 'temp_med_freq' not in st.session_state: st.session_state.temp_med_freq = 1
        if 'temp_med_times' not in st.session_state: st.session_state.temp_med_times = ["08:00"]

        st.session_state.temp_med_name = st.text_input("药品名称", value=st.session_state.temp_med_name,
                                                       placeholder="例如：硝苯地平控释片")
        st.session_state.temp_med_dosage = st.text_input("单次剂量", value=st.session_state.temp_med_dosage,
                                                         placeholder="例如：30mg")

        freq_options = ["每天 1 次", "每天 2 次", "每天 3 次", "每天 4 次"]
        selected_label = st.selectbox("每天几次", freq_options, index=st.session_state.temp_med_freq - 1)
        new_freq = freq_options.index(selected_label) + 1

        if new_freq != st.session_state.temp_med_freq:
            st.session_state.temp_med_freq = new_freq
            current_len = len(st.session_state.temp_med_times)
            if new_freq > current_len:
                st.session_state.temp_med_times.extend(["08:00"] * (new_freq - current_len))
            elif new_freq < current_len:
                st.session_state.temp_med_times = st.session_state.temp_med_times[:new_freq]
            st.rerun()

        st.markdown("**服药时间点：**")
        for i in range(st.session_state.temp_med_freq):
            try:
                default_time_obj = datetime.strptime(st.session_state.temp_med_times[i], "%H:%M").time()
            except Exception:
                default_time_obj = datetime.now().replace(minute=0, second=0).time()
            t_val = st.time_input(f"第 {i + 1} 次服药时间", value=default_time_obj, key=f"time_input_{i}")
            st.session_state.temp_med_times[i] = t_val.strftime("%H:%M")

        with st.form("save_med_form", clear_on_submit=False):
            submitted = st.form_submit_button("💾 保存计划", use_container_width=True)

            if submitted:
                med_name = st.session_state.temp_med_name.strip()
                if not med_name:
                    st.error("❌ 请输入药品名称")
                else:
                    # ================= 🛡️ 启动药物冲突检测 =================
                    st.info("🔍 正在进行安全筛查...")

                    # 1. 收集当前用户所有正在服用的药物名称
                    existing_drugs = set()

                    # 从档案中获取
                    profile = load_user_profile(st.session_state.current_user)
                    if profile.get("current_medications"):
                        meds = re.split(r'[,\n,]', profile["current_medications"])
                        for m in meds:
                            if m.strip(): existing_drugs.add(m.strip())

                    # 从已有的用药计划中获取
                    current_med_data = load_medication_data(st.session_state.current_user)
                    for plan in current_med_data.get("plans", []):
                        if plan.get("name"):
                            existing_drugs.add(plan["name"].strip())

                    # 移除新药本身
                    existing_drugs.discard(med_name)

                    has_conflict = False
                    conflict_details = []

                    # 只有当知识库存在且有对比药物时才检测
                    vs = load_vector_store()
                    if vs and len(existing_drugs) > 0:
                        has_conflict, conflict_details = check_drug_interaction(med_name, list(existing_drugs), vs)

                    # ================= 处理检测结果 =================
                    force_save = False
                    if has_conflict:
                        st.error("⚠️ **检测到潜在药物冲突！请谨慎操作！**")
                        for item in conflict_details:
                            st.markdown(f"""
                            <div style="background-color: #FEF2F2; border-left: 4px solid #DC2626; padding: 10px; margin: 5px 0; border-radius: 4px;">
                                <strong>❌ 冲突组合</strong>: {item['drug_pair']}<br>
                                <strong>风险关键词</strong>: {item['risk_keyword']}<br>
                                <small>📖 依据：{item['evidence']}</small>
                            </div>
                            """, unsafe_allow_html=True)

                        st.warning("💡 建议：请咨询医生或药师确认是否可以联用。如果确认无误，请勾选下方选项强制保存。")

                        force_save = st.checkbox("✅ 我已咨询医生，确认可以联用，强制保存", key="force_save_check")
                    else:
                        if vs:
                            st.success("✅ 安全筛查通过：未在知识库中发现明显冲突。")
                        else:
                            st.caption("ℹ️ 知识库未加载，跳过自动筛查。")

                    # ================= 执行保存 =================
                    # 有冲突时必须勾选强制保存才写入；无冲突直接保存。
                    # 不使用 st.stop()，避免阻断 tab1/tab2 渲染。
                    if not has_conflict or force_save:
                        new_plan = {
                            "id": f"plan_{datetime.now().timestamp()}",
                            "name": med_name,
                            "dosage": st.session_state.temp_med_dosage,
                            "frequency": st.session_state.temp_med_freq,
                            "times": st.session_state.temp_med_times.copy()
                        }

                        final_med_data = load_medication_data(st.session_state.current_user)
                        final_med_data["plans"].append(new_plan)
                        save_medication_data(final_med_data, st.session_state.current_user)

                        st.balloons()
                        st.success(f"✅ 已添加：{new_plan['name']}")

                        # 重置状态
                        st.session_state.temp_med_name = ""
                        st.session_state.temp_med_dosage = ""
                        st.session_state.temp_med_freq = 1
                        st.session_state.temp_med_times = ["08:00"]
                        st.rerun()

    if med_data["plans"]:
        st.caption(f"📋 **{st.session_state.current_user}** 的当前计划 (点击删除)")
        for i, plan in enumerate(med_data["plans"]):
            cols = st.columns([4, 1])
            with cols[0]:
                st.write(f"**{plan['name']}** ({plan['dosage']})\n⏰ {'、'.join(plan['times'])}")
            with cols[1]:
                if st.button("🗑️", key=f"del_{plan['id']}_{st.session_state.current_user}"):
                    med_data["plans"].pop(i)
                    save_medication_data(med_data, st.session_state.current_user)
                    st.rerun()
    else:
        st.info("暂无用药计划，请在上方添加。")

    st.divider()
    with st.expander("🚗 路线预估设置", expanded=False):
        start_input = st.text_input("起点地址", placeholder="例如：福州市万达广场", key="start_addr")
        start_lat, start_lon = None, None
        if start_input:
            with st.spinner("正在解析地址..."):
                start_lat, start_lon = geocode_address(start_input, GAODE_MAP_KEY)
            # 用 `is not None` 判断，避免坐标为 0.0 时被误判为解析失败
            if start_lat is not None and start_lon is not None:
                st.success("✅ 起点定位成功")
            else:
                st.error("❌ 地址解析失败")
        st.session_state.start_input_for_hospital = start_input
        st.session_state.start_coords_for_hospital = (start_lat, start_lon)

    st.divider()
    if os.path.exists(VECTOR_STORE_PATH):
        st.markdown(
            "<div style='background-color: #ECFDF5; color: #047857; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold;'>✅ 知识库已就绪</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background-color: #FEF2F2; color: #DC2626; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold;'>❌ 无知识库</div>",
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🤖 医疗问答", "🏥 附近医院"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": DEFAULT_GREETING}]
    if "enable_web_search" not in st.session_state: st.session_state.enable_web_search = False

    vs = load_vector_store()
    if not vs:
        st.warning("⚠️ **知识库为空**：请先在左侧上传 PDF 并重建索引。")
    else:

        active_profile = load_user_profile(st.session_state.current_user)
        med_data = load_medication_data(st.session_state.current_user)

        if "perf_stats" not in st.session_state:
            st.session_state.perf_stats = {
                "qa_count": 0,
                "first_answer_ms": None,
                "avg_total_ms": 0.0,
                "avg_retrieval_ms": 0.0,
                "avg_generation_ms": 0.0,
                "with_evidence_count": 0,
            }

        stats = st.session_state.perf_stats
        total_questions = stats["qa_count"]
        evidence_rate = (stats["with_evidence_count"] / total_questions * 100) if total_questions else 0.0
        metric_cols = st.columns(4)
        metric_cols[0].metric("首问耗时", f"{stats['first_answer_ms']:.0f} ms" if stats["first_answer_ms"] else "-")
        metric_cols[1].metric("平均总耗时", f"{stats['avg_total_ms']:.0f} ms" if total_questions else "-")
        metric_cols[2].metric("平均检索耗时", f"{stats['avg_retrieval_ms']:.0f} ms" if total_questions else "-")
        metric_cols[3].metric("检索命中率", f"{evidence_rate:.1f}%")
        st.caption("说明：检索命中率=有返回证据片段的问题数/总提问数。")
        st.divider()

        # ================= 今日用药打卡面板 =================
        st.markdown("### 📅 今日用药打卡")
        today_str = get_today_date_str()
        now = datetime.now()

        if today_str not in med_data["logs"]:
            med_data["logs"][today_str] = {}

        plans = med_data.get("plans", [])
        today_logs = med_data["logs"].get(today_str, {})

        if not plans:
            st.caption("💡 暂无用药计划。请在左侧侧边栏添加。")
        else:
            tasks = []
            for plan in plans:
                for t_str in plan['times']:
                    status = today_logs.get(f"{plan['id']}_{t_str}", "pending")
                    tasks.append({
                        "plan_id": plan['id'],
                        "time_str": t_str,
                        "name": plan['name'],
                        "dosage": plan['dosage'],
                        "status": status,
                        "full_key": f"{plan['id']}_{t_str}"
                    })

            tasks.sort(key=lambda x: x['time_str'])
            cols = st.columns(min(len(tasks), 3))
            has_overdue = False

            for idx, task in enumerate(tasks):
                col = cols[idx % len(cols)]
                with col:
                    is_taken = task['status'] == 'taken'
                    try:
                        scheduled_time = datetime.strptime(task['time_str'], "%H:%M").time()
                        scheduled_dt = datetime.combine(now.date(), scheduled_time)
                        if scheduled_dt > now and (scheduled_dt - now).total_seconds() > 43200:
                            scheduled_dt = scheduled_dt - timedelta(days=1)

                        window_minutes = 60
                        start_window = scheduled_dt - timedelta(minutes=window_minutes)
                        end_window = scheduled_dt + timedelta(minutes=window_minutes)

                        if is_taken:
                            is_overdue = False
                        elif now > end_window:
                            is_overdue = True
                        elif start_window <= now <= end_window:
                            is_overdue = False
                            st.warning(f"⏰ **现在**: {task['name']}")
                            st.caption(f"剂量：{task['dosage']}")
                            if st.button("✅ 打卡", key=f"btn_{task['full_key']}_{st.session_state.current_user}",
                                         use_container_width=True):
                                today_logs[task['full_key']] = "taken"
                                med_data["logs"][today_str] = today_logs
                                save_medication_data(med_data, st.session_state.current_user)
                                st.balloons()
                                st.rerun()
                            continue
                        else:
                            is_overdue = False

                        if is_overdue:
                            has_overdue = True
                            st.error(f"⚠️ {task['time_str']} {task['name']}")
                        elif not is_taken:
                            st.info(f"⏳ {task['time_str']} {task['name']}")

                        st.caption(f"剂量：{task['dosage']}")

                        if not is_taken:
                            if st.button("✅ 打卡", key=f"btn_{task['full_key']}_{st.session_state.current_user}",
                                         use_container_width=True):
                                today_logs[task['full_key']] = "taken"
                                med_data["logs"][today_str] = today_logs
                                save_medication_data(med_data, st.session_state.current_user)
                                st.balloons()
                                st.rerun()
                        else:
                            if st.button("↩️ 撤销", key=f"undo_{task['full_key']}_{st.session_state.current_user}",
                                         use_container_width=True):
                                today_logs[task['full_key']] = "pending"
                                med_data["logs"][today_str] = today_logs
                                save_medication_data(med_data, st.session_state.current_user)
                                st.rerun()
                    except Exception as e:
                        st.error(f"⚠️ 时间解析错误：{task['time_str']}")
                        st.info(f"⏳ {task['time_str']} {task['name']}")
                        st.caption(f"剂量：{task['dosage']}")

            if has_overdue:
                st.toast("⚠️ 您有药物尚未服用，请注意时间！", icon="⚠️")

        st.divider()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        col_input, col_toggle, col_clear = st.columns([4, 1, 1])
        with col_input:
            prompt = st.chat_input("请输入问题... (例如：我头痛能吃布洛芬吗？)")
        with col_toggle:
            use_web = st.toggle("🌐 联网", value=st.session_state.enable_web_search, help="开启后将搜索全网最新医疗资讯")
            st.session_state.enable_web_search = use_web
        with col_clear:
            if st.button("🧹 清空", use_container_width=True, key="clear_chat_btn"):
                st.session_state.messages = [{"role": "assistant", "content": DEFAULT_GREETING}]
                st.rerun()

        if prompt:
            invoke_start = time.perf_counter()
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                status_msg = "🤔 正在结合您的健康档案分析..."
                if st.session_state.enable_web_search: status_msg += " & 🌐 联网搜索..."
                with st.spinner(status_msg):
                    try:
                        history_context = ""
                        recent_msgs = st.session_state.messages[-6:]
                        for msg in recent_msgs:
                            role_label = "用户问" if msg["role"] == "user" else "助手答"
                            history_context += f"{role_label}：{msg['content']}\n"
                        rag_chain = get_rag_chain(vs, history=history_context,
                                                  enable_web_search=st.session_state.enable_web_search,
                                                  user_profile=active_profile)
                        resp = rag_chain.invoke({"question": prompt, "history": history_context,
                                                 "enable_web_search": st.session_state.enable_web_search})
                        total_ms = (time.perf_counter() - invoke_start) * 1000
                        perf_snapshot = st.session_state.get("_last_rag_metrics", {})
                        retrieval_ms = float(perf_snapshot.get("retrieval_ms", 0.0))
                        generation_ms = max(total_ms - retrieval_ms, 0.0)

                        stats = st.session_state.perf_stats
                        prev_n = stats["qa_count"]
                        new_n = prev_n + 1
                        stats["qa_count"] = new_n
                        stats["avg_total_ms"] = ((stats["avg_total_ms"] * prev_n) + total_ms) / new_n
                        stats["avg_retrieval_ms"] = ((stats["avg_retrieval_ms"] * prev_n) + retrieval_ms) / new_n
                        stats["avg_generation_ms"] = ((stats["avg_generation_ms"] * prev_n) + generation_ms) / new_n
                        if perf_snapshot.get("retrieved_docs_count", 0) > 0:
                            stats["with_evidence_count"] += 1
                        if stats["first_answer_ms"] is None:
                            stats["first_answer_ms"] = total_ms
                        st.session_state.perf_stats = stats

                        st.markdown(resp)
                        st.caption(
                            f"⏱️ 本次耗时：总计 {total_ms:.0f} ms | 检索 {retrieval_ms:.0f} ms | 生成 {generation_ms:.0f} ms"
                        )
                        st.session_state.messages.append({"role": "assistant", "content": resp})
                    except Exception as e:
                        if str(e) == "knowledge_base_unavailable":
                            error_msg = "⚠️ 当前未加载知识库，请先在左侧『共享知识库』上传 PDF 并构建知识库后再提问。"
                        else:
                            error_msg = f"❌ 发生错误：{str(e)}"
                        st.error(error_msg)
                        print(traceback.format_exc())
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

with tab2:
    st.markdown("### 📍 推荐附近医院")
    st.info("🤖 **智能推荐**：系统将自动查找并显示距离您**最近的 3 家**医疗机构。")
    default_loc = st.session_state.get("start_input_for_hospital", "")
    loc_input = st.text_input("📍 您当前所在位置", placeholder="例如：福州市闽侯县闽江学院", value=default_loc,
                              key="hospital_loc")
    precise_origin_input = st.text_input(
        "🎯 精准起点（可选）",
        placeholder="例如：闽江大学旗山校区(东南门)对面",
        key="hospital_precise_origin",
        help="用于路线计算。若输入校区/小区等大范围地点，建议补充到门口/路口以减少偏差。",
    )
    selected_origin_candidate = None
    if precise_origin_input.strip():
        origin_candidates = search_poi_candidates(precise_origin_input.strip(), GAODE_MAP_KEY, limit=8)
        if origin_candidates:
            option_labels = [
                f"{idx + 1}. {c['name']} | {c['address'] or '地址未知'} | {c['location']}"
                for idx, c in enumerate(origin_candidates)
            ]
            selected_label = st.selectbox(
                "🧭 请选择路线起点候选",
                options=option_labels,
                index=0,
                key="hospital_origin_candidate_select",
                help="路线计算将严格使用你选中的这个起点坐标。",
            )
            selected_index = option_labels.index(selected_label)
            selected_origin_candidate = origin_candidates[selected_index]
            st.caption(
                f"已选起点：{selected_origin_candidate['name']}（POI ID: {selected_origin_candidate['id'] or '-'}）"
            )
        else:
            st.caption("未找到可用候选起点，将按文字地址解析。")

    if st.button("🔍 查找最近 3 家医院", type="primary", use_container_width=True):
        if not loc_input:
            st.warning("⚠️ 请输入位置信息")
        else:
            with st.spinner("正在定位、排序并筛选..."):
                all_results = search_nearby_hospitals(loc_input, radius=10000)
                # 默认以本次检索输入位置作为路线起点，避免沿用旧起点造成距离偏差
                route_origin_text = precise_origin_input.strip() if precise_origin_input.strip() else loc_input
                if selected_origin_candidate and selected_origin_candidate.get("location"):
                    try:
                        origin_lon, origin_lat = map(float, selected_origin_candidate["location"].split(","))
                        query_lat, query_lon = origin_lat, origin_lon
                        route_origin_text = selected_origin_candidate.get("name", route_origin_text)
                    except Exception:
                        print("[hospital_tab] 起点候选坐标解析失败，回退到 geocode")
                        print(traceback.format_exc())
                        query_lat, query_lon = geocode_address(route_origin_text, GAODE_MAP_KEY)
                else:
                    query_lat, query_lon = geocode_address(route_origin_text, GAODE_MAP_KEY)
                valid_results = [h for h in all_results if
                                 "错误" not in h['name'] and "未配置" not in h['name'] and h.get('location')]
                if not valid_results:
                    st.info("🔍 该区域附近未找到符合条件的医疗机构。")
                else:
                    # 不再硬过滤掉非“医院”机构，改为按类型评分参与排序：
                    # 医院(5) > 急救/疾控(4) > 社区卫生服务中心(3) > 卫生院(2) > 门诊/诊所(1) > 其他(0)
                    # 复用 search_nearby_hospitals 已预填的 _type_score，避免重复计算；
                    # 若上游未填（旧调用路径或测试构造），再回退到现场计算。
                    for h in valid_results:
                        if "_type_score" not in h:
                            h["_type_score"] = score_medical_institution_type(h.get("name", ""))

                    def parse_distance(d_str):
                        try:
                            d_str = str(d_str).strip()
                            if '公里' in d_str:
                                return float(d_str.replace('公里', '')) * 1000
                            elif '米' in d_str:
                                return float(d_str.replace('米', ''))
                            else:
                                return float(d_str)
                        except Exception:
                            return 999999


                    # 先按 POI 距离 + 类型评分粗排，减少路线规划调用次数，再按驾车路线距离精排
                    coarse_sorted = sorted(
                        valid_results,
                        key=lambda x: (-x.get("_type_score", 0), parse_distance(x['distance'])),
                    )
                    candidate_results = coarse_sorted[:10]
                    ranked_results = []
                    for h in candidate_results:
                        route_info_html = ""
                        route_distance_m = None
                        # 用 `is not None` 判断坐标有效性，避免坐标为 0.0 时被误判为无效
                        start_lat, start_lon = (query_lat, query_lon) if (query_lat is not None and query_lon is not None) else st.session_state.get(
                            "start_coords_for_hospital", (None, None)
                        )
                        if start_lat is not None and start_lon is not None and h.get('location'):
                            try:
                                dest_lon, dest_lat = map(float, h['location'].split(','))
                                d_text, w_text, route_distance_m = get_route_info(
                                    start_lat, start_lon, dest_lat, dest_lon, GAODE_MAP_KEY
                                )
                                if "--" not in d_text or "--" not in w_text:
                                    route_info_html = f"<div style='background-color:#EFF6FF; border-left: 4px solid #2563EB; padding:10px; margin:10px 0; border-radius:4px;'><span style='font-weight:bold; color:#1E3A8A;'>🚗 驾车:</span> {d_text} &nbsp;&nbsp;|&nbsp;&nbsp; <span style='font-weight:bold; color:#1E3A8A;'>🚶 步行:</span> {w_text}</div>"
                            except Exception:
                                print(f"[hospital_tab] 路线计算失败 hospital={h.get('name')}")
                                print(traceback.format_exc())
                        # 驾车距离可用时优先按驾车距离排序；不可用时回退 POI 距离
                        rank_distance = route_distance_m if route_distance_m is not None else parse_distance(h["distance"])
                        ranked_results.append({
                            **h,
                            "_route_info_html": route_info_html,
                            "_route_distance_m": route_distance_m,
                            "_rank_distance": rank_distance,
                            "_route_source": "driving" if route_distance_m is not None else "poi",
                        })

                    # 排序：类型评分倒序 -> 距离正序（驾车距离优先，不可用回退 POI 距离）
                    sorted_results = sorted(
                        ranked_results,
                        key=lambda x: (-x.get("_type_score", 0), x["_rank_distance"]),
                    )
                    # 在类型 + 距离综合排序后，仍优先展示可计算驾车路线的机构
                    driving_ready = [h for h in sorted_results if h.get("_route_source") == "driving"]
                    hos_list = driving_ready[:3] if driving_ready else sorted_results[:3]
                    st.success(f"✅ 找到距离最近的 {len(hos_list)} 家机构：")
                    if query_lat is not None and query_lon is not None:
                        st.caption(f"🧭 路线起点：{route_origin_text}（{query_lon:.6f},{query_lat:.6f}）")
                    else:
                        st.caption(f"🧭 路线起点：{route_origin_text}（解析失败，已回退为参考直线距离）")

                    for i, h in enumerate(hos_list):
                        safe_name = urllib.parse.quote(h['name'])
                        nav_url = f"https://uri.amap.com/marker?position={h['location']}&name={safe_name}"
                        route_info_html = h.get("_route_info_html", "")
                        route_distance_m = h.get("_route_distance_m")
                        route_distance_text = (
                            f"{(route_distance_m / 1000):.1f}km" if route_distance_m is not None else str(h["distance"])
                        )
                        route_source = h.get("_route_source", "poi")
                        distance_label = "驾车路线距离" if route_source == "driving" else "参考直线距离(驾车不可达)"

                        with st.expander(f"{i + 1}. **{h['name']}** ({distance_label}：{route_distance_text})", expanded=(i == 0)):
                            st.write(f"**📍 地址**: {h['address']}")
                            st.write(f"**📞 电话**: `{h['tel']}`")
                            if route_info_html:
                                st.markdown(route_info_html, unsafe_allow_html=True)
                            elif query_lat is not None and query_lon is not None:
                                st.caption("ℹ️ 暂时无法计算路线时间。")
                            else:
                                st.caption("💡 **提示**：在左侧输入**起点地址**查看预计时间。")
                            st.markdown(f"""
                            <div style="text-align: right; margin-top: 15px; border-top: 1px solid #eee; padding-top: 10px;">
                                <a href="{nav_url}" target="_blank" style="text-decoration: none;">
                                    <button style="background-color: #2563EB; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: bold;">🧭 点击这里导航 (高德地图)</button>
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
