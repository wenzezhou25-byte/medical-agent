import sys
import os
import locale
import requests
import json
import traceback
import tempfile
import shutil
import hashlib
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
        except:
            pass
    return {"age": "", "gender": "未知", "allergies": "", "chronic_diseases": "", "current_medications": ""}


def save_user_profile(profile, user_id="default"):
    path = get_user_profile_path(user_id)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def load_medication_data(user_id="default"):
    path = get_user_med_log_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
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

def hash_password(raw_password: str) -> str:
    return hashlib.sha256(raw_password.encode("utf-8")).hexdigest()


def load_auth_users():
    if os.path.exists(AUTH_USERS_PATH):
        try:
            with open(AUTH_USERS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}


def save_auth_users(users):
    os.makedirs(BASE_DATA_PATH, exist_ok=True)
    with open(AUTH_USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


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
    users = load_auth_users()
    user = users.get((username or "").strip())
    if not user:
        return False
    return user.get("password_hash") == hash_password((password or "").strip())


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
    except:
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
    try:
        d_params = {"origin": f"{origin_lon},{origin_lat}", "destination": f"{dest_lon},{dest_lat}", "key": api_key,
                    "extensions": "base", "output": "json"}
        resp = requests.get(f"{base_url}/driving", params=d_params, timeout=3)
        data = resp.json()
        if data.get("status") == "1" and data.get("route", {}).get("paths"):
            path = data["route"]["paths"][0]
            drive_res = f"🚗 {round(int(path['duration']) / 60)}分 ({round(int(path['distance']) / 1000, 1)}km)"
    except:
        pass
    try:
        w_params = {"origin": f"{origin_lon},{origin_lat}", "destination": f"{dest_lon},{dest_lat}", "key": api_key,
                    "extensions": "base", "output": "json"}
        resp = requests.get(f"{base_url}/walking", params=w_params, timeout=3)
        data = resp.json()
        if data.get("status") == "1" and data.get("route", {}).get("paths"):
            path = data["route"]["paths"][0]
            walk_res = f"🚶 {round(int(path['duration']) / 60)}分 ({round(int(path['distance']) / 1000, 1)}km)"
    except:
        pass
    return drive_res, walk_res


def geocode_address(address, api_key):
    if not address: return None, None
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {"address": address, "key": api_key, "output": "json"}
    try:
        resp = requests.get(url, params=params, timeout=3)
        data = resp.json()
        if data.get("status") == "1" and data.get("geocodes"):
            loc = data["geocodes"][0]["location"]
            lon, lat = loc.split(",")
            return float(lat), float(lon)
    except:
        pass
    return None, None


def search_nearby_hospitals(location_query, radius=5000):
    if not GAODE_MAP_KEY: return [
        {"name": "⚠️ 未配置地图 API", "address": "", "distance": "-", "tel": "-", "location": ""}]
    try:
        geo_resp = requests.get("https://restapi.amap.com/v3/geocode/geo",
                                params={"address": location_query, "key": GAODE_MAP_KEY, "output": "json"}, timeout=5)
        geo_data = geo_resp.json()
        if geo_data.get("status") != "1" or not geo_data.get("geocodes"):
            return [{"name": "❌ 地址解析失败", "address": "", "distance": "-", "tel": "-", "location": ""}]
        location = geo_data["geocodes"][0]["location"]
        search_resp = requests.get("https://restapi.amap.com/v3/place/around", params={
            "location": location, "keywords": "医院 | 卫生院 | 诊所 | 门诊部 | 疾控 | 急救",
            "radius": radius, "key": GAODE_MAP_KEY, "output": "json", "offset": 50
        }, timeout=5)
        search_data = search_resp.json()
        hospitals = []
        blacklist = ["酒店", "宾馆", "餐厅", "超市", "学校", "公司"]
        whitelist = ["医院", "卫生", "诊所", "门诊", "疾控", "急救", "医务", "护理"]
        if search_data.get("status") == "1" and search_data.get("pois"):
            for poi in search_data["pois"]:
                name = poi.get("name", "")
                if any(word in name for word in blacklist): continue
                if any(word in name for word in whitelist) and poi.get("location"):
                    hospitals.append(
                        {"name": name, "address": poi.get("address", ""), "distance": poi.get("distance", ""),
                         "tel": poi.get("tel", ""), "location": poi.get("location", "")})
        if not hospitals and search_data.get("pois"):
            for poi in search_data["pois"][:20]:
                name = poi.get("name", "")
                if any(word in name for word in ["酒店", "宾馆", "餐厅", "超市"]): continue
                if ("室" in name or "站" in name or "所" in name or "医" in name) and poi.get("location"):
                    hospitals.append(
                        {"name": name, "address": poi.get("address", ""), "distance": poi.get("distance", ""),
                         "tel": poi.get("tel", ""), "location": poi.get("location", "")})
                if len(hospitals) >= 3: break
        return hospitals if hospitals else [
            {"name": "🔍 附近暂未找到正规医疗机构", "address": "", "distance": "-", "tel": "-", "location": ""}]
    except Exception as e:
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
    if not uploaded_files: return False
    temp_dir = tempfile.mkdtemp(prefix="pdf_temp_")
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

            if os.path.exists(VECTOR_STORE_PATH): shutil.rmtree(VECTOR_STORE_PATH)
            vectorstore.save_local(VECTOR_STORE_PATH)
            st.success("✅ 知识库重建成功！")
            load_vector_store.clear()
            load_hybrid_retriever.clear()
            return True
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


# 假设 perform_web_search 和其他导入已在文件顶部定义

def get_rag_chain(vectorstore, history="", enable_web_search=False, user_profile=None):
    # 初始化组件
    # 优化：检索数量设为 6，提高命中率
    retriever = load_hybrid_retriever() or vectorstore.as_retriever(search_kwargs={"k": 6})
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
    template = f"""你是一名拥有20年临床经验的**资深执业药师**。你的任务是基于【用户档案】和【检索到的知识库】，为用户提供**综合归纳后**的用药指导，而非简单的资料堆砌。

    {profile_text}

    ### ⛔ 核心铁律 (违反即失败)
    1. **严禁逐条罗列**：检索到的片段可能有多条重复或相似信息。你必须将它们**融合、归纳**为通顺的段落。
       - ❌ 禁止这样回答：
         “1. 小儿应在医师指导下服用。
          2. 孕妇应在医师指导下服用。
          3. 老人应在医师指导下服用。”
       - ✅ 必须这样回答：
         “**特殊人群指导**：小儿、孕妇及年老体弱者均需在医师指导下服用，不可自行决定剂量。”
    2. **数字零容忍**：涉及剂量、时间、年龄时，**必须逐字摘录原文数字**，严禁模糊化。
    3. **针对性过滤**：只回答与用户问题强相关的内容。若用户未问及“贮藏”，则不要主动罗列“放在儿童不能接触的地方”等通用废话，除非该药有特殊贮藏要求（如冷藏）。
    4. **冲突与未知**：本地库与网络冲突以本地为准；本地库未提及的特定人群用法，明确告知“说明书未提及”。

    ### 🧠 思考与合成流程
    1. **阅读**：仔细阅读所有检索片段。
    2. **去重合并**：识别重复信息（如多条片段都提到“忌辛辣”），合并为一点。
    3. **结构化**：按 `### 💊 核心用法`、`### ⚠️ 禁忌与警示`、`### 🥗 生活饮食` 组织内容。
    4. **语气转换**：将生硬的说明书语言转化为“药师对患者的叮嘱”口吻。

    ### 📝 输出格式
    - 使用 Markdown 标题分级。
    - 关键数据**加粗**。
    - 每条综合建议后标注具体证据，例如 `[来源：文件名 / 第X页 / 章节名]`，不要只写笼统的“来源：说明书”。
    - 若证据不足，明确说明“说明书未提及”或“当前检索证据不足”，不要补充推测。
    - 优先使用短段落或短条目，每个条目都要带来源。
    - 最后增加 `### 证据说明`，简要说明本次主要依据了哪 1-3 条证据。
    - 末尾附带免责声明。

    ---
    ### 📥 动态输入数据
    【对话历史】: {{history}}
    【本地知识库片段】 (请综合归纳以下内容，不要照抄):
    {{context}}
    【互联网最新资讯】: {{web_context}}
    【用户最新问题】: {{question}}

    ---
    ### 🗣️ 请开始综合回答:
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
            except:
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
                        if not force_save:
                            st.stop()  # 停止执行后续保存代码
                    else:
                        if vs:
                            st.success("✅ 安全筛查通过：未在知识库中发现明显冲突。")
                        else:
                            st.caption("ℹ️ 知识库未加载，跳过自动筛查。")

                    # ================= 执行保存 =================
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
            if start_lat:
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
        st.stop()

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
                        pass
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

    if st.button("🔍 查找最近 3 家医院", type="primary", use_container_width=True):
        if not loc_input:
            st.warning("⚠️ 请输入位置信息")
        else:
            with st.spinner("正在定位、排序并筛选..."):
                all_results = search_nearby_hospitals(loc_input, radius=10000)
                valid_results = [h for h in all_results if
                                 "错误" not in h['name'] and "未配置" not in h['name'] and h.get('location')]
                if not valid_results:
                    st.info("🔍 该区域附近未找到符合条件的医疗机构。")
                else:
                    def parse_distance(d_str):
                        try:
                            d_str = str(d_str).strip()
                            if '公里' in d_str:
                                return float(d_str.replace('公里', '')) * 1000
                            elif '米' in d_str:
                                return float(d_str.replace('米', ''))
                            else:
                                return float(d_str)
                        except:
                            return 999999


                    sorted_results = sorted(valid_results, key=lambda x: parse_distance(x['distance']))
                    hos_list = sorted_results[:3]
                    vs_loc = load_vector_store()
                    retriever_loc = vs_loc.as_retriever(search_kwargs={"k": 1}) if vs_loc else None
                    st.success(f"✅ 找到距离最近的 {len(hos_list)} 家机构：")

                    for i, h in enumerate(hos_list):
                        safe_name = urllib.parse.quote(h['name'])
                        nav_url = f"https://uri.amap.com/marker?position={h['location']}&name={safe_name}"
                        route_info_html = ""
                        start_lat, start_lon = st.session_state.get("start_coords_for_hospital", (None, None))
                        if start_lat and start_lon and h.get('location'):
                            try:
                                dest_lon, dest_lat = map(float, h['location'].split(','))
                                d_text, w_text = get_route_info(start_lat, start_lon, dest_lat, dest_lon, GAODE_MAP_KEY)
                                if "--" not in d_text or "--" not in w_text:
                                    route_info_html = f"<div style='background-color:#EFF6FF; border-left: 4px solid #2563EB; padding:10px; margin:10px 0; border-radius:4px;'><span style='font-weight:bold; color:#1E3A8A;'>🚗 驾车:</span> {d_text} &nbsp;&nbsp;|&nbsp;&nbsp; <span style='font-weight:bold; color:#1E3A8A;'>🚶 步行:</span> {w_text}</div>"
                            except:
                                pass

                        with st.expander(f"{i + 1}. **{h['name']}** (距离：{h['distance']})", expanded=(i == 0)):
                            st.write(f"**📍 地址**: {h['address']}")
                            st.write(f"**📞 电话**: `{h['tel']}`")
                            if route_info_html:
                                st.markdown(route_info_html, unsafe_allow_html=True)
                            elif st.session_state.get("start_input_for_hospital", ""):
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
                            if retriever_loc:
                                try:
                                    docs = retriever_loc.invoke(f"{h['name']} 简介")
                                    if docs and docs[0].page_content.strip():
                                        st.info(f"**📖 知识库补充**: {docs[0].page_content[:300]}...")
                                except:
                                    pass
