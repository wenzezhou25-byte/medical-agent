"""从 CHIP-2025 药品说明书数据集构建本地知识库。

数据源：https://huggingface.co/datasets/szk123/chip-2025
字段结构：instruction(固定模板) / input(药名) / output(说明书正文)

与 build_knowledge_base.py 的差别在于：本脚本不再解析 data/*.pdf，
而是直接读取清洗后的药品说明书文本，过滤出“常见家庭用药”，
按药名作为 source_name、按说明书字段切分为 section，复用现有
FAISS + 混合检索链路构建 vector_store。
"""
import json
import os
import re
import sys
from pathlib import Path

import requests

from config import BASE_DATA_PATH, VECTOR_STORE_PATH
from embedding_provider import get_embeddings
from retrieval_core import Chunk
from vector_store import VectorStore
from rag_utils import sanitize_untrusted_text

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

RAW_FILENAME = "chip-2025-raw.json"
RAW_PATH = Path(BASE_DATA_PATH) / RAW_FILENAME
DOWNLOAD_URL = (
    "https://hf-mirror.com/datasets/szk123/chip-2025/"
    "resolve/main/merged_three_json_files-2.json"
)

FIELDS = ["规格", "成份", "成分", "适应症", "用法用量", "不良反应", "禁忌", "注意事项"]
_FIELD_ALT = {"成份": "成分"}
_FIELD_NAMES = r"规格|成份|成分|适应症|用法用量|不良反应|禁忌|注意事项"
FIELD_PATTERN = re.compile(
    r"(?:^|[\n;；])\s*(规格|成份|成分|适应症|用法用量|不良反应|禁忌|注意事项)\s*[:：]"
)

# ---- 常见家庭用药过滤 ----
# 命中任一条即排除（含注射剂型、肿瘤/重症等家庭不建议自用的场景）
EXCLUDE_KEYWORDS = [
    "注射液", "注射用", "静脉", "静滴", "静注", "滴注", "肌注", "皮下",
    "癌", "肿瘤", "恶性", "淋巴瘤", "白血病", "肉瘤", "化疗", "放疗",
    "器官移植", "透析", "麻醉", "造影", "抗逆转录", "抗疟", "促红",
    "结核", "癫痫", "帕金森",
]

# 命中任一条则视为常见家庭用药，予以保留（聚焦具体药名/成分/精准分类词，
# 不把“片胶囊”剂型词、也不把“高血压/钙/胃”等宽泛词作为命中条件，避免误收）
INCLUDE_KEYWORDS = [
    # 感冒 / 退热 / 解热镇痛
    "感冒", "流感", "退热", "解热", "清热", "感冒灵", "感康", "快克", "泰诺",
    "布洛芬", "对乙酰氨基酚", "阿司匹林", "萘普生", "扑热息痛", "镇痛", "止痛",
    # 止咳化痰
    "止咳", "化痰", "祛痰", "咳嗽", "氨溴索", "右美沙芬", "川贝", "枇杷",
    "甘草", "急支", "肺力咳",
    # 肠胃
    "蒙脱石", "健胃", "消食", "益生菌", "双歧", "乳果糖", "开塞露",
    "小檗碱", "黄连素", "铝碳酸镁", "奥美拉唑", "雷贝拉唑", "泮托拉唑",
    "法莫替丁", "西咪替丁", "雷尼替丁", "多潘立酮", "莫沙必利",
    "止泻", "便秘", "腹泻", "消化不良", "藿香正气", "肠炎宁",
    # 抗过敏
    "氯雷他定", "西替利嗪", "氯苯那敏", "扑尔敏", "抗过敏", "过敏",
    # 维生素 / 矿物质
    "维生素", "多维", "叶酸", "鱼肝油", "钙尔奇", "碳酸钙", "葡萄糖酸钙",
    "硫酸亚铁", "葡萄糖酸锌",
    # 常用口服抗生素
    "阿莫西林", "头孢", "阿奇霉素", "罗红霉素", "左氧氟沙星", "莫西沙星",
    "克拉霉素", "诺氟沙星", "红霉素",
    # 慢性病家庭常备（具体药名）
    "二甲双胍", "格列", "阿卡波糖", "氨氯地平", "硝苯地平", "厄贝沙坦",
    "缬沙坦", "氯沙坦", "替米沙坦", "美托洛尔", "比索洛尔", "卡托普利",
    "依那普利", "贝那普利", "吲达帕胺", "氢氯噻嗪",
    "阿托伐他汀", "辛伐他汀", "瑞舒伐他汀", "非诺贝特", "氯吡格雷",
    # 中成药 / 外用 / 急救常备
    "板蓝根", "连花", "双黄连", "蒲地蓝", "银翘", "桑菊", "清开灵",
    "清凉油", "风油精", "碘伏", "酒精", "创可贴", "云南白药", "红花油",
    "活络油", "伤湿止痛", "跌打", "皮炎平", "达克宁", "莫匹罗星", "炉甘石",
    "安宫牛黄", "速效救心", "硝酸甘油",
]


def download() -> Path:
    """下载原始数据集到本地（若已存在则跳过）。"""
    if RAW_PATH.exists():
        print(f"✅ 已存在本地数据集：{RAW_PATH}")
        return RAW_PATH

    print(f"⬇️  开始下载数据集到 {RAW_PATH} ...")
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(DOWNLOAD_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(RAW_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {100 * done / total:.1f}%", end="", flush=True)
    print(f"\n✅ 下载完成：{RAW_PATH} ({done} bytes)")
    return RAW_PATH


def iter_rows(path: Path):
    """流式逐条解析 JSON 数组合集，避免一次性把全部数据读入内存。

    原实现用 f.read() 整文件载入；数据集可达数百 MB，这里改为「分块读 + 滚动缓冲」：
    raw_decode 逐条解析出完整 JSON 元素后，丢弃已消费头部、保留尾部半截，
    再补充下一个分块继续，整个过程内存峰值仅约 1 个缓冲块 + 单个元素大小。

    返回 (药名, 说明书正文) 迭代器。
    """
    decoder = json.JSONDecoder()
    buffer = ""
    buf_chunk = 1 * 1024 * 1024  # 每次补充约 1MB
    saw_open = False  # 是否已定位到数组开头的 '['

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(buf_chunk)
            if chunk:
                buffer += chunk

            # 尚未定位数组开头：跳过开头的空白与 '['（允许 '[' 跨分块迟到）
            if not saw_open:
                stripped = buffer.lstrip(" \t\r\n")
                if stripped.startswith("["):
                    buffer = stripped[1:]
                    saw_open = True
                elif not chunk:
                    return  # EOF 仍未见到数组开头，视为非法 JSON，安全退出
                else:
                    # 分块里还没有 '['，但缓冲区可能仍被空白占满，只保留尾部继续读
                    buffer = buffer[-1000:] if buffer else ""
                    continue

            # 在当前缓冲中尽量多地解析出完整元素
            n = len(buffer)
            consumed = 0
            i = 0
            while i < n:
                while i < n and buffer[i] in " \t\r\n,":
                    i += 1
                if i >= n or buffer[i] == "]":
                    break
                try:
                    item, nxt = decoder.raw_decode(buffer, i)
                except json.JSONDecodeError:
                    break  # 尾部半截，等下一个分块补全
                name = item.get("input") or item.get("drug") or ""
                output = item.get("output") or ""
                if name and output:
                    yield name, output
                i = nxt
                consumed = nxt

            # 丢弃已消费头部，保留未消费尾部（半截元素得以跨分块累积）
            buffer = buffer[consumed:]

            if not chunk:
                # EOF：剩余若尽是空白/逗号/] 则正常结束
                if not buffer.strip(" \t\r\n,]"):
                    return
                return  # 残余为不完整元素，安全退出


def clean_output(text: str) -> str:
    """清洗说明书正文：处理问号乱码、异体顿号与分隔符。"""
    if not text:
        return ""
    text = text.replace("?", "、")  # 问号乱码，多为枚举顿号
    text = text.replace("\ufe51", "、")  # 小型顿号 U+FE51
    text = text.replace("\ufe50", "、")
    text = text.replace("\\|", "\n").replace("|", "\n")
    text = re.sub(
        rf"[；;]\s*(?=(?:{_FIELD_NAMES})[:：])", "\n", text
    )
    return text.strip()


def _family_text(name: str, output: str) -> str:
    """取药名 + 适应症 + 成分作为判断依据。

    刻意避开「不良反应/禁忌/注意事项/药物相互作用」等字段，因为那些字段会
    大量提到其它药名和症状词（如“对本品过敏者禁用”“与阿司匹林合用”），
    若全文匹配会交叉误命中其它药物。
    """
    sections = dict(split_fields(clean_output(output)))
    indication = sections.get("适应症", "")
    composition = sections.get("成分", "")
    return f"{name} {indication} {composition}"


def is_plausible_drug_name(name: str) -> bool:
    """校验 entry 的 input 是否为“药名”，剔除混入的问答/对话噪声。

    chip-2025 的 input 字段除药名外还混有问答文本，例如“宝宝感冒吃什么食物”。
    这类记录会因命中药品关键词被误选，需在此剔除（只保留短小、无标点/问号的名称）。
    """
    n = (name or "").strip()
    if not n:
        return False
    if "？" in n or "?" in n:
        return False
    if re.search(r"[，,、。；;！!：:（）()]", n):
        return False
    if re.search(r"(宝宝|孩子|小孩|我家|怎么办|吃什么|有什么|可以喝|拉肚子|发烧|想要|需要治疗)", n):
        return False
    if len(n) > 25:
        return False
    return True


def is_family_drug(name: str, output: str) -> bool:
    """判断是否为常见家庭用药：先排除禁忌场景，再命中保留关键词。"""
    if not is_plausible_drug_name(name):
        return False
    text = _family_text(name, output)
    for kw in EXCLUDE_KEYWORDS:
        if kw in text:
            return False
    for kw in INCLUDE_KEYWORDS:
        if kw in text:
            return True
    return False


def split_fields(output: str):
    """按说明书字段切分，返回 [(字段名, 内容)]。"""
    matches = list(FIELD_PATTERN.finditer(output))
    if not matches:
        return [("未分类", output)]
    sections = []
    for i, m in enumerate(matches):
        title = m.group(1)
        title = _FIELD_ALT.get(title, title)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(output)
        content = output[start:end].strip().strip(";；").strip()
        if content:
            sections.append((title, content))
    return sections


def build_structured_drug_documents(rows) -> list[Chunk]:
    """把每条药品数据转成带 section 元数据的 Chunk。"""
    docs: list[Chunk] = []
    for name, output in rows:
        cleaned = clean_output(output)
        if len(cleaned.strip()) <= 20:
            continue
        for section_index, (title, content) in enumerate(split_fields(cleaned)):
            # P1-10：入库前净化（去除控制字符 + 中和指令注入片段）
            content = sanitize_untrusted_text(content)
            if len(content.strip()) <= 20:
                continue
            docs.append(
                Chunk(
                    page_content=content,
                    metadata={
                        "source": name,
                        "source_name": name,
                        "page": "1",
                        "section_title": title,
                        "section_index": section_index,
                        "document_type": "drug_insert",
                    },
                )
            )
    return docs


def main():
    print("🚀 开始从 CHIP-2025 数据集构建家庭用药知识库...", flush=True)

    path = download()

    print("🔍 读取并过滤常见家庭用药（按药名去重）...", flush=True)
    seen_names: set[str] = set()
    selected_rows: list[tuple[str, str]] = []
    total = 0
    for name, output in iter_rows(path):
        total += 1
        if name in seen_names:
            continue
        if is_family_drug(name, output):
            seen_names.add(name)
            selected_rows.append((name, output))

    print(f"📊 总记录 {total} 条，命中家庭用药 {len(selected_rows)} 种", flush=True)

    if not selected_rows:
        print("❌ 没有筛选到任何家庭用药，请检查过滤关键词。", flush=True)
        return

    docs = build_structured_drug_documents(selected_rows)
    print(f"✂️  生成 {len(docs)} 个结构化片段。", flush=True)

    embeddings = get_embeddings()

    # 分批向量化，避免一次性处理全部片段导致内存峰值过高
    # 注意：batch 过大（如 3000）会让 FastEmbed/ONNX 在长文本上出现 Attention
    # OOM 并整批回退到 hashing，造成向量空间不一致；故取较小 batch。
    batch_size = 500
    print(f"🧠 正在分批生成向量索引（每批 {batch_size} 条）...", flush=True)
    vectorstore = None
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        if vectorstore is None:
            vectorstore = VectorStore.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        # 一旦 embedding 发生降级回退，本批向量与其他批不在同一向量空间，
        # 继续写入会让整库检索失真；必须在落盘前中止。
        if getattr(embeddings, "degraded", False):
            print("❌ embedding 发生降级回退，向量空间可能不一致，已中止构建。"
                  "请检查模型文件/内存占用后重试（必要时先清理 .cache/fastembed）。", flush=True)
            return
        done = min(start + batch_size, len(docs))
        print(f"  进度 {done}/{len(docs)}", flush=True)

    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"💾 知识库已保存至：{VECTOR_STORE_PATH}", flush=True)
    print("🎉 构建完成！", flush=True)


if __name__ == "__main__":
    main()