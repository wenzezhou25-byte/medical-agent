import os
import sys
import json  # 新增：用于保存统计结果
import traceback
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from config import BASE_DATA_PATH, STATS_SAVE_PATH, VECTOR_STORE_PATH, get_required_env
from embedding_provider import get_embeddings
from rag_utils import build_structured_documents

# 设置 UTF-8 编码环境
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'zh_CN.UTF-8'

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

DATA_PATH = BASE_DATA_PATH


# ================= 新增：统计函数 =================
def calculate_knowledge_base_stats(splits):
    """
    计算知识库核心统计指标
    :param splits: 切割后的Document对象列表
    :return: 统计结果字典
    """
    if not splits:
        return {"error": "知识库文本片段为空"}

    # 提取纯文本列表
    text_chunks = [doc.page_content for doc in splits]

    # 核心统计指标
    stats = {
        "知识库构建时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "总文本片段数量": len(text_chunks),
        "总字符数（去空格换行）": sum(len(chunk.replace(" ", "").replace("\n", "")) for chunk in text_chunks),
        "平均每片段字符数": round(sum(len(chunk) for chunk in text_chunks) / len(text_chunks), 2),
        "最长片段字符数": max(len(chunk) for chunk in text_chunks),
        "最短片段字符数": min(len(chunk) for chunk in text_chunks),
        "有效片段占比(字符数>50)": round(len([c for c in text_chunks if len(c.strip()) > 50]) / len(text_chunks) * 100,
                                         2),
        "原始文档总页数": len([doc for doc in splits])  # 基于加载的PDF页数
    }
    return stats


def save_stats_to_file(stats, save_path):
    """保存统计结果到JSON文件"""
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        print(f"📊 统计结果已保存至：{save_path}")
    except Exception as e:
        print(f"⚠️  保存统计结果失败：{e}")


def clean_pdf_text(text):
    """清洗 PDF 解析文本并强化章节边界。"""
    import re

    if not text:
        return ""

    section_titles = [
        "适应症",
        "功能主治",
        "用法用量",
        "不良反应",
        "禁忌",
        "注意事项",
        "药物相互作用",
        "药理作用",
        "贮藏",
        "孕妇及哺乳期妇女用药",
        "儿童用药",
        "老年用药",
        "临床表现",
        "症状",
        "治疗",
        "饮食",
        "生活方式",
        "就医指征",
    ]

    # 1) 统一常见空白与标点
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("【", "[").replace("】", "]")

    # 2) 去除 OCR 夸张重复（保留正常叠词）
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"(.{2,8})\1{2,}", r"\1", text)
    text = re.sub(r"([,\.!?;:，。！？；：])\1+", r"\1", text)

    # 3) 合并中文字符之间被错误打断的空格
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)

    # 4) 对标题前强制换行，便于后续 section 正则识别
    for title in section_titles:
        escaped = re.escape(title)
        text = re.sub(
            rf"(?<!\n)(\s*(?:\[{escaped}\]|{escaped}\s*[:：])\s*)",
            r"\n\1",
            text,
        )

    # 5) 压缩空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    print("🚀 开始构建医疗知识库...")

    # 1. 加载文档 - 改用更稳健的方式处理中文路径
    documents = []
    pdf_files = list(Path(DATA_PATH).rglob("*.pdf"))

    print(f"📂 发现 {len(pdf_files)} 个 PDF文件")

    # 如果有 PDF文件，先复制到临时目录（使用英文文件名）
    temp_dir = None
    if pdf_files:
        temp_dir = tempfile.mkdtemp(prefix="pdf_build_")
        file_name_map = {}
        for idx, pdf_file in enumerate(pdf_files):
            try:
                safe_name = f"{idx}.pdf"
                temp_path = Path(temp_dir) / safe_name
                shutil.copy2(pdf_file, temp_path)
                file_name_map[safe_name] = pdf_file.name
                print(f"✅ 已复制：{pdf_file.name} -> {safe_name}")
            except Exception as e:
                print(f"⚠️  复制失败 {pdf_file.name}: {e}")

        pdf_files = list(Path(temp_dir).glob("*.pdf"))
        print(f"📂 临时目录中有 {len(pdf_files)} 个 PDF文件\n")

    try:
        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                print(f"({i}/{len(pdf_files)}) 📄 正在加载 PDF #{i}...")
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
        
                # 关键：清洗每个文档的内容
                for doc in docs:
                    doc.page_content = clean_pdf_text(doc.page_content)
                    doc.metadata = {
                        "source": f"pdf_{i}",
                        "source_name": file_name_map.get(pdf_file.name, pdf_file.name),
                        "page": str(doc.metadata.get("page", i)),
                    }
        
                documents.extend(docs)
                print(f"   ✅ 成功加载 {len(docs)} 页")
            except Exception as e:
                print(f"   ⚠️  加载失败 PDF #{i}: {e}")
                continue

        if not documents:
            print("❌ 没有成功加载任何文档！请检查 PDF文件是否损坏")
            return

        print(f"\n✅ 总共成功加载 {len(documents)} 页文档。")

        # 2. 结构化切块
        splits = build_structured_documents(documents, clean_pdf_text)
        print(f"✂️  文档已被切割为 {len(splits)} 个片段。")

        # ================= 新增：计算并输出统计结果 =================
        print("\n📊 开始计算知识库统计指标...")
        stats = calculate_knowledge_base_stats(splits)
        # 打印统计结果
        print("\n=== 医疗知识库统计报告 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        # 保存统计结果到文件
        save_stats_to_file(stats, STATS_SAVE_PATH)

        # 3. 生成向量并存储 (Embedding & Storage)
        embeddings = get_embeddings()

        print("\n🧠 正在生成向量索引 (这可能需要几分钟)...")
        vectorstore = FAISS.from_documents(splits, embeddings)

        # 4. 保存到本地
        vectorstore.save_local(VECTOR_STORE_PATH)
        print(f"💾 知识库已保存至：{VECTOR_STORE_PATH}")
        print("🎉 知识库构建完成！现在可以开始问答了。")

    finally:
        # 清理临时目录
        if temp_dir:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


if __name__ == "__main__":
    main()
