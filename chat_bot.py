import os
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import VECTOR_STORE_PATH, get_required_env
from embedding_provider import get_embeddings


def create_rag_chain():
    print("🔄 正在加载知识库和模型...")

    # 1. 加载向量库
    # 注意：allow_dangerous_deserialization=True 是加载本地 FAISS 必须的，因为是可信的本地文件
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

    # 2. 设置检索器 (Retriever)
    # k=3 表示每次检索最相关的 3 个片段
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. 定义大模型 (LLM)
    # 使用通义千问-plus，性价比高，适合医疗问答
    llm = ChatTongyi(model="qwen-plus", dashscope_api_key=get_required_env("DASHSCOPE_API_KEY"))

    # 4. 设计提示词模板 (Prompt Template)
    # 这是 RAG 的灵魂！告诉 AI 如何利用检索到的信息
    template = """
    你是一名专业的医疗助手。请根据以下【参考信息】回答用户的问题。
    如果【参考信息】中没有答案，请直接说“抱歉，根据目前的资料库，我无法回答这个问题”，不要编造内容。
    回答要简洁、准确、语气亲切。

    【参考信息】:
    {context}

    【用户问题】:
    {question}

    【你的回答】:
    """

    prompt = PromptTemplate.from_template(template)

    # 5. 构建链条 (Chain)
    # 流程：输入问题 -> 检索文档(context) -> 填入 Prompt -> 发送给 LLM -> 输出结果
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print("✅ 系统初始化完成！可以开始对话了。\n")
    return rag_chain


def main():
    try:
        chain = create_rag_chain()

        while True:
            user_input = input("👤 请输入问题 (输入 'q' 退出): ").strip()
            if user_input.lower() == 'q':
                print("👋 再见！")
                break

            if not user_input:
                continue

            print("🤖 思考中...", end="\r")
            try:
                response = chain.invoke(user_input)
                print(f"🤖 AI: {response}\n")
            except Exception as e:
                print(f"❌ 生成回答时出错: {e}")

    except FileNotFoundError:
        print("❌ 错误：未找到向量库文件夹 './vector_store'。请先运行 build_knowledge_base.py 构建知识库！")
    except Exception as e:
        print(f"❌ 启动失败: {e}")


if __name__ == "__main__":
    main()
