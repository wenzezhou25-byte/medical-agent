from langchain_community.llms import Tongyi
from config import get_required_env

# 1. 初始化模型
llm = Tongyi(model="qwen-max", dashscope_api_key=get_required_env("DASHSCOPE_API_KEY"))

# 2. 调用模型并打印结果
response = llm.invoke("你好，请用医生的口吻做一下自我介绍")
print(response)
