# -*- coding: utf-8 -*-
"""联网搜索：requests 直连 Tavily API（原生实现，不依赖 langchain）。

从 app.py 拆出。不依赖 Streamlit。
"""
import os
import requests


def perform_web_search(query):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.startswith("tvly-dev-PLACEHOLDER"):
        return "⚠️ 联网搜索未配置有效 API Key。"
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "max_results": 3,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        context_parts = []
        for i, res in enumerate(data.get("results", [])):
            title = res.get("title", "无标题")
            snippet = res.get("content", res.get("snippet", ""))
            url = res.get("url", "")
            context_parts.append(f"{i + 1}. 【{title}】: {snippet} (来源：{url})")
        return "【互联网最新资讯】:\n" + "\n".join(context_parts) + "\n"
    except Exception as e:
        return f"⚠️ 联网搜索出错：{str(e)}"