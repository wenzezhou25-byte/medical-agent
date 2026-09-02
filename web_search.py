# -*- coding: utf-8 -*-
"""联网搜索：requests 直连 Tavily API（原生实现，不依赖 langchain）。

从 app.py 拆出。不依赖 Streamlit。
"""
import os
import traceback

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
            title = res.get("title", "无标题")[:80]
            # P2-19：单条 snippet 截到 500 字，避免 advanced 深度内容撑爆上下文
            snippet = (res.get("content") or res.get("snippet") or "")[:500]
            url = res.get("url", "")
            context_parts.append(f"{i + 1}. 【{title}】: {snippet} (来源：{url})")
        # 总返回也控制在合理范围：仅保留前 3 条已截断的结果，防止极端情况膨胀
        return "【互联网最新资讯】:\n" + "\n".join(context_parts[:3]) + "\n"
    except Exception as e:
        # P1-16：异常细节只进日志，回灌通用话术，避免泄漏内部信息
        traceback.print_exc()
        print(f"[web_search] 联网搜索失败：{e}")
        return "⚠️ 联网搜索暂时不可用，请稍后重试。"