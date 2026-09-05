# -*- coding: utf-8 -*-
"""联网搜索：requests 直连 Tavily API（原生实现，不依赖 langchain）。

从 app.py 拆出。不依赖 Streamlit。
"""
import re
import traceback

import requests

from config import TAVILY_API_KEY


def perform_web_search(query):
    api_key = TAVILY_API_KEY
    # 空值或占位符（your-* / PLACEHOLDER / xxx）均视为未配置（P2-23/C）
    if not api_key or api_key.strip().lower().startswith(("your-", "tvly-dev-placeholder", "xxx")) or "placeholder" in api_key.lower():
        return "⚠️ 联网搜索未配置有效 API Key。"
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": 3,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        context_parts = []
        for i, res in enumerate(data.get("results", [])):
            title = (res.get("title") or "无标题")[:80]
            # P2-19：Tavily 片段字段为 content（无 snippet 字段），折叠空白后
            # 截到 500 字并加省略号，避免 advanced 深度内容撑爆上下文。
            snippet = re.sub(r"\s+", " ", (res.get("content") or "").strip())
            if len(snippet) > 500:
                snippet = snippet[:500] + "…"
            url = (res.get("url") or "")[:100]
            context_parts.append(f"{i + 1}. 【{title}】: {snippet} (来源：{url})")
        # 单条正文 ≤500 字 + 标题 ≤80 + 链接 ≤100，3 条合计约 2000 字内
        return "【互联网最新资讯】:\n" + "\n".join(context_parts[:3]) + "\n"
    except Exception as e:
        # P1-16：异常细节只进日志，回灌通用话术，避免泄漏内部信息
        traceback.print_exc()
        print(f"[web_search] 联网搜索失败：{e}")
        return "⚠️ 联网搜索暂时不可用，请稍后重试。"