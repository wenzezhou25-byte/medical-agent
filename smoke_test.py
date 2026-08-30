# -*- coding: utf-8 -*-
"""Agent 层三项优化（#7/#8/#9）的最小冒烟测试。

只测试纯逻辑（无需向量库/联网/UI），LLM 实调为可选步骤（网络不可用时跳过）。

运行：
    D:\\ananconda3\\python.exe smoke_test.py
"""
from drug_interaction import (
    _build_interaction_prompt,
    _classify_interaction_llm,
    _keyword_heuristic,
    _parse_interaction_json,
)
from agent_core import AgentCore, AgentState
from rag_utils import extract_drug_name_candidates, register_known_source_names

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


class _FakeDoc:
    def __init__(self, content, score=0.5):
        self.page_content = content
        self.metadata = {"rerank_score": score}


print("== #9 药物相互作用：LLM 结果 JSON 解析 ==")
v = _parse_interaction_json('{"has_conflict": true, "risk_level": "严重", "reason": "a", "suggestion": "b"}')
check("纯 JSON 解析", v["has_conflict"] is True and v["risk_level"] == "严重")

v = _parse_interaction_json('```json\n{"has_conflict": false, "risk_level": "无", "reason": "", "suggestion": ""}\n```')
check("markdown 围栏 JSON 解析", v["has_conflict"] is False and v["risk_level"] == "无")

v = _parse_interaction_json('结论如下：{"has_conflict": true, "risk_level": "中度", "reason": "r", "suggestion": "s"}')
check("带前缀文字 JSON 解析", v["has_conflict"] is True and v["risk_level"] == "中度")

v = _parse_interaction_json('{"has_conflict": false, "risk_level": "严重", "reason": "", "suggestion": ""}')
check("无冲突时风险等级强置为无", v["risk_level"] == "无")

try:
    _parse_interaction_json("这不是 json")
    check("非法 JSON 抛异常", False)
except ValueError:
    check("非法 JSON 抛异常", True)

print("== #9 关键词启发式 fallback ==")
docs_hit = [_FakeDoc("本品与华法林合用可能增加出血风险，属禁忌。")]
v = _keyword_heuristic("布洛芬", "华法林", docs_hit)
check("命中风险关键词判定冲突", v["has_conflict"] is True and v["risk_level"] == "中度")

docs_miss = [_FakeDoc("本品用于缓解头痛。")]
v = _keyword_heuristic("布洛芬", "维生素C", docs_miss)
check("无风险关键词不判定冲突", v["has_conflict"] is False and v["risk_keyword"] == "")

print("== #9 prompt 构造 ==")
p = _build_interaction_prompt("布洛芬", "华法林", "证据文本")
check("prompt 含两药名", "布洛芬" in p and "华法林" in p)
check("prompt 含 JSON 字段约束", "has_conflict" in p and "risk_level" in p)

print("== #8 指代消解 ==")
cache = ["布洛芬缓释胶囊", "二甲双胍"]
r = AgentCore._resolve_coreference("这个药能吃吗？", cache)
check("含指代注入候选药名", "布洛芬缓释胶囊" in r and "上下文提示" in r)

r2 = AgentCore._resolve_coreference("这个药能吃吗？", [])
check("空缓存不注入", "上下文提示" not in r2)

r3 = AgentCore._resolve_coreference("布洛芬缓释胶囊一次吃几片？", cache)
check("无指代原样返回", r3 == "布洛芬缓释胶囊一次吃几片？")

print("== #8 AgentState 药名缓存 ==")
register_known_source_names(["布洛芬缓释胶囊", "盐酸二甲双胍片", "二甲双胍"])
extracted = extract_drug_name_candidates("我之前吃布洛芬缓释胶囊")
check("正则剥离前缀噪声（提取到标准名）", "布洛芬缓释胶囊" in extracted)
check("药名不含句首噪声", all("我之前吃" not in n for n in extracted))

st = AgentState(max_rounds=3)
st.add_history([
    {"role": "user", "content": "我之前吃布洛芬缓释胶囊"},
    {"role": "assistant", "content": "好的，布洛芬对退热有效。"},
])
check("从历史提取到布洛芬药名", any("布洛芬" in n for n in st.drug_cache))

before = len(st.drug_cache)
st.merge_drug_cache(["布洛芬", "布洛芬缓释胶囊"])  # 与历史重复，应全被去重
check("merge 去重（重复项不新增）", len(st.drug_cache) == before)

print("== #9 LLM 实调（可选，网络不可用时跳过） ==")
try:
    v = _classify_interaction_llm("布洛芬", "华法林", "布洛芬与华法林合用可能增加出血风险。")
    if isinstance(v, dict) and "has_conflict" in v and "risk_level" in v:
        check("LLM 实调返回结构化 verdict", True)
        print(f"      -> {v}")
    else:
        check("LLM 实调返回结构化 verdict", False)
except Exception as e:
    print(f"  [SKIP] LLM 实调不可用：{e}")

print()
print(f"结果：{PASS} 通过, {FAIL} 失败")
if FAIL:
    raise SystemExit(1)