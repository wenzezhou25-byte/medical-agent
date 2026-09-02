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
from rag_utils import (
    extract_drug_name_candidates,
    upsert_drug_cache,
)

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

print("== #8 指代消解（确定性替换） ==")
cache = ["布洛芬缓释胶囊", "二甲双胍"]  # 末尾 = 最近
r = AgentCore._resolve_coreference("这个药能吃吗？", cache)
check("单数指代替换为最近药", "「二甲双胍」" in r and "这个药" not in r)
check("替换后附带指代说明", "指代说明" in r and "二甲双胍" in r)

r2 = AgentCore._resolve_coreference("这个药能吃吗？", [])
check("空缓存不替换", r2 == "这个药能吃吗？")

r3 = AgentCore._resolve_coreference("布洛芬缓释胶囊一次吃几片？", cache)
check("无指代原样返回", r3 == "布洛芬缓释胶囊一次吃几片？")

# 复数两药：最近两个 → "A 和 B"
r4 = AgentCore._resolve_coreference("这两个药能一起吃吗？", cache)
check("两药指代展开为最近两个", "「布洛芬缓释胶囊」和「二甲双胍」" in r4)

# 这个药 + 那个药 同现：这个→最近，那个→次近
r5 = AgentCore._resolve_coreference("这个药和那个药能一起吃吗？", cache)
check("这个/那个分别映射最近与次近", "「二甲双胍」和「布洛芬缓释胶囊」" in r5)

# 两药指代但缓存不足 → 降级声明
r6 = AgentCore._resolve_coreference("这两个药能一起吃吗？", ["二甲双胍"])
check("缓存不足两药时降级声明", "仅提到" in r6 and "二甲双胍" in r6)

# 两药指代变体触发词：这两种药 / 这两个
r9 = AgentCore._resolve_coreference("这两种药能一起吃吗？", cache)
check("两种药变体同样展开为最近两个", "「布洛芬缓释胶囊」和「二甲双胍」" in r9)

r10 = AgentCore._resolve_coreference("这两个能一起吃吗？", cache)
check("这两个变体同样展开为最近两个", "「布洛芬缓释胶囊」和「二甲双胍」" in r10)

# 缓存多于两个药时取最近两个（末尾=最近），而非最早两个
cache3 = ["布洛芬缓释胶囊", "二甲双胍", "阿莫西林胶囊"]
r11 = AgentCore._resolve_coreference("这两个药能一起吃吗？", cache3)
check("多药缓存取最近两个", "「二甲双胍」和「阿莫西林胶囊」" in r11)

# 空缓存时两药指代保持原样（无缓存可消解）
r12 = AgentCore._resolve_coreference("这两个药能一起吃吗？", [])
check("空缓存两药指代不替换", r12 == "这两个药能一起吃吗？")

# 时间状语类指代（非药内容）不替换
r7 = AgentCore._resolve_coreference("刚才说的注意事项再说一遍", cache)
check("时间状语非药内容不替换", r7 == "刚才说的注意事项再说一遍")

# 当前问题新药名不抢占焦点：传入缓存只有历史药时，指代按历史最近药解析
r8 = AgentCore._resolve_coreference("布洛芬和这个药能一起吃吗？", ["二甲双胍"])
check("单数指代按传入缓存最近药解析", "「二甲双胍」" in r8 and "布洛芬" in r8)

print("== #8 两药指代补充边界用例 ==")
# 替换后触发词被完全清除，句中不残留「这两个药」
r20 = AgentCore._resolve_coreference("这两个药能一起吃吗？", cache)
check("两药替换后触发词被清除", "这两个药" not in r20)

# 多个两药触发词同现（这两个药 + 这两种药）：全部替换为同一对药
r21 = AgentCore._resolve_coreference("这两个药和这两种药能一起吃吗？", cache)
check("两药触发词同现全部替换",
      "这两种药" not in r21 and "「布洛芬缓释胶囊」和「二甲双胍」" in r21)

# 句中带标点/插入语：两药指代仍能命中
r22 = AgentCore._resolve_coreference("我刚吃了这两种药，能一起吗？", cache)
check("句中带标点的两药指代命中", "「布洛芬缓释胶囊」和「二甲双胍」" in r22)

# 两药分支优先于单数分支：句中同时含「这两个药」与「这个药」时走两药分支
r23 = AgentCore._resolve_coreference("这两个药和这个药一起有关系吗？", cache)
check("两药分支优先于单数分支", "「布洛芬缓释胶囊」和「二甲双胍」" in r23)

# 指代说明含两个药名，便于模型回读校验
r24 = AgentCore._resolve_coreference("这两个药能一起吃吗？", cache)
check("两药指代说明含两个药名",
      "指代说明" in r24 and "布洛芬缓释胶囊" in r24 and "二甲双胍" in r24)

# 核心去重后的缓存参与两药指代：同一药物不同写法已合并，不影响双药判定
cache_dedup2 = []
for n in ["布洛芬", "阿莫西林胶囊", "布洛芬缓释胶囊", "阿莫西林"]:
    upsert_drug_cache(cache_dedup2, n)
check("upsert 后缓存无同核心重复", cache_dedup2 == ["布洛芬缓释胶囊", "阿莫西林胶囊"])
r25 = AgentCore._resolve_coreference("这两个药能一起吃吗？", cache_dedup2)
check("经 upsert 去重后两药指代正确",
      r25.startswith("「布洛芬缓释胶囊」和「阿莫西林胶囊」"))

# 缓存三药但最近两个恰好是同药不同写法（未去重的原始缓存）：按原样取最近两个，
# 说明这依赖上游 upsert 去重；此处仅验证解析器按「末尾=最近」取位序
cache_raw3 = ["阿莫西林胶囊", "布洛芬", "布洛芬缓释胶囊"]
r26 = AgentCore._resolve_coreference("这两个药能一起吃吗？", cache_raw3)
check("解析器按位序取最近两个",
      "「布洛芬」和「布洛芬缓释胶囊」" in r26)

print("== #8 AgentState 药名缓存（MRU） ==")
_KNOWN1 = ("布洛芬缓释胶囊", "盐酸二甲双胍片", "二甲双胍")
extracted = extract_drug_name_candidates("我之前吃布洛芬缓释胶囊", _KNOWN1)
check("正则剥离前缀噪声（提取到标准名）", "布洛芬缓释胶囊" in extracted)
check("药名不含句首噪声", all("我之前吃" not in n for n in extracted))

st = AgentState(max_rounds=3, known_names=_KNOWN1)
st.add_history([
    {"role": "user", "content": "我之前吃布洛芬缓释胶囊"},
    {"role": "assistant", "content": "好的，布洛芬对退热有效。"},
])
check("从历史提取到布洛芬药名", any("布洛芬" in n for n in st.drug_cache))

before = len(st.drug_cache)
st.merge_drug_cache(["布洛芬", "布洛芬缓释胶囊"])  # 与历史重复，应全被去重
check("merge 去重（重复项不新增）", len(st.drug_cache) == before)

st2 = AgentState(max_rounds=3, known_names=_KNOWN1)
st2.drug_cache = ["甲药", "乙药"]
st2.merge_drug_cache(["甲药"])  # 重复提及 → 移到末尾（最近焦点）
check("merge MRU：重复提及移到末尾", st2.drug_cache == ["乙药", "甲药"])

st3 = AgentState(max_rounds=3, known_names=_KNOWN1)
st3.drug_cache = ["阿莫西林胶囊", "二甲双胍"]  # 二甲双胍最近
st3.ingest_drug_names("阿莫西林胶囊怎么吃？")  # 再次提到阿莫西林胶囊 → 移到末尾（最近焦点）
check("ingest MRU：重复提及刷新最近焦点", st3.drug_cache == ["二甲双胍", "阿莫西林胶囊"])

print("== #8 泛名提取 + 缓存核心去重（对话日志回归） ==")
_KNOWN2 = (
    "布洛芬缓释胶囊", "盐酸二甲双胍片", "二甲双胍",
    "阿莫西林胶囊", "阿莫西林片", "阿莫西林克拉维酸钾片",
)

# 根因1：裸泛名（无剂型）也能被提取进候选（如「阿莫西林」）
extracted2 = extract_drug_name_candidates("这个药能和阿莫西林一起吃吗？", _KNOWN2)
check("裸泛名阿莫西林被提取", "阿莫西林" in extracted2)

# 根因2：同一药物的剂型名与泛名合并为一条（「布洛芬缓释胶囊」+「布洛芬」不再被当两个药）
st4 = AgentState(max_rounds=3, known_names=_KNOWN2)
st4.ingest_drug_names("布洛芬缓释胶囊一次吃几片？")
check("剂型名+泛名合并为一条", st4.drug_cache == ["布洛芬缓释胶囊"])

st4.ingest_drug_names("这个药能和阿莫西林一起吃吗？")
check("阿莫西林进入缓存", any("阿莫西林" in n for n in st4.drug_cache))

# 完整对话轨迹：Q3「这两个药」应指向布洛芬缓释胶囊 + 阿莫西林（缓存中是标准名形式）
rq3 = AgentCore._resolve_coreference("这两个药能一起吃吗？", st4.drug_cache)
check("Q3 两药指代展开为布洛芬+阿莫西林",
      rq3.startswith("「布洛芬缓释胶囊」和「阿莫西林"))

# Q4「这个药和那个药」：指向最近两个，且说明不再附带未出现触发词的噪音
rq4 = AgentCore._resolve_coreference("我吃了二甲双胍，这个药和那个药能一起吃吗", st4.drug_cache)
check("Q4 指代说明无多余触发词噪音", "这种药" not in rq4 and "该药" not in rq4)
check("Q4 指向布洛芬缓释胶囊与阿莫西林", "「阿莫西林" in rq4 and "「布洛芬缓释胶囊」" in rq4)

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