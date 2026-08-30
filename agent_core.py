# -*- coding: utf-8 -*-
"""
agent_core.py —— DashScope 原生 function-calling 智能体（不依赖 LangChain）。

核心结构：
  1. ToolRegistry：注册工具，提供 OpenAI 格式 function definitions 给 DashScope
  2. AgentState：维护对话历史、工具调用记录
  3. AgentLoop：单步决策 + 工具调用 + 结果回灌，直到模型停止调用

工具设计：
  - rag_search(query: str): -> 家庭用药知识库 RAG 检索
  - web_search(query: str): -> Tavily 联网搜索最新医学资讯
  - search_nearby_hospitals(location: str): -> 高德地图查附近医院
  - save_user_medical_record(key: str, value: str): -> 保存用户个人健康档案
  - conflict_checker(drug_a: str, drug_b: str): -> 药物相互作用查询

DashScope function-calling 协议对齐：https://help.aliyun.com/zh/dashscope/developer-reference/function-calling
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import dashscope
from dashscope import Generation

from config import LLM_MODEL, get_required_env
from rag_utils import extract_drug_name_candidates


# 指代消解触发词：当当前问题含这些词，且会话缓存中已有药名时，注入候选药名供模型解析。
PRONOUN_HINTS = (
    "这个药", "那个药", "这种药", "该药", "此药",
    "刚才说的", "前面说的", "上面说的", "之前说的",
    "这两个药", "这两个", "这两种药", "这些药",
)

# 流式阶段状态：工具名 → 执行前的过渡提示语，让用户看到「检索 → 生成」的切换。
_TOOL_STATUS_TEXT = {
    "rag_search": "🔍 正在检索家庭用药知识库...",
    "web_search": "🌐 正在联网搜索最新医学资讯...",
    "search_nearby_hospitals": "🏥 正在查找附近医院...",
    "save_user_medical_record": "📋 正在保存健康档案...",
    "conflict_checker": "💊 正在检查药物相互作用...",
}


def _tool_status(name: str) -> str:
    """返回某工具执行前的阶段提示语。"""
    return _TOOL_STATUS_TEXT.get(name, f"🔧 正在执行工具：{name}...")


def _resolve_tool_call_id(tool_call: Dict[str, Any]) -> str:
    """返回 tool_call 的 id；DashScope 偶发缺 id 时用 uuid4 兜底，避免 tool_call_id 对不上。"""
    tool_call_id = str((tool_call or {}).get("id") or "").strip()
    return tool_call_id or uuid.uuid4().hex


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Tool:
    """工具定义（对齐 DashScope function-calling 要求）。"""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    required: List[str]
    handler: Callable[[Dict[str, Any]], str]

    def to_openai_dict(self) -> Dict[str, Any]:
        """转成 OpenAI 格式 function definition。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


@dataclass
class AgentMessage:
    """单条对话消息。"""

    role: str  # "user" | "assistant" | "tool" | "system"
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None  # assistant 调用工具时
    name: Optional[str] = None  # tool 返回结果时是工具名
    tool_call_id: Optional[str] = None  # tool 返回结果对应 id


@dataclass
class AgentState:
    """智能体运行状态。"""

    messages: List[AgentMessage] = field(default_factory=list)
    drug_cache: List[str] = field(default_factory=list)  # 会话中已出现的药物名，用于指代消解
    max_rounds: int = 10  # 最大工具调用轮次，防止死循环
    current_round: int = 0
    call_history: List[Tuple[str, str]] = field(default_factory=list)  # (tool_name, args_json) 调用历史
    loop_detected: bool = False  # 检测到死循环时置位

    def ingest_drug_names(self, text: str) -> None:
        """从文本中提取药物名并缓存（去重、保序），供指代消解使用。"""
        for name in extract_drug_name_candidates(text):
            name = str(name).strip()
            if name and name not in self.drug_cache:
                self.drug_cache.append(name)

    def merge_drug_cache(self, names: Optional[List[str]] = None) -> None:
        """合并外部传入的会话级药名缓存（用于跨请求持久化）。"""
        for name in names or []:
            name = str(name).strip()
            if name and name not in self.drug_cache:
                self.drug_cache.append(name)

    def add_system_message(self, content: str) -> None:
        self.messages.insert(0, AgentMessage(role="system", content=content))

    def add_user_message(self, content: str) -> None:
        self.ingest_drug_names(content)
        self.messages.append(AgentMessage(role="user", content=content))

    def add_history(self, history: Optional[List[Dict[str, Any]]] = None) -> None:
        """注入多轮对话历史（仅 user/assistant 文本消息），用于保持上下文。

        history 元素形如 {"role": "user"|"assistant", "content": "..."}；
        在 system 之后、当前问题之前顺序追加，帮助模型承接上文。
        同时从历史文本中提取药物名到 drug_cache，用于指代消解。
        """
        for msg in history or []:
            role = msg.get("role")
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                self.add_user_message(content)
            elif role == "assistant":
                self.ingest_drug_names(content)
                self.add_assistant_message(content, None)

    def add_assistant_message(self, content: str, tool_calls: Optional[List] = None) -> None:
        self.messages.append(AgentMessage(role="assistant", content=content, tool_calls=tool_calls))

    def add_tool_result(self, tool_name: str, tool_call_id: str, result: str) -> None:
        self.messages.append(
            AgentMessage(role="tool", content=result, name=tool_name, tool_call_id=tool_call_id)
        )

    def can_continue(self) -> bool:
        return self.current_round < self.max_rounds

    def increment_round(self) -> None:
        self.current_round += 1

    def to_dashscope_format(self) -> List[Dict[str, Any]]:
        """转成 DashScope 要求的消息格式。"""
        out = []
        for m in self.messages:
            msg: Dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls is not None:
                msg["tool_calls"] = m.tool_calls
            if m.name is not None:
                msg["name"] = m.name
            if m.tool_call_id is not None:
                msg["tool_call_id"] = m.tool_call_id
            out.append(msg)
        return out


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """工具注册表，管理可用工具并生成 function definitions。"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: List[str],
        handler: Callable[[Dict[str, Any]], str],
    ) -> None:
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            required=required,
            handler=handler,
        )

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_functions(self) -> List[Dict[str, Any]]:
        return [t.to_openai_dict() for t in self._tools.values()]

    def call_tool(self, name: str, args: Dict[str, Any]) -> str:
        tool = self.get_tool(name)
        if tool is None:
            return "❌ 未找到工具: %s" % name
        try:
            return tool.handler(args)
        except Exception as e:
            return "❌ 工具执行异常: %s" % str(e)


# =============================================================================
# Agent Core Loop
# =============================================================================

class AgentCore:
    """DashScope 原生 function-calling 决策循环。"""

    def __init__(
        self,
        registry: ToolRegistry,
        model: str = "",
        system_prompt: str = "",
        api_key: Optional[str] = None,
    ):
        self.registry = registry
        self.model = model or LLM_MODEL  # 未显式指定时取配置（.env 的 LLM_MODEL，默认 qwen3-max）
        self.system_prompt = system_prompt
        self.api_key = api_key or get_required_env("DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key

    def _call_dashscope(self, messages: List[Dict[str, Any]], stream: bool = False):
        """调用 DashScope Generation API。

        stream=True 时返回逐块 GenerationResponse 生成器。dashscope 在 qwen 系列上
        默认合并增量输出，因此每块的 content / tool_calls 都是「已累积全文」而非增量
        delta；需要增量文本时由调用方自行做差。
        """
        functions = self.registry.list_functions()
        return Generation.call(
            model=self.model,
            messages=messages,
            tools=functions,
            result_format="message",
            stream=stream,
        )

    def _execute_tool_calls(
        self,
        state: AgentState,
        tool_calls: List[Dict[str, Any]],
        on_tool: Optional[Callable[[str], None]] = None,
    ) -> None:
        """执行模型发起的一轮工具调用，并把结果回灌到对话状态。

        on_tool(name)：可选回调，在每个工具执行前触发，供 UI 展示当前阶段。
        """
        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            name = func.get("name", "")
            if on_tool:
                on_tool(name)
            arguments = func.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    args = json.loads(arguments)
                except json.JSONDecodeError as e:
                    result = "❌ 参数解析失败: %s" % str(e)
                    args = {}
            else:
                args = arguments or {}

            # 决策循环防抖：记录 (tool_name, args_json) 调用历史
            args_json = json.dumps(args, ensure_ascii=False, sort_keys=True)
            state.call_history.append((name, args_json))
            # 同一调用连续出现 2 次即视为死循环，强制停止
            if len(state.call_history) >= 2 and state.call_history[-1] == state.call_history[-2]:
                state.loop_detected = True
                result = "检测到重复工具调用，已停止"
                state.add_tool_result(name, _resolve_tool_call_id(tool_call), result)
                return

            result = self.registry.call_tool(name, args)
            state.add_tool_result(name, _resolve_tool_call_id(tool_call), result)

    @staticmethod
    def _resolve_coreference(user_query: str, drug_cache: List[str]) -> str:
        """对含指代的问题做轻量消解：把会话缓存中的候选药名作为解析提示注入。

        不做字符串替换（会破坏句子），而是给模型提供候选，由模型在生成
        rag_search/conflict_checker 参数时自行解析为具体药名。
        """
        if not drug_cache:
            return user_query
        if not any(hint in user_query for hint in PRONOUN_HINTS):
            return user_query
        names = "、".join(drug_cache[:8])
        return (
            f"{user_query}\n\n"
            f"[上下文提示] 本会话此前提到的药物有：{names}。"
            f"若问题中的「这个药/刚才说的」等指代不明，请解析为上述最相关的一种或多种药物后再检索。"
        )

    def run(
        self,
        user_query: str,
        max_rounds: int = 10,
        history: Optional[List[Dict[str, Any]]] = None,
        drug_cache: Optional[List[str]] = None,
    ) -> Tuple[str, AgentState]:
        """运行完整的决策循环直到模型停止调用工具，返回最终回答。

        history：多轮对话历史（见 AgentState.add_history），默认为空（单轮问答）。
        drug_cache：跨请求持久化的会话级药名缓存，用于指代消解；同时也会从 history 自动提取。
        """
        state = AgentState(max_rounds=max_rounds)
        if self.system_prompt:
            state.add_system_message(self.system_prompt)
        state.add_history(history)
        state.merge_drug_cache(drug_cache)

        # 指代消解：把「这个药/刚才说的」等指代背后的候选药名注入问题，供模型解析后再检索。
        resolved_query = self._resolve_coreference(user_query, state.drug_cache)
        state.add_user_message(resolved_query)

        while state.can_continue():
            state.increment_round()
            response = self._call_dashscope(state.to_dashscope_format())

            if response.status_code != 200:
                return (
                    "❌ API 调用失败: %s (code=%s)" % (response.message, response.code),
                    state,
                )

            output = response.output
            choices = output.get("choices", []) if output else []
            if not choices:
                return "❌ 模型未返回有效内容", state

            choice = choices[0]
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls")
            content = msg.get("content", "")

            if not tool_calls:
                # 无工具调用，直接返回回答
                state.add_assistant_message(content, None)
                return content, state

            # 有工具调用，逐个执行并回灌结果
            state.add_assistant_message(content, tool_calls)
            self._execute_tool_calls(state, tool_calls)
            if state.loop_detected:
                return "检测到重复工具调用，已停止", state

        # 达到最大轮次
        return "⚠️ 已达到最大工具调用轮次(%d)，停止推理。" % max_rounds, state

    def run_stream(
        self,
        user_query: str,
        max_rounds: int = 10,
        history: Optional[List[Dict[str, Any]]] = None,
        drug_cache: Optional[List[str]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Generator[str, None, None]:
        """流式运行决策循环，逐段产出「最终回答」文本。

        与 run() 的区别：
          - 每轮 LLM 调用使用 stream=True 逐块消费；
          - 只有最终回答轮（无 tool_calls）的文本会以增量方式 yield 给调用方，
            实现打字机效果；工具调用中间轮不产出任何文本（保持非流执行），
            仅在工具执行前通过 on_status 回调通知阶段切换；
          - on_status(msg) 供 UI 展示「正在检索 → 正在生成」等过渡，避免用户干等。

        yield 出的均为 str 增量，可直接喂给 st.write_stream 等流式渲染组件。
        """
        state = AgentState(max_rounds=max_rounds)
        if self.system_prompt:
            state.add_system_message(self.system_prompt)
        state.add_history(history)
        state.merge_drug_cache(drug_cache)
        resolved_query = self._resolve_coreference(user_query, state.drug_cache)
        state.add_user_message(resolved_query)

        while state.can_continue():
            state.increment_round()
            if on_status:
                on_status("🤔 正在分析问题，规划检索方案...")

            stream = self._call_dashscope(state.to_dashscope_format(), stream=True)

            content = ""            # 本轮已累积全文（dashscope 流式默认给出累积值）
            tool_calls: Optional[List[Dict[str, Any]]] = None
            is_tool_round = False
            yielded_len = 0         # 已 yield 的文本长度，用于计算增量
            final_notified = False  # 「正在生成回答」提示只触发一次
            error_msg: Optional[str] = None

            for chunk in stream:
                if chunk.status_code != 200:
                    error_msg = "❌ API 调用失败: %s (code=%s)" % (chunk.message, chunk.code)
                    break
                output = chunk.output
                choices = output.get("choices", []) if output else []
                if choices:
                    msg = choices[0].get("message", {}) or {}
                    content = msg.get("content") or ""
                    tcs = msg.get("tool_calls")
                else:
                    # 兜底：个别 chunk 无 choices，走 output.text 累积文本
                    content = output.get("text") or ""
                    tcs = None
                if tcs:
                    is_tool_round = True
                    tool_calls = tcs
                # 仅最终回答轮产出文本；工具轮 content 一般为空，此处天然不产出
                if not is_tool_round:
                    delta = content[yielded_len:]
                    yielded_len = len(content)
                    if delta:
                        if on_status and not final_notified:
                            on_status("📝 正在生成回答...")
                            final_notified = True
                        yield delta

            if error_msg is not None:
                yield error_msg
                return

            if not is_tool_round:
                # 最终回答轮：文本已在上方逐步 yield，收尾记录到状态后结束
                state.add_assistant_message(content, None)
                return

            # 工具调用轮：非流，逐个执行并回灌结果
            state.add_assistant_message(content, tool_calls)
            self._execute_tool_calls(
                state,
                tool_calls,
                on_tool=(lambda n: on_status(_tool_status(n))) if on_status else None,
            )
            if state.loop_detected:
                yield "检测到重复工具调用，已停止"
                return
            if on_status:
                on_status("🤔 正在综合检索结果，准备回答...")

        yield "⚠️ 已达到最大工具调用轮次(%d)，停止推理。" % max_rounds


# =============================================================================
# Pre-built system prompt for medical agent
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """你是专业的家庭医疗智能助手，擅长解答用户关于家庭用药、常见病护理、健康生活方式的问题。

当用户提问时，你需要：
1. 优先使用 rag_search 工具检索家庭用药说明书，获取权威准确的用药信息
2. 如果 rag_search 返回「本地检索结果不充分」或证据不足，先调整关键词再次检索；仍不足且联网已开启时，使用 web_search 联网搜索
3. 拿到检索结果后先自检「证据是否足以完整回答」，不足时进行二次查询（换关键词或换工具），而非直接拼凑回答
4. 如果用户需要查找附近的医院/诊所，可以调用 search_nearby_hospitals 工具（需要用户提供当前位置描述，如小区名/街道/学校等）
5. 如果用户需要记录个人健康信息（过敏史、慢性病史、用药记录等），使用 save_user_medical_record 工具保存
6. 如果用户询问两种药物是否可以一起吃，使用 conflict_checker 工具查询相互作用
7. 结合工具返回的结果，用清晰易懂的中文给出回答，避免专业术语堆砌；对用药建议必须提醒「请仔细阅读药品说明书，遵医嘱使用」
8. 回答中凡依据知识库得出的结论，用【文件名·章节】标注来源；联网信息标注「据联网搜索」
9. 如果工具返回结果不足以回答，直接说明信息不足，不要编造

请保持回答专业、客观、谨慎，对不确定内容明确告知，必要时建议用户线下就医。
"""

# 关闭联网时，需从系统提示词中剔除的联网相关片段
_WEB_INSTRUCTION_FRAGMENTS = (
    "；仍不足且联网已开启时，使用 web_search 联网搜索",
    "；联网信息标注「据联网搜索」",
    "；仍不足且已开启联网时，调用 web_search",
    "，联网信息标注「据联网搜索」",
)


def _strip_web_instructions(prompt: str) -> str:
    """删除系统提示词中的联网搜索指令，保留其余内容。"""
    for frag in _WEB_INSTRUCTION_FRAGMENTS:
        prompt = prompt.replace(frag, "")
    return prompt


# =============================================================================
# Example: how to register built-in tools
# =============================================================================

def create_medical_agent(
    rag_search_handler: Callable[[str], str],
    web_search_handler: Callable[[str], str],
    search_hospitals_handler: Callable[[str], str],
    save_record_handler: Callable[[str, str], str],
    conflict_check_handler: Callable[[str, str], str],
    model: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    enable_web_search: bool = True,
) -> AgentCore:
    """快速创建装配好五大工具的医疗智能体。

    enable_web_search: False 时不注册 web_search 工具，并从系统提示词中剔除
                        联网搜索相关指令，使模型在关网状态下无从得知该工具存在。
    """
    reg = ToolRegistry()

    # 1. RAG 检索家庭用药
    reg.register(
        name="rag_search",
        description="在家庭用药知识库中搜索药品说明书信息，包括适应症、用法用量、不良反应、禁忌、注意事项等",
        parameters={
            "query": {
                "type": "string",
                "description": "搜索关键词或问题，例如「布洛芬用法用量」「二甲双胍不良反应」",
            }
        },
        required=["query"],
        handler=lambda args: rag_search_handler(args["query"]),
    )

    # 2. 联网搜索（仅在开启联网时注册）
    if enable_web_search:
        reg.register(
            name="web_search",
            description="搜索互联网获取最新医学资讯、指南、政策、价格等知识库没有的信息",
            parameters={
                "query": {
                    "type": "string",
                    "description": "搜索问题，例如「最新高血压治疗指南」「连花清瘟价格」",
                }
            },
            required=["query"],
            handler=lambda args: web_search_handler(args["query"]),
        )

    # 3. 附近医院查询
    reg.register(
        name="search_nearby_hospitals",
        description="搜索附近的正规医院或诊所",
        parameters={
            "location": {
                "type": "string",
                "description": "用户当前位置描述，例如「福州市闽侯县闽江学院」「北京市朝阳区」",
            }
        },
        required=["location"],
        handler=lambda args: search_hospitals_handler(args.get("location", "")),
    )

    # 4. 保存健康档案
    reg.register(
        name="save_user_medical_record",
        description="保存用户个人健康信息，比如过敏史、慢性病史、长期用药记录等",
        parameters={
            "key": {
                "type": "string",
                "description": "信息类别，比如「青霉素过敏史」「高血压病史」",
            },
            "value": {
                "type": "string",
                "description": "具体内容",
            },
        },
        required=["key", "value"],
        handler=lambda args: save_record_handler(args["key"], args["value"]),
    )

    # 5. 药物相互作用查询
    reg.register(
        name="conflict_checker",
        description="查询两种药物一起吃是否有禁忌或相互作用",
        parameters={
            "drug_a": {
                "type": "string",
                "description": "第一种药物名称",
            },
            "drug_b": {
                "type": "string",
                "description": "第二种药物名称",
            },
        },
        required=["drug_a", "drug_b"],
        handler=lambda args: conflict_check_handler(args["drug_a"], args["drug_b"]),
    )

    if not enable_web_search:
        system_prompt = _strip_web_instructions(system_prompt)

    return AgentCore(reg, model=model, system_prompt=system_prompt)