# -*- coding: utf-8 -*-
"""drug_interaction.py 的 _parse_interaction_json 单元测试。

被测函数是纯函数，不使用 dashscope。但模块顶层会 `import dashscope`，
而本机未安装且测试不允许联网/依赖 SDK，因此在导入前用 sys.modules 注入
一个占位 stub，不修改被测源码。
"""
import os
import sys
import types

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _P not in sys.path:
    sys.path.insert(0, _P)

# --- 注入 dashscope 占位，避免 import 报错（_parse_interaction_json 不会用到它） ---
if "dashscope" not in sys.modules:
    _dashscope = types.ModuleType("dashscope")
    _dashscope.api_key = ""
    _dashscope.Generation = types.SimpleNamespace(call=lambda *a, **k: None)
    sys.modules["dashscope"] = _dashscope

import drug_interaction  # noqa: E402


def _ok_json(has_conflict, risk_level, reason="依据说明", suggestion="建议"):
    return {
        "has_conflict": has_conflict,
        "risk_level": risk_level,
        "reason": reason,
        "suggestion": suggestion,
    }


def test_normal_json_parses_conflict_and_level():
    result = drug_interaction._parse_interaction_json(
        '{"has_conflict": true, "risk_level": "严重", "reason": "机制A", "suggestion": "看医生"}'
    )
    assert result["has_conflict"] is True
    assert result["risk_level"] == "严重"
    assert result["reason"] == "机制A"
    assert result["suggestion"] == "看医生"


def test_normal_json_parses_no_conflict():
    result = drug_interaction._parse_interaction_json(
        '{"has_conflict": false, "risk_level": "无", "reason": "", "suggestion": ""}'
    )
    assert result["has_conflict"] is False
    assert result["risk_level"] == "无"


def test_false_conflict_forces_risk_level_none():
    # has_conflict=false 时，即使 LLM 写了 risk_level=严重，也必须强制为「无」
    result = drug_interaction._parse_interaction_json(
        '{"has_conflict": false, "risk_level": "严重", "reason": "x", "suggestion": "y"}'
    )
    assert result["has_conflict"] is False
    assert result["risk_level"] == "无"


def test_invalid_risk_level_falls_back_to_moderate():
    # has_conflict=true 但 risk_level=高危（不在白名单）→ 回退「中度」
    result = drug_interaction._parse_interaction_json(
        '{"has_conflict": true, "risk_level": "高危", "reason": "x", "suggestion": "y"}'
    )
    assert result["has_conflict"] is True
    assert result["risk_level"] == "中度"


def test_risk_level_whitelist_values_kept():
    for level in ("严重", "中度", "轻微"):
        result = drug_interaction._parse_interaction_json(
            f'{{"has_conflict": true, "risk_level": "{level}", "reason": "x", "suggestion": "y"}}'
        )
        assert result["risk_level"] == level


def test_markdown_fence_is_stripped():
    content = '```json\n{"has_conflict": true, "risk_level": "轻微", "reason": "r", "suggestion": "s"}\n```'
    result = drug_interaction._parse_interaction_json(content)
    assert result["has_conflict"] is True
    assert result["risk_level"] == "轻微"
    assert result["reason"] == "r"


def test_surrounding_text_ignored():
    content = '解释：{"has_conflict": true, "risk_level": "中度", "reason": "r", "suggestion": "s"}（完毕）'
    result = drug_interaction._parse_interaction_json(content)
    assert result["has_conflict"] is True
    assert result["risk_level"] == "中度"


def test_missing_json_raises():
    import pytest
    with pytest.raises(ValueError):
        drug_interaction._parse_interaction_json("没有 JSON 对象")
    with pytest.raises(ValueError):
        drug_interaction._parse_interaction_json("")
    with pytest.raises(ValueError):
        drug_interaction._parse_interaction_json(None)