"""Unit tests for klovis_agent.llm (types, gateway JSON extraction).

ModelRequest / ModelResponse / ModelRoutingPolicy are pure Pydantic models.
_extract_json is a standalone function that only needs structlog at gateway
module level — we importorskip it.
"""

from __future__ import annotations

import pytest

from klovis_agent.llm.types import ModelRequest, ModelResponse, ModelRoutingPolicy


class TestModelRequest:
    def test_minimal(self):
        r = ModelRequest(purpose="test", system_prompt="sys", user_prompt="usr")
        assert r.temperature is None
        assert r.max_tokens is None
        assert r.structured_output_schema is None

    def test_full(self):
        r = ModelRequest(
            purpose="planning",
            system_prompt="sys",
            user_prompt="usr",
            temperature=0.5,
            max_tokens=1000,
            reasoning_effort="high",
            structured_output_schema={"type": "object"},
        )
        assert r.reasoning_effort == "high"


class TestModelResponse:
    def test_structured(self):
        r = ModelResponse(
            structured_output={"key": "val"},
            model_name="gpt-4o",
        )
        assert r.raw_text is None
        assert r.structured_output["key"] == "val"

    def test_raw_text(self):
        r = ModelResponse(raw_text="hello", model_name="gpt-4o")
        assert r.structured_output is None


class TestModelRoutingPolicy:
    def test_defaults(self):
        p = ModelRoutingPolicy()
        assert p.model_for_purpose("planning", "gpt-4o") == "gpt-4o"
        assert p.base_url_for_purpose("planning") is None
        assert p.max_tokens_for_purpose("planning") is None

    def test_overrides(self):
        p = ModelRoutingPolicy(
            planning_model="gpt-3.5",
            execution_max_tokens=8000,
            check_reasoning_effort="low",
        )
        assert p.model_for_purpose("planning", "gpt-4o") == "gpt-3.5"
        assert p.model_for_purpose("execution", "gpt-4o") == "gpt-4o"
        assert p.max_tokens_for_purpose("execution") == 8000
        assert p.reasoning_effort_for_purpose("check") == "low"

    def test_unknown_purpose_falls_back(self):
        p = ModelRoutingPolicy()
        assert p.model_for_purpose("unknown", "default") == "default"


structlog = pytest.importorskip("structlog")

from klovis_agent.llm.gateway import _extract_json


class TestExtractJson:
    def test_raw_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self):
        text = '```json\n{"a": 1}\n```'
        assert _extract_json(text) == {"a": 1}

    def test_markdown_fenced_no_lang(self):
        text = '```\n{"a": 1}\n```'
        assert _extract_json(text) == {"a": 1}

    def test_json_in_text(self):
        text = 'Here is the result:\n{"status": "ok", "value": 42}\nDone.'
        result = _extract_json(text)
        assert result["status"] == "ok"

    def test_nested_braces(self):
        text = '{"outer": {"inner": "val"}}'
        result = _extract_json(text)
        assert result["outer"]["inner"] == "val"

    def test_no_json(self):
        assert _extract_json("just plain text") is None

    def test_empty_string(self):
        assert _extract_json("") is None

    def test_whitespace(self):
        assert _extract_json('  \n  {"x": 1}  \n  ') == {"x": 1}

    def test_reasoning_before_json(self):
        text = "Let me think about this...\nOkay here:\n{\"answer\": 42}"
        result = _extract_json(text)
        assert result["answer"] == 42
