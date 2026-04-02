"""Unit tests for config, result, and decision modules.

- LLMConfig / SandboxConfig depend on pydantic_settings (AgentConfig only)
- DecisionOutput depends on structlog
- AgentResult is pure Python + Pydantic
"""

from __future__ import annotations

import pytest

from klovis_agent.models.state import AgentState
from klovis_agent.models.task import Task
from klovis_agent.result import AgentResult


class TestAgentResult:
    def _make_state(self, **kwargs) -> AgentState:
        task = Task(task_id="t1", goal="test goal")
        defaults = {
            "run_id": "r1",
            "task": task,
            "status": "completed",
            "iteration_count": 3,
        }
        defaults.update(kwargs)
        return AgentState(**defaults)

    def test_basic_properties(self):
        state = self._make_state()
        result = AgentResult(state)
        assert result.run_id == "r1"
        assert result.status == "completed"
        assert result.goal == "test goal"
        assert result.iteration_count == 3

    def test_summary_from_dict(self):
        state = self._make_state(
            artifacts={"_final_summary": {"summary": "All done", "overall_status": "success"}}
        )
        result = AgentResult(state)
        assert result.summary == "All done"

    def test_summary_from_string(self):
        state = self._make_state(artifacts={"_final_summary": "plain text"})
        result = AgentResult(state)
        assert result.summary == "plain text"

    def test_summary_empty(self):
        state = self._make_state()
        result = AgentResult(state)
        assert result.summary == ""

    def test_str(self):
        state = self._make_state(
            artifacts={"_final_summary": {"summary": "Done!"}}
        )
        result = AgentResult(state)
        assert str(result) == "Done!"

    def test_str_no_summary(self):
        state = self._make_state()
        result = AgentResult(state)
        assert "test goal" in str(result)

    def test_repr(self):
        state = self._make_state()
        result = AgentResult(state)
        assert "AgentResult" in repr(result)
        assert "r1" in repr(result)

    def test_raw_state(self):
        state = self._make_state()
        result = AgentResult(state)
        assert result.raw_state is state


pydantic_settings = pytest.importorskip("pydantic_settings")

from klovis_agent.config import AgentConfig, LLMConfig, SandboxConfig


class TestLLMConfig:
    def test_defaults(self):
        c = LLMConfig()
        assert c.default_model == "gpt-4o"
        assert c.temperature == 0.2
        assert c.max_tokens == 4096

    def test_custom(self):
        c = LLMConfig(api_key="sk-test", default_model="gpt-3.5", temperature=0.8)
        assert c.api_key == "sk-test"
        assert c.temperature == 0.8


class TestSandboxConfig:
    def test_defaults(self):
        c = SandboxConfig()
        assert c.backend == "local"
        assert c.timeout == 30

    def test_opensandbox(self):
        c = SandboxConfig(backend="opensandbox", domain="my.host:9090")
        assert c.domain == "my.host:9090"


class TestAgentConfig:
    def test_defaults(self):
        c = AgentConfig(llm=LLMConfig(api_key="test"))
        assert c.max_iterations == 25
        assert c.verbose is False


structlog = pytest.importorskip("structlog")

from klovis_agent.decision import DecisionOutput


class TestDecisionOutput:
    def test_should_act(self):
        d = DecisionOutput(should_act=True, reasoning="urgent", goal="do X", priority="high")
        assert "ACT" in d.label
        assert "do X" in d.label

    def test_idle(self):
        d = DecisionOutput(should_act=False, reasoning="nothing to do")
        assert "IDLE" in d.label
        assert d.goal == ""

    def test_coherence_goal_without_act(self):
        d = DecisionOutput(should_act=False, reasoning="nah", goal="but I set a goal")
        assert d.goal == ""

    def test_coherence_act_without_goal(self):
        d = DecisionOutput(should_act=True, reasoning="want to act", goal="")
        assert d.should_act is False
        assert "forced idle" in d.reasoning
