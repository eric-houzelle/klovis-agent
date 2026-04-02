"""Unit tests for klovis_agent.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from klovis_agent.models.artifact import CodeArtifact, ExecutionResult
from klovis_agent.models.plan import ExecutionPlan, PlanStepResult
from klovis_agent.models.state import AgentState
from klovis_agent.models.step import StepResult, StepSpec
from klovis_agent.models.task import Task


class TestTask:
    def test_minimal(self):
        t = Task(task_id="1", goal="do something")
        assert t.task_id == "1"
        assert t.goal == "do something"
        assert t.context == {}
        assert t.constraints == {}
        assert t.success_criteria == []

    def test_full(self):
        t = Task(
            task_id="abc",
            goal="build X",
            context={"key": "val"},
            constraints={"time": 60},
            success_criteria=["works", "fast"],
        )
        assert t.context["key"] == "val"
        assert len(t.success_criteria) == 2

    def test_roundtrip_json(self):
        t = Task(task_id="rt", goal="roundtrip", context={"a": 1})
        data = t.model_dump_json()
        t2 = Task.model_validate_json(data)
        assert t == t2


class TestStepSpec:
    def test_defaults(self):
        s = StepSpec(step_id="1", step_type="execute", title="T", objective="O")
        assert s.status == "pending"
        assert s.allowed_tools == []
        assert s.depends_on == []
        assert s.retry_limit == 0

    def test_invalid_step_type(self):
        with pytest.raises(ValidationError):
            StepSpec(step_id="1", step_type="invalid", title="T", objective="O")


class TestStepResult:
    def test_success(self):
        r = StepResult(step_id="1", status="success", outputs={"x": 42}, tool_used="web_search")
        assert r.status == "success"
        assert r.tool_used == "web_search"

    def test_failed_with_observations(self):
        r = StepResult(
            step_id="2",
            status="failed",
            observations=["timeout", "retried"],
        )
        assert len(r.observations) == 2

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            StepResult(step_id="1", status="unknown")


class TestExecutionPlan:
    def test_empty_plan(self):
        p = ExecutionPlan(plan_id="p1", goal="test")
        assert p.version == 1
        assert p.steps == []
        assert p.status == "active"

    def test_with_steps(self):
        steps = [
            StepSpec(step_id="1", step_type="execute", title="A", objective="do A"),
            StepSpec(step_id="2", step_type="execute", title="B", objective="do B"),
        ]
        p = ExecutionPlan(plan_id="p2", goal="multi", steps=steps)
        assert len(p.steps) == 2
        assert p.steps[0].title == "A"


class TestAgentState:
    def test_defaults(self):
        t = Task(task_id="1", goal="g")
        s = AgentState(run_id="r1", task=t)
        assert s.status == "running"
        assert s.iteration_count == 0
        assert s.max_iterations == 25
        assert s.plan is None
        assert s.step_results == []
        assert s.artifacts == {}

    def test_roundtrip(self):
        t = Task(task_id="1", goal="g")
        s = AgentState(run_id="r1", task=t, verbose=True, max_iterations=10)
        d = s.model_dump()
        s2 = AgentState(**d)
        assert s2.verbose is True
        assert s2.max_iterations == 10


class TestCodeArtifact:
    def test_basic(self):
        a = CodeArtifact(
            artifact_id="a1",
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('hi')"},
        )
        assert "main.py" in a.files


class TestExecutionResult:
    def test_success(self):
        r = ExecutionResult(status="success", stdout="hello")
        assert r.stderr == ""

    def test_error(self):
        r = ExecutionResult(status="error", stderr="boom")
        assert r.status == "error"

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            ExecutionResult(status="unknown")
