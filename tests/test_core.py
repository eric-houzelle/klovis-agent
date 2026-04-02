"""Unit tests for klovis_agent.core (graph builder, routing logic).

Depends on langgraph.
"""

from __future__ import annotations

import pytest

langgraph = pytest.importorskip("langgraph")

from klovis_agent.core.graph import (
    MAX_REPLANS,
    MAX_RETRIES_PER_STEP,
    _check_router,
    _consecutive_attempts_on_step,
    _total_replans,
)


class TestConsecutiveAttempts:
    def test_no_results(self):
        state = {"current_step_id": "1", "step_results": []}
        assert _consecutive_attempts_on_step(state) == 0

    def test_no_current_step(self):
        state = {"current_step_id": None, "step_results": []}
        assert _consecutive_attempts_on_step(state) == 0

    def test_single_attempt(self):
        state = {
            "current_step_id": "1",
            "step_results": [{"step_id": "1", "status": "failed"}],
        }
        assert _consecutive_attempts_on_step(state) == 1

    def test_multiple_attempts(self):
        state = {
            "current_step_id": "2",
            "step_results": [
                {"step_id": "1", "status": "success"},
                {"step_id": "2", "status": "failed"},
                {"step_id": "2", "status": "failed"},
            ],
        }
        assert _consecutive_attempts_on_step(state) == 2

    def test_mixed_steps(self):
        state = {
            "current_step_id": "2",
            "step_results": [
                {"step_id": "2", "status": "failed"},
                {"step_id": "1", "status": "success"},
                {"step_id": "2", "status": "failed"},
            ],
        }
        assert _consecutive_attempts_on_step(state) == 1


class TestTotalReplans:
    def test_no_plan(self):
        assert _total_replans({}) == 0

    def test_version_1(self):
        assert _total_replans({"plan": {"version": 1}}) == 0

    def test_version_3(self):
        assert _total_replans({"plan": {"version": 3}}) == 2


class TestCheckRouter:
    def test_max_iterations(self):
        state = {"iteration_count": 25, "max_iterations": 25}
        assert _check_router(state) == "finish"

    def test_failed_status(self):
        state = {"status": "failed", "iteration_count": 0, "max_iterations": 25}
        assert _check_router(state) == "finish"

    def test_finish_decision(self):
        state = {
            "_check_decision": "finish",
            "iteration_count": 0,
            "max_iterations": 25,
        }
        assert _check_router(state) == "finish"

    def test_continue_with_next_step(self):
        state = {
            "_check_decision": "continue",
            "iteration_count": 1,
            "max_iterations": 25,
            "current_step_id": "2",
        }
        assert _check_router(state) == "execute"

    def test_continue_no_more_steps(self):
        state = {
            "_check_decision": "continue",
            "iteration_count": 1,
            "max_iterations": 25,
            "current_step_id": None,
        }
        assert _check_router(state) == "finish"

    def test_retry(self):
        state = {
            "_check_decision": "retry",
            "iteration_count": 1,
            "max_iterations": 25,
            "current_step_id": "1",
            "step_results": [{"step_id": "1", "status": "failed"}],
        }
        assert _check_router(state) == "execute"

    def test_retry_max_reached_triggers_replan(self):
        state = {
            "_check_decision": "retry",
            "iteration_count": 1,
            "max_iterations": 25,
            "current_step_id": "1",
            "step_results": [{"step_id": "1"}] * MAX_RETRIES_PER_STEP,
            "plan": {"version": 1},
        }
        assert _check_router(state) == "replan"

    def test_retry_max_reached_max_replans_finishes(self):
        state = {
            "_check_decision": "retry",
            "iteration_count": 1,
            "max_iterations": 25,
            "current_step_id": "1",
            "step_results": [{"step_id": "1"}] * MAX_RETRIES_PER_STEP,
            "plan": {"version": MAX_REPLANS + 1},
        }
        assert _check_router(state) == "finish"

    def test_replan_decision(self):
        state = {
            "_check_decision": "replan",
            "iteration_count": 1,
            "max_iterations": 25,
            "current_step_id": "1",
            "step_results": [{"step_id": "1"}],
            "plan": {"version": 1},
        }
        assert _check_router(state) == "replan"

    def test_replan_max_reached(self):
        state = {
            "_check_decision": "replan",
            "iteration_count": 1,
            "max_iterations": 25,
            "current_step_id": "1",
            "step_results": [{"step_id": "1"}],
            "plan": {"version": MAX_REPLANS + 1},
        }
        assert _check_router(state) == "finish"
