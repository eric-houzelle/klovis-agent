"""User-friendly wrapper around the internal AgentState."""

from __future__ import annotations

from typing import Any

from klovis_agent.models.state import AgentState
from klovis_agent.models.step import StepResult


class AgentResult:
    """Public result of an agent run.

    Wraps the internal ``AgentState`` with a cleaner API for library consumers.
    """

    def __init__(self, state: AgentState) -> None:
        self._state = state

    @property
    def run_id(self) -> str:
        return self._state.run_id

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def summary(self) -> str:
        raw = self._state.artifacts.get("_final_summary")
        if isinstance(raw, dict):
            return raw.get("summary", "")
        return str(raw) if raw else ""

    @property
    def steps(self) -> list[StepResult]:
        return list(self._state.step_results)

    @property
    def artifacts(self) -> dict[str, Any]:
        return dict(self._state.artifacts)

    @property
    def iteration_count(self) -> int:
        return self._state.iteration_count

    @property
    def goal(self) -> str:
        return self._state.task.goal

    @property
    def raw_state(self) -> AgentState:
        """Access the underlying AgentState for advanced use cases."""
        return self._state

    def __repr__(self) -> str:
        return (
            f"AgentResult(run_id={self.run_id!r}, status={self.status!r}, "
            f"steps={len(self.steps)}, iterations={self.iteration_count})"
        )

    def __str__(self) -> str:
        return self.summary or f"[{self.status}] {self.goal}"
