from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ReasoningEffort = Literal["low", "medium", "high"]


class ModelRequest(BaseModel):
    """Normalized request to an LLM."""

    purpose: str
    system_prompt: str
    user_prompt: str
    structured_output_schema: dict[str, object] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None


class ModelResponse(BaseModel):
    """Normalized response from an LLM."""

    raw_text: str | None = None
    structured_output: dict[str, object] | None = None
    tool_calls: list[dict[str, object]] = Field(default_factory=list)
    usage: dict[str, object] = Field(default_factory=dict)
    model_name: str


class ModelRoutingPolicy(BaseModel):
    """Routing policy: which model and limits to use for each phase.

    Per-phase fields default to None, falling back to the global defaults
    set on LLMRouter / LLMConfig.
    """

    planning_model: str | None = None
    execution_model: str | None = None
    check_model: str | None = None
    finish_model: str | None = None
    narration_model: str | None = None

    planning_base_url: str | None = None
    execution_base_url: str | None = None
    check_base_url: str | None = None
    finish_base_url: str | None = None
    narration_base_url: str | None = None

    planning_max_tokens: int | None = None
    execution_max_tokens: int | None = None
    check_max_tokens: int | None = None
    finish_max_tokens: int | None = None
    narration_max_tokens: int | None = None

    planning_reasoning_effort: ReasoningEffort | None = None
    execution_reasoning_effort: ReasoningEffort | None = None
    check_reasoning_effort: ReasoningEffort | None = None
    finish_reasoning_effort: ReasoningEffort | None = None
    narration_reasoning_effort: ReasoningEffort | None = None

    def _phase_lookup(self, purpose: str, suffix: str) -> object | None:
        return getattr(self, f"{purpose}_{suffix}", None)

    def model_for_purpose(self, purpose: str, default_model: str) -> str:
        return self._phase_lookup(purpose, "model") or default_model  # type: ignore[return-value]

    def base_url_for_purpose(self, purpose: str) -> str | None:
        """Return per-phase base_url override, or None to use global default."""
        return self._phase_lookup(purpose, "base_url")  # type: ignore[return-value]

    def max_tokens_for_purpose(self, purpose: str) -> int | None:
        """Return per-phase max_tokens override, or None to use global default."""
        return self._phase_lookup(purpose, "max_tokens")  # type: ignore[return-value]

    def reasoning_effort_for_purpose(self, purpose: str) -> ReasoningEffort | None:
        """Return per-phase reasoning_effort, or None to skip the parameter."""
        return self._phase_lookup(purpose, "reasoning_effort")  # type: ignore[return-value]
