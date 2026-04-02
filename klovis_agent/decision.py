"""Decision module — should the agent act?

Given a list of perceived events and recalled memories, uses the LLM to
decide whether to launch a run, and if so, generates a concrete goal.
This module is source-agnostic: it doesn't know if events come from
Moltbook, email, RSS, or anything else.
"""

from __future__ import annotations

import json
from typing import Literal

import structlog
from pydantic import BaseModel, Field, model_validator

from klovis_agent.llm.router import LLMRouter
from klovis_agent.llm.types import ModelRequest
from klovis_agent.perception.base import PerceptionResult

logger = structlog.get_logger(__name__)


class DecisionOutput(BaseModel):
    """Structured output expected from the decision LLM call."""

    model_config = {"extra": "forbid"}

    should_act: bool = Field(
        description="True if the agent should launch a run right now",
    )
    reasoning: str = Field(
        description="Brief explanation of why the agent should or should not act",
    )
    goal: str = Field(
        default="",
        description="The concrete goal for the run (only meaningful when should_act is true)",
    )
    priority: Literal["low", "medium", "high"] = Field(
        default="low",
        description="How urgent is this action",
    )

    @model_validator(mode="after")
    def _coherence_check(self) -> DecisionOutput:
        """Fix incoherent responses: goal set but should_act is False."""
        if not self.should_act and self.goal:
            logger.warning(
                "decision_incoherent",
                goal=self.goal[:80],
                msg="LLM provided a goal but should_act=False — clearing goal",
            )
            self.goal = ""
        if self.should_act and not self.goal:
            logger.warning(
                "decision_incoherent",
                msg="LLM set should_act=True but provided no goal — forcing idle",
            )
            self.should_act = False
            self.reasoning = f"(forced idle: no goal provided) {self.reasoning}"
        return self

    @property
    def label(self) -> str:
        if not self.should_act:
            return f"IDLE ({self.reasoning})"
        return f"ACT [{self.priority}]: {self.goal}"


DECISION_SCHEMA: dict[str, object] = DecisionOutput.model_json_schema()


DECISION_SYSTEM_PROMPT = """\
You are the decision engine of an autonomous AI agent.
You are evaluating whether to take action based on events perceived from
your environment (social platforms, messages, scheduled tasks, etc.).

You must decide: should you launch a task right now, or stay silent?

Guidelines:
- Act if an event genuinely warrants a response (a question directed at you,
  a meaningful comment, a DM, a scheduled task that's due).
- Do NOT act on trivial events (someone upvoted something — nice, but no
  action needed).
- Do NOT act if you recently acted on the same topic (avoid spam / loops).
  CHECK THE MEMORIES BELOW CAREFULLY: if a memory says you already replied
  to a post, commented on a thread, or handled a DM, do NOT do it again.
  Memories tagged "action_taken" are records of past actions — respect them.
- Prefer quality over quantity. One thoughtful action is better than five
  generic ones.
- If you decide to act, formulate a SPECIFIC goal. Not "check notifications"
  but "Reply to the comment from X about Y" or "Write the scheduled weekly
  summary".
- Your goal should reference the tools available to the agent — it will be
  passed as the goal of a full agent run.

You MUST respond with valid JSON matching this schema:
{schema}
"""

DECISION_USER_TEMPLATE = """\
## Perceived Events
{events_text}

## Relevant Memories (including past actions — do NOT repeat them)
{recalled_memories}

Based on these events and your past actions, should you act now?
If a memory shows you already handled an event, skip it.
If yes, what specific NEW goal?
"""


Decision = DecisionOutput


async def decide(
    perception: PerceptionResult,
    recalled_memories: str,
    llm: LLMRouter,
    soul: str = "",
) -> DecisionOutput:
    """Evaluate perceived events and decide whether to act."""

    if not perception.has_events:
        logger.info("decision_skip_no_events")
        return DecisionOutput(
            should_act=False,
            goal="",
            reasoning="No events detected",
            priority="low",
        )

    schema_text = json.dumps(DECISION_SCHEMA, indent=2)

    system = DECISION_SYSTEM_PROMPT.format(schema=schema_text)
    if soul:
        system = f"{system}\n\n{soul}"

    request = ModelRequest(
        purpose="planning",
        system_prompt=system,
        user_prompt=DECISION_USER_TEMPLATE.format(
            events_text=perception.as_text(),
            recalled_memories=recalled_memories or "(no prior memories)",
        ),
        structured_output_schema=DECISION_SCHEMA,
        temperature=0.3,
    )

    try:
        response = await llm.invoke(request)
    except Exception as exc:
        logger.warning("decision_llm_failed", error=str(exc))
        return DecisionOutput(
            should_act=False,
            goal="",
            reasoning=f"LLM error: {exc}",
            priority="low",
        )

    data = response.structured_output or {}

    try:
        decision = DecisionOutput.model_validate(data)
    except Exception as exc:
        logger.warning("decision_validation_failed", error=str(exc), raw=data)
        decision = DecisionOutput(
            should_act=False,
            goal="",
            reasoning=f"Validation error: {exc}",
            priority="low",
        )

    logger.info(
        "decision_made",
        should_act=decision.should_act,
        priority=decision.priority,
        goal=decision.goal[:80] if decision.goal else "",
        reasoning=decision.reasoning[:120],
    )
    return decision
