from __future__ import annotations

import pytest

pytest.importorskip("structlog")

from klovis_agent.models.state import AgentState
from klovis_agent.models.step import StepResult
from klovis_agent.models.task import Task
from klovis_agent.result import AgentResult
from klovis_agent.tools.builtin.discord_bot import format_discord_reply


def _result(
    *,
    summary: str,
    steps: list[StepResult],
) -> AgentResult:
    state = AgentState(
        run_id="r1",
        task=Task(task_id="t1", goal="test"),
        step_results=steps,
        artifacts={
            "_final_summary": {
                "summary": summary,
                "overall_status": "success",
                "artifacts_produced": [],
                "limitations": [],
            },
        },
        status="completed",
    )
    return AgentResult(state)


def test_format_discord_reply_keeps_summary_and_adds_direct_response() -> None:
    result = _result(
        summary="J'ai préparé le post.",
        steps=[
            StepResult(step_id="1", status="success", outputs={"response": "Voici ton post LinkedIn prêt à publier."}),
        ],
    )

    text = format_discord_reply(result)
    assert text.startswith("> 💭 *J'ai préparé le post.*")
    assert "Voici ton post LinkedIn prêt à publier." in text


def test_format_discord_reply_uses_content_when_no_response() -> None:
    result = _result(
        summary="Post généré dans un fichier.",
        steps=[
            StepResult(
                step_id="1",
                status="success",
                outputs={
                    "path": "linkedin_post.md",
                    "content": "Titre accrocheur\n\nTexte du post...",
                },
                tool_used="file_write",
            ),
        ],
    )

    text = format_discord_reply(result)
    assert text.startswith("> 💭 *Post généré dans un fichier.*")
    assert "Titre accrocheur" in text


def test_format_discord_reply_avoids_duplicate_when_direct_equals_summary() -> None:
    result = _result(
        summary="Task completed successfully",
        steps=[
            StepResult(
                step_id="1",
                status="success",
                outputs={"response": "Task completed successfully"},
            ),
        ],
    )

    text = format_discord_reply(result)
    assert text == "> 💭 *Task completed successfully*"
