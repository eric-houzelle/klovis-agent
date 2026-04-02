"""End-to-end test of the agentic loop with a mocked LLM."""

from __future__ import annotations

import pytest

from klovis_agent import Agent, LLMConfig
from klovis_agent.llm.gateway import ModelGateway
from klovis_agent.llm.types import ModelRequest, ModelResponse


class FakeGateway:
    """LLM gateway that returns predefined responses based on the phase."""

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        if request.purpose == "planning":
            return self._plan_response()
        if request.purpose == "execution":
            return self._execute_response()
        if request.purpose == "check":
            return self._check_response()
        return self._finish_response()

    def _plan_response(self) -> ModelResponse:
        return ModelResponse(
            structured_output={
                "reasoning_summary": "Single step plan for testing",
                "steps": [
                    {
                        "step_id": "step-1",
                        "step_type": "execute",
                        "title": "Compute result",
                        "objective": "Return a direct response",
                        "success_criteria": ["Response produced"],
                        "allowed_tools": [],
                        "depends_on": [],
                    }
                ],
            },
            model_name="fake",
        )

    def _execute_response(self) -> ModelResponse:
        return ModelResponse(
            structured_output={
                "action": "direct_response",
                "direct_response": "42",
            },
            model_name="fake",
        )

    def _check_response(self) -> ModelResponse:
        return ModelResponse(
            structured_output={
                "status": "success",
                "observations": ["Result looks correct"],
                "next_action": "finish",
            },
            model_name="fake",
        )

    def _finish_response(self) -> ModelResponse:
        return ModelResponse(
            structured_output={
                "summary": "Task completed successfully",
                "overall_status": "success",
                "artifacts_produced": [],
                "limitations": [],
            },
            model_name="fake",
        )


assert isinstance(FakeGateway(), ModelGateway)


@pytest.fixture
def agent() -> Agent:
    """Agent with a mocked LLM and real LocalSandbox — no network calls."""
    from klovis_agent.sandbox.service import LocalSandbox

    config = LLMConfig(api_key="fake", base_url="http://fake")
    ag = Agent(llm=config, sandbox=LocalSandbox(), ephemeral=True)

    fake = FakeGateway()
    ag._llm._get_gateway = lambda model, base_url=None: fake  # type: ignore[method-assign]
    return ag


@pytest.mark.asyncio
async def test_full_loop(agent: Agent) -> None:
    result = await agent.run("Compute the answer to everything")

    assert result.status == "completed"
    assert result.iteration_count >= 1
    assert len(result.steps) >= 1
    assert result.steps[0].status == "success"
    assert result.summary
