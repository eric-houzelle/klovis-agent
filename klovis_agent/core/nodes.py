from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic import BaseModel, ValidationError
from ulid import ULID

from klovis_agent.console import get_console
from klovis_agent.core.prompts import (
    CHECK_SYSTEM_PROMPT,
    CHECK_USER_TEMPLATE,
    EXECUTE_SYSTEM_PROMPT,
    EXECUTE_USER_TEMPLATE,
    FINISH_SYSTEM_PROMPT,
    FINISH_USER_TEMPLATE,
    PLAN_SYSTEM_PROMPT,
    PLAN_USER_TEMPLATE,
    REPLAN_SYSTEM_PROMPT,
    REPLAN_USER_TEMPLATE,
)
from klovis_agent.core.schemas import (
    CHECK_OUTPUT_SCHEMA,
    EXECUTE_OUTPUT_SCHEMA,
    FINISH_OUTPUT_SCHEMA,
    PLAN_OUTPUT_SCHEMA,
    REPLAN_OUTPUT_SCHEMA,
    CheckOutput,
    ExecuteOutput,
    FinishOutput,
    PlanOutput,
    ReplanOutput,
)
from klovis_agent.llm.router import LLMRouter
from klovis_agent.llm.types import ModelRequest
from klovis_agent.models.plan import ExecutionPlan
from klovis_agent.models.state import AgentState
from klovis_agent.models.step import StepResult, StepSpec
from klovis_agent.tools.docs import format_tool_catalog
from klovis_agent.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


def _append_schema_hint(system_prompt: str, schema: dict[str, object]) -> str:
    schema_text = json.dumps(schema, indent=2)
    return (
        f"{system_prompt}\n\n"
        f"You MUST respond with valid JSON matching this schema:\n{schema_text}"
    )


def _validate_output(
    model_cls: type[BaseModel], data: dict[str, object],
) -> BaseModel | None:
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        logger.warning(
            "structured_output_validation_failed",
            model=model_cls.__name__,
            error=str(exc),
        )
        return None



# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _format_prior_results(agent_state: AgentState) -> str:
    lines: list[str] = []

    successful = [r for r in agent_state.step_results if r.status == "success"]
    if successful:
        lines.append(
            "Prior step results (use this data, do NOT re-fetch or guess):"
        )
        for r in successful:
            step = (
                _find_step(agent_state.plan.steps, r.step_id)
                if agent_state.plan else None
            )
            label = step.title if step else r.step_id
            compact = json.dumps(r.outputs, ensure_ascii=False, default=str)
            if len(compact) > 3500:
                compact = compact[:3500] + "..."
            lines.append(
                f"  - [{label}] (tool: {r.tool_used or 'n/a'}) -> {compact}"
            )

    current_id = agent_state.current_step_id
    if current_id:
        failures = [
            r for r in agent_state.step_results
            if r.step_id == current_id and r.status == "failed"
        ]
        if failures:
            last_fail = failures[-1]
            lines.append("")
            lines.append(
                f"LAST FAILED ATTEMPT on this step (attempt {len(failures)}):"
            )
            lines.append(f"  Tool: {last_fail.tool_used or 'n/a'}")
            fail_out = json.dumps(
                last_fail.outputs, ensure_ascii=False, default=str,
            )
            if len(fail_out) > 3500:
                fail_out = fail_out[:3500] + "..."
            lines.append(f"  Output: {fail_out}")
            if last_fail.observations:
                lines.append(
                    f"  Observations: {'; '.join(last_fail.observations)}"
                )
            lines.append(
                "  FIX the issue described above. "
                "Do NOT repeat the same mistake."
            )

    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _format_failure_report(agent_state: AgentState) -> str:
    failed = [r for r in agent_state.step_results if r.status == "failed"]
    if not failed:
        return "No failures recorded."

    lines: list[str] = []
    for r in failed:
        step = (
            _find_step(agent_state.plan.steps, r.step_id)
            if agent_state.plan else None
        )
        label = step.title if step else r.step_id
        lines.append(f"FAILED: {label} (step_id: {r.step_id})")
        lines.append(f"  Tool used: {r.tool_used or 'n/a'}")
        out = json.dumps(r.outputs, ensure_ascii=False, default=str)
        if len(out) > 3500:
            out = out[:3500] + "..."
        lines.append(f"  Output: {out}")
        if r.observations:
            lines.append(f"  Observations: {'; '.join(r.observations)}")
        lines.append("")

    return "\n".join(lines)


def _step_number(agent_state: AgentState) -> tuple[int, int]:
    """Return (current_step_index, total_steps) for display."""
    if agent_state.plan is None:
        return 0, 0
    steps = agent_state.plan.steps
    total = len(steps)
    idx = next(
        (
            i for i, s in enumerate(steps)
            if s.step_id == agent_state.current_step_id
        ),
        0,
    )
    return idx + 1, total


def _output_preview(outputs: dict[str, Any]) -> str:
    """Build a short human-readable preview of tool outputs."""
    if not outputs:
        return ""
    if "response" in outputs:
        return str(outputs["response"])[:200]
    if "content" in outputs:
        return str(outputs["content"])[:200]
    if "stdout" in outputs:
        return str(outputs["stdout"]).strip()[:200]
    keys = list(outputs.keys())
    if len(keys) <= 4:
        return ", ".join(f"{k}={_short(outputs[k])}" for k in keys)
    return f"{len(keys)} fields returned"


def _short(val: Any) -> str:
    s = str(val)
    return s[:60] + "..." if len(s) > 60 else s


_NARRATE_INTENT_SYSTEM = (
    "You are a concise narrator for an autonomous agent. "
    "Summarize in ONE short sentence (max 15 words) what the agent is about to do. "
    "Be friendly and clear. Write in the same language as the objective."
)

_NARRATE_OUTCOME_SYSTEM = (
    "You are a concise narrator for an autonomous agent. "
    "Summarize in ONE short sentence (max 15 words) what just happened. "
    "Be friendly and clear. Write in the same language as the objective."
)


async def _stream_narration(
    llm: LLMRouter,
    con: Any,
    system: str,
    user: str,
) -> None:
    """Stream a short LLM-generated summary to the console."""
    request = ModelRequest(
        purpose="narration",
        system_prompt=system,
        user_prompt=user,
        max_tokens=80,
        temperature=0.3,
    )
    try:
        async for token in llm.invoke_stream(request):
            con.stream_token(token)
    except Exception:
        logger.debug("narration_stream_failed", exc_info=True)


def _record_llm_usage(
    state: dict[str, Any],
    *,
    phase: str,
    usage: dict[str, Any],
    con: Any,
) -> None:
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    total = int(usage.get("total_tokens", prompt + completion) or 0)
    if total <= 0 and prompt <= 0 and completion <= 0:
        return

    token_usage = state.get("_token_usage", {})
    token_usage["prompt_tokens"] = int(token_usage.get("prompt_tokens", 0)) + prompt
    token_usage["completion_tokens"] = int(
        token_usage.get("completion_tokens", 0),
    ) + completion
    token_usage["total_tokens"] = int(token_usage.get("total_tokens", 0)) + total
    token_usage["calls"] = int(token_usage.get("calls", 0)) + 1
    state["_token_usage"] = token_usage

    if con:
        con.llm_usage(phase, prompt, completion, total)


# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------

async def plan_node(
    state: dict[str, Any],
    *,
    llm: LLMRouter,
    tool_registry: ToolRegistry,
) -> dict[str, Any]:
    agent_state = AgentState(**state)
    con = get_console(state)
    task = agent_state.task

    all_tool_specs = tool_registry.list_specs()
    tools_doc = format_tool_catalog(all_tool_specs)

    system = PLAN_SYSTEM_PROMPT
    soul = state.get("soul", "")
    if soul:
        system = f"{system}\n\n{soul}"
    system = _append_schema_hint(system, PLAN_OUTPUT_SCHEMA)

    request = ModelRequest(
        purpose="planning",
        system_prompt=system,
        user_prompt=PLAN_USER_TEMPLATE.format(
            goal=task.goal,
            context=json.dumps(task.context),
            constraints=json.dumps(task.constraints),
            success_criteria=json.dumps(task.success_criteria),
            available_tools=tools_doc,
        ),
        structured_output_schema=PLAN_OUTPUT_SCHEMA,
    )

    response = await llm.invoke(request)
    _record_llm_usage(
        state,
        phase="plan",
        usage=response.usage,
        con=con,
    )

    if response.structured_output is None:
        con.plan_failed("LLM returned no structured output")
        logger.error("plan_node_no_structured_output")
        state["status"] = "failed"
        return state

    data = response.structured_output
    con._debug_json("PLAN", data)

    validated = _validate_output(PlanOutput, data)
    if validated is None:
        con.plan_failed("Structured output failed validation")
        logger.error("plan_node_validation_failed")
        state["status"] = "failed"
        return state

    raw_steps = data.get("steps", [])
    steps = [
        StepSpec(
            step_id=str(i + 1),
            step_type=s.get("step_type", "execute"),
            title=s["title"],
            objective=s["objective"],
            success_criteria=s.get("success_criteria", []),
            allowed_tools=s.get("allowed_tools", []),
            depends_on=s.get("depends_on", []),
        )
        for i, s in enumerate(raw_steps)
    ]

    plan = ExecutionPlan(
        plan_id=str(ULID()),
        version=1,
        goal=task.goal,
        steps=steps,
    )

    state["plan"] = plan.model_dump()
    if steps:
        state["current_step_id"] = steps[0].step_id

    con.plan(
        [{"title": s.title, "step_id": s.step_id} for s in steps],
        data.get("reasoning_summary", ""),
    )

    logger.info("plan_generated", num_steps=len(steps))
    return state


async def execute_node(
    state: dict[str, Any],
    *,
    llm: LLMRouter,
    tool_registry: ToolRegistry,
) -> dict[str, Any]:
    agent_state = AgentState(**state)
    con = get_console(state)

    if agent_state.plan is None or agent_state.current_step_id is None:
        state["status"] = "failed"
        return state

    step = _find_step(agent_state.plan.steps, agent_state.current_step_id)
    if step is None:
        state["status"] = "failed"
        return state

    step_num, total = _step_number(agent_state)
    con.step_start(step_num, total, step.title)

    # --- Stream intent narration ---
    if not con.quiet:
        con.step_intent_start()
        await _stream_narration(
            llm, con,
            system=_NARRATE_INTENT_SYSTEM,
            user=f"Step: {step.title}\nObjective: {step.objective}",
        )
        con.stream_end()

    step.status = "running"
    all_tool_specs = tool_registry.list_specs()
    tools_doc = format_tool_catalog(all_tool_specs)
    prior_results = _format_prior_results(agent_state)

    exec_system = EXECUTE_SYSTEM_PROMPT
    soul = state.get("soul", "")
    if soul:
        exec_system = f"{exec_system}\n\n{soul}"
    exec_system = _append_schema_hint(exec_system, EXECUTE_OUTPUT_SCHEMA)

    request = ModelRequest(
        purpose="execution",
        system_prompt=exec_system,
        user_prompt=EXECUTE_USER_TEMPLATE.format(
            step_title=step.title,
            step_objective=step.objective,
            inputs=json.dumps(step.inputs),
            prior_results=prior_results,
            tools_catalog=tools_doc,
            max_tokens=llm.effective_max_tokens("execution"),
        ),
        structured_output_schema=EXECUTE_OUTPUT_SCHEMA,
    )

    step_tokens_before = int(state.get("_token_usage", {}).get("total_tokens", 0))

    response = await llm.invoke(request)
    _record_llm_usage(
        state,
        phase="execute",
        usage=response.usage,
        con=con,
    )

    if response.structured_output is None:
        con.step_failed("LLM returned no structured output")
        _record_step_result(
            state, step.step_id, "failed",
            observations=["No LLM output"],
        )
        return state

    data = response.structured_output
    con._debug_json(f"EXECUTE ({step.title})", data)
    _validate_output(ExecuteOutput, data)

    outputs: dict[str, Any] = {}
    tool_used: str | None = None

    if data.get("action") == "tool_call" and data.get("tool_name"):
        tool_name = data["tool_name"]
        tool_input = data.get("tool_input", {})

        validation_error = _validate_tool_call(
            tool_registry, tool_name, tool_input,
        )
        if validation_error:
            logger.warning(
                "tool_call_invalid", tool=tool_name, error=validation_error,
            )
            con.step_failed(validation_error)
            _record_step_result(
                state, step.step_id, "failed",
                outputs={
                    "requested_tool": tool_name,
                    "requested_input": tool_input,
                },
                tool_used=tool_name,
                observations=[validation_error],
            )
            return state

        con.step_tool_call(tool_name, list(tool_input.keys()))

        logger.info(
            "execute_tool_input",
            tool=tool_name,
            input_keys=list(tool_input.keys()),
            has_files=bool(tool_input.get("files")),
            has_code=bool(tool_input.get("code")),
        )
        result = await tool_registry.invoke(tool_name, tool_input)
        tool_used = tool_name
        outputs = result.output
        logger.info(
            "execute_tool_output",
            tool=tool_name,
            success=result.success,
            output_keys=list(outputs.keys()),
        )
        if not result.success:
            con.step_failed(result.error or "Tool execution failed")
            _record_step_result(
                state, step.step_id, "failed",
                outputs=outputs, tool_used=tool_used,
                observations=[result.error or "Tool execution failed"],
            )
            return state
    else:
        direct = data.get("direct_response", "")
        outputs = {"response": direct}
        con.step_direct_response(direct)
        logger.info(
            "execute_direct_response",
            response_len=len(direct),
        )

    _record_step_result(
        state, step.step_id, "success",
        outputs=outputs, tool_used=tool_used,
    )

    preview = _output_preview(outputs)
    con.step_success(tool_used, preview)

    # --- Stream outcome narration ---
    if not con.quiet:
        step_tokens_after = int(
            state.get("_token_usage", {}).get("total_tokens", 0),
        )
        step_tokens = step_tokens_after - step_tokens_before

        con.step_outcome_start(success=True)
        await _stream_narration(
            llm, con,
            system=_NARRATE_OUTCOME_SYSTEM,
            user=(
                f"Step: {step.title}\n"
                f"Tool used: {tool_used or 'direct response'}\n"
                f"Result: {preview}"
            ),
        )
        con.step_outcome_end(step_tokens)

    logger.info("step_executed", step_id=step.step_id, tool=tool_used)
    return state


async def check_node(
    state: dict[str, Any],
    *,
    llm: LLMRouter,
) -> dict[str, Any]:
    agent_state = AgentState(**state)
    con = get_console(state)

    if not agent_state.step_results:
        state["status"] = "failed"
        return state

    last_result = agent_state.step_results[-1]
    step = _find_step(
        agent_state.plan.steps if agent_state.plan else [],
        last_result.step_id,
    )
    if step is None:
        state["status"] = "failed"
        return state

    check_system = _append_schema_hint(CHECK_SYSTEM_PROMPT, CHECK_OUTPUT_SCHEMA)

    request = ModelRequest(
        purpose="check",
        system_prompt=check_system,
        user_prompt=CHECK_USER_TEMPLATE.format(
            step_title=step.title,
            step_objective=step.objective,
            expected_outputs=json.dumps(step.expected_outputs),
            actual_outputs=json.dumps(last_result.outputs),
            success_criteria=json.dumps(step.success_criteria),
        ),
        structured_output_schema=CHECK_OUTPUT_SCHEMA,
    )

    response = await llm.invoke(request)
    _record_llm_usage(
        state,
        phase="check",
        usage=response.usage,
        con=con,
    )

    if response.structured_output is None:
        state["_check_decision"] = "replan"
        return state

    data = response.structured_output
    con._debug_json("CHECK", data)
    _validate_output(CheckOutput, data)

    next_action = data.get("next_action", "continue")
    state["_check_decision"] = next_action

    if data.get("observations"):
        last_result.observations.extend(data["observations"])
        results = state.get("step_results", [])
        if results:
            results[-1] = last_result.model_dump()

    if next_action == "continue":
        _advance_to_next_step(state, agent_state)

    con.check_result(
        data.get("status", "?"),
        next_action,
        data.get("observations", []),
    )

    state["iteration_count"] = agent_state.iteration_count + 1
    logger.info(
        "check_completed",
        decision=next_action,
        step_id=last_result.step_id,
    )
    return state


async def replan_node(
    state: dict[str, Any],
    *,
    llm: LLMRouter,
    tool_registry: ToolRegistry,
) -> dict[str, Any]:
    agent_state = AgentState(**state)
    con = get_console(state)

    if agent_state.plan is None:
        state["status"] = "failed"
        return state

    current_steps_desc = "\n".join(
        f"  - [{s.status}] {s.step_id}: {s.title}"
        for s in agent_state.plan.steps
    )

    failure_report = _format_failure_report(agent_state)

    all_tool_specs = tool_registry.list_specs()
    tools_doc = format_tool_catalog(all_tool_specs)

    replan_system = REPLAN_SYSTEM_PROMPT
    soul = state.get("soul", "")
    if soul:
        replan_system = f"{replan_system}\n\n{soul}"
    replan_system = _append_schema_hint(replan_system, REPLAN_OUTPUT_SCHEMA)

    request = ModelRequest(
        purpose="planning",
        system_prompt=replan_system,
        user_prompt=REPLAN_USER_TEMPLATE.format(
            goal=agent_state.plan.goal,
            version=agent_state.plan.version,
            current_steps=current_steps_desc,
            failed_steps=failure_report,
            observations="(included in failure report above)",
            available_tools=tools_doc,
        ),
        structured_output_schema=REPLAN_OUTPUT_SCHEMA,
    )

    response = await llm.invoke(request)
    _record_llm_usage(
        state,
        phase="replan",
        usage=response.usage,
        con=con,
    )

    if response.structured_output is None:
        state["status"] = "failed"
        return state

    data = response.structured_output
    con._debug_json("REPLAN", data)
    _validate_output(ReplanOutput, data)

    existing_max = max(
        (
            int(r.step_id)
            for r in agent_state.step_results
            if r.step_id.isdigit()
        ),
        default=0,
    )
    raw_updated = data.get("updated_steps", [])
    new_steps = [
        StepSpec(
            step_id=str(existing_max + i + 1),
            step_type=s.get("step_type", "execute"),
            title=s["title"],
            objective=s["objective"],
            success_criteria=s.get("success_criteria", []),
            allowed_tools=s.get("allowed_tools", []),
            depends_on=s.get("depends_on", []),
        )
        for i, s in enumerate(raw_updated)
    ]

    plan_data = state.get("plan", {})
    plan_data["version"] = agent_state.plan.version + 1
    plan_data["steps"] = [s.model_dump() for s in new_steps]
    plan_data["status"] = "active"
    state["plan"] = plan_data

    if new_steps:
        pending = [s for s in new_steps if s.status == "pending"]
        state["current_step_id"] = (
            pending[0].step_id if pending else new_steps[0].step_id
        )

    con.replan(
        agent_state.plan.version + 1,
        [{"title": s.title, "step_id": s.step_id} for s in new_steps],
        data.get("reasoning_summary", ""),
    )

    logger.info(
        "replanned",
        new_version=agent_state.plan.version + 1,
        num_steps=len(new_steps),
    )
    return state


async def finish_node(
    state: dict[str, Any],
    *,
    llm: LLMRouter,
) -> dict[str, Any]:
    agent_state = AgentState(**state)
    con = get_console(state)

    finish_system = FINISH_SYSTEM_PROMPT
    soul = state.get("soul", "")
    if soul:
        finish_system = f"{finish_system}\n\n{soul}"
    finish_system = _append_schema_hint(finish_system, FINISH_OUTPUT_SCHEMA)

    request = ModelRequest(
        purpose="finish",
        system_prompt=finish_system,
        user_prompt=FINISH_USER_TEMPLATE.format(
            goal=agent_state.task.goal,
            step_results=json.dumps(
                [r.model_dump() for r in agent_state.step_results],
            ),
            artifacts=json.dumps(agent_state.artifacts),
        ),
        structured_output_schema=FINISH_OUTPUT_SCHEMA,
    )

    response = await llm.invoke(request)
    _record_llm_usage(
        state,
        phase="finish",
        usage=response.usage,
        con=con,
    )

    if response.structured_output:
        con._debug_json("FINISH", response.structured_output)
        _validate_output(FinishOutput, response.structured_output)
        state["artifacts"]["_final_summary"] = response.structured_output

        summary = response.structured_output.get("summary", "")
        status = response.structured_output.get("overall_status", "?")
        con.finish(
            status,
            summary,
            agent_state.iteration_count,
            response.structured_output.get("limitations"),
        )

        if not con.quiet and summary:
            con.finish_narration_start()
            await _stream_narration(
                llm, con,
                system=(
                    "You are a concise narrator for an autonomous agent. "
                    "Rephrase this execution summary in 2-3 short, friendly sentences. "
                    "Keep it informative but concise. Write in the same language as the goal."
                ),
                user=f"Goal: {agent_state.task.goal}\nSummary: {summary}",
            )
            con.stream_end()

    state["status"] = "completed"
    state["artifacts"]["_token_usage"] = state.get("_token_usage", {})
    logger.info("agent_finished", run_id=agent_state.run_id)
    return state


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _validate_tool_call(
    registry: ToolRegistry, tool_name: str, tool_input: dict[str, Any],
) -> str | None:
    tool = registry.get(tool_name)
    if tool is None:
        available = [s.name for s in registry.list_specs()]
        return (
            f"Tool '{tool_name}' does not exist. "
            f"Available tools: {', '.join(available)}"
        )

    spec = tool.spec()
    required = spec.input_schema.get("required", [])
    missing = [f for f in required if f not in tool_input]
    if missing:
        return (
            f"Tool '{tool_name}' missing required fields: {missing}. "
            f"Expected schema: {json.dumps(spec.input_schema, default=str)}"
        )

    return None


def _find_step(steps: list[StepSpec], step_id: str) -> StepSpec | None:
    return next((s for s in steps if s.step_id == step_id), None)


def _record_step_result(
    state: dict[str, Any],
    step_id: str,
    status: str,
    outputs: dict[str, Any] | None = None,
    tool_used: str | None = None,
    observations: list[str] | None = None,
) -> None:
    result = StepResult(
        step_id=step_id,
        status=status,  # type: ignore[arg-type]
        outputs=outputs or {},
        tool_used=tool_used,
        observations=observations or [],
    )
    results = state.get("step_results", [])
    results.append(result.model_dump())
    state["step_results"] = results


def _advance_to_next_step(
    state: dict[str, Any], agent_state: AgentState,
) -> None:
    if agent_state.plan is None:
        return
    steps = agent_state.plan.steps
    current_idx = next(
        (
            i for i, s in enumerate(steps)
            if s.step_id == agent_state.current_step_id
        ),
        -1,
    )
    if current_idx >= 0 and current_idx + 1 < len(steps):
        state["current_step_id"] = steps[current_idx + 1].step_id
    else:
        state["current_step_id"] = None
