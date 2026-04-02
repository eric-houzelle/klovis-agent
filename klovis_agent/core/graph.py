from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph

from klovis_agent.core.nodes import check_node, execute_node, finish_node, plan_node, replan_node
from klovis_agent.llm.router import LLMRouter
from klovis_agent.tools.registry import ToolRegistry


MAX_RETRIES_PER_STEP = 3
MAX_REPLANS = 3


def _consecutive_attempts_on_step(state: dict[str, Any]) -> int:
    """Count consecutive results (success or failed) on the current step.

    This captures *all* attempts — including ones where the tool succeeded but
    the checker asked to retry anyway.
    """
    current = state.get("current_step_id")
    if not current:
        return 0
    results = state.get("step_results", [])
    count = 0
    for r in reversed(results):
        if r.get("step_id") == current:
            count += 1
        else:
            break
    return count


def _total_replans(state: dict[str, Any]) -> int:
    """Count how many times the plan has been revised."""
    plan = state.get("plan")
    if not plan:
        return 0
    return max(0, plan.get("version", 1) - 1)


def _check_router(state: dict[str, Any]) -> str:
    """Decide the transition after the check node."""
    decision = state.get("_check_decision", "continue")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 25)

    if iteration >= max_iter:
        return "finish"

    if state.get("status") == "failed":
        return "finish"

    if decision == "finish":
        return "finish"

    if decision in ("retry", "replan"):
        attempts = _consecutive_attempts_on_step(state)

        if attempts >= MAX_RETRIES_PER_STEP:
            if _total_replans(state) >= MAX_REPLANS:
                return "finish"
            return "replan"

        if decision == "replan":
            if _total_replans(state) >= MAX_REPLANS:
                return "finish"
            return "replan"

        return "execute"

    if state.get("current_step_id") is None:
        return "finish"

    return "execute"


def build_agent_graph(
    llm: LLMRouter,
    tool_registry: ToolRegistry,
) -> StateGraph:
    """Build the LangGraph graph for the agentic loop.

    Flow:
        plan → execute → check
        check → execute  (continue / retry)
        check → replan   (replan)
        replan → execute
        check → finish   (finish / max iterations)
    """
    graph = StateGraph(dict)

    graph.add_node("plan", partial(plan_node, llm=llm, tool_registry=tool_registry))
    graph.add_node("execute", partial(execute_node, llm=llm, tool_registry=tool_registry))
    graph.add_node("check", partial(check_node, llm=llm))
    graph.add_node("replan", partial(replan_node, llm=llm, tool_registry=tool_registry))
    graph.add_node("finish", partial(finish_node, llm=llm))

    graph.set_entry_point("plan")

    graph.add_edge("plan", "execute")
    graph.add_conditional_edges(
        "check",
        _check_router,
        {
            "execute": "execute",
            "replan": "replan",
            "finish": "finish",
        },
    )
    graph.add_edge("execute", "check")
    graph.add_edge("replan", "execute")
    graph.add_edge("finish", END)

    return graph
