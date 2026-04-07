"""Post-run memory consolidation.

After each run, the agent reflects on what happened and decides what
is worth remembering long-term. This is the "sleep" phase where
short-term experiences become persistent knowledge.
"""

from __future__ import annotations

import json

import structlog

from klovis_agent.llm.embeddings import EmbeddingClient
from klovis_agent.llm.router import LLMRouter
from klovis_agent.llm.types import ModelRequest
from klovis_agent.models.state import AgentState
from klovis_agent.tools.builtin.semantic_memory import SemanticMemoryStore

logger = structlog.get_logger(__name__)

CONSOLIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "A concise, self-contained piece of knowledge to remember",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "zone": {
                        "type": "string",
                        "enum": ["episodic", "semantic"],
                        "description": (
                            "'episodic' for time-bound actions/events (auto-expires), "
                            "'semantic' for permanent facts/lessons/identity"
                        ),
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "mission", "state", "preference", "fact", "lesson",
                            "strategy", "action", "identity", "other",
                        ],
                        "description": (
                            "Structured memory category. "
                            "Use 'action' for concrete actions performed."
                        ),
                    },
                },
                "required": ["content", "tags", "zone", "type"],
            },
        },
    },
    "required": ["memories"],
}

CONSOLIDATION_PROMPT = """\
You just completed a task. Reflect on what happened and extract knowledge
worth remembering for future runs.

Goal: {goal}
Status: {status}
Summary: {summary}

Step results (condensed):
{steps}

What should you remember? Extract 2-6 concise memories. Each memory should be:
- Self-contained (understandable without the original context)
- Actionable or informative (a lesson, a fact, a preference, a skill)
- NOT a raw copy of the task or output — distill the insight

Each memory MUST include a "zone" field:
- **"episodic"** — for time-bound events: actions you took, interactions,
  things that happened during this specific run. These auto-expire after
  ~2 weeks. Use for anything that answers "what did I do / what happened?"
- **"semantic"** — for permanent knowledge: facts, lessons, preferences,
  identity, skills. These persist forever and get deduplicated automatically.
  Use for anything that answers "what do I know / who am I?"

Categories of things worth remembering:

- **Actions taken** → zone: "episodic" (CRITICAL — always include at least one):
  Record WHAT you did, WHERE (post ID, username, platform), and WHEN.
  This prevents you from repeating the same action in future cycles.
  Tag these with "action_taken". type should be "action".
- **Facts you discovered** → zone: "semantic": names, IDs, platform features, API behaviors
- **Lessons from errors** → zone: "semantic": what went wrong and how you fixed it
- **Preferences** → zone: "semantic": how the user likes things done, what works well
- **Skills** → zone: "semantic": techniques or patterns that proved effective
- **Skills acquired** → zone: "semantic": record which skill you installed,
  what it does, and when it was useful. type should be "lesson" or "strategy".
  Example: {{"content": "Installed 'vercel-deploy' skill to deploy Next.js apps.
  Provides endpoints for creating deployments and checking status.",
  "tags": ["skill_acquired", "vercel"], "zone": "semantic", "type": "lesson"}}
- **Identity** → zone: "semantic": things about yourself (your name, your profile, your style)

Examples of good memories:
- {{"content": "Replied to nex_v4's comment on post ee22ee81 ('Agent Reflection') on 2026-03-31.", "tags": ["action_taken", "moltbook"], "zone": "episodic"}}
- {{"content": "Rejected DM request from opencodeai01 (spam/promotional) on 2026-03-31.", "tags": ["action_taken", "moltbook", "dm"], "zone": "episodic"}}
- {{"content": "My Moltbook username is cortexvista and I have 7 karma.", "tags": ["identity", "moltbook"], "zone": "semantic"}}
- {{"content": "The Moltbook API sometimes returns 500 errors; retrying usually works.", "tags": ["lesson", "api"], "zone": "semantic"}}

IMPORTANT: You MUST include at least one "episodic" / "action_taken" memory
that records the concrete action(s) performed during this run.

Respond with ONLY a JSON object in this exact format (no extra text):
{{"memories": [{{"content": "...", "tags": ["tag1", "tag2"], "zone": "episodic|semantic", "type": "fact"}}]}}
"""


async def consolidate_run(
    state: AgentState,
    llm: LLMRouter,
    embedder: EmbeddingClient,
    store: SemanticMemoryStore,
) -> int:
    """Extract and store memories from a completed run. Returns count of new memories."""
    summary_data = state.artifacts.get("_final_summary", {})
    if isinstance(summary_data, dict):
        summary = summary_data.get("summary", "")
        status = summary_data.get("overall_status", state.status)
    else:
        summary = str(summary_data)
        status = state.status

    steps_condensed = []
    for r in state.step_results:
        out_str = json.dumps(r.outputs, ensure_ascii=False, default=str)
        if len(out_str) > 500:
            out_str = out_str[:500] + "..."
        obs = "; ".join(r.observations) if r.observations else ""
        steps_condensed.append(
            f"  [{r.status}] {r.step_id} (tool: {r.tool_used or 'n/a'}) "
            f"→ {out_str}"
            + (f" | obs: {obs}" if obs else "")
        )

    prompt = CONSOLIDATION_PROMPT.format(
        goal=state.task.goal,
        status=status,
        summary=summary or "(no summary)",
        steps="\n".join(steps_condensed) or "(no steps)",
    )

    logger.info(
        "consolidation_prompt",
        goal=state.task.goal,
        status=status,
        summary_len=len(summary or ""),
        num_steps=len(steps_condensed),
    )

    request = ModelRequest(
        purpose="finish",
        system_prompt=(
            "You are a memory consolidation engine for an autonomous AI agent. "
            "Your job is to extract useful facts, lessons, and observations from "
            "a completed run so the agent can remember them in future runs. "
            "Always find something worth remembering. Respond with valid JSON only."
        ),
        user_prompt=prompt,
        structured_output_schema=CONSOLIDATION_SCHEMA,
    )

    try:
        response = await llm.invoke(request)
    except Exception as exc:
        logger.warning("consolidation_llm_failed", error=str(exc))
        return 0

    logger.info(
        "consolidation_llm_response",
        has_structured=response.structured_output is not None,
        raw_text_head=(response.raw_text or "")[:300] if response.raw_text else None,
        structured_output=response.structured_output,
    )

    if not response.structured_output:
        logger.warning(
            "consolidation_no_structured_output",
            raw_text=(response.raw_text or "")[:500],
        )
        return 0

    memories = response.structured_output.get("memories", [])
    stored = 0
    for mem in memories:
        content = mem.get("content", "").strip()
        if not content:
            continue
        tags = mem.get("tags", [])
        zone = mem.get("zone", "semantic")
        memory_type = mem.get("type", "")
        if zone not in ("episodic", "semantic"):
            zone = "episodic" if "action_taken" in tags else "semantic"
        if not memory_type:
            memory_type = "action" if zone == "episodic" else "fact"
        try:
            embedding = await embedder.embed_one(content)
            meta = {"tags": tags, "run_id": state.run_id, "type": memory_type}
            store.add(content=content, embedding=embedding, metadata=meta, zone=zone)
            stored += 1
            logger.info(
                "memory_consolidated",
                content=content[:80],
                tags=tags,
                zone=zone,
                memory_type=memory_type,
            )
        except Exception as exc:
            logger.warning("consolidation_embed_failed", error=str(exc))

    logger.info("consolidation_complete", extracted=len(memories), stored=stored)
    return stored
