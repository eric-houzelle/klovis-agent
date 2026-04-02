"""Pre-run memory recall.

Before planning, the agent recalls relevant memories from past runs
and injects them into the task context. This gives the planner
awareness of prior experiences, learned lessons, and accumulated knowledge.

Recall queries both memory zones independently:
  - **episodic** — recent actions/events, ranked by similarity × recency
  - **semantic** — permanent facts/lessons, ranked by pure similarity
Results are merged so the planner always sees both "what I did recently"
and "what I know".
"""

from __future__ import annotations

import structlog

from klovis_agent.llm.embeddings import EmbeddingClient
from klovis_agent.tools.builtin.semantic_memory import SemanticMemoryStore

logger = structlog.get_logger(__name__)

MAX_RECALLED_EPISODIC = 4
MAX_RECALLED_SEMANTIC = 4
MIN_SIMILARITY = 0.30


async def recall_for_task(
    goal: str,
    embedder: EmbeddingClient,
    store: SemanticMemoryStore,
    k_episodic: int = MAX_RECALLED_EPISODIC,
    k_semantic: int = MAX_RECALLED_SEMANTIC,
) -> str:
    """Recall relevant memories for a task goal. Returns formatted context string."""
    store.prune_episodic()

    if store.count() == 0:
        return ""

    try:
        query_embedding = await embedder.embed_one(goal)
    except Exception as exc:
        logger.warning("recall_embed_failed", error=str(exc))
        return ""

    results = store.search_zones(
        query_embedding,
        k_episodic=k_episodic,
        k_semantic=k_semantic,
        min_similarity=MIN_SIMILARITY,
    )

    if not results:
        logger.info("recall_no_matches", goal=goal[:80])
        return ""

    episodic = [r for r in results if r.get("zone") == "episodic"]
    semantic = [r for r in results if r.get("zone") == "semantic"]

    lines: list[str] = []

    if episodic:
        lines.append("Recent actions & events (episodic memory):")
        for r in episodic:
            tags = r.get("metadata", {}).get("tags", [])
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(
                f"  - {r['content']}{tag_str} "
                f"(relevance: {r['similarity']}, score: {r.get('score', r['similarity'])})"
            )

    if semantic:
        lines.append("Permanent knowledge (semantic memory):")
        for r in semantic:
            tags = r.get("metadata", {}).get("tags", [])
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(
                f"  - {r['content']}{tag_str} (relevance: {r['similarity']})"
            )

    logger.info(
        "recall_complete",
        goal=goal[:80],
        episodic_found=len(episodic),
        semantic_found=len(semantic),
    )
    return "\n".join(lines)
