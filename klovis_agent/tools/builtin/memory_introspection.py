"""Memory introspection tool — visual snapshot of the agent's memory state.

Provides a structured, human-readable overview of what the agent
currently "has in mind": counts by zone and type, top tags, and
short previews per category.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec
from klovis_agent.tools.builtin.semantic_memory import SemanticMemoryStore

logger = structlog.get_logger(__name__)

_ZONE_ICONS = {"semantic": "🧠", "episodic": "⏳"}
_TYPE_ICONS = {
    "mission": "🎯",
    "state": "📍",
    "preference": "💡",
    "fact": "📚",
    "lesson": "🎓",
    "strategy": "♟️",
    "action": "⚡",
    "identity": "🪞",
    "other": "📦",
}


def _ts_ago(ts: float | None) -> str:
    if ts is None:
        return "n/a"
    delta = time.time() - ts
    if delta < 3600:
        return f"{int(delta / 60)}min ago"
    if delta < 86400:
        return f"{delta / 3600:.1f}h ago"
    return f"{delta / 86400:.1f}d ago"


def _render_text(data: dict[str, Any]) -> str:
    """Build a visual text report from introspection data."""
    lines: list[str] = []
    total = data["total_memories"]

    lines.append("╔══════════════════════════════════════════╗")
    lines.append(f"║  MEMORY STATE — {total} memories total       ║")
    lines.append("╚══════════════════════════════════════════╝")
    lines.append("")

    if data["oldest_memory_ts"] or data["newest_memory_ts"]:
        lines.append(
            f"  Oldest: {_ts_ago(data['oldest_memory_ts'])}  "
            f"│  Newest: {_ts_ago(data['newest_memory_ts'])}"
        )
        lines.append("")

    by_zone = data.get("by_zone", {})
    by_zone_type = data.get("by_zone_and_type", {})
    previews = data.get("previews", {})

    for zone in ("semantic", "episodic"):
        zone_count = by_zone.get(zone, 0)
        if zone_count == 0:
            continue
        icon = _ZONE_ICONS.get(zone, "")
        lines.append(f"  {icon} {zone.upper()} ({zone_count})")
        lines.append(f"  {'─' * 38}")

        type_counts = by_zone_type.get(zone, {})
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

        for mtype, count in sorted_types:
            ticon = _TYPE_ICONS.get(mtype, "•")
            bar_len = min(count, 20)
            bar = "█" * bar_len + ("+" if count > 20 else "")
            lines.append(f"    {ticon} {mtype:<12} {bar} ({count})")

            zone_previews = previews.get(zone, {}).get(mtype, [])
            for p in zone_previews:
                snippet = p["snippet"].replace("\n", " ")
                tags_str = ""
                if p.get("tags"):
                    tags_str = f" [{', '.join(p['tags'][:3])}]"
                lines.append(f"       └─ \"{snippet}\"{tags_str}")

        lines.append("")

    top_tags = data.get("top_tags", [])
    if top_tags:
        lines.append("  🏷️  TOP TAGS")
        lines.append(f"  {'─' * 38}")
        for entry in top_tags[:10]:
            tag, count = entry["tag"], entry["count"]
            bar = "▪" * min(count, 15) + ("+" if count > 15 else "")
            lines.append(f"    #{tag:<20} {bar} ({count})")
        lines.append("")

    return "\n".join(lines)


class MemoryIntrospectionTool(BaseTool):
    """Lets the agent (or a human observer) see a snapshot of memory state.

    Returns both structured data and a formatted text visualization.
    """

    def __init__(self, store: SemanticMemoryStore) -> None:
        super().__init__()
        self._store = store

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="memory_map",
            description=(
                "Visualize the current state of long-term memory. "
                "Shows a breakdown by zone (semantic/episodic), by type "
                "(fact, lesson, action, identity, …), top tags, and "
                "short previews per category. Use this to understand "
                "what the agent currently 'knows' and how its memory "
                "is distributed. No parameters needed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["visual", "data", "both"],
                        "description": (
                            "'visual' for a formatted text report (default), "
                            "'data' for raw structured JSON, "
                            "'both' for both."
                        ),
                    },
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "report": {"type": "string"},
                    "data": {"type": "object"},
                },
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        fmt = inputs.get("format", "visual")
        if fmt not in ("visual", "data", "both"):
            fmt = "visual"

        try:
            data = self._store.introspect()
        except Exception as exc:
            logger.warning("memory_introspection_failed", error=str(exc))
            return ToolResult(success=False, error=f"Introspection failed: {exc}")

        output: dict[str, Any] = {}

        if fmt in ("visual", "both"):
            output["report"] = _render_text(data)
        if fmt in ("data", "both"):
            output["data"] = data

        logger.info(
            "memory_introspection",
            total=data["total_memories"],
            zones=data["by_zone"],
        )
        return ToolResult(success=True, output=output)
