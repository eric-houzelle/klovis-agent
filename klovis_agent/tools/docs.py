"""Generate human-readable tool documentation from ToolSpec objects."""

from __future__ import annotations

from klovis_agent.tools.base import ToolSpec


def _format_schema_fields(schema: dict[str, object], indent: int = 4) -> str:
    """Render a JSON-schema 'properties' block as a readable parameter list."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))  # type: ignore[arg-type]
    if not properties:
        return f"{' ' * indent}(none)"

    lines: list[str] = []
    pad = " " * indent
    for name, prop in properties.items():  # type: ignore[union-attr]
        if not isinstance(prop, dict):
            continue
        ptype = prop.get("type", "any")
        tag = "REQUIRED" if name in required else "optional"
        desc = prop.get("description", "")
        enum_vals = prop.get("enum")

        parts = [f"{pad}- {name} ({ptype}, {tag})"]
        if desc:
            parts.append(f": {desc}")
        if enum_vals:
            parts.append(f"  [values: {', '.join(str(v) for v in enum_vals)}]")
        lines.append("".join(parts))
    return "\n".join(lines)


def format_tool_doc(spec: ToolSpec) -> str:
    """Format a single ToolSpec into a readable documentation block."""
    lines = [
        f"### {spec.name}",
        f"  {spec.description}",
    ]

    if spec.input_schema.get("properties"):
        lines.append("  Parameters:")
        lines.append(_format_schema_fields(spec.input_schema))
    else:
        lines.append("  Parameters: (none)")

    if spec.output_schema.get("properties"):
        lines.append("  Returns:")
        lines.append(_format_schema_fields(spec.output_schema))

    return "\n".join(lines)


def format_tool_catalog(specs: list[ToolSpec]) -> str:
    """Format all tool specs into a single readable catalog for LLM prompts."""
    if not specs:
        return "(no tools available)"

    groups: dict[str, list[ToolSpec]] = {}
    for spec in specs:
        prefix = spec.name.split("_")[0] if "_" in spec.name else "_other"
        groups.setdefault(prefix, []).append(spec)

    sections: list[str] = []
    for group_name, group_specs in groups.items():
        section_lines = [f"## {group_name} tools"]
        for spec in group_specs:
            section_lines.append(format_tool_doc(spec))
        sections.append("\n".join(section_lines))

    header = (
        f"# Available Tools ({len(specs)} total)\n"
        "Call these via action=\"tool_call\", tool_name=\"<name>\", "
        "tool_input={{...}} matching the parameters below.\n"
        "Do NOT write code that imports these tools — they are native agent tools.\n"
    )
    return header + "\n\n".join(sections)


def format_tool_summary(specs: list[ToolSpec]) -> str:
    """Short summary (name + one-liner) for planning prompts."""
    if not specs:
        return "(no tools available)"

    lines: list[str] = []
    for spec in specs:
        lines.append(f"- {spec.name}: {spec.description}")
    return "\n".join(lines)
