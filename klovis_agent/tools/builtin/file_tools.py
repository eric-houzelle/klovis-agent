from __future__ import annotations

from typing import Any

from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec
from klovis_agent.tools.workspace import AgentWorkspace

MAX_READ_CHARS = 100_000
MAX_WRITE_PREVIEW_CHARS = 15_000


class FileReadTool(BaseTool):
    """Read file contents from the shared agent workspace."""

    def __init__(self, workspace: AgentWorkspace) -> None:
        self._workspace = workspace

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_read",
            description=(
                "Read the contents of a file from the workspace. "
                "Use this to inspect files created by previous steps. "
                "Supports optional offset and limit for large files."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file within the workspace",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start reading from",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return",
                    },
                },
                "required": ["path"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        path_str = inputs.get("path", "")
        if not path_str:
            return ToolResult(success=False, error="Missing required field: 'path'")

        try:
            filepath = self._workspace.resolve(path_str)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        if not filepath.exists():
            return ToolResult(success=False, error=f"File not found: {path_str}")
        if not filepath.is_file():
            return ToolResult(success=False, error=f"Not a file: {path_str}")

        try:
            raw = filepath.read_text(encoding="utf-8")
            lines = raw.splitlines(keepends=True)
            offset = max(1, inputs.get("offset", 1))
            limit = inputs.get("limit")
            selected = lines[offset - 1:]
            if limit is not None and limit > 0:
                selected = selected[:limit]

            content = "".join(selected)[:MAX_READ_CHARS]
            return ToolResult(
                success=True,
                output={
                    "content": content,
                    "total_lines": len(lines),
                    "returned_from_line": offset,
                    "returned_lines": len(selected),
                    "size_bytes": filepath.stat().st_size,
                    "truncated": len("".join(selected)) > MAX_READ_CHARS,
                },
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Read error: {exc}")


class FileWriteTool(BaseTool):
    """Write file contents to the shared agent workspace."""

    def __init__(self, workspace: AgentWorkspace) -> None:
        self._workspace = workspace

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_write",
            description=(
                "Write content to a file in the workspace. "
                "Creates parent directories automatically. "
                "Use this to create or overwrite files that persist across steps."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path for the file within the workspace",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write to the file",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, append to existing file instead of overwriting",
                    },
                },
                "required": ["path", "content"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        path_str = inputs.get("path", "")
        content = inputs.get("content", "")
        append = inputs.get("append", False)

        if not path_str:
            return ToolResult(success=False, error="Missing required field: 'path'")

        try:
            filepath = self._workspace.resolve(path_str)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(filepath, mode, encoding="utf-8") as f:
                f.write(content)

            final_content = filepath.read_text(encoding="utf-8")
            preview = final_content[:MAX_WRITE_PREVIEW_CHARS]

            return ToolResult(
                success=True,
                output={
                    "path": path_str,
                    "size_bytes": filepath.stat().st_size,
                    "action": "appended" if append else "written",
                    "content": preview,
                    "truncated": len(final_content) > MAX_WRITE_PREVIEW_CHARS,
                },
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Write error: {exc}")


class FileEditTool(BaseTool):
    """Edit files in the workspace via search-and-replace or insertion."""

    def __init__(self, workspace: AgentWorkspace) -> None:
        self._workspace = workspace

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_edit",
            description=(
                "Edit an existing file in the workspace. Two modes:\n"
                "  - 'replace': find old_content and replace with new_content.\n"
                "  - 'insert': insert new_content before or after a marker string, "
                "or at a specific line number.\n"
                "Use this instead of file_write when you need to modify part of "
                "a file without rewriting the whole thing."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file within the workspace",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["replace", "insert"],
                        "description": "'replace' for search-and-replace, 'insert' for insertion",
                    },
                    "old_content": {
                        "type": "string",
                        "description": "(replace mode) Exact text to find and replace",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "Replacement text (replace mode) or text to insert (insert mode)",
                    },
                    "marker": {
                        "type": "string",
                        "description": "(insert mode) Text to insert before/after",
                    },
                    "position": {
                        "type": "string",
                        "enum": ["before", "after"],
                        "description": "(insert mode) Insert before or after the marker",
                    },
                    "line": {
                        "type": "integer",
                        "description": "(insert mode) 1-based line number to insert at (alternative to marker)",
                    },
                },
                "required": ["path", "mode", "new_content"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        path_str = inputs.get("path", "")
        mode = inputs.get("mode", "")
        new_content = inputs.get("new_content", "")

        if not path_str:
            return ToolResult(success=False, error="Missing required field: 'path'")
        if mode not in ("replace", "insert"):
            return ToolResult(success=False, error="mode must be 'replace' or 'insert'")

        try:
            filepath = self._workspace.resolve(path_str)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        if not filepath.exists():
            return ToolResult(success=False, error=f"File not found: {path_str}")
        if not filepath.is_file():
            return ToolResult(success=False, error=f"Not a file: {path_str}")

        try:
            original = filepath.read_text(encoding="utf-8")
        except Exception as exc:
            return ToolResult(success=False, error=f"Read error: {exc}")

        if mode == "replace":
            return self._do_replace(filepath, path_str, original, inputs, new_content)
        return self._do_insert(filepath, path_str, original, inputs, new_content)

    def _do_replace(
        self,
        filepath: "Path",  # noqa: F821
        path_str: str,
        original: str,
        inputs: dict[str, Any],
        new_content: str,
    ) -> ToolResult:
        old_content = inputs.get("old_content", "")
        if not old_content:
            return ToolResult(
                success=False,
                error="replace mode requires 'old_content'",
            )

        count = original.count(old_content)
        if count == 0:
            return ToolResult(
                success=False,
                error="old_content not found in file",
                output={"hint": "Make sure old_content matches the file exactly (whitespace matters)"},
            )

        updated = original.replace(old_content, new_content, 1)
        filepath.write_text(updated, encoding="utf-8")

        return ToolResult(
            success=True,
            output={
                "path": path_str,
                "replacements": 1,
                "occurrences_found": count,
                "size_bytes": filepath.stat().st_size,
            },
        )

    def _do_insert(
        self,
        filepath: "Path",  # noqa: F821
        path_str: str,
        original: str,
        inputs: dict[str, Any],
        new_content: str,
    ) -> ToolResult:
        marker = inputs.get("marker")
        position = inputs.get("position", "after")
        line_num = inputs.get("line")

        if marker:
            if marker not in original:
                return ToolResult(
                    success=False,
                    error="marker not found in file",
                )
            if position == "before":
                updated = original.replace(marker, new_content + marker, 1)
            else:
                updated = original.replace(marker, marker + new_content, 1)
        elif line_num is not None:
            lines = original.splitlines(keepends=True)
            idx = max(0, min(line_num - 1, len(lines)))
            lines.insert(idx, new_content if new_content.endswith("\n") else new_content + "\n")
            updated = "".join(lines)
        else:
            return ToolResult(
                success=False,
                error="insert mode requires either 'marker' or 'line'",
            )

        filepath.write_text(updated, encoding="utf-8")

        return ToolResult(
            success=True,
            output={
                "path": path_str,
                "action": "inserted",
                "size_bytes": filepath.stat().st_size,
            },
        )
