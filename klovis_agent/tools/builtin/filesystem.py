"""Filesystem tools — operate anywhere on the host machine.

Unlike the workspace-scoped file_read / file_write tools, these give the
agent access to the real filesystem.  Destructive operations (write, delete,
move) require user confirmation by default; this can be toggled per-instance.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec

MAX_READ_CHARS = 100_000


class FsReadTool(BaseTool):
    """Read a file from anywhere on the filesystem."""

    requires_confirmation = False

    def __init__(self, *, requires_confirmation: bool | None = None) -> None:
        super().__init__(requires_confirmation=requires_confirmation)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fs_read",
            description=(
                "Read the contents of a file at an absolute path on the host "
                "filesystem. Supports optional line offset and limit for large "
                "files."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file",
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

        filepath = Path(path_str).expanduser().resolve()
        if not filepath.exists():
            return ToolResult(success=False, error=f"File not found: {filepath}")
        if not filepath.is_file():
            return ToolResult(success=False, error=f"Not a file: {filepath}")

        try:
            raw = filepath.read_text(encoding="utf-8")
        except Exception as exc:
            return ToolResult(success=False, error=f"Read error: {exc}")

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
                "truncated": len("".join(selected)) > MAX_READ_CHARS,
                "size_bytes": filepath.stat().st_size,
            },
        )

    def describe_action(self, inputs: dict[str, Any]) -> str:
        return f"Read file: {inputs.get('path', '?')}"


class FsListTool(BaseTool):
    """List directory contents anywhere on the filesystem."""

    requires_confirmation = False

    def __init__(self, *, requires_confirmation: bool | None = None) -> None:
        super().__init__(requires_confirmation=requires_confirmation)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fs_list",
            description=(
                "List files and directories at an absolute path. "
                "Returns names, types, and sizes."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true, list contents recursively (max 500 entries)",
                    },
                },
                "required": ["path"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        path_str = inputs.get("path", "")
        if not path_str:
            return ToolResult(success=False, error="Missing required field: 'path'")

        dirpath = Path(path_str).expanduser().resolve()
        if not dirpath.exists():
            return ToolResult(success=False, error=f"Path not found: {dirpath}")
        if not dirpath.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {dirpath}")

        recursive = inputs.get("recursive", False)
        max_entries = 500
        entries: list[dict[str, Any]] = []

        try:
            iterator = dirpath.rglob("*") if recursive else dirpath.iterdir()
            for item in sorted(iterator):
                if len(entries) >= max_entries:
                    break
                rel = item.relative_to(dirpath)
                entry: dict[str, Any] = {
                    "name": str(rel),
                    "type": "dir" if item.is_dir() else "file",
                }
                if item.is_file():
                    entry["size_bytes"] = item.stat().st_size
                entries.append(entry)
        except PermissionError as exc:
            return ToolResult(success=False, error=f"Permission denied: {exc}")

        return ToolResult(
            success=True,
            output={
                "path": str(dirpath),
                "entries": entries,
                "count": len(entries),
                "truncated": len(entries) >= max_entries,
            },
        )

    def describe_action(self, inputs: dict[str, Any]) -> str:
        return f"List directory: {inputs.get('path', '?')}"


class FsMkdirTool(BaseTool):
    """Create directories anywhere on the filesystem."""

    requires_confirmation = False

    def __init__(self, *, requires_confirmation: bool | None = None) -> None:
        super().__init__(requires_confirmation=requires_confirmation)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fs_mkdir",
            description=(
                "Create a directory (and parents) at an absolute path."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path of the directory to create",
                    },
                },
                "required": ["path"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        path_str = inputs.get("path", "")
        if not path_str:
            return ToolResult(success=False, error="Missing required field: 'path'")

        dirpath = Path(path_str).expanduser().resolve()
        try:
            dirpath.mkdir(parents=True, exist_ok=True)
            return ToolResult(
                success=True,
                output={"path": str(dirpath), "created": True},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"mkdir error: {exc}")

    def describe_action(self, inputs: dict[str, Any]) -> str:
        return f"Create directory: {inputs.get('path', '?')}"


class FsWriteTool(BaseTool):
    """Write a file anywhere on the filesystem."""

    requires_confirmation = True

    def __init__(self, *, requires_confirmation: bool | None = None) -> None:
        super().__init__(requires_confirmation=requires_confirmation)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fs_write",
            description=(
                "Write content to a file at an absolute path. Creates parent "
                "directories automatically. Requires user confirmation by default."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path for the file",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Append instead of overwrite",
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

        filepath = Path(path_str).expanduser().resolve()
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with filepath.open(mode, encoding="utf-8") as f:
                f.write(content)
            return ToolResult(
                success=True,
                output={
                    "path": str(filepath),
                    "size_bytes": filepath.stat().st_size,
                    "action": "appended" if append else "written",
                },
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Write error: {exc}")

    def describe_action(self, inputs: dict[str, Any]) -> str:
        action = "Append to" if inputs.get("append") else "Write"
        size = len(inputs.get("content", ""))
        return f"{action} file: {inputs.get('path', '?')} ({size} chars)"


class FsDeleteTool(BaseTool):
    """Delete a file or directory anywhere on the filesystem."""

    requires_confirmation = True

    def __init__(self, *, requires_confirmation: bool | None = None) -> None:
        super().__init__(requires_confirmation=requires_confirmation)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fs_delete",
            description=(
                "Delete a file or directory at an absolute path. "
                "Directories are removed recursively. "
                "Requires user confirmation by default."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to delete",
                    },
                },
                "required": ["path"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        path_str = inputs.get("path", "")
        if not path_str:
            return ToolResult(success=False, error="Missing required field: 'path'")

        target = Path(path_str).expanduser().resolve()
        if not target.exists():
            return ToolResult(success=False, error=f"Path not found: {target}")

        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            return ToolResult(
                success=True,
                output={"path": str(target), "deleted": True},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Delete error: {exc}")

    def describe_action(self, inputs: dict[str, Any]) -> str:
        return f"DELETE: {inputs.get('path', '?')}"


class FsMoveTool(BaseTool):
    """Move or rename a file/directory anywhere on the filesystem."""

    requires_confirmation = True

    def __init__(self, *, requires_confirmation: bool | None = None) -> None:
        super().__init__(requires_confirmation=requires_confirmation)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fs_move",
            description=(
                "Move or rename a file/directory. "
                "Requires user confirmation by default."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Absolute path of the source",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Absolute path of the destination",
                    },
                },
                "required": ["source", "destination"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        src = inputs.get("source", "")
        dst = inputs.get("destination", "")
        if not src or not dst:
            return ToolResult(success=False, error="Missing source or destination")

        src_path = Path(src).expanduser().resolve()
        dst_path = Path(dst).expanduser().resolve()

        if not src_path.exists():
            return ToolResult(success=False, error=f"Source not found: {src_path}")

        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
            return ToolResult(
                success=True,
                output={"source": str(src_path), "destination": str(dst_path)},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Move error: {exc}")

    def describe_action(self, inputs: dict[str, Any]) -> str:
        return f"Move: {inputs.get('source', '?')} → {inputs.get('destination', '?')}"


class FsCopyTool(BaseTool):
    """Copy a file or directory anywhere on the filesystem."""

    requires_confirmation = True

    def __init__(self, *, requires_confirmation: bool | None = None) -> None:
        super().__init__(requires_confirmation=requires_confirmation)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fs_copy",
            description=(
                "Copy a file or directory. Directories are copied recursively. "
                "Requires user confirmation by default."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Absolute path of the source",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Absolute path of the destination",
                    },
                },
                "required": ["source", "destination"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        src = inputs.get("source", "")
        dst = inputs.get("destination", "")
        if not src or not dst:
            return ToolResult(success=False, error="Missing source or destination")

        src_path = Path(src).expanduser().resolve()
        dst_path = Path(dst).expanduser().resolve()

        if not src_path.exists():
            return ToolResult(success=False, error=f"Source not found: {src_path}")

        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if src_path.is_dir():
                shutil.copytree(str(src_path), str(dst_path))
            else:
                shutil.copy2(str(src_path), str(dst_path))
            return ToolResult(
                success=True,
                output={"source": str(src_path), "destination": str(dst_path)},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Copy error: {exc}")

    def describe_action(self, inputs: dict[str, Any]) -> str:
        return f"Copy: {inputs.get('source', '?')} → {inputs.get('destination', '?')}"
