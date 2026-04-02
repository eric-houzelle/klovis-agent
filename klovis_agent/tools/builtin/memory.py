from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from klovis_agent.paths import data_home
from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec

logger = structlog.get_logger(__name__)


class MemoryTool(BaseTool):
    """Persistent key-value memory that survives across runs.

    Data is stored in a JSON file on disk. Each write flushes to disk
    immediately so nothing is lost if the process crashes.
    """

    def __init__(self, memory_dir: Path | None = None) -> None:
        self._dir = memory_dir or (data_home() / "memory")
        self._path = self._dir / "store.json"
        self._store = self._load()

    def _load(self) -> dict[str, Any]:
        if self._path.is_file():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                logger.info(
                    "memory_loaded",
                    path=str(self._path),
                    num_keys=len(data),
                )
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("memory_load_failed", error=str(exc))
        return {}

    def _flush(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._store, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="memory",
            description=(
                "Persistent key-value memory that survives across runs. "
                "Use to store facts, preferences, learned lessons, or any data "
                "the agent should remember long-term. "
                "Operations: 'set' (store), 'get' (retrieve), 'delete' (remove), "
                "'list' (show all keys). Data persists on disk between sessions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["set", "get", "delete", "list"],
                    },
                    "key": {
                        "type": "string",
                        "description": "The key to operate on (not needed for 'list')",
                    },
                    "value": {
                        "description": "The value to store (only for 'set')",
                    },
                },
                "required": ["operation"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "value": {},
                    "keys": {"type": "array", "items": {"type": "string"}},
                },
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        operation = inputs.get("operation", "")
        key = inputs.get("key", "")

        if operation == "set":
            if not key:
                return ToolResult(success=False, error="'set' requires a 'key'")
            value = inputs.get("value")
            self._store[key] = value
            self._flush()
            return ToolResult(
                success=True,
                output={"key": key, "stored": True, "total_keys": len(self._store)},
            )

        if operation == "get":
            if not key:
                return ToolResult(success=False, error="'get' requires a 'key'")
            if key not in self._store:
                return ToolResult(success=False, error=f"Key not found: '{key}'")
            return ToolResult(
                success=True,
                output={"key": key, "value": self._store[key]},
            )

        if operation == "delete":
            if not key:
                return ToolResult(success=False, error="'delete' requires a 'key'")
            if key not in self._store:
                return ToolResult(success=False, error=f"Key not found: '{key}'")
            del self._store[key]
            self._flush()
            return ToolResult(
                success=True,
                output={"key": key, "deleted": True, "total_keys": len(self._store)},
            )

        if operation == "list":
            return ToolResult(
                success=True,
                output={
                    "keys": list(self._store.keys()),
                    "total_keys": len(self._store),
                },
            )

        return ToolResult(
            success=False,
            error=f"Unknown operation: '{operation}'. Use: set, get, delete, list",
        )
