from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec

DEFAULT_TIMEOUT = 30
MAX_OUTPUT_BYTES = 500_000


@runtime_checkable
class _HasRoot(Protocol):
    @property
    def root(self) -> Path: ...


class ShellCommandTool(BaseTool):
    """Execute shell commands in a scoped directory.

    By default the working directory is the agent scratch workspace.
    The caller can override this per-invocation via the ``cwd`` input
    parameter (useful when working on a project outside the workspace).
    """

    def __init__(
        self,
        workspace: _HasRoot,
        timeout: int = DEFAULT_TIMEOUT,
        *,
        requires_confirmation: bool | None = None,
    ) -> None:
        super().__init__(requires_confirmation=requires_confirmation)
        self._workspace = workspace
        self._timeout = timeout

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="shell_command",
            description=(
                "Execute a shell command. By default the working directory is "
                "the agent scratch workspace. Set 'cwd' to an absolute path to "
                "run the command elsewhere (e.g. in a project directory)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Absolute path to use as working directory. "
                            "Defaults to the agent scratch workspace."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30)",
                    },
                },
                "required": ["command"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "exit_code": {"type": "integer"},
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                },
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        command = inputs.get("command", "")
        if not command:
            return ToolResult(success=False, error="Missing required field: 'command'")

        timeout = inputs.get("timeout", self._timeout)

        cwd_input = inputs.get("cwd")
        if cwd_input:
            cwd = Path(cwd_input).expanduser().resolve()
            if not cwd.is_dir():
                return ToolResult(
                    success=False,
                    error=f"cwd is not a directory: {cwd}",
                )
        else:
            cwd = self._workspace.root

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT_BYTES]
            stderr = stderr_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT_BYTES]
            exit_code = proc.returncode or 0

            return ToolResult(
                success=exit_code == 0,
                output={
                    "exit_code": exit_code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "cwd": str(cwd),
                },
                error=stderr if exit_code != 0 else None,
            )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout}s",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Shell error: {exc}")

    def describe_action(self, inputs: dict[str, Any]) -> str:
        cmd = inputs.get("command", "?")
        cwd = inputs.get("cwd", "(workspace)")
        return f"Run in {cwd}: {cmd[:120]}"
