from __future__ import annotations

from typing import Any

from klovis_agent.models.artifact import CodeArtifact
from klovis_agent.sandbox.service import SandboxExecutionService
from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec


class CodeExecutionTool(BaseTool):
    """Code execution tool via isolated sandbox."""

    def __init__(self, sandbox: SandboxExecutionService) -> None:
        self._sandbox = sandbox

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_execution",
            description="Execute code in an isolated sandbox environment",
            input_schema={
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "entrypoint": {"type": "string"},
                    "files": {"type": "object", "additionalProperties": {"type": "string"}},
                },
                "required": ["language", "entrypoint", "files"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                },
            },
            requires_sandbox=True,
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        language = inputs.get("language", "python")
        files = inputs.get("files", {})
        entrypoint = inputs.get("entrypoint", "")

        if not files:
            code = inputs.get("code", "")
            if code:
                entrypoint = entrypoint or "main.py"
                files = {entrypoint: code}

        if not files:
            return ToolResult(
                success=False,
                error="No files or code provided. Required: 'files' dict or 'code' string.",
            )

        if not entrypoint:
            entrypoint = next(iter(files))

        artifact = CodeArtifact(
            artifact_id="runtime",
            language=language,
            entrypoint=entrypoint,
            files=files,
        )
        result = await self._sandbox.run_code(artifact=artifact, constraints={})

        output: dict[str, Any] = {
            "status": result.status,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "files_written": list(files.keys()),
        }
        for name, content in files.items():
            preview = content[:2000] + ("..." if len(content) > 2000 else "")
            output[f"file:{name}"] = preview

        return ToolResult(
            success=result.status == "success",
            output=output,
            error=result.stderr if result.status == "error" else None,
        )


class TextAnalysisTool(BaseTool):
    """Text analysis tool (no sandbox required)."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="text_analysis",
            description="Analyze and extract information from text",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "operation": {
                        "type": "string",
                        "enum": ["summarize", "extract_entities", "classify"],
                    },
                },
                "required": ["text", "operation"],
            },
            output_schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=True,
            output={"result": f"[{inputs['operation']}] processed {len(inputs['text'])} chars"},
        )
