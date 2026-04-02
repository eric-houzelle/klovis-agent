"""Unit tests for klovis_agent.tools (base, docs).

NOTE: ToolRegistry and AgentWorkspace depend on structlog, so they are
tested only when structlog is available.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec


class DummyTool(BaseTool):
    def __init__(self, name: str = "dummy", should_fail: bool = False):
        self._name = name
        self._should_fail = should_fail

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description="A dummy tool for testing",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self._should_fail:
            return ToolResult(success=False, error="forced failure")
        return ToolResult(success=True, output={"result": inputs.get("x", 0) * 2})


class TestToolSpec:
    def test_defaults(self):
        s = ToolSpec(name="t", description="d")
        assert s.requires_sandbox is False
        assert s.input_schema == {}

    def test_full(self):
        s = ToolSpec(
            name="t",
            description="d",
            input_schema={"type": "object"},
            requires_sandbox=True,
        )
        assert s.requires_sandbox is True


class TestToolResult:
    def test_success(self):
        r = ToolResult(success=True, output={"a": 1})
        assert r.error is None

    def test_failure(self):
        r = ToolResult(success=False, error="bad")
        assert r.output == {}


structlog = pytest.importorskip("structlog")


class TestToolRegistry:
    def setup_method(self):
        from klovis_agent.tools.registry import ToolRegistry
        self.ToolRegistry = ToolRegistry

    def test_register_and_get(self):
        reg = self.ToolRegistry()
        reg.register(DummyTool("alpha"))
        assert reg.get("alpha") is not None
        assert reg.get("nonexistent") is None

    def test_duplicate_raises(self):
        reg = self.ToolRegistry()
        reg.register(DummyTool("dup"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(DummyTool("dup"))

    def test_list_specs(self):
        reg = self.ToolRegistry()
        reg.register(DummyTool("a"))
        reg.register(DummyTool("b"))
        specs = reg.list_specs()
        assert len(specs) == 2

    def test_list_specs_filtered(self):
        reg = self.ToolRegistry()
        reg.register(DummyTool("a"))
        reg.register(DummyTool("b"))
        specs = reg.list_specs(allowed=["a"])
        assert len(specs) == 1
        assert specs[0].name == "a"

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        reg = self.ToolRegistry()
        reg.register(DummyTool("calc"))
        result = await reg.invoke("calc", {"x": 5})
        assert result.success
        assert result.output["result"] == 10

    @pytest.mark.asyncio
    async def test_invoke_not_found(self):
        reg = self.ToolRegistry()
        result = await reg.invoke("missing", {})
        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_invoke_not_allowed(self):
        reg = self.ToolRegistry()
        reg.register(DummyTool("restricted"))
        result = await reg.invoke("restricted", {}, allowed_tools=["other"])
        assert not result.success
        assert "allowlist" in result.error

    @pytest.mark.asyncio
    async def test_invoke_tool_failure(self):
        reg = self.ToolRegistry()
        reg.register(DummyTool("fail", should_fail=True))
        result = await reg.invoke("fail", {"x": 1})
        assert not result.success


class TestAgentWorkspace:
    def setup_method(self):
        from klovis_agent.tools.workspace import AgentWorkspace
        self.AgentWorkspace = AgentWorkspace

    def test_create_with_path(self, tmp_path: Path):
        ws = self.AgentWorkspace(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
        assert ws.root.exists()
        assert ws.content.root.exists()
        assert ws.scratch.root.exists()

    def test_create_ephemeral(self):
        ws = self.AgentWorkspace(ephemeral=True)
        assert ws.root.exists()
        assert ws.content.root.exists()
        assert ws.scratch.root.exists()
        ws.cleanup()

    def test_resolve_valid(self, tmp_path: Path):
        ws = self.AgentWorkspace(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
        resolved = ws.resolve("subdir/file.txt")
        assert str(resolved).startswith(str(ws.root))

    def test_resolve_escape_raises(self, tmp_path: Path):
        ws = self.AgentWorkspace(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
        with pytest.raises(ValueError, match="escapes workspace"):
            ws.resolve("../../etc/passwd")

    def test_content_and_scratch_are_separate(self, tmp_path: Path):
        ws = self.AgentWorkspace(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
        assert ws.content.root != ws.scratch.root


class TestToolDocs:
    def setup_method(self):
        from klovis_agent.tools.docs import (
            format_tool_catalog,
            format_tool_doc,
            format_tool_summary,
        )
        self.format_tool_catalog = format_tool_catalog
        self.format_tool_doc = format_tool_doc
        self.format_tool_summary = format_tool_summary

    def test_format_tool_doc(self):
        spec = ToolSpec(
            name="test_tool",
            description="Does testing",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        )
        doc = self.format_tool_doc(spec)
        assert "test_tool" in doc
        assert "query" in doc
        assert "REQUIRED" in doc

    def test_format_tool_catalog_empty(self):
        assert self.format_tool_catalog([]) == "(no tools available)"

    def test_format_tool_catalog(self):
        specs = [
            ToolSpec(name="web_search", description="Search the web"),
            ToolSpec(name="web_fetch", description="Fetch a URL"),
            ToolSpec(name="memory", description="KV store"),
        ]
        catalog = self.format_tool_catalog(specs)
        assert "web tools" in catalog
        assert "3 total" in catalog

    def test_format_tool_summary(self):
        specs = [ToolSpec(name="a", description="does A")]
        summary = self.format_tool_summary(specs)
        assert "- a: does A" in summary

    def test_format_tool_summary_empty(self):
        assert self.format_tool_summary([]) == "(no tools available)"
