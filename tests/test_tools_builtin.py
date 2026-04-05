"""Unit tests for klovis_agent.tools.builtin.

All builtin tools depend on structlog (via workspace/registry/sandbox).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

structlog = pytest.importorskip("structlog")

from klovis_agent.tools.builtin.code_execution import CodeExecutionTool, TextAnalysisTool
from klovis_agent.tools.builtin.file_tools import FileReadTool, FileWriteTool
from klovis_agent.tools.builtin.memory import MemoryTool
from klovis_agent.tools.builtin.shell import ShellCommandTool
from klovis_agent.tools.builtin.skills import (
    SkillStore,
    _parse_frontmatter,
    _source_to_candidates,
)
from klovis_agent.tools.workspace import AgentWorkspace


@pytest.fixture
def workspace(tmp_path: Path) -> AgentWorkspace:
    return AgentWorkspace(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")


class TestFileReadTool:
    @pytest.mark.asyncio
    async def test_read_existing(self, workspace: AgentWorkspace):
        (workspace.root / "hello.txt").write_text("world")
        tool = FileReadTool(workspace)
        result = await tool.execute({"path": "hello.txt"})
        assert result.success
        assert result.output["content"] == "world"

    @pytest.mark.asyncio
    async def test_read_missing(self, workspace: AgentWorkspace):
        tool = FileReadTool(workspace)
        result = await tool.execute({"path": "nope.txt"})
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_no_path(self, workspace: AgentWorkspace):
        tool = FileReadTool(workspace)
        result = await tool.execute({})
        assert not result.success

    @pytest.mark.asyncio
    async def test_read_escape(self, workspace: AgentWorkspace):
        tool = FileReadTool(workspace)
        result = await tool.execute({"path": "../../etc/passwd"})
        assert not result.success


class TestFileWriteTool:
    @pytest.mark.asyncio
    async def test_write_new(self, workspace: AgentWorkspace):
        tool = FileWriteTool(workspace)
        result = await tool.execute({"path": "out.txt", "content": "hello"})
        assert result.success
        assert (workspace.root / "out.txt").read_text() == "hello"

    @pytest.mark.asyncio
    async def test_write_creates_dirs(self, workspace: AgentWorkspace):
        tool = FileWriteTool(workspace)
        result = await tool.execute({"path": "sub/dir/f.txt", "content": "nested"})
        assert result.success
        assert (workspace.root / "sub" / "dir" / "f.txt").exists()

    @pytest.mark.asyncio
    async def test_write_append(self, workspace: AgentWorkspace):
        (workspace.root / "log.txt").write_text("line1\n")
        tool = FileWriteTool(workspace)
        result = await tool.execute({"path": "log.txt", "content": "line2\n", "append": True})
        assert result.success
        assert "line1\nline2\n" in (workspace.root / "log.txt").read_text()

    @pytest.mark.asyncio
    async def test_write_no_path(self, workspace: AgentWorkspace):
        tool = FileWriteTool(workspace)
        result = await tool.execute({"content": "x"})
        assert not result.success


class TestShellCommandTool:
    @pytest.mark.asyncio
    async def test_echo(self, workspace: AgentWorkspace):
        tool = ShellCommandTool(workspace.scratch)
        result = await tool.execute({"command": "echo hello"})
        assert result.success
        assert "hello" in result.output["stdout"]

    @pytest.mark.asyncio
    async def test_failing_command(self, workspace: AgentWorkspace):
        tool = ShellCommandTool(workspace.scratch)
        result = await tool.execute({"command": "false"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_no_command(self, workspace: AgentWorkspace):
        tool = ShellCommandTool(workspace.scratch)
        result = await tool.execute({})
        assert not result.success

    @pytest.mark.asyncio
    async def test_timeout(self, workspace: AgentWorkspace):
        tool = ShellCommandTool(workspace.scratch, timeout=1)
        result = await tool.execute({"command": "sleep 10", "timeout": 1})
        assert not result.success
        assert "timed out" in result.error.lower()


class TestMemoryTool:
    @pytest.mark.asyncio
    async def test_set_get(self, tmp_path: Path):
        tool = MemoryTool(memory_dir=tmp_path)
        await tool.execute({"operation": "set", "key": "name", "value": "klovis"})
        result = await tool.execute({"operation": "get", "key": "name"})
        assert result.success
        assert result.output["value"] == "klovis"

    @pytest.mark.asyncio
    async def test_list(self, tmp_path: Path):
        tool = MemoryTool(memory_dir=tmp_path)
        await tool.execute({"operation": "set", "key": "a", "value": 1})
        await tool.execute({"operation": "set", "key": "b", "value": 2})
        result = await tool.execute({"operation": "list"})
        assert result.success
        assert set(result.output["keys"]) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path: Path):
        tool = MemoryTool(memory_dir=tmp_path)
        await tool.execute({"operation": "set", "key": "x", "value": 1})
        result = await tool.execute({"operation": "delete", "key": "x"})
        assert result.success
        result = await tool.execute({"operation": "get", "key": "x"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_get_missing(self, tmp_path: Path):
        tool = MemoryTool(memory_dir=tmp_path)
        result = await tool.execute({"operation": "get", "key": "nope"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_unknown_op(self, tmp_path: Path):
        tool = MemoryTool(memory_dir=tmp_path)
        result = await tool.execute({"operation": "explode"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path: Path):
        tool1 = MemoryTool(memory_dir=tmp_path)
        await tool1.execute({"operation": "set", "key": "persist", "value": "yes"})
        tool2 = MemoryTool(memory_dir=tmp_path)
        result = await tool2.execute({"operation": "get", "key": "persist"})
        assert result.success
        assert result.output["value"] == "yes"


class TestCodeExecutionTool:
    @pytest.mark.asyncio
    async def test_no_files(self):
        from klovis_agent.sandbox.service import LocalSandbox

        tool = CodeExecutionTool(LocalSandbox())
        result = await tool.execute({"language": "python"})
        assert not result.success
        assert "No files" in result.error

    @pytest.mark.asyncio
    async def test_code_shorthand(self):
        from klovis_agent.sandbox.service import LocalSandbox

        tool = CodeExecutionTool(LocalSandbox())
        result = await tool.execute({
            "language": "python",
            "code": "print('hello')",
        })
        assert result.success
        assert "hello" in result.output["stdout"]

    @pytest.mark.asyncio
    async def test_files_dict(self):
        from klovis_agent.sandbox.service import LocalSandbox

        tool = CodeExecutionTool(LocalSandbox())
        result = await tool.execute({
            "language": "python",
            "entrypoint": "main.py",
            "files": {"main.py": "print(2+2)"},
        })
        assert result.success
        assert "4" in result.output["stdout"]


class TestTextAnalysisTool:
    @pytest.mark.asyncio
    async def test_basic(self):
        tool = TextAnalysisTool()
        result = await tool.execute({"text": "Hello world", "operation": "summarize"})
        assert result.success
        assert "11 chars" in result.output["result"]


class TestSkillStore:
    def test_empty_dir(self, tmp_path: Path):
        store = SkillStore(tmp_path / "nope")
        assert store.list_skills() == []

    def test_load_skill(self, tmp_path: Path):
        skill_dir = tmp_path / "myskill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: myskill\ndescription: A test skill\n"
            "api_base: https://api.example.com\n---\n# Docs\n"
        )
        store = SkillStore(tmp_path)
        skills = store.list_skills()
        assert len(skills) == 1
        assert skills[0].name == "myskill"
        assert store.get_content("myskill") is not None

    def test_get_auth_no_match(self, tmp_path: Path):
        store = SkillStore(tmp_path)
        assert store.get_auth_for_url("https://random.com") is None

    def test_priority_order_first_dir_wins(self, tmp_path: Path):
        high = tmp_path / "high"
        low = tmp_path / "low"
        (high / "shared").mkdir(parents=True)
        (low / "shared").mkdir(parents=True)
        (high / "shared" / "SKILL.md").write_text(
            "---\nname: shared\ndescription: high priority\n---\n# High\n"
        )
        (low / "shared" / "SKILL.md").write_text(
            "---\nname: shared\ndescription: low priority\n---\n# Low\n"
        )

        store = SkillStore([high, low])
        meta = store.get_meta("shared")
        assert meta is not None
        assert meta.description == "high priority"
        assert "# High" in (store.get_content("shared") or "")


class TestParseFrontmatter:
    def test_basic(self):
        text = "---\nname: test\nversion: 1.0\n---\nBody"
        fm = _parse_frontmatter(text)
        assert fm["name"] == "test"
        assert fm["version"] == "1.0"

    def test_no_frontmatter(self):
        assert _parse_frontmatter("Just text") == {}

    def test_quoted_values(self):
        text = '---\nname: "quoted"\n---\nBody'
        fm = _parse_frontmatter(text)
        assert fm["name"] == "quoted"


class TestSkillSourceParsing:
    def test_skills_sh_owner_skills_pattern(self):
        c = _source_to_candidates("https://skills.sh/vercel-labs/skills/find-skills")
        assert c
        assert c[0].endswith("/vercel-labs/skills/main/skills/find-skills/SKILL.md")

    def test_github_path_pattern(self):
        c = _source_to_candidates("owner/repo/skills/my-skill")
        assert c
        assert c[0].endswith("/owner/repo/main/skills/my-skill/SKILL.md")

    def test_direct_skill_md_url(self):
        url = "https://raw.githubusercontent.com/a/b/main/skills/x/SKILL.md"
        c = _source_to_candidates(url)
        assert c == [url]
