"""Unit tests for klovis_agent.sandbox (LocalSandbox).

Depends on structlog.
"""

from __future__ import annotations

import pytest

structlog = pytest.importorskip("structlog")

from klovis_agent.models.artifact import CodeArtifact
from klovis_agent.sandbox.service import LocalSandbox


class TestLocalSandbox:
    @pytest.mark.asyncio
    async def test_python_success(self):
        sandbox = LocalSandbox()
        artifact = CodeArtifact(
            artifact_id="t1",
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('hello sandbox')"},
        )
        result = await sandbox.run_code(artifact, {})
        assert result.status == "success"
        assert "hello sandbox" in result.stdout

    @pytest.mark.asyncio
    async def test_python_error(self):
        sandbox = LocalSandbox()
        artifact = CodeArtifact(
            artifact_id="t2",
            language="python",
            entrypoint="bad.py",
            files={"bad.py": "raise ValueError('boom')"},
        )
        result = await sandbox.run_code(artifact, {})
        assert result.status == "error"
        assert "boom" in result.stderr

    @pytest.mark.asyncio
    async def test_bash(self):
        sandbox = LocalSandbox()
        artifact = CodeArtifact(
            artifact_id="t3",
            language="bash",
            entrypoint="run.sh",
            files={"run.sh": "echo 'from bash'"},
        )
        result = await sandbox.run_code(artifact, {})
        assert result.status == "success"
        assert "from bash" in result.stdout

    @pytest.mark.asyncio
    async def test_unsupported_language(self):
        sandbox = LocalSandbox()
        artifact = CodeArtifact(
            artifact_id="t4",
            language="cobol",
            entrypoint="main.cob",
            files={"main.cob": "DISPLAY 'HI'."},
        )
        result = await sandbox.run_code(artifact, {})
        assert result.status == "error"
        assert "Unsupported" in result.stderr

    @pytest.mark.asyncio
    async def test_timeout(self):
        sandbox = LocalSandbox(timeout=1)
        artifact = CodeArtifact(
            artifact_id="t5",
            language="python",
            entrypoint="slow.py",
            files={"slow.py": "import time; time.sleep(10)"},
        )
        result = await sandbox.run_code(artifact, {})
        assert result.status == "timeout"

    @pytest.mark.asyncio
    async def test_workspace_persistence(self, tmp_path):
        sandbox = LocalSandbox(workspace_root=tmp_path)
        artifact = CodeArtifact(
            artifact_id="t6",
            language="python",
            entrypoint="write.py",
            files={"write.py": "open('output.txt','w').write('persisted')"},
        )
        await sandbox.run_code(artifact, {})
        assert (tmp_path / "output.txt").read_text() == "persisted"

    @pytest.mark.asyncio
    async def test_multiple_files(self):
        sandbox = LocalSandbox()
        artifact = CodeArtifact(
            artifact_id="t7",
            language="python",
            entrypoint="main.py",
            files={
                "helper.py": "def greet(): return 'hi'",
                "main.py": "from helper import greet; print(greet())",
            },
        )
        result = await sandbox.run_code(artifact, {})
        assert result.status == "success"
        assert "hi" in result.stdout

    @pytest.mark.asyncio
    async def test_cleanup_noop(self):
        sandbox = LocalSandbox()
        await sandbox.cleanup()
