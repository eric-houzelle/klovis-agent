from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog

from klovis_agent.models.artifact import CodeArtifact, ExecutionResult

logger = structlog.get_logger(__name__)


@runtime_checkable
class SandboxExecutionService(Protocol):
    """Protocol that all sandbox backends must implement."""

    async def run_code(
        self,
        artifact: CodeArtifact,
        constraints: dict[str, Any],
    ) -> ExecutionResult: ...

    async def cleanup(self) -> None: ...


DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_OUTPUT_BYTES = 1_000_000


class LocalSandbox:
    """Isolated code execution via restricted subprocesses.

    Works on any machine without Docker. When a workspace_root is provided,
    files persist across executions (shared workspace). Otherwise a temporary
    directory is created per execution (legacy behaviour).
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        workspace_root: Path | None = None,
    ) -> None:
        self._timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._workspace_root = workspace_root

    async def run_code(
        self,
        artifact: CodeArtifact,
        constraints: dict[str, Any],
    ) -> ExecutionResult:
        timeout = constraints.get("timeout", self._timeout)

        if self._workspace_root is not None:
            return await self._run_in_dir(self._workspace_root, artifact, timeout)

        with tempfile.TemporaryDirectory(prefix="sandbox_") as tmpdir:
            return await self._run_in_dir(Path(tmpdir), artifact, timeout)

    async def _run_in_dir(
        self,
        workdir: Path,
        artifact: CodeArtifact,
        timeout: int,
    ) -> ExecutionResult:
        for filename, content in artifact.files.items():
            filepath = workdir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content)

        cmd = self._build_command(artifact.language, artifact.entrypoint)
        if cmd is None:
            return ExecutionResult(
                status="error",
                stderr=f"Unsupported language: {artifact.language}",
            )

        logger.info(
            "sandbox_exec",
            language=artifact.language,
            entrypoint=artifact.entrypoint,
            timeout=timeout,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(workdir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._restricted_env(),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")[
                : self._max_output_bytes
            ]
            stderr = stderr_bytes.decode("utf-8", errors="replace")[
                : self._max_output_bytes
            ]

            status = "success" if proc.returncode == 0 else "error"
            return ExecutionResult(status=status, stdout=stdout, stderr=stderr)

        except asyncio.TimeoutError:
            proc.kill()  # type: ignore[possibly-undefined]
            logger.warning("sandbox_timeout", timeout=timeout)
            return ExecutionResult(
                status="timeout", stderr=f"Execution timed out after {timeout}s"
            )

        except Exception as exc:
            logger.error("sandbox_error", error=str(exc))
            return ExecutionResult(status="error", stderr=str(exc))

    async def cleanup(self) -> None:
        pass

    @staticmethod
    def _build_command(language: str, entrypoint: str) -> list[str] | None:
        runners: dict[str, list[str]] = {
            "python": ["python3", entrypoint],
            "javascript": ["node", entrypoint],
            "typescript": ["npx", "tsx", entrypoint],
            "bash": ["bash", entrypoint],
        }
        return runners.get(language)

    @staticmethod
    def _restricted_env() -> dict[str, str]:
        return {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "LANG": "en_US.UTF-8",
        }


class OpenSandboxService:
    """Isolated code execution backed by OpenSandbox containers.

    Requires a running opensandbox-server with Docker access.
    """

    def __init__(
        self,
        domain: str = "localhost:8080",
        api_key: str = "",
        protocol: str = "http",
        image: str = "opensandbox/code-interpreter:v1.0.2",
        timeout_minutes: int = 10,
        python_version: str = "3.11",
        keep_alive: bool = False,
    ) -> None:
        from datetime import timedelta

        from opensandbox.config import ConnectionConfig

        self._connection_config = ConnectionConfig(
            domain=domain,
            api_key=api_key,
            protocol=protocol,
            request_timeout=timedelta(seconds=120),
        )
        self._image = image
        self._timeout = timedelta(minutes=timeout_minutes)
        self._python_version = python_version
        self.keep_alive = keep_alive

        self._sandbox = None
        self._interpreter = None

    async def _ensure_sandbox(self):  # type: ignore[no-untyped-def]
        if self._sandbox is not None and self._interpreter is not None:
            return self._sandbox, self._interpreter

        from code_interpreter import CodeInterpreter
        from opensandbox import Sandbox

        logger.info(
            "sandbox_creating",
            image=self._image,
            python_version=self._python_version,
        )

        self._sandbox = await Sandbox.create(
            self._image,
            connection_config=self._connection_config,
            entrypoint=["/opt/opensandbox/code-interpreter.sh"],
            env={"PYTHON_VERSION": self._python_version},
            timeout=self._timeout,
        )
        self._interpreter = await CodeInterpreter.create(sandbox=self._sandbox)

        logger.info("sandbox_ready", sandbox_id=self._sandbox.id)
        return self._sandbox, self._interpreter

    async def run_code(
        self,
        artifact: CodeArtifact,
        constraints: dict[str, Any],
    ) -> ExecutionResult:
        from code_interpreter import SupportedLanguage
        from opensandbox.models import WriteEntry

        language_map: dict[str, SupportedLanguage] = {
            "python": SupportedLanguage.PYTHON,
            "java": SupportedLanguage.JAVA,
            "javascript": SupportedLanguage.TYPESCRIPT,
            "typescript": SupportedLanguage.TYPESCRIPT,
            "go": SupportedLanguage.GO,
        }

        sandbox, interpreter = await self._ensure_sandbox()

        if artifact.files:
            entries = [
                WriteEntry(path=f"/workspace/{name}", data=content, mode=644)
                for name, content in artifact.files.items()
            ]
            await sandbox.files.write_files(entries)
            logger.info("sandbox_files_written", count=len(entries))

        lang = language_map.get(artifact.language.lower())

        if lang is not None:
            code = artifact.files.get(artifact.entrypoint, "")
            if not code:
                return ExecutionResult(
                    status="error",
                    stderr=f"Entrypoint '{artifact.entrypoint}' not found in files",
                )

            logger.info(
                "sandbox_exec_interpreter",
                language=artifact.language,
                entrypoint=artifact.entrypoint,
            )

            result = await interpreter.codes.run(code, language=lang)

            stdout_parts: list[str] = []
            if result.logs and result.logs.stdout:
                stdout_parts.extend(msg.text for msg in result.logs.stdout)

            stderr_parts: list[str] = []
            if result.logs and result.logs.stderr:
                stderr_parts.extend(msg.text for msg in result.logs.stderr)

            result_text = ""
            if result.result:
                result_text = "\n".join(r.text for r in result.result if r.text)

            stdout = "\n".join(stdout_parts)
            if result_text:
                stdout = f"{stdout}\n{result_text}" if stdout else result_text

            stderr = "\n".join(stderr_parts)
            has_error = bool(result.error) or bool(stderr)

            return ExecutionResult(
                status="error" if has_error else "success",
                stdout=stdout,
                stderr=stderr or (result.error.value if result.error else ""),
            )

        runners = {"python": "python3", "bash": "bash", "sh": "sh"}
        runner = runners.get(artifact.language.lower())
        if not runner:
            return ExecutionResult(
                status="error",
                stderr=f"Unsupported language: {artifact.language}",
            )

        cmd = f"cd /workspace && {runner} {artifact.entrypoint}"
        logger.info("sandbox_exec_command", cmd=cmd)

        execution = await sandbox.commands.run(cmd)

        stdout = "\n".join(msg.text for msg in (execution.logs.stdout or []))
        stderr = "\n".join(msg.text for msg in (execution.logs.stderr or []))
        has_error = (
            execution.exit_code != 0
            if hasattr(execution, "exit_code")
            else bool(stderr)
        )

        return ExecutionResult(
            status="error" if has_error else "success",
            stdout=stdout,
            stderr=stderr,
        )

    async def cleanup(self) -> None:
        if self._sandbox is not None:
            try:
                await self._sandbox.kill()
                logger.info("sandbox_killed", sandbox_id=self._sandbox.id)
            except Exception as exc:
                logger.warning("sandbox_kill_failed", error=str(exc))
            finally:
                self._sandbox = None
                self._interpreter = None
