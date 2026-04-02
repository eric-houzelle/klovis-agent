"""Agent workspace — sandboxed filesystem areas for content and execution.

``AgentWorkspace`` is the single object that owns both:

* **content** — persistent files the agent produces (articles, notes, …).
* **scratch** — throwaway area for sandbox / shell execution.

Both areas prevent path-traversal escapes.  The split keeps generated
content safe from rogue code execution while staying simple to wire up.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import structlog

from klovis_agent.paths import resolve_cache_dir, resolve_data_dir

logger = structlog.get_logger(__name__)


class _ScopedDir:
    """A directory with path-escape protection."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, relative_path: str) -> Path:
        resolved = (self._root / relative_path).resolve()
        if not str(resolved).startswith(str(self._root.resolve())):
            raise ValueError(f"Path escapes workspace: {relative_path}")
        return resolved


class AgentWorkspace:
    """Shared filesystem workspace that persists across steps within a run.

    Parameters
    ----------
    data_dir:
        Root for persistent content.  When *None* the XDG data home is used
        (``~/.local/share/klovis``).
    cache_dir:
        Root for ephemeral scratch / sandbox files.  When *None* the XDG
        cache home is used (``~/.cache/klovis``).
    ephemeral:
        When *True* both areas are created in a temporary directory that is
        cleaned up on :meth:`cleanup`.  Useful for tests and one-shot runs.
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
        *,
        ephemeral: bool = False,
    ) -> None:
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

        if ephemeral:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="klovis_ws_")
            base = Path(self._tmpdir.name)
            self._content = _ScopedDir(base / "content")
            self._scratch = _ScopedDir(base / "scratch")
        else:
            self._content = _ScopedDir(resolve_data_dir(data_dir) / "content")
            self._scratch = _ScopedDir(
                resolve_cache_dir(cache_dir) / "scratch"
            )

        logger.info(
            "workspace_created",
            content=str(self._content.root),
            scratch=str(self._scratch.root),
        )

    # -- Backward-compatible ``root`` keeps existing code working during
    #    the transition.  It points to the *content* area.
    @property
    def root(self) -> Path:
        return self._content.root

    # -- Public scoped directories ----------------------------------------

    @property
    def content(self) -> _ScopedDir:
        """Persistent files the agent produces (articles, notes, …)."""
        return self._content

    @property
    def scratch(self) -> _ScopedDir:
        """Throwaway area for sandbox / shell execution."""
        return self._scratch

    # -- Convenience delegations (keep old call-sites working) ------------

    def resolve(self, relative_path: str) -> Path:
        """Resolve inside the *content* area (backward-compat)."""
        return self._content.resolve(relative_path)

    # -- Lifecycle --------------------------------------------------------

    def cleanup(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            logger.info("workspace_cleaned")
