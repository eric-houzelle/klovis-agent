"""Inbox perception source — external requests for the daemon.

Watches ~/.config/agent/inbox/ for request files. Any .txt or .md file
dropped there becomes a REQUEST event. After processing, files are moved
to inbox/done/ so they aren't picked up again.

This lets humans (or scripts, cron jobs, webhooks…) send goals to the
agent while it runs in daemon mode.

Usage:
    echo "Write a blog post about autonomous agents" > ~/.config/agent/inbox/task.txt
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import structlog

from klovis_agent.perception.base import Event, EventKind, PerceptionSource

logger = structlog.get_logger(__name__)

_INBOX_DIR = Path.home() / ".config" / "agent" / "inbox"
_DONE_DIR = _INBOX_DIR / "done"
_EXTENSIONS = {".txt", ".md"}


class InboxPerceptionSource(PerceptionSource):
    """Watches a local folder for request files."""

    def __init__(self, inbox_dir: Path | None = None) -> None:
        self._inbox = inbox_dir or _INBOX_DIR
        self._done = self._inbox / "done"
        self._inbox.mkdir(parents=True, exist_ok=True)
        self._done.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "inbox"

    async def poll(self) -> list[Event]:
        events: list[Event] = []

        for path in sorted(self._inbox.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _EXTENSIONS:
                continue

            try:
                content = path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                logger.warning("inbox_read_error", file=path.name, error=str(exc))
                continue

            if not content:
                self._archive(path)
                continue

            events.append(Event(
                source="inbox",
                kind=EventKind.REQUEST,
                title=content[:200],
                detail=content,
                metadata={"file": path.name, "path": str(path)},
                timestamp=path.stat().st_mtime,
            ))

            logger.info("inbox_request_found", file=path.name, length=len(content))

        return events

    def archive(self, filename: str) -> None:
        """Move a processed request file to done/."""
        path = self._inbox / filename
        if path.exists():
            self._archive(path)

    def _archive(self, path: Path) -> None:
        dest = self._done / f"{int(time.time())}_{path.name}"
        shutil.move(str(path), str(dest))
        logger.info("inbox_archived", file=path.name, dest=dest.name)
