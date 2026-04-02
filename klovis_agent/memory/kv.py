"""Key-value memory backend — thin wrapper around the JSON store on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_MEMORY_DIR = Path.home() / ".config" / "agent" / "memory"


class KeyValueMemory:
    """Persistent key-value store backed by a JSON file."""

    def __init__(self, memory_dir: Path | None = None) -> None:
        self._dir = memory_dir or _DEFAULT_MEMORY_DIR
        self._path = self._dir / "store.json"
        self._store = self._load()

    def _load(self) -> dict[str, Any]:
        if self._path.is_file():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                logger.info("memory_loaded", path=str(self._path), num_keys=len(data))
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

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value
        self._flush()

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            self._flush()
            return True
        return False

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def count(self) -> int:
        return len(self._store)
