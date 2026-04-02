from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klovis_agent.perception.base import (
        Event,
        EventKind,
        PerceptionResult,
        PerceptionSource,
        perceive,
    )
    from klovis_agent.perception.inbox import InboxPerceptionSource

__all__ = [
    "Event",
    "EventKind",
    "InboxPerceptionSource",
    "PerceptionResult",
    "PerceptionSource",
    "perceive",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Event": ("klovis_agent.perception.base", "Event"),
    "EventKind": ("klovis_agent.perception.base", "EventKind"),
    "InboxPerceptionSource": ("klovis_agent.perception.inbox", "InboxPerceptionSource"),
    "PerceptionResult": ("klovis_agent.perception.base", "PerceptionResult"),
    "PerceptionSource": ("klovis_agent.perception.base", "PerceptionSource"),
    "perceive": ("klovis_agent.perception.base", "perceive"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'klovis_agent.perception' has no attribute {name!r}")
