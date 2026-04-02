from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klovis_agent.memory.kv import KeyValueMemory
    from klovis_agent.memory.semantic import SemanticMemoryStore

__all__ = [
    "KeyValueMemory",
    "SemanticMemoryStore",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "KeyValueMemory": ("klovis_agent.memory.kv", "KeyValueMemory"),
    "SemanticMemoryStore": ("klovis_agent.memory.semantic", "SemanticMemoryStore"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'klovis_agent.memory' has no attribute {name!r}")
