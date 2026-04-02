from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec
    from klovis_agent.tools.registry import ToolRegistry

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseTool": ("klovis_agent.tools.base", "BaseTool"),
    "ToolResult": ("klovis_agent.tools.base", "ToolResult"),
    "ToolSpec": ("klovis_agent.tools.base", "ToolSpec"),
    "ToolRegistry": ("klovis_agent.tools.registry", "ToolRegistry"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'klovis_agent.tools' has no attribute {name!r}")
