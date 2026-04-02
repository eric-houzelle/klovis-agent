from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klovis_agent.sandbox.service import (
        LocalSandbox,
        OpenSandboxService,
        SandboxExecutionService,
    )

__all__ = [
    "LocalSandbox",
    "OpenSandboxService",
    "SandboxExecutionService",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "LocalSandbox": ("klovis_agent.sandbox.service", "LocalSandbox"),
    "OpenSandboxService": ("klovis_agent.sandbox.service", "OpenSandboxService"),
    "SandboxExecutionService": ("klovis_agent.sandbox.service", "SandboxExecutionService"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'klovis_agent.sandbox' has no attribute {name!r}")
