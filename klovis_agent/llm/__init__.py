from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klovis_agent.llm.embeddings import EmbeddingClient
    from klovis_agent.llm.gateway import ModelGateway, OpenAIGateway
    from klovis_agent.llm.router import LLMRouter
    from klovis_agent.llm.types import ModelRequest, ModelResponse, ModelRoutingPolicy

__all__ = [
    "EmbeddingClient",
    "LLMRouter",
    "ModelGateway",
    "ModelRequest",
    "ModelResponse",
    "ModelRoutingPolicy",
    "OpenAIGateway",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "EmbeddingClient": ("klovis_agent.llm.embeddings", "EmbeddingClient"),
    "ModelGateway": ("klovis_agent.llm.gateway", "ModelGateway"),
    "OpenAIGateway": ("klovis_agent.llm.gateway", "OpenAIGateway"),
    "LLMRouter": ("klovis_agent.llm.router", "LLMRouter"),
    "ModelRequest": ("klovis_agent.llm.types", "ModelRequest"),
    "ModelResponse": ("klovis_agent.llm.types", "ModelResponse"),
    "ModelRoutingPolicy": ("klovis_agent.llm.types", "ModelRoutingPolicy"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'klovis_agent.llm' has no attribute {name!r}")
