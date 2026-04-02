"""Semantic memory backend — SQLite-backed vector store.

Re-exports SemanticMemoryStore from the builtin tool module so that
both the tool and the standalone memory backend share the same class.
"""

from klovis_agent.tools.builtin.semantic_memory import MemoryZone, SemanticMemoryStore

__all__ = ["MemoryZone", "SemanticMemoryStore"]
