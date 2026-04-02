from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec

if TYPE_CHECKING:
    from klovis_agent.llm.embeddings import EmbeddingClient

logger = structlog.get_logger(__name__)

_DEFAULT_DB_DIR = Path.home() / ".config" / "agent" / "memory"

MemoryZone = Literal["semantic", "episodic"]

EPISODIC_TTL_DAYS = 14
DEDUP_SIMILARITY_THRESHOLD = 0.9


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _recency_weight(created_at: float, now: float) -> float:
    """Score between 0 and 1 that decays over hours."""
    hours = max(0.0, (now - created_at) / 3600.0)
    return 1.0 / (1.0 + hours / 24.0)


class SemanticMemoryStore:
    """SQLite-backed vector store with episodic and semantic zones.

    * **semantic** — permanent facts, lessons, identity.  Ranked by pure
      cosine similarity.  Duplicates (similarity > 0.9) are updated
      in-place instead of creating new rows.
    * **episodic** — actions taken, interactions, events.  Ranked by a
      blend of similarity and recency.  Automatically pruned after
      ``EPISODIC_TTL_DAYS``.
    """

    def __init__(self, db_dir: Path | None = None) -> None:
        self._dir = db_dir or _DEFAULT_DB_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "semantic.db"
        self._conn = sqlite3.connect(str(self._db_path))
        self._has_zone = False
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding TEXT NOT NULL,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                zone TEXT NOT NULL DEFAULT 'semantic'
            )
        """)
        self._conn.commit()
        self._migrate_add_zone()
        self._has_zone = self._check_has_zone()

    def _check_has_zone(self) -> bool:
        cursor = self._conn.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}
        return "zone" in columns

    def _migrate_add_zone(self) -> None:
        """Add the zone column if it doesn't exist (backward compat)."""
        cursor = self._conn.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}
        if "zone" not in columns:
            try:
                self._conn.execute(
                    "ALTER TABLE memories ADD COLUMN zone TEXT NOT NULL DEFAULT 'semantic'"
                )
                self._conn.commit()
                logger.info("memory_schema_migrated", added_column="zone")
            except sqlite3.OperationalError as exc:
                logger.warning("memory_migration_skipped", reason=str(exc))

    def add(
        self,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        zone: MemoryZone = "semantic",
    ) -> int:
        meta = metadata or {}
        now = time.time()

        if zone == "semantic" and self._has_zone:
            existing = self._find_duplicate(embedding, zone="semantic")
            if existing is not None:
                self._conn.execute(
                    """UPDATE memories
                       SET content = ?, metadata = ?, embedding = ?,
                           accessed_at = ?, access_count = access_count + 1
                       WHERE id = ?""",
                    (
                        content,
                        json.dumps(meta, ensure_ascii=False, default=str),
                        json.dumps(embedding),
                        now,
                        existing,
                    ),
                )
                self._conn.commit()
                logger.info("memory_dedup_updated", id=existing, zone=zone)
                return existing

        if self._has_zone:
            row = self._conn.execute(
                """INSERT INTO memories
                   (content, metadata, embedding, created_at, accessed_at, zone)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    content,
                    json.dumps(meta, ensure_ascii=False, default=str),
                    json.dumps(embedding),
                    now,
                    now,
                    zone,
                ),
            )
        else:
            row = self._conn.execute(
                """INSERT INTO memories
                   (content, metadata, embedding, created_at, accessed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    content,
                    json.dumps(meta, ensure_ascii=False, default=str),
                    json.dumps(embedding),
                    now,
                    now,
                ),
            )
        self._conn.commit()
        return row.lastrowid  # type: ignore[return-value]

    def _find_duplicate(
        self,
        embedding: list[float],
        zone: MemoryZone,
    ) -> int | None:
        """Return the ID of an existing memory with similarity > threshold, or None."""
        if self._has_zone:
            rows = self._conn.execute(
                "SELECT id, embedding FROM memories WHERE zone = ?",
                (zone,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, embedding FROM memories"
            ).fetchall()
        best_id: int | None = None
        best_sim = 0.0
        for row_id, emb_json in rows:
            emb = json.loads(emb_json)
            sim = _cosine_similarity(embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_id = row_id
        if best_sim >= DEDUP_SIMILARITY_THRESHOLD:
            return best_id
        return None

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        min_similarity: float = 0.3,
        zone: MemoryZone | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories by similarity.

        When *zone* is ``None``, searches all zones.  Episodic memories
        are scored with a recency blend; semantic memories use pure
        cosine similarity.
        """
        if self._has_zone:
            if zone:
                rows = self._conn.execute(
                    "SELECT id, content, metadata, embedding, created_at, access_count, zone "
                    "FROM memories WHERE zone = ?",
                    (zone,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, content, metadata, embedding, created_at, access_count, zone "
                    "FROM memories"
                ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, content, metadata, embedding, created_at, access_count "
                "FROM memories"
            ).fetchall()
            rows = [(*r, "semantic") for r in rows]

        now = time.time()
        scored: list[tuple[float, dict[str, Any]]] = []
        for row_id, content, meta_json, emb_json, created_at, access_count, row_zone in rows:
            emb = json.loads(emb_json)
            sim = _cosine_similarity(query_embedding, emb)
            if sim < min_similarity:
                continue

            if row_zone == "episodic":
                recency = _recency_weight(created_at, now)
                score = sim * 0.6 + recency * 0.4
            else:
                score = sim

            scored.append((score, {
                "id": row_id,
                "content": content,
                "metadata": json.loads(meta_json),
                "similarity": round(sim, 4),
                "score": round(score, 4),
                "zone": row_zone,
                "created_at": created_at,
                "access_count": access_count,
            }))

        scored.sort(key=lambda x: x[0], reverse=True)

        for _, item in scored[:k]:
            self._conn.execute(
                "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                (now, item["id"]),
            )
        if scored:
            self._conn.commit()

        return [item for _, item in scored[:k]]

    def search_zones(
        self,
        query_embedding: list[float],
        k_episodic: int = 3,
        k_semantic: int = 3,
        min_similarity: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search both zones independently and merge results.

        Returns up to *k_episodic* + *k_semantic* memories, ensuring
        representation from both zones when available.
        """
        episodic = self.search(
            query_embedding, k=k_episodic,
            min_similarity=min_similarity, zone="episodic",
        )
        semantic = self.search(
            query_embedding, k=k_semantic,
            min_similarity=min_similarity, zone="semantic",
        )

        seen_ids = set()
        merged: list[dict[str, Any]] = []
        for item in episodic + semantic:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                merged.append(item)

        return merged

    def prune_episodic(self, ttl_days: int = EPISODIC_TTL_DAYS) -> int:
        """Delete episodic memories older than *ttl_days*. Returns count deleted."""
        if not self._has_zone:
            return 0
        cutoff = time.time() - (ttl_days * 86400)
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE zone = 'episodic' AND created_at < ?",
            (cutoff,),
        )
        self._conn.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.info("episodic_memories_pruned", deleted=deleted, ttl_days=ttl_days)
        return deleted

    def delete(self, memory_id: int) -> bool:
        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self, zone: MemoryZone | None = None) -> int:
        if zone and self._has_zone:
            return self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE zone = ?", (zone,),
            ).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def list_recent(self, limit: int = 10, zone: MemoryZone | None = None) -> list[dict[str, Any]]:
        if self._has_zone:
            if zone:
                rows = self._conn.execute(
                    "SELECT id, content, metadata, created_at, access_count, zone "
                    "FROM memories WHERE zone = ? ORDER BY created_at DESC LIMIT ?",
                    (zone, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, content, metadata, created_at, access_count, zone "
                    "FROM memories ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        else:
            rows_raw = self._conn.execute(
                "SELECT id, content, metadata, created_at, access_count "
                "FROM memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            rows = [(*r, "semantic") for r in rows_raw]
        return [
            {
                "id": r[0],
                "content": r[1],
                "metadata": json.loads(r[2]),
                "created_at": r[3],
                "access_count": r[4],
                "zone": r[5],
            }
            for r in rows
        ]


class SemanticMemoryTool(BaseTool):
    """Semantic memory: remember and recall information by meaning, not exact keys.

    Memories live in two zones:
    - **semantic** (default) — permanent facts, lessons, identity.
      Duplicates are merged automatically.
    - **episodic** — time-bound actions and events. Auto-pruned after ~2 weeks.
    """

    def __init__(
        self,
        embedder: EmbeddingClient,
        db_dir: Path | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = SemanticMemoryStore(db_dir=db_dir)

    @property
    def store(self) -> SemanticMemoryStore:
        return self._store

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="semantic_memory",
            description=(
                "Long-term memory with two zones: "
                "'semantic' (permanent facts, lessons, identity — duplicates auto-merged) "
                "and 'episodic' (actions taken, events — auto-expires after ~2 weeks). "
                "Operations: "
                "'remember' — store knowledge (set zone='episodic' for actions). "
                "'recall' — search by meaning across both zones. "
                "'forget' — delete a specific memory by ID. "
                "'stats' — show memory counts per zone and recent entries."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["remember", "recall", "forget", "stats"],
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "For 'remember': the text to memorize. "
                            "For 'recall': the search query describing what you want to find."
                        ),
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for categorization (only for 'remember')",
                    },
                    "zone": {
                        "type": "string",
                        "enum": ["semantic", "episodic"],
                        "description": (
                            "Memory zone. 'semantic' (default) for permanent knowledge, "
                            "'episodic' for time-bound actions/events. "
                            "For 'recall': filter to a specific zone, or omit to search both."
                        ),
                    },
                    "memory_id": {
                        "type": "integer",
                        "description": "The memory ID to delete (only for 'forget')",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return for 'recall' (default: 5)",
                    },
                },
                "required": ["operation"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "memories": {"type": "array"},
                    "memory_id": {"type": "integer"},
                    "total_memories": {"type": "integer"},
                },
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        operation = inputs.get("operation", "")

        if operation == "remember":
            content = inputs.get("content", "")
            if not content:
                return ToolResult(success=False, error="'remember' requires 'content'")
            tags = inputs.get("tags", [])
            zone: MemoryZone = inputs.get("zone", "semantic")  # type: ignore[assignment]
            if zone not in ("semantic", "episodic"):
                zone = "semantic"
            try:
                embedding = await self._embedder.embed_one(content)
            except Exception as exc:
                return ToolResult(success=False, error=f"Embedding failed: {exc}")

            memory_id = self._store.add(
                content=content,
                embedding=embedding,
                metadata={"tags": tags},
                zone=zone,
            )
            logger.info("semantic_memory_stored", memory_id=memory_id, tags=tags, zone=zone)
            return ToolResult(
                success=True,
                output={
                    "memory_id": memory_id,
                    "zone": zone,
                    "stored": True,
                    "total_memories": self._store.count(),
                },
            )

        if operation == "recall":
            query = inputs.get("content", "")
            if not query:
                return ToolResult(success=False, error="'recall' requires 'content' (the search query)")
            k = inputs.get("k", 5)
            zone_filter: MemoryZone | None = inputs.get("zone")  # type: ignore[assignment]
            if zone_filter and zone_filter not in ("semantic", "episodic"):
                zone_filter = None
            try:
                query_embedding = await self._embedder.embed_one(query)
            except Exception as exc:
                return ToolResult(success=False, error=f"Embedding failed: {exc}")

            if zone_filter:
                results = self._store.search(
                    query_embedding, k=k, zone=zone_filter,
                )
            else:
                results = self._store.search_zones(
                    query_embedding,
                    k_episodic=max(1, k // 2),
                    k_semantic=max(1, k - k // 2),
                )
            logger.info("semantic_memory_recall", query=query[:80], results=len(results))
            return ToolResult(
                success=True,
                output={"memories": results, "query": query, "total_found": len(results)},
            )

        if operation == "forget":
            memory_id = inputs.get("memory_id")
            if memory_id is None:
                return ToolResult(success=False, error="'forget' requires 'memory_id'")
            deleted = self._store.delete(int(memory_id))
            if not deleted:
                return ToolResult(success=False, error=f"Memory ID {memory_id} not found")
            return ToolResult(
                success=True,
                output={"memory_id": memory_id, "deleted": True, "total_memories": self._store.count()},
            )

        if operation == "stats":
            total = self._store.count()
            episodic_count = self._store.count(zone="episodic")
            semantic_count = self._store.count(zone="semantic")
            recent = self._store.list_recent(limit=5)
            return ToolResult(
                success=True,
                output={
                    "total_memories": total,
                    "episodic_count": episodic_count,
                    "semantic_count": semantic_count,
                    "recent": recent,
                },
            )

        return ToolResult(
            success=False,
            error=f"Unknown operation: '{operation}'. Use: remember, recall, forget, stats",
        )
