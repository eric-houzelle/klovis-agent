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
MemoryType = Literal[
    "mission",
    "state",
    "preference",
    "fact",
    "lesson",
    "strategy",
    "action",
    "identity",
    "other",
]

EPISODIC_TTL_DAYS = 14
DEDUP_SIMILARITY_THRESHOLD = 0.9
_VALID_MEMORY_TYPES = {
    "mission",
    "state",
    "preference",
    "fact",
    "lesson",
    "strategy",
    "action",
    "identity",
    "other",
}


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


def _normalize_metadata(
    metadata: dict[str, Any] | None,
    zone: MemoryZone,
) -> dict[str, Any]:
    meta = dict(metadata or {})
    tags = meta.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t) for t in tags]
    meta["tags"] = tags

    mtype = str(meta.get("type", "")).strip().lower()
    if not mtype:
        mtype = "action" if (zone == "episodic" or "action_taken" in tags) else "fact"
    if mtype not in _VALID_MEMORY_TYPES:
        mtype = "other"
    meta["type"] = mtype

    # Dynamic "cases" layer on top of core memory_type.
    category = str(meta.get("category", "")).strip().lower()
    if not category:
        category = mtype
    meta["category"] = category

    subcategory = str(meta.get("subcategory", "")).strip().lower()
    if subcategory:
        meta["subcategory"] = subcategory
    else:
        meta.pop("subcategory", None)

    namespace = str(meta.get("namespace", "")).strip().lower()
    if namespace:
        meta["namespace"] = namespace
    else:
        meta.pop("namespace", None)
    return meta


def _matches_type_filter(
    metadata: dict[str, Any],
    memory_type: MemoryType | None,
    memory_types: list[str] | None,
) -> bool:
    if not memory_type and not memory_types:
        return True
    mtype = str(metadata.get("type", "")).strip().lower()
    if memory_type and mtype != memory_type:
        return False
    if memory_types:
        allowed = {str(t).strip().lower() for t in memory_types if str(t).strip()}
        if allowed and mtype not in allowed:
            return False
    return True


def _matches_category_filter(
    metadata: dict[str, Any],
    category: str | None,
    categories: list[str] | None,
    namespace: str | None,
    subcategory: str | None,
) -> bool:
    if not category and not categories and not namespace and not subcategory:
        return True

    m_category = str(metadata.get("category", "")).strip().lower()
    m_namespace = str(metadata.get("namespace", "")).strip().lower()
    m_subcategory = str(metadata.get("subcategory", "")).strip().lower()

    if category and m_category != str(category).strip().lower():
        return False
    if categories:
        allowed = {str(c).strip().lower() for c in categories if str(c).strip()}
        if allowed and m_category not in allowed:
            return False
    if namespace and m_namespace != str(namespace).strip().lower():
        return False
    if subcategory and m_subcategory != str(subcategory).strip().lower():
        return False
    return True


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
        meta = _normalize_metadata(metadata, zone)
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
        memory_type: MemoryType | None = None,
        memory_types: list[str] | None = None,
        category: str | None = None,
        categories: list[str] | None = None,
        namespace: str | None = None,
        subcategory: str | None = None,
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
            metadata = json.loads(meta_json)
            if not _matches_type_filter(metadata, memory_type, memory_types):
                continue
            if not _matches_category_filter(
                metadata,
                category=category,
                categories=categories,
                namespace=namespace,
                subcategory=subcategory,
            ):
                continue

            if row_zone == "episodic":
                recency = _recency_weight(created_at, now)
                score = sim * 0.6 + recency * 0.4
            else:
                score = sim

            scored.append((score, {
                "id": row_id,
                "content": content,
                "metadata": metadata,
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
        memory_type: MemoryType | None = None,
        memory_types: list[str] | None = None,
        category: str | None = None,
        categories: list[str] | None = None,
        namespace: str | None = None,
        subcategory: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search both zones independently and merge results.

        Returns up to *k_episodic* + *k_semantic* memories, ensuring
        representation from both zones when available.
        """
        episodic = self.search(
            query_embedding, k=k_episodic,
            min_similarity=min_similarity, zone="episodic",
            memory_type=memory_type, memory_types=memory_types,
            category=category, categories=categories,
            namespace=namespace, subcategory=subcategory,
        )
        semantic = self.search(
            query_embedding, k=k_semantic,
            min_similarity=min_similarity, zone="semantic",
            memory_type=memory_type, memory_types=memory_types,
            category=category, categories=categories,
            namespace=namespace, subcategory=subcategory,
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

    def count_by_type(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT metadata FROM memories").fetchall()
        counts: dict[str, int] = {}
        for (meta_json,) in rows:
            try:
                meta = json.loads(meta_json)
            except Exception:
                meta = {}
            mtype = str(meta.get("type", "other")).strip().lower() or "other"
            if mtype not in _VALID_MEMORY_TYPES:
                mtype = "other"
            counts[mtype] = counts.get(mtype, 0) + 1
        return counts

    def count_by_category(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT metadata FROM memories").fetchall()
        counts: dict[str, int] = {}
        for (meta_json,) in rows:
            try:
                meta = json.loads(meta_json)
            except Exception:
                meta = {}
            category = str(meta.get("category", "")).strip().lower()
            if not category:
                category = str(meta.get("type", "other")).strip().lower() or "other"
            counts[category] = counts.get(category, 0) + 1
        return counts

    def list_categories(self, limit: int = 50) -> list[dict[str, Any]]:
        counts = self.count_by_category()
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [{"category": name, "count": count} for name, count in ranked[:limit]]

    def list_recent(
        self,
        limit: int = 10,
        zone: MemoryZone | None = None,
        memory_type: MemoryType | None = None,
        memory_types: list[str] | None = None,
        category: str | None = None,
        categories: list[str] | None = None,
        namespace: str | None = None,
        subcategory: str | None = None,
    ) -> list[dict[str, Any]]:
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
        items = [
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
        filtered = [
            item for item in items
            if _matches_type_filter(item["metadata"], memory_type, memory_types)
            and _matches_category_filter(
                item["metadata"],
                category=category,
                categories=categories,
                namespace=namespace,
                subcategory=subcategory,
            )
        ]
        return filtered[:limit]

    def reclassify(
        self,
        memory_ids: list[int],
        *,
        memory_type: str | None = None,
        category: str | None = None,
        subcategory: str | None = None,
        namespace: str | None = None,
        tags: list[str] | None = None,
        merge_tags: bool = True,
    ) -> list[dict[str, Any]]:
        """Update metadata fields for existing memories and return updated rows."""
        if not memory_ids:
            return []

        unique_ids = sorted({int(i) for i in memory_ids})
        if not unique_ids:
            return []

        qmarks = ",".join(["?"] * len(unique_ids))
        rows = self._conn.execute(
            f"SELECT id, content, metadata, created_at, access_count, zone FROM memories WHERE id IN ({qmarks})",  # noqa: S608
            unique_ids,
        ).fetchall()
        if not rows:
            return []

        updated: list[dict[str, Any]] = []
        for row_id, content, meta_json, created_at, access_count, zone in rows:
            try:
                meta = json.loads(meta_json)
            except Exception:
                meta = {}

            if memory_type is not None:
                mtype = str(memory_type).strip().lower()
                if mtype not in _VALID_MEMORY_TYPES:
                    mtype = "other"
                meta["type"] = mtype
            if category is not None:
                cat = str(category).strip().lower()
                if cat:
                    meta["category"] = cat
                else:
                    meta.pop("category", None)
            if subcategory is not None:
                sub = str(subcategory).strip().lower()
                if sub:
                    meta["subcategory"] = sub
                else:
                    meta.pop("subcategory", None)
            if namespace is not None:
                ns = str(namespace).strip().lower()
                if ns:
                    meta["namespace"] = ns
                else:
                    meta.pop("namespace", None)
            if tags is not None:
                cleaned = [str(t).strip() for t in tags if str(t).strip()]
                if merge_tags:
                    prev = meta.get("tags", [])
                    if not isinstance(prev, list):
                        prev = []
                    meta["tags"] = sorted({*map(str, prev), *cleaned})
                else:
                    meta["tags"] = cleaned

            normalized = _normalize_metadata(meta, zone=zone)
            self._conn.execute(
                "UPDATE memories SET metadata = ? WHERE id = ?",
                (json.dumps(normalized, ensure_ascii=False, default=str), row_id),
            )
            updated.append(
                {
                    "id": row_id,
                    "content": content,
                    "metadata": normalized,
                    "created_at": created_at,
                    "access_count": access_count,
                    "zone": zone,
                }
            )

        self._conn.commit()
        return updated


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
                "'stats' — show memory counts per zone and recent entries. "
                "'cases' — list dynamic memory categories currently in use. "
                "'reclassify' — reorganize memory metadata for existing IDs."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["remember", "recall", "forget", "stats", "cases", "reclassify"],
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
                    "memory_type": {
                        "type": "string",
                        "enum": [
                            "mission", "state", "preference", "fact", "lesson",
                            "strategy", "action", "identity", "other",
                        ],
                        "description": (
                            "Structured category of the memory. "
                            "Use for remember and optional recall filtering."
                        ),
                    },
                    "memory_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "mission", "state", "preference", "fact", "lesson",
                                "strategy", "action", "identity", "other",
                            ],
                        },
                        "description": "Optional list of categories to include during recall.",
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Dynamic memory case name (free-form). "
                            "Use for remember, recall filters, or reclassify."
                        ),
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of dynamic categories to include during recall.",
                    },
                    "subcategory": {
                        "type": "string",
                        "description": "Optional subcategory case (free-form).",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace to group memory cases (e.g. project slug).",
                    },
                    "memory_id": {
                        "type": "integer",
                        "description": "The memory ID to delete (only for 'forget')",
                    },
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Memory IDs to update (for reclassify).",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return for 'recall' (default: 5)",
                    },
                    "merge_tags": {
                        "type": "boolean",
                        "description": "For reclassify: merge new tags with existing ones (default: true).",
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
            memory_type = inputs.get("memory_type")
            if memory_type and memory_type not in _VALID_MEMORY_TYPES:
                memory_type = "other"
            category = inputs.get("category")
            subcategory = inputs.get("subcategory")
            namespace = inputs.get("namespace")
            try:
                embedding = await self._embedder.embed_one(content)
            except Exception as exc:
                return ToolResult(success=False, error=f"Embedding failed: {exc}")

            memory_id = self._store.add(
                content=content,
                embedding=embedding,
                metadata={
                    "tags": tags,
                    "type": memory_type,
                    "category": category,
                    "subcategory": subcategory,
                    "namespace": namespace,
                },
                zone=zone,
            )
            logger.info("semantic_memory_stored", memory_id=memory_id, tags=tags, zone=zone)
            return ToolResult(
                success=True,
                output={
                    "memory_id": memory_id,
                    "zone": zone,
                    "memory_type": memory_type,
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
            memory_type: MemoryType | None = inputs.get("memory_type")  # type: ignore[assignment]
            if memory_type and memory_type not in _VALID_MEMORY_TYPES:
                memory_type = None
            memory_types = inputs.get("memory_types")
            if not isinstance(memory_types, list):
                memory_types = None
            category = inputs.get("category")
            if category is not None:
                category = str(category).strip().lower() or None
            categories = inputs.get("categories")
            if not isinstance(categories, list):
                categories = None
            namespace = inputs.get("namespace")
            if namespace is not None:
                namespace = str(namespace).strip().lower() or None
            subcategory = inputs.get("subcategory")
            if subcategory is not None:
                subcategory = str(subcategory).strip().lower() or None
            try:
                query_embedding = await self._embedder.embed_one(query)
            except Exception as exc:
                return ToolResult(success=False, error=f"Embedding failed: {exc}")

            if zone_filter:
                results = self._store.search(
                    query_embedding,
                    k=k,
                    zone=zone_filter,
                    memory_type=memory_type,
                    memory_types=memory_types,
                    category=category,
                    categories=categories,
                    namespace=namespace,
                    subcategory=subcategory,
                )
            else:
                results = self._store.search_zones(
                    query_embedding,
                    k_episodic=max(1, k // 2),
                    k_semantic=max(1, k - k // 2),
                    memory_type=memory_type,
                    memory_types=memory_types,
                    category=category,
                    categories=categories,
                    namespace=namespace,
                    subcategory=subcategory,
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
                    "counts_by_type": self._store.count_by_type(),
                    "counts_by_category": self._store.count_by_category(),
                    "recent": recent,
                },
            )

        if operation == "cases":
            cases = self._store.list_categories(limit=100)
            return ToolResult(
                success=True,
                output={
                    "cases": cases,
                    "total_cases": len(cases),
                },
            )

        if operation == "reclassify":
            memory_id = inputs.get("memory_id")
            memory_ids = inputs.get("memory_ids")
            ids: list[int] = []
            if memory_id is not None:
                ids.append(int(memory_id))
            if isinstance(memory_ids, list):
                ids.extend(int(v) for v in memory_ids)
            ids = sorted({*ids})
            if not ids:
                return ToolResult(
                    success=False,
                    error="'reclassify' requires 'memory_id' or 'memory_ids'",
                )

            memory_type = inputs.get("memory_type")
            category = inputs.get("category")
            subcategory = inputs.get("subcategory")
            namespace = inputs.get("namespace")
            tags = inputs.get("tags")
            if tags is not None and not isinstance(tags, list):
                return ToolResult(success=False, error="'tags' must be a list when provided")
            merge_tags = bool(inputs.get("merge_tags", True))
            if all(v is None for v in (memory_type, category, subcategory, namespace, tags)):
                return ToolResult(
                    success=False,
                    error="'reclassify' requires at least one field to update",
                )

            updated = self._store.reclassify(
                ids,
                memory_type=memory_type,
                category=category,
                subcategory=subcategory,
                namespace=namespace,
                tags=tags,
                merge_tags=merge_tags,
            )
            if not updated:
                return ToolResult(
                    success=False,
                    error="No matching memory IDs found",
                )
            return ToolResult(
                success=True,
                output={
                    "updated_count": len(updated),
                    "updated": updated,
                },
            )

        return ToolResult(
            success=False,
            error=(
                f"Unknown operation: '{operation}'. "
                "Use: remember, recall, forget, stats, cases, reclassify"
            ),
        )
