"""Unit tests for klovis_agent.memory.

KeyValueMemory and SemanticMemoryStore depend on structlog.
"""

from __future__ import annotations

from pathlib import Path

import pytest

structlog = pytest.importorskip("structlog")

from klovis_agent.memory.kv import KeyValueMemory
from klovis_agent.tools.builtin.semantic_memory import (
    DEDUP_SIMILARITY_THRESHOLD,
    SemanticMemoryStore,
    _cosine_similarity,
    _recency_weight,
)


class TestKeyValueMemory:
    def test_set_get(self, tmp_path: Path):
        m = KeyValueMemory(memory_dir=tmp_path)
        m.set("name", "klovis")
        assert m.get("name") == "klovis"

    def test_get_missing(self, tmp_path: Path):
        m = KeyValueMemory(memory_dir=tmp_path)
        assert m.get("nope") is None

    def test_delete(self, tmp_path: Path):
        m = KeyValueMemory(memory_dir=tmp_path)
        m.set("x", 1)
        assert m.delete("x") is True
        assert m.get("x") is None

    def test_delete_missing(self, tmp_path: Path):
        m = KeyValueMemory(memory_dir=tmp_path)
        assert m.delete("nope") is False

    def test_keys(self, tmp_path: Path):
        m = KeyValueMemory(memory_dir=tmp_path)
        m.set("a", 1)
        m.set("b", 2)
        assert set(m.keys()) == {"a", "b"}

    def test_count(self, tmp_path: Path):
        m = KeyValueMemory(memory_dir=tmp_path)
        assert m.count() == 0
        m.set("x", 1)
        assert m.count() == 1

    def test_persistence(self, tmp_path: Path):
        m1 = KeyValueMemory(memory_dir=tmp_path)
        m1.set("persist", "yes")
        m2 = KeyValueMemory(memory_dir=tmp_path)
        assert m2.get("persist") == "yes"


class TestCosineSimilarity:
    def test_identical(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestSemanticMemoryStore:
    def test_add_and_count(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        assert store.count() == 0
        store.add("hello world", [1.0, 0.0, 0.0])
        assert store.count() == 1

    def test_search(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("about cats", [1.0, 0.0, 0.0])
        store.add("about dogs", [0.0, 1.0, 0.0])
        store.add("about fish", [0.0, 0.0, 1.0])

        results = store.search([1.0, 0.0, 0.0], k=2, min_similarity=0.5)
        assert len(results) == 1
        assert results[0]["content"] == "about cats"
        assert results[0]["similarity"] == pytest.approx(1.0)

    def test_search_with_metadata(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("tagged", [1.0, 0.0], metadata={"tags": ["test"]})
        results = store.search([1.0, 0.0], k=1, min_similarity=0.5)
        assert results[0]["metadata"]["tags"] == ["test"]

    def test_delete(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        mid = store.add("to delete", [1.0, 0.0])
        assert store.count() == 1
        assert store.delete(mid) is True
        assert store.count() == 0

    def test_delete_nonexistent(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        assert store.delete(999) is False

    def test_list_recent(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("first", [1.0, 0.0])
        store.add("second", [0.0, 1.0])
        recent = store.list_recent(limit=1)
        assert len(recent) == 1
        assert recent[0]["content"] == "second"


class TestMemoryZones:
    """Tests for the episodic/semantic zone system."""

    def test_default_zone_is_semantic(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("a fact", [1.0, 0.0])
        recent = store.list_recent(limit=1)
        assert recent[0]["zone"] == "semantic"

    def test_add_episodic(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("I replied to post X", [1.0, 0.0], zone="episodic")
        assert store.count(zone="episodic") == 1
        assert store.count(zone="semantic") == 0

    def test_add_semantic(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("My name is Klovis", [0.0, 1.0], zone="semantic")
        assert store.count(zone="semantic") == 1
        assert store.count(zone="episodic") == 0

    def test_count_by_zone(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("episodic 1", [1.0, 0.0], zone="episodic")
        store.add("episodic 2", [0.9, 0.1], zone="episodic")
        store.add("semantic 1", [0.0, 1.0], zone="semantic")
        assert store.count() == 3
        assert store.count(zone="episodic") == 2
        assert store.count(zone="semantic") == 1

    def test_search_by_zone(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("episodic fact", [1.0, 0.0, 0.0], zone="episodic")
        store.add("semantic fact", [1.0, 0.0, 0.0], zone="semantic")

        ep_results = store.search([1.0, 0.0, 0.0], k=5, zone="episodic")
        assert len(ep_results) == 1
        assert ep_results[0]["zone"] == "episodic"

        sem_results = store.search([1.0, 0.0, 0.0], k=5, zone="semantic")
        assert len(sem_results) == 1
        assert sem_results[0]["zone"] == "semantic"

    def test_search_all_zones(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("episodic", [1.0, 0.0], zone="episodic")
        store.add("semantic", [0.9, 0.1], zone="semantic")
        results = store.search([1.0, 0.0], k=5, zone=None)
        assert len(results) == 2

    def test_search_zones_merges(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("ep1", [1.0, 0.0, 0.0], zone="episodic")
        store.add("ep2", [0.9, 0.1, 0.0], zone="episodic")
        store.add("sem1", [1.0, 0.0, 0.0], zone="semantic")
        store.add("sem2", [0.0, 1.0, 0.0], zone="semantic")

        merged = store.search_zones(
            [1.0, 0.0, 0.0],
            k_episodic=1, k_semantic=1,
            min_similarity=0.3,
        )
        zones = {r["zone"] for r in merged}
        assert "episodic" in zones
        assert "semantic" in zones

    def test_list_recent_by_zone(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("ep", [1.0, 0.0], zone="episodic")
        store.add("sem", [0.0, 1.0], zone="semantic")

        ep = store.list_recent(limit=10, zone="episodic")
        assert len(ep) == 1
        assert ep[0]["zone"] == "episodic"

        sem = store.list_recent(limit=10, zone="semantic")
        assert len(sem) == 1
        assert sem[0]["zone"] == "semantic"


class TestSemanticDeduplication:
    """Tests for duplicate detection in the semantic zone."""

    def test_duplicate_semantic_updates_in_place(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        id1 = store.add("My name is Klovis", [1.0, 0.0, 0.0], zone="semantic")
        id2 = store.add("My name is Klovis", [1.0, 0.0, 0.0], zone="semantic")
        assert id1 == id2
        assert store.count(zone="semantic") == 1

    def test_similar_semantic_updates_in_place(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.99, 0.01, 0.0]
        sim = _cosine_similarity(v1, v2)
        assert sim >= DEDUP_SIMILARITY_THRESHOLD

        id1 = store.add("fact v1", v1, zone="semantic")
        id2 = store.add("fact v2 (updated)", v2, zone="semantic")
        assert id1 == id2
        assert store.count(zone="semantic") == 1
        recent = store.list_recent(limit=1, zone="semantic")
        assert recent[0]["content"] == "fact v2 (updated)"

    def test_different_semantic_creates_new(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("about cats", [1.0, 0.0, 0.0], zone="semantic")
        store.add("about dogs", [0.0, 1.0, 0.0], zone="semantic")
        assert store.count(zone="semantic") == 2

    def test_episodic_never_deduplicates(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        id1 = store.add("replied to post X", [1.0, 0.0], zone="episodic")
        id2 = store.add("replied to post X", [1.0, 0.0], zone="episodic")
        assert id1 != id2
        assert store.count(zone="episodic") == 2


class TestEpisodicPruning:
    """Tests for TTL-based episodic memory pruning."""

    def test_prune_old_episodic(self, tmp_path: Path):
        import time

        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("old action", [1.0, 0.0], zone="episodic")

        store._conn.execute(
            "UPDATE memories SET created_at = ? WHERE zone = 'episodic'",
            (time.time() - 20 * 86400,),
        )
        store._conn.commit()

        deleted = store.prune_episodic(ttl_days=14)
        assert deleted == 1
        assert store.count(zone="episodic") == 0

    def test_prune_keeps_recent_episodic(self, tmp_path: Path):
        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("recent action", [1.0, 0.0], zone="episodic")
        deleted = store.prune_episodic(ttl_days=14)
        assert deleted == 0
        assert store.count(zone="episodic") == 1

    def test_prune_does_not_touch_semantic(self, tmp_path: Path):
        import time

        store = SemanticMemoryStore(db_dir=tmp_path)
        store.add("permanent fact", [1.0, 0.0], zone="semantic")

        store._conn.execute(
            "UPDATE memories SET created_at = ? WHERE zone = 'semantic'",
            (time.time() - 100 * 86400,),
        )
        store._conn.commit()

        deleted = store.prune_episodic(ttl_days=14)
        assert deleted == 0
        assert store.count(zone="semantic") == 1


class TestRecencyWeight:
    def test_now_is_one(self):
        import time
        now = time.time()
        assert _recency_weight(now, now) == pytest.approx(1.0)

    def test_24h_is_half(self):
        import time
        now = time.time()
        assert _recency_weight(now - 86400, now) == pytest.approx(0.5)

    def test_old_approaches_zero(self):
        import time
        now = time.time()
        w = _recency_weight(now - 30 * 86400, now)
        assert w < 0.05


class TestSchemaMigration:
    def test_migration_adds_zone_column(self, tmp_path: Path):
        """Simulate an old DB without the zone column."""
        import sqlite3

        db_path = tmp_path / "semantic.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding TEXT NOT NULL,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute(
            "INSERT INTO memories (content, metadata, embedding, created_at, accessed_at) "
            "VALUES ('old memory', '{}', '[1.0, 0.0]', 0, 0)"
        )
        conn.commit()
        conn.close()

        store = SemanticMemoryStore(db_dir=tmp_path)
        assert store.count() == 1
        recent = store.list_recent(limit=1)
        assert recent[0]["zone"] == "semantic"
