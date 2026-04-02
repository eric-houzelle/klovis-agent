from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import structlog

from klovis_agent.models.state import AgentState
from klovis_agent.paths import data_home

logger = structlog.get_logger(__name__)

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    task_goal TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS step_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_step_logs_run ON step_logs(run_id);
"""


class AgentStore:
    """SQLite persistence for agent runs (V1)."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            default = data_home() / "agent.db"
            default.parent.mkdir(parents=True, exist_ok=True)
            db_path = str(default)
        self._db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(CREATE_TABLES_SQL)
            await db.commit()
        logger.info("db_initialized", path=self._db_path)

    async def save_run(self, state: AgentState) -> None:
        now = datetime.now(timezone.utc).isoformat()
        state_json = state.model_dump_json()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO agent_runs (run_id, task_goal, status, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (state.run_id, state.task.goal, state.status, state_json, now, now),
            )
            await db.commit()

    async def load_run(self, run_id: str) -> AgentState | None:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT state_json FROM agent_runs WHERE run_id = ?",
                (run_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return AgentState.model_validate_json(row[0])

    async def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT run_id, task_goal, status, created_at, updated_at "
                "FROM agent_runs ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def log_step_event(
        self,
        run_id: str,
        step_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO step_logs (run_id, step_id, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, step_id, event_type, json.dumps(payload), now),
            )
            await db.commit()

    async def get_run_logs(self, run_id: str) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM step_logs WHERE run_id = ? ORDER BY created_at",
                (run_id,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
