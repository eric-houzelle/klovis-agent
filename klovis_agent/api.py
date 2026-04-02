from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from klovis_agent.agent import Agent
from klovis_agent.config import AgentConfig
from klovis_agent.infra.persistence import AgentStore

logger = structlog.get_logger(__name__)

config = AgentConfig()
store = AgentStore(db_path=config.db_url.replace("sqlite+aiosqlite:///", "") if config.db_url else None)
agent: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    global agent
    await store.initialize()

    agent = Agent(
        llm=config.llm,
        sandbox=config.sandbox,
        max_iterations=config.max_iterations,
        verbose=config.verbose,
        data_dir=config.data_dir or None,
    )
    logger.info("api_started")
    yield
    logger.info("api_shutdown")


app = FastAPI(
    title="Klovis Agent",
    version="0.2.0",
    description="Composable autonomous agent library",
    lifespan=lifespan,
)


class CreateTaskRequest(BaseModel):
    goal: str
    context: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)


class RunResponse(BaseModel):
    run_id: str
    status: str
    iteration_count: int
    num_step_results: int
    artifacts: dict[str, Any]


@app.post("/runs", response_model=RunResponse)
async def create_run(req: CreateTaskRequest) -> RunResponse:
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    result = await agent.run(
        goal=req.goal,
        context=req.context,
        constraints=req.constraints,
        success_criteria=req.success_criteria,
    )
    await store.save_run(result.raw_state)

    return RunResponse(
        run_id=result.run_id,
        status=result.status,
        iteration_count=result.iteration_count,
        num_step_results=len(result.steps),
        artifacts=result.artifacts,
    )


@app.get("/runs")
async def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    return await store.list_runs(limit=limit)


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    state = await store.load_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return state.model_dump()


@app.get("/runs/{run_id}/logs")
async def get_run_logs(run_id: str) -> list[dict[str, Any]]:
    return await store.get_run_logs(run_id)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
