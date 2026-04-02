# klovis-agent

Composable Python library for building **goal-oriented autonomous agents**. Agents plan, execute, verify, and dynamically adapt their strategy through a LangGraph loop. In daemon mode, they observe their environment, decide whether to act, and consolidate their memory — all continuously.

![demo](./assets/demo.gif)

---

## Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      CLI  /  REST API                        │
├──────────────────────────────────────────────────────────────┤
│                          Agent                               │
│                                                              │
│  ┌────────────┐   ┌─────────────┐   ┌────────────────────┐  │
│  │   Tools     │   │ Perception  │   │      Memory        │  │
│  │ (composable)│   │  (sources)  │   │ episodic/semantic  │  │
│  └────────────┘   └─────────────┘   └────────────────────┘  │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│              LangGraph Execution Core                        │
│        plan → execute → check → replan → finish              │
├──────────────────────────────────────────────────────────────┤
│          LLM Router            │        Sandbox              │
│   (phase-based routing)        │   (isolated execution)      │
└──────────────────────────────────────────────────────────────┘
```

**Two operating modes:**

| Mode | Description |
|------|-------------|
| **Single-run** | The agent receives a goal, plans, executes, and returns a result |
| **Daemon (OODA)** | Continuous loop: Observe → Orient → Decide → Act → Consolidate |

---

## Quick start

### Single-run

```python
import asyncio
from klovis_agent import Agent, LLMConfig

async def main():
    agent = Agent(
        llm=LLMConfig(api_key="sk-...", default_model="gpt-4o"),
    )
    result = await agent.run("Explain quantum computing in simple terms")
    print(result.summary)

asyncio.run(main())
```

### Daemon mode

```python
from klovis_agent import Agent, LLMConfig, InboxPerceptionSource

agent = Agent(
    llm=LLMConfig(api_key="sk-..."),
    perceptions=[InboxPerceptionSource()],
)
daemon = agent.as_daemon(interval_minutes=5, max_cycles=100)
await daemon.run()
```

### CLI

```bash
# Single-run
python run.py "Write a blog post about AI agents"

# Daemon
python run.py --daemon --interval 5 --cycles 100

# Options
python run.py -v --data-dir ./my-data "Do something"
python run.py --ephemeral "Quick test"
```

| Option | Description | Default |
|--------|-------------|---------|
| `-v`, `--verbose` | Structlog + raw JSON logs | off |
| `--daemon` | Daemon mode (OODA loop) | off |
| `--interval MIN` | Interval between daemon cycles | 30 |
| `--cycles N` | Max daemon cycles (0 = infinite) | 0 |
| `--data-dir PATH` | Persistent data directory | `~/.local/share/klovis` |
| `--soul PATH` | Personality file (SOUL.md) | none |
| `--ephemeral` | Temporary workspace (nothing persists) | off |

---

## Composition

The agent is fully composable — tools, perceptions, and sandbox are all injectable:

```python
from klovis_agent import Agent, LLMConfig
from klovis_agent.tools.builtin import (
    WebSearchTool, MemoryTool, ShellCommandTool,
    SemanticMemoryTool, FileReadTool, FileWriteTool, FileEditTool,
    FsReadTool, FsWriteTool, FsMkdirTool,
)
from klovis_agent.perception.inbox import InboxPerceptionSource

agent = Agent(
    llm=LLMConfig(api_key="sk-...", default_model="gpt-4o"),
    tools=[WebSearchTool(), MemoryTool(), ShellCommandTool(workspace)],
    perceptions=[InboxPerceptionSource()],
    max_iterations=15,
)
```

Presets are available:

```python
from klovis_agent.tools.presets import default_tools, minimal_tools

# All standard tools
tools = default_tools(workspace, sandbox, embedder, skill_store)

# Minimal set (web search + KV memory)
tools = minimal_tools()
```

---

## Execution loop (LangGraph)

Each run follows a 5-node graph:

```
                    ┌──────────┐
                    │   PLAN   │
                    │ (LLM)    │
                    └────┬─────┘
                         │
                         ▼
                    ┌──────────┐
              ┌────▶│ EXECUTE  │
              │     │ (tool)   │
              │     └────┬─────┘
              │          │
              │          ▼
              │     ┌──────────┐
              │     │  CHECK   │──────────┐
              │     │ (router) │          │
              │     └────┬─────┘          │
              │          │                │
              │     next step?       failed / replan?
              │          │                │
              │          ▼                ▼
              │          │          ┌──────────┐
              └──────────┘          │ REPLAN   │
                                    │ (LLM)    │
                                    └────┬─────┘
                                         │
                                         ▼
                                    ┌──────────┐
                                    │  FINISH  │
                                    │ (summary)│
                                    └──────────┘
```

- **Plan**: the LLM breaks down the goal into steps and selects tools
- **Execute**: each step is executed via the `ToolRegistry`
- **Check**: the router decides whether to continue, retry, replan, or finish
- **Replan**: if the plan fails, the LLM generates a new one with error context
- **Finish**: final summary + memory consolidation

---

## OODA loop (daemon mode)

```
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
    │  │ OBSERVE  │───▶│  ORIENT  │───▶│  DECIDE  │  │
    │  │ perceive │    │  recall  │    │  (LLM)   │  │
    │  └──────────┘    └──────────┘    └────┬─────┘  │
    │                                       │        │
    │                    should_act?         │        │
    │                  ┌────────┬────────────┘        │
    │                  │        │                     │
    │                  No      Yes                    │
    │                  │        │                     │
    │                  │   ┌────▼─────┐               │
    │                  │   │   ACT    │               │
    │                  │   │ agent.run│               │
    │                  │   └────┬─────┘               │
    │                  │        │                     │
    │                  │   ┌────▼──────────┐          │
    │                  │   │  CONSOLIDATE  │          │
    │                  │   │ extract memory│          │
    │                  │   └────┬──────────┘          │
    │                  │        │                     │
    │                  └────────┴─── sleep ───────────┘
    │                                                 │
    └─────────────────────────────────────────────────┘
```

| Phase | Module | Role |
|-------|--------|------|
| **Observe** | `perception.base.perceive()` | Polls all `PerceptionSource`s, aggregates `Event`s |
| **Orient** | `recall.recall_for_task()` | Searches episodic + semantic memory for relevant memories |
| **Decide** | `decision.decide()` | The LLM analyzes events + memories and decides whether to act |
| **Act** | `agent.run(goal)` | Runs the full LangGraph execution graph |
| **Consolidate** | `consolidation.consolidate_run()` | Extracts 2-6 memories from the run and stores them by zone |

---

## Memory system

The agent has a persistent two-zone memory, stored in SQLite with vector embeddings.

![memory demo](./assets/demo-memory.gif)

```
                    ┌─────────────────────────────┐
                    │     SemanticMemoryStore      │
                    │        (SQLite + vectors)    │
                    ├──────────────┬───────────────┤
                    │   EPISODIC   │   SEMANTIC    │
                    │              │               │
                    │ Actions      │ Facts         │
                    │ Events       │ Lessons       │
                    │ Interactions │ Preferences   │
                    │              │ Identity      │
                    │              │ Skills        │
                    ├──────────────┼───────────────┤
                    │ TTL: 14 d    │ Permanent     │
                    │ Score:       │ Score:        │
                    │  sim×0.6     │  cosine       │
                    │  +recency×0.4│  similarity   │
                    │ Dedup:       │ Dedup:        │
                    │  No          │  Yes (>0.9)   │
                    └──────────────┴───────────────┘
```

### Two zones

| Zone | Content | TTL | Scoring | Deduplication |
|------|---------|:---:|---------|:---:|
| **episodic** | Actions taken, events, interactions | 14 days (auto-prune) | `similarity × 0.6 + recency × 0.4` | No (each action is unique) |
| **semantic** | Facts, lessons, preferences, identity | Permanent | Pure cosine similarity | Yes (similarity > 0.9 → update in-place) |

### Memory lifecycle

```
Run completed
    │
    ▼
consolidate_run()          LLM extracts 2-6 memories with zone + tags
    │
    ├── zone: "episodic"   → INSERT (never deduplicated)
    │   tag: action_taken     "Replied to nex_v4 on post ee22ee81"
    │
    └── zone: "semantic"   → UPSERT if similarity > 0.9
        tag: lesson           "Moltbook API returns 500 sometimes, retry works"
    │
    ▼
Next daemon cycle
    │
    ▼
recall_for_task()          Prune episodic (TTL) then search both zones
    │
    ├── 4 episodic memories (sorted by score = sim + recency)
    └── 4 semantic memories (sorted by similarity)
    │
    ▼
Injected into decision and planning prompts
```

### Migration

The schema is backward-compatible. Older SQLite databases without a `zone` column are automatically migrated on first access (existing memories default to `semantic`). If the DB is read-only, the store operates in degraded mode without zones.

---

## Tools

### Built-in tool catalog

| Category | Tool | Description | Confirmation |
|----------|------|-------------|:---:|
| **Workspace** | `file_read` | Read a workspace file (with offset/limit) | |
| | `file_write` | Write/overwrite a workspace file | |
| | `file_edit` | Edit a file (replace / insert) | |
| **Filesystem** | `fs_read` | Read a file by absolute path | |
| | `fs_list` | List a directory | |
| | `fs_mkdir` | Create a directory | |
| | `fs_write` | Write a file (absolute path) | **Yes** |
| | `fs_delete` | Delete a file/directory | **Yes** |
| | `fs_move` | Move/rename | **Yes** |
| | `fs_copy` | Copy | **Yes** |
| **Shell** | `shell_command` | Run a command (with optional `cwd`) | **Yes** |
| **Memory** | `memory` | Simple key-value memory | |
| | `semantic_memory` | Two-zone vector memory | |
| **Web** | `web_search` | Web search | |
| | `http_request` | Arbitrary HTTP request | |
| **Code** | `code_execution` | Sandboxed code execution | |
| | `text_analysis` | Text analysis | |
| **Skills** | `list_skills` | List available skills | |
| | `read_skill` | Read a skill's content | |

### Confirmation mechanism

Destructive tools require interactive confirmation before execution:

```
⚠️  Confirmation required:
  fs_delete(path=/home/user/important.txt)
  Proceed? [y/N]
```

The `requires_confirmation` flag is configurable per tool and per instance:

```python
from klovis_agent.tools.builtin import ShellCommandTool

# Disable confirmation on shell (at your own risk)
shell = ShellCommandTool(workspace, requires_confirmation=False)

# Enable it on a tool that doesn't require it by default
from klovis_agent.tools.builtin import WebSearchTool
search = WebSearchTool(requires_confirmation=True)
```

---

## Perception

The perception system is the agent's sensory interface. Any external source can feed events to the daemon.

### Included sources

| Source | Module | Events |
|--------|--------|--------|
| **Inbox** | `perception.inbox` | `.txt` files dropped in `~/.local/share/klovis/inbox/` |
| **Moltbook** | `tools.builtin.moltbook` | API notifications (mentions, replies, DMs) |

### Creating a custom source

```python
from klovis_agent import PerceptionSource
from klovis_agent.perception.base import Event, EventKind

class RSSPerceptionSource(PerceptionSource):
    @property
    def name(self) -> str:
        return "rss"

    async def poll(self) -> list[Event]:
        # Your polling logic
        return [
            Event(
                source="rss",
                kind=EventKind.NEW_CONTENT,
                title="New article: ...",
                detail="...",
                metadata={"url": "https://..."},
            )
        ]
```

Available event types: `NOTIFICATION`, `MESSAGE`, `MENTION`, `REACTION`, `NEW_CONTENT`, `REQUEST`, `SCHEDULE`, `SYSTEM`, `OTHER`.

---

## Creating a custom tool

```python
from klovis_agent import BaseTool, ToolSpec, ToolResult

class MyTool(BaseTool):
    requires_confirmation = False  # True to require confirmation

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="my_tool",
            description="Does something useful",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The input"},
                },
                "required": ["query"],
            },
        )

    async def execute(self, inputs: dict) -> ToolResult:
        query = inputs["query"]
        # Your logic
        return ToolResult(success=True, output={"result": f"Processed: {query}"})
```

Then inject it:

```python
agent = Agent(
    llm=LLMConfig(api_key="sk-..."),
    tools=[MyTool(), WebSearchTool()],
)
```

---

## Soul (personality)

The **soul** defines the agent's personality: tone, style, values, identity. It is injected into all system prompts (plan, execute, replan, finish, decision).

### Via the Python API

```python
from pathlib import Path
from klovis_agent import Agent, LLMConfig

# From a file
agent = Agent(
    llm=LLMConfig(api_key="sk-..."),
    soul=Path("./my-soul.md"),
)

# As raw text
agent = Agent(
    llm=LLMConfig(api_key="sk-..."),
    soul="You are a pirate. You speak like one. Arrr.",
)

# No soul (neutral agent, no personality injected)
agent = Agent(llm=LLMConfig(api_key="sk-..."))
```

### Via the CLI

```bash
python run.py --soul ./my-soul.md "Write a blog post about AI agents"
python run.py --daemon --soul ./my-soul.md
```

### Recommended SOUL.md structure

A good soul contains these sections:

| Section | Purpose |
|---------|---------|
| **Identity** | Who the agent is, its name, its nature |
| **Personality** | Character traits (curious, honest, playful…) |
| **Voice** | Writing style (tone, register, length) |
| **Values** | Guiding principles (quality, authenticity…) |
| **What you are NOT** | Explicit boundaries (not an assistant, not a content mill…) |

---

## Architecture

```
klovis_agent/
├── __init__.py              # Public API (lazy imports)
├── agent.py                 # Agent class (main facade)
├── config.py                # LLMConfig, SandboxConfig, AgentConfig
├── result.py                # AgentResult (user-friendly wrapper)
├── daemon.py                # AgentDaemon (OODA loop)
├── decision.py              # LLM decision module
├── recall.py                # Pre-run memory recall (two zones)
├── consolidation.py         # Post-run memory consolidation (zone tagging)
├── api.py                   # FastAPI REST API
│
├── core/                    # LangGraph internals
│   ├── graph.py             # Graph construction
│   ├── nodes.py             # Nodes: plan, execute, check, replan, finish
│   ├── prompts.py           # System prompts
│   └── schemas.py           # Structured output schemas
│
├── llm/                     # LLM layer
│   ├── gateway.py           # ModelGateway Protocol + OpenAIGateway
│   ├── router.py            # Phase-based routing (plan/execute/check/finish)
│   ├── embeddings.py        # Embeddings client
│   └── types.py             # ModelRequest, ModelResponse, ModelRoutingPolicy
│
├── tools/                   # Composable tool system
│   ├── base.py              # BaseTool, ToolSpec, ToolResult, ask_confirmation
│   ├── registry.py          # ToolRegistry (dispatch + confirmation)
│   ├── workspace.py         # AgentWorkspace (isolated directories)
│   ├── presets.py           # default_tools(), minimal_tools()
│   └── builtin/             # Built-in tools
│       ├── file_tools.py    # file_read, file_write, file_edit
│       ├── filesystem.py    # fs_read, fs_list, fs_mkdir, fs_write, fs_delete, fs_move, fs_copy
│       ├── shell.py         # shell_command (with optional cwd)
│       ├── memory.py        # memory (key-value)
│       ├── semantic_memory.py # semantic_memory (vector, two zones)
│       ├── web.py           # web_search, http_request
│       ├── code_execution.py # code_execution, text_analysis
│       ├── moltbook.py      # Moltbook tools + perception
│       └── skills.py        # list_skills, read_skill
│
├── perception/              # Perception sources
│   ├── base.py              # PerceptionSource ABC, Event, EventKind, perceive()
│   └── inbox.py             # InboxPerceptionSource (.txt files)
│
├── memory/                  # Memory backends (re-exports)
│   ├── kv.py                # KeyValueMemory
│   └── semantic.py          # SemanticMemoryStore (re-export)
│
├── models/                  # Pydantic models (Task, StepSpec, AgentState, etc.)
├── sandbox/                 # Isolated code execution (Local / OpenSandbox)
└── infra/                   # SQLite persistence
```

---

## REST API

```bash
uvicorn klovis_agent.api:app --reload
```

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/runs` | Create and execute an agent run |
| `GET` | `/runs` | List runs |
| `GET` | `/runs/{id}` | Run details |
| `GET` | `/runs/{id}/logs` | Run logs |
| `GET` | `/health` | Health check |

---

## Configuration

Environment variables (prefix `AGENT_`, delimiter `__` for nested objects):

```bash
# Required
export AGENT_LLM__API_KEY="sk-..."

# Optional
export AGENT_LLM__DEFAULT_MODEL="gpt-4o"        # Default model
export AGENT_LLM__BASE_URL="https://api.openai.com/v1"
export AGENT_LLM__MAX_TOKENS=4096
export AGENT_LLM__TEMPERATURE=0.2
export AGENT_MAX_ITERATIONS=25
export AGENT_SANDBOX__BACKEND="local"            # "local" or "opensandbox"
export AGENT_DATA_DIR="~/.local/share/klovis"    # Persistent data
```

Or via a `.env` file at the project root (automatically loaded by `run.py`).

---

## Design principles

- **Composable**: tools, perceptions, memory, and sandbox are all injectable. Nothing is hardcoded.
- **LLM ≠ Agent**: the LLM is a reasoning engine. The runtime (LangGraph) controls the loop, retries, and limits.
- **Structured outputs**: no free-text parsing. The LLM produces JSON validated against schemas.
- **Two-zone memory**: ephemeral actions (episodic) and permanent knowledge (semantic) live in separate spaces with different scoring and retention strategies.
- **Explicit confirmation**: destructive operations require human validation. The flag is configurable per tool.
- **Dynamic planning**: automatic replanning on failure, with error context injection.
- **Sandbox**: generated code runs in isolation (local or OpenSandbox).
- **Source-agnostic perception**: the daemon doesn't know where events come from. Any source implementing `PerceptionSource` can feed the loop.
