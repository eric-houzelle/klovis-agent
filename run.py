"""CLI entry point for running a klovis-agent.

This file demonstrates how to use klovis-agent as a library.
Perception sources are instantiated here and injected into the Agent,
giving the caller full control over what the agent can observe.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from klovis_agent import Agent, AgentConfig, InboxPerceptionSource

# ---------------------------------------------------------------------------
# GitHub repos to watch in daemon mode.
# Each entry creates a GitHubPerceptionSource.  Add, remove, or edit entries
# as needed — the agent will observe all of them during its OODA loop.
# ---------------------------------------------------------------------------
_GITHUB_REPOS: list[dict] = [
    {
        "owner": "eric-houzelle",
        "repo": "klovis-agent",
        "issue_labels": ["agent"],
    },
    # {"owner": "eric-houzelle", "repo": "other-project"},
]

_HELP = """\
Usage: python run.py [OPTIONS] [GOAL...]

Options:
  -v, --verbose         Verbose output (show structlog + raw JSON)
  --daemon              Run in daemon mode (continuous OODA loop)
  --interval MINUTES    Daemon check interval (default: 30)
  --cycles N            Max daemon cycles, 0 = infinite (default: 0)
  --data-dir PATH       Persistent data directory (default: ~/.local/share/klovis)
  --soul PATH           Path to a SOUL.md file defining the agent's personality
  --ephemeral           Use a temporary directory (nothing persists)
  -h, --help            Show this help
"""


def _parse_args(argv: list[str]) -> dict:
    opts: dict = {
        "verbose": False,
        "daemon": False,
        "interval": 30.0,
        "cycles": 0,
        "data_dir": "",
        "soul": "",
        "ephemeral": False,
        "goal_parts": [],
    }
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-v", "--verbose"):
            opts["verbose"] = True
        elif arg == "--daemon":
            opts["daemon"] = True
        elif arg == "--interval" and i + 1 < len(argv):
            i += 1
            opts["interval"] = float(argv[i])
        elif arg == "--cycles" and i + 1 < len(argv):
            i += 1
            opts["cycles"] = int(argv[i])
        elif arg == "--data-dir" and i + 1 < len(argv):
            i += 1
            opts["data_dir"] = argv[i]
        elif arg == "--soul" and i + 1 < len(argv):
            i += 1
            opts["soul"] = argv[i]
        elif arg == "--ephemeral":
            opts["ephemeral"] = True
        elif arg in ("-h", "--help"):
            print(_HELP)
            sys.exit(0)
        else:
            opts["goal_parts"].append(arg)
        i += 1
    return opts


async def _run_single(
    config: AgentConfig,
    goal: str,
    verbose: bool,
    data_dir: str = "",
    soul: str = "",
    ephemeral: bool = False,
) -> None:
    agent = Agent(
        llm=config.llm,
        embedding=config.embedding,
        sandbox=config.sandbox,
        max_iterations=config.max_iterations,
        verbose=verbose,
        data_dir=data_dir or config.data_dir or None,
        soul=Path(soul) if soul else None,
        ephemeral=ephemeral,
    )

    agent.console.banner(config.llm.default_model, config.llm.base_url)
    agent.console.banner_detail(
        config.llm.max_tokens,
        config.llm.temperature,
        config.max_iterations,
        config.sandbox.backend,
    )

    result = await agent.run(
        goal, success_criteria=["The answer is correct and well explained"],
    )

    agent.console.run_result(
        result.status,
        result.iteration_count,
        len(result.steps),
        result.summary or "",
    )


def _build_perception_sources() -> list:
    """Instantiate the perception sources for daemon mode.

    This is where you add or remove sources depending on your setup.
    """
    sources = [InboxPerceptionSource()]

    try:
        from klovis_agent.tools.builtin.moltbook import (
            MoltbookPerceptionSource,
            load_credentials,
        )

        creds = load_credentials()
        if creds.get("api_key"):
            sources.append(MoltbookPerceptionSource())
    except Exception:
        pass

    try:
        from klovis_agent.tools.builtin.github import (
            GitHubPerceptionSource,
            _GitHubAuth,
            _load_github_config,
        )

        gh_config = _load_github_config()
        if gh_config:
            auth = _GitHubAuth(gh_config)
            for repo_spec in _GITHUB_REPOS:
                sources.append(
                    GitHubPerceptionSource(
                        owner=repo_spec["owner"],
                        repo=repo_spec["repo"],
                        auth=auth,
                        issue_labels=repo_spec.get("issue_labels", []),
                    )
                )
    except Exception:
        pass

    return sources


async def _run_daemon(
    config: AgentConfig,
    interval: float,
    max_cycles: int,
    verbose: bool,
    data_dir: str = "",
    soul: str = "",
) -> None:
    sources = _build_perception_sources()

    agent = Agent(
        llm=config.llm,
        embedding=config.embedding,
        sandbox=config.sandbox,
        max_iterations=config.max_iterations,
        verbose=verbose,
        data_dir=data_dir or config.data_dir or None,
        soul=Path(soul) if soul else None,
        perceptions=sources,
    )

    agent.console.banner(config.llm.default_model, config.llm.base_url)
    agent.console.banner_detail(
        config.llm.max_tokens,
        config.llm.temperature,
        config.max_iterations,
        config.sandbox.backend,
    )

    daemon = agent.as_daemon(
        interval_minutes=interval,
        max_cycles=max_cycles,
    )
    await daemon.run()


async def main() -> None:
    load_dotenv()

    opts = _parse_args(sys.argv[1:])
    verbose = opts["verbose"]

    config = AgentConfig()

    if not config.llm.api_key:
        print("Error: AGENT_LLM__API_KEY not set in .env")
        sys.exit(1)

    if opts["daemon"]:
        await _run_daemon(
            config, opts["interval"], opts["cycles"], verbose,
            data_dir=opts["data_dir"],
            soul=opts["soul"],
        )
    else:
        goal = (
            " ".join(opts["goal_parts"])
            if opts["goal_parts"]
            else "Explain what 2+2 equals and why."
        )
        await _run_single(
            config, goal, verbose,
            data_dir=opts["data_dir"],
            soul=opts["soul"],
            ephemeral=opts["ephemeral"],
        )


if __name__ == "__main__":
    asyncio.run(main())
