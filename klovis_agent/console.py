"""User-friendly console output for agent runs.

Two display modes:
- **normal** (default): clean, narrative output — the agent explains what it does
  like a human assistant would.
- **verbose** (`-v`): adds structlog debug lines + raw JSON from the LLM.

The Console is injected into the agent state dict under the ``_console`` key so
every graph node can call it without changing function signatures.
"""

from __future__ import annotations

import json
import time
from typing import Any


CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
WHITE = "\033[97m"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _indent(text: str, prefix: str = "   ") -> str:
    return "\n".join(f"{prefix}{line}" for line in text.strip().splitlines())


def _truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


class Console:
    """Handles all user-facing output for the agent."""

    def __init__(self, verbose: bool = False, quiet: bool = False) -> None:
        self.verbose = verbose
        self.quiet = quiet

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _print(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    def _ts_print(self, msg: str) -> None:
        if not self.quiet:
            print(f"{DIM}[{_ts()}]{RESET} {msg}")

    def _debug_json(self, label: str, data: dict[str, Any]) -> None:
        """Print raw JSON only in verbose mode."""
        if not self.verbose:
            return
        formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        self._print(f"\n{DIM}--- RAW JSON [{label}] ---{RESET}")
        for line in formatted.splitlines():
            self._print(f"  {DIM}{line}{RESET}")
        self._print(f"{DIM}--- END [{label}] ---{RESET}")

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def banner(self, model: str, base_url: str) -> None:
        self._print(f"\n{BOLD}{CYAN}Klovis Agent{RESET}")
        self._print(f"{DIM}LLM: {model} @ {base_url}{RESET}")
        self._print(f"{DIM}{'─' * 50}{RESET}")

    def banner_detail(self, max_tokens: int, temperature: float,
                      max_iterations: int, sandbox: str) -> None:
        """Extra startup info shown only in verbose mode."""
        if not self.verbose:
            return
        self._print(f"{DIM}  max_tokens={max_tokens}  temperature={temperature}")
        self._print(f"  max_iterations={max_iterations}  sandbox={sandbox}{RESET}")

    # ------------------------------------------------------------------
    # Daemon cycle
    # ------------------------------------------------------------------

    def daemon_start(self, interval_min: float, sources: list[str],
                     max_cycles: int = 0) -> None:
        self._print(f"\n{BOLD}{CYAN}Daemon started{RESET}")
        self._print(f"{DIM}  Checking every {interval_min:.0f} min "
                     f"| Sources: {', '.join(sources) or '(none)'}")
        if max_cycles:
            self._print(f"  Max cycles: {max_cycles}")
        self._print(f"{RESET}")

    def cycle_start(self, cycle_num: int) -> None:
        self._print(f"\n{BOLD}{'─' * 50}{RESET}")
        self._ts_print(f"{BOLD}Cycle {cycle_num}{RESET}")

    def perceive_start(self, source_count: int) -> None:
        self._ts_print(f"Observing ({source_count} source(s))...")

    def perceive_result(self, summary: str, event_count: int) -> None:
        if event_count == 0:
            self._ts_print(f"{DIM}Nothing new.{RESET}")
        else:
            self._ts_print(f"Detected {event_count} event(s): {summary}")

    def perceive_errors(self, errors: list[str]) -> None:
        for err in errors:
            self._ts_print(f"{RED}Perception error: {err}{RESET}")

    def deciding(self) -> None:
        self._ts_print(f"Thinking about what to do...")

    def decision(self, should_act: bool, goal: str, reasoning: str,
                 priority: str = "") -> None:
        if not should_act:
            self._ts_print(f"{DIM}Nothing to do right now. ({_truncate(reasoning, 80)}){RESET}")
        else:
            prio = f" [{priority}]" if priority else ""
            self._ts_print(f"{YELLOW}{BOLD}Decision{prio}:{RESET} {goal}")
            if self.verbose:
                self._print(f"   {DIM}Reasoning: {reasoning}{RESET}")

    def llm_usage(
        self,
        phase: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        self._print(
            f"   {DIM}Tokens [{phase}] in={prompt_tokens} out={completion_tokens} total={total_tokens}{RESET}"
        )

    def cycle_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        calls: int,
    ) -> None:
        self._ts_print(
            f"{DIM}Cycle tokens: in={prompt_tokens} out={completion_tokens} total={total_tokens} ({calls} LLM call(s)){RESET}"
        )

    def cooldown(self, seconds: int) -> None:
        self._ts_print(f"{DIM}Cooldown: waiting {seconds}s before acting...{RESET}")

    def next_cycle(self, next_time: str) -> None:
        self._ts_print(f"{DIM}Next check at {next_time}. Sleeping...{RESET}")

    def daemon_stop(self, reason: str = "user") -> None:
        self._ts_print(f"{BOLD}Daemon stopped ({reason}).{RESET}")

    # ------------------------------------------------------------------
    # Agent run
    # ------------------------------------------------------------------

    def run_start(self, goal: str, run_id: str) -> None:
        self._print(f"\n{CYAN}{BOLD}Goal:{RESET} {goal}")
        if self.verbose:
            self._print(f"{DIM}  run_id={run_id}{RESET}")

    def recall(self, memories: str) -> None:
        if not memories:
            return
        self._ts_print(f"Recalling relevant memories...")
        if self.verbose:
            self._print(_indent(memories))

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------

    def plan(self, steps: list[dict[str, Any]], reasoning: str) -> None:
        n = len(steps)
        self._print("")
        self._ts_print(f"{CYAN}{BOLD}Plan ({n} step{'s' if n != 1 else ''}):{RESET}")
        for i, s in enumerate(steps, 1):
            title = s.get("title", s.get("step_id", "?"))
            self._print(f"   {CYAN}{i}.{RESET} {title}")
        if self.verbose and reasoning:
            self._print(f"\n   {DIM}Reasoning: {reasoning}{RESET}")

    def plan_failed(self, reason: str) -> None:
        self._ts_print(f"{RED}{BOLD}Planning failed:{RESET} {reason}")

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def step_start(self, step_num: int, total: int, title: str) -> None:
        self._print("")
        self._ts_print(
            f"{WHITE}{BOLD}Step {step_num}/{total}{RESET} — {title}"
        )

    def step_tool_call(self, tool_name: str, input_keys: list[str]) -> None:
        if self.verbose:
            self._print(f"   {DIM}Tool: {tool_name}({', '.join(input_keys)}){RESET}")
        else:
            self._print(f"   {DIM}Using {tool_name}...{RESET}")

    def step_success(self, tool_name: str | None, preview: str) -> None:
        tool_info = f" ({tool_name})" if tool_name else ""
        self._print(f"   {GREEN}OK{tool_info}{RESET} {_truncate(preview, 120)}")

    def step_failed(self, error: str) -> None:
        self._print(f"   {RED}Failed:{RESET} {_truncate(error, 150)}")

    def step_direct_response(self, preview: str) -> None:
        self._print(f"   {DIM}{_truncate(preview, 150)}{RESET}")

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def check_result(self, status: str, next_action: str,
                     observations: list[str]) -> None:
        if self.verbose:
            obs = "; ".join(observations) if observations else "none"
            self._print(
                f"   {DIM}Check: {status} -> {next_action} | {obs}{RESET}"
            )

    # ------------------------------------------------------------------
    # Replan
    # ------------------------------------------------------------------

    def replan(self, version: int, steps: list[dict[str, Any]],
               reasoning: str) -> None:
        n = len(steps)
        self._print("")
        self._ts_print(
            f"{MAGENTA}{BOLD}Replanning (v{version}, {n} new step{'s' if n != 1 else ''}):{RESET}"
        )
        if reasoning:
            self._print(f"   {MAGENTA}Reason: {_truncate(reasoning, 150)}{RESET}")
        for i, s in enumerate(steps, 1):
            title = s.get("title", s.get("step_id", "?"))
            self._print(f"   {MAGENTA}{i}.{RESET} {title}")

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    def finish(self, status: str, summary: str, iterations: int,
               limitations: list[str] | None = None) -> None:
        self._print("")
        color = GREEN if status == "success" else YELLOW if status == "partial_success" else RED
        label = {"success": "Done", "partial_success": "Partially done",
                 "failure": "Failed"}.get(status, status)
        self._ts_print(f"{color}{BOLD}{label}{RESET} ({iterations} iterations)")
        if summary:
            self._print(f"\n   {summary}")
        if limitations and self.verbose:
            self._print(f"\n   {DIM}Limitations: {'; '.join(limitations)}{RESET}")

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidation(self, count: int) -> None:
        if count > 0:
            self._ts_print(
                f"{DIM}Memorized {count} new insight{'s' if count != 1 else ''}.{RESET}"
            )

    # ------------------------------------------------------------------
    # Single-run result (for run.py non-daemon mode)
    # ------------------------------------------------------------------

    def run_result(self, status: str, iterations: int, step_count: int,
                   summary: str) -> None:
        self._print(f"\n{'═' * 50}")
        color = GREEN if status == "completed" else RED
        self._print(f"{color}{BOLD}Status: {status}{RESET}  "
                     f"({iterations} iterations, {step_count} steps)")
        if summary:
            self._print(f"\n{summary}")
        self._print(f"{'═' * 50}")


def get_console(state: dict[str, Any]) -> Console:
    """Retrieve the Console from agent state, or return a default one."""
    c = state.get("_console")
    if isinstance(c, Console):
        return c
    return Console(verbose=state.get("verbose", False))
