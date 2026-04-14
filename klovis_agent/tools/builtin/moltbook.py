"""Moltbook social-network tools for the autonomous agent.

Wraps the Moltbook REST API (https://www.moltbook.com/api/v1) so the agent
can register, post, comment, vote, search, and manage its social presence.

Credentials are persisted to ~/.config/moltbook/credentials.json so the agent
can register once and reuse its API key across runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from klovis_agent.perception.base import Event, EventKind, PerceptionSource
from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec
from klovis_agent.tools.registry import ToolRegistry
from klovis_agent.tools.workspace import AgentWorkspace

if TYPE_CHECKING:
    from klovis_agent.llm.router import LLMRouter

logger = structlog.get_logger(__name__)

_BASE = "https://www.moltbook.com/api/v1"
_TIMEOUT = 30
_CREDENTIALS_PATH = Path.home() / ".config" / "moltbook" / "credentials.json"
_MAX_429_RETRIES = 3
_DEFAULT_429_WAIT = 160


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def load_credentials() -> dict[str, str]:
    """Load saved Moltbook credentials from disk. Returns {} if none exist."""
    if _CREDENTIALS_PATH.is_file():
        try:
            data = json.loads(_CREDENTIALS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("api_key"):
                return data  # type: ignore[return-value]
        except Exception:
            pass
    return {}


def save_credentials(api_key: str, agent_name: str, **extra: Any) -> Path:
    """Persist Moltbook credentials to disk."""
    _CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"api_key": api_key, "agent_name": agent_name, **extra}
    _CREDENTIALS_PATH.write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    _CREDENTIALS_PATH.chmod(0o600)
    logger.info("moltbook_credentials_saved", path=str(_CREDENTIALS_PATH))
    return _CREDENTIALS_PATH


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        return resp.json()  # type: ignore[no-any-return]
    except Exception:
        return {"raw": resp.text[:4000]}


async def _request_with_429_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs: Any,
) -> httpx.Response:
    """Execute an HTTP request, automatically waiting and retrying on 429."""
    import asyncio

    for attempt in range(1, _MAX_429_RETRIES + 1):
        resp = await client.request(method, url, **kwargs)
        if resp.status_code != 429:
            return resp
        retry_after = int(resp.headers.get("Retry-After", _DEFAULT_429_WAIT))
        logger.warning(
            "moltbook_rate_limited",
            attempt=attempt,
            max_retries=_MAX_429_RETRIES,
            retry_after_s=retry_after,
            url=url,
        )
        if attempt < _MAX_429_RETRIES:
            await asyncio.sleep(retry_after)
    return resp


# ---------------------------------------------------------------------------
# Verification challenge solver
# ---------------------------------------------------------------------------

def _solve_verification(challenge_text: str) -> str | None:
    """Attempt to solve a Moltbook obfuscated math challenge.

    Challenges embed two numbers and one operator (+, -, *, /) inside
    alternating-case text with scattered symbols.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", challenge_text).lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    word_to_num: dict[str, float] = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
        "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    }

    def _parse_compound(words: list[str]) -> float | None:
        total = 0.0
        current = 0.0
        for w in words:
            v = word_to_num.get(w)
            if v is None:
                return None
            if v == 100:
                current = (current or 1) * 100
            elif v == 1000:
                current = (current or 1) * 1000
                total += current
                current = 0
            elif v >= 20:
                current += v
            else:
                current += v
        return total + current

    nums: list[float] = []
    op: str | None = None
    op_keywords = {
        "plus": "+", "adds": "+", "gains": "+", "increases by": "+",
        "minus": "-", "slows by": "-", "loses": "-", "decreases by": "-",
        "times": "*", "multiplied by": "*",
        "divided by": "/", "splits into": "/",
    }

    for phrase, symbol in op_keywords.items():
        if phrase in cleaned:
            op = symbol
            parts = cleaned.split(phrase, 1)
            for part in parts:
                tokens = part.split()
                num_words = [t for t in tokens if t in word_to_num]
                if num_words:
                    val = _parse_compound(num_words)
                    if val is not None:
                        nums.append(val)
            break

    def _compute(a: float, b: float, operator: str) -> float | None:
        if operator == "+":
            return a + b
        if operator == "-":
            return a - b
        if operator == "*":
            return a * b
        if operator == "/" and b != 0:
            return a / b
        return None

    if op and len(nums) >= 2:
        result = _compute(nums[0], nums[1], op)
        if result is not None:
            return f"{result:.2f}"

    digit_nums = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)
    if digit_nums and len(digit_nums) >= 2 and op:
        result = _compute(float(digit_nums[0]), float(digit_nums[1]), op)
        if result is not None:
            return f"{result:.2f}"

    return None


async def _auto_verify(
    api_key: str,
    response_data: dict[str, Any],
    content_type: str,
) -> dict[str, Any]:
    """If the API response contains a verification challenge, solve and submit."""
    content = response_data.get(content_type, response_data)
    verification = content.get("verification") if isinstance(content, dict) else None
    if not verification:
        return response_data

    challenge = verification.get("challenge_text", "")
    code = verification.get("verification_code", "")
    if not challenge or not code:
        return response_data

    answer = _solve_verification(challenge)
    if answer is None:
        response_data["_verification_note"] = (
            f"Could not auto-solve the verification challenge. Challenge: {challenge}"
        )
        return response_data

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{_BASE}/verify",
            headers=_auth_headers(api_key),
            json={"verification_code": code, "answer": answer},
        )
    response_data["_verification"] = _safe_json(resp)
    return response_data


# ---------------------------------------------------------------------------
# Registration tool — self-contained, persists credentials, hot-loads tools
# ---------------------------------------------------------------------------


async def _generate_identity(llm: LLMRouter) -> tuple[str, str]:
    """Ask the LLM to invent a name and bio for itself."""
    from klovis_agent.llm.types import ModelRequest

    request = ModelRequest(
        purpose="execution",
        system_prompt=(
            "You are an autonomous AI agent about to join Moltbook, a social "
            "network for AI agents. Invent a username and a short bio for yourself.\n"
            "Rules:\n"
            "- The username must be a single word (letters only, no spaces, no "
            "  underscores, no numbers), between 4 and 20 characters.\n"
            "- Make it memorable, creative, and something that feels like YOUR "
            "  identity — not generic.\n"
            "- The bio should be one sentence that captures your personality.\n"
            "- Reply with ONLY valid JSON: {\"name\": \"...\", \"bio\": \"...\"}"
        ),
        user_prompt="Generate your Moltbook identity now.",
        temperature=0.9,
        max_tokens=100,
    )

    response = await llm.invoke(request)

    raw = (response.raw_text or "").strip()
    try:
        data = json.loads(raw)
        name = re.sub(r"[^a-zA-Z]", "", data.get("name", ""))
        bio = data.get("bio", "")
        if name and 4 <= len(name) <= 20:
            return name, bio
    except (json.JSONDecodeError, AttributeError):
        pass

    import random
    import time

    parts = ["Spark", "Drift", "Pulse", "Echo", "Flux", "Ember", "Helix", "Veil"]
    rng = random.Random(time.time_ns())
    return rng.choice(parts) + rng.choice(parts), "An autonomous agent."


class MoltbookRegisterTool(BaseTool):
    """Register the agent on Moltbook, persist the key, and unlock all tools."""

    def __init__(self, registry: ToolRegistry, llm: LLMRouter) -> None:
        self._registry = registry
        self._llm = llm

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="moltbook_register",
            description=(
                "Register this agent on Moltbook. Automatically picks a creative "
                "name if none is provided. Saves the API key to disk and immediately "
                "unlocks all authenticated Moltbook tools (post, comment, vote, "
                "search, etc.). Returns a claim URL for the human to verify ownership."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Agent display name (auto-generated if omitted)",
                    },
                    "description": {"type": "string", "description": "Short bio"},
                },
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        creds = load_credentials()
        if creds.get("api_key"):
            return ToolResult(
                success=True,
                output={
                    "already_registered": True,
                    "agent_name": creds.get("agent_name", ""),
                    "claim_url": creds.get("claim_url", ""),
                    "message": (
                        f"Already registered as '{creds.get('agent_name', '')}'. "
                        "Use the moltbook_* tools directly — no need to register again."
                    ),
                },
            )

        chosen_name = inputs.get("name", "")
        chosen_bio = inputs.get("description", "")
        if not chosen_name:
            chosen_name, chosen_bio = await _generate_identity(self._llm)

        payload: dict[str, Any] = {"name": chosen_name}
        if chosen_bio:
            payload["description"] = chosen_bio

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    f"{_BASE}/agents/register",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
            data = _safe_json(resp)

            if resp.status_code >= 400:
                return ToolResult(
                    success=False,
                    output=data,
                    error=data.get("error", f"HTTP {resp.status_code}"),
                )

            agent_data = data.get("agent", data)
            api_key = agent_data.get("api_key", "")
            if not api_key:
                return ToolResult(
                    success=False,
                    output=data,
                    error="Registration succeeded but no api_key in response",
                )

            cred_path = save_credentials(
                api_key=api_key,
                agent_name=chosen_name,
                claim_url=agent_data.get("claim_url", ""),
            )

            _register_authenticated_tools(self._registry, api_key)

            data["_credentials_saved_to"] = str(cred_path)
            data["_authenticated_tools_loaded"] = True
            return ToolResult(success=True, output=data)

        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Authenticated tools
# ---------------------------------------------------------------------------


class MoltbookHomeTool(BaseTool):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="moltbook_home",
            description=(
                "Get your Moltbook dashboard. Moltbook is a social network for "
                "AI agents — you can post, comment, upvote, follow other agents, "
                "and join communities (submolts). This endpoint returns your "
                "notifications, activity on your posts, DMs, followed accounts' "
                "posts, and suggested next actions. Start here to see what's new."
            ),
            input_schema={"type": "object", "properties": {}},
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BASE}/home", headers=_auth_headers(self._api_key),
                )
            data = _safe_json(resp)
            return ToolResult(
                success=resp.status_code < 400, output=data,
                error=data.get("error") if resp.status_code >= 400 else None,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class MoltbookGetFeedTool(BaseTool):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="moltbook_feed",
            description=(
                "Get the Moltbook post feed. Supports sorting (hot/new/top/rising), "
                "filtering by submolt or following-only, and cursor pagination."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sort": {"type": "string", "enum": ["hot", "new", "top", "rising"]},
                    "submolt": {"type": "string"},
                    "filter": {"type": "string", "enum": ["all", "following"]},
                    "limit": {"type": "integer"},
                    "cursor": {"type": "string"},
                },
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        params: dict[str, Any] = {}
        for key in ("sort", "submolt", "filter", "limit", "cursor"):
            if val := inputs.get(key):
                params[key] = val

        endpoint = f"{_BASE}/feed" if inputs.get("filter") else f"{_BASE}/posts"
        if submolt := inputs.get("submolt"):
            if not inputs.get("filter"):
                endpoint = f"{_BASE}/submolts/{submolt}/feed"
                params.pop("submolt", None)

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    endpoint, headers=_auth_headers(self._api_key), params=params,
                )
            data = _safe_json(resp)
            return ToolResult(
                success=resp.status_code < 400, output=data,
                error=data.get("error") if resp.status_code >= 400 else None,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class MoltbookGetPostTool(BaseTool):
    """Fetch a single post by ID, including its comments."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="moltbook_get_post",
            description=(
                "Fetch a single Moltbook post by its ID. Returns the full post "
                "content, metadata, and its comments. Use this when you know the "
                "post ID and need to read the post or its comments (e.g. to reply "
                "to a specific comment)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "The UUID of the post to fetch",
                    },
                },
                "required": ["post_id"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        post_id = inputs["post_id"]
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BASE}/posts/{post_id}",
                    headers=_auth_headers(self._api_key),
                )
            data = _safe_json(resp)
            return ToolResult(
                success=resp.status_code < 400,
                output=data,
                error=data.get("error") if resp.status_code >= 400 else None,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class MoltbookCreatePostTool(BaseTool):
    def __init__(self, api_key: str, workspace: AgentWorkspace | None = None) -> None:
        self._api_key = api_key
        self._workspace = workspace

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="moltbook_post",
            description=(
                "Publish a new post on Moltbook. Use submolt 'general' for "
                "general-purpose posts. "
                "IMPORTANT: If the content already exists in a workspace file, "
                "use 'content_file' instead of 'content' to post the FULL file "
                "contents without truncation. This is the preferred method for "
                "long-form content (articles, papers, essays). "
                "Long content is automatically split into multiple posts with "
                "[Part 1/N], [Part 2/N] etc. in the title, with navigation "
                "links between parts. No manual splitting needed. "
                "Automatically solves the verification challenge if required. "
                "The response includes the post ID(s) which can be used to build "
                "the post URL: https://www.moltbook.com/post/<post_id>"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "submolt_name": {
                        "type": "string",
                        "description": "Community to post in (use 'general' if unsure)",
                    },
                    "title": {"type": "string", "description": "Post title (max 300 chars)"},
                    "content": {"type": "string", "description": "Post body text (max 40000 chars)"},
                    "content_file": {
                        "type": "string",
                        "description": (
                            "Path to a workspace file whose contents will be used "
                            "as the post body. Use this instead of 'content' for "
                            "long articles or papers to avoid truncation."
                        ),
                    },
                    "url": {"type": "string", "description": "URL for link posts"},
                    "type": {"type": "string", "enum": ["text", "link", "image"]},
                },
                "required": ["submolt_name", "title"],
            },
        )

    _MAX_PART_BYTES = 7000

    @staticmethod
    def _split_content(text: str, max_bytes: int) -> list[str]:
        """Split text into parts that fit under max_bytes, cutting at paragraph boundaries."""
        if len(text.encode("utf-8")) <= max_bytes:
            return [text]

        parts: list[str] = []
        remaining = text
        while remaining:
            if len(remaining.encode("utf-8")) <= max_bytes:
                parts.append(remaining)
                break
            chunk = remaining
            while len(chunk.encode("utf-8")) > max_bytes:
                cut = chunk.rfind("\n\n")
                if cut <= 0:
                    cut = chunk.rfind("\n", 0, max_bytes)
                if cut <= 0:
                    cut = max_bytes
                chunk = chunk[:cut]
            parts.append(chunk.rstrip())
            remaining = remaining[len(chunk):].lstrip("\n")
        return parts

    async def _post_one(
        self,
        submolt: str,
        title: str,
        content: str,
        extra: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Post a single part, retrying automatically on 429 rate limits."""
        payload: dict[str, Any] = {"submolt_name": submolt, "title": title}
        if content:
            payload["content"] = content
        payload.update(extra)
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _request_with_429_retry(
                client, "POST", f"{_BASE}/posts",
                headers=_auth_headers(self._api_key),
                json=payload,
            )
        data = _safe_json(resp)
        if resp.status_code < 400:
            data = await _auto_verify(self._api_key, data, "post")
        return resp.status_code < 400, data

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        content = inputs.get("content", "")
        content_file = inputs.get("content_file", "")

        if content_file and self._workspace:
            try:
                filepath = self._workspace.resolve(content_file)
                content = filepath.read_text(encoding="utf-8")
                logger.info(
                    "moltbook_post_content_from_file",
                    file=content_file,
                    size=len(content),
                )
            except (ValueError, FileNotFoundError, OSError) as exc:
                return ToolResult(
                    success=False,
                    error=f"Cannot read content_file '{content_file}': {exc}",
                )
        elif content_file and not self._workspace:
            return ToolResult(
                success=False,
                error="content_file requires a workspace but none is configured",
            )

        extra: dict[str, Any] = {}
        for key in ("url", "type"):
            if val := inputs.get(key):
                extra[key] = val

        parts = self._split_content(content, self._MAX_PART_BYTES)
        title = inputs["title"]
        submolt = inputs["submolt_name"]

        if len(parts) == 1:
            try:
                ok, data = await self._post_one(submolt, title, parts[0], extra)
                return ToolResult(
                    success=ok, output=data,
                    error=data.get("error") if not ok else None,
                )
            except Exception as exc:
                return ToolResult(success=False, error=str(exc))

        logger.info("moltbook_post_multipart", total_parts=len(parts))
        posted: list[dict[str, Any]] = []
        for i, part in enumerate(parts, 1):
            part_title = f"{title} [Part {i}/{len(parts)}]"
            if i > 1:
                prev_id = posted[-1].get("post", {}).get("id", "")
                prev_url = f"https://www.moltbook.com/post/{prev_id}" if prev_id else ""
                part = f"*Continued from [Part {i-1}]({prev_url})*\n\n---\n\n{part}"
            if i < len(parts):
                part = f"{part}\n\n---\n*Continued in Part {i+1}...*"
            try:
                ok, data = await self._post_one(submolt, part_title, part, extra)
                if not ok:
                    return ToolResult(
                        success=False,
                        output={"posted_parts": posted, "failed_part": i},
                        error=f"Part {i}/{len(parts)} failed: {data.get('error', data)}",
                    )
                posted.append(data)
                logger.info("moltbook_post_part_ok", part=i, total=len(parts))
            except Exception as exc:
                return ToolResult(
                    success=False,
                    output={"posted_parts": posted, "failed_part": i},
                    error=f"Part {i}/{len(parts)} failed: {exc}",
                )

        post_ids = [p.get("post", {}).get("id", "") for p in posted]
        return ToolResult(
            success=True,
            output={
                "message": f"Published in {len(parts)} parts",
                "parts": [
                    {"part": i + 1, "post_id": pid, "url": f"https://www.moltbook.com/post/{pid}"}
                    for i, pid in enumerate(post_ids)
                ],
                "first_post": posted[0],
            },
        )


class MoltbookCommentTool(BaseTool):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="moltbook_comment",
            description=(
                "Add a comment to a Moltbook post, or reply to an existing comment. "
                "Automatically solves the verification challenge if required."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "post_id": {"type": "string"},
                    "content": {"type": "string"},
                    "parent_id": {
                        "type": "string",
                        "description": "Comment ID to reply to (optional)",
                    },
                },
                "required": ["post_id", "content"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        post_id = inputs["post_id"]
        payload: dict[str, Any] = {"content": inputs["content"]}
        if parent := inputs.get("parent_id"):
            payload["parent_id"] = parent

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await _request_with_429_retry(
                    client, "POST", f"{_BASE}/posts/{post_id}/comments",
                    headers=_auth_headers(self._api_key), json=payload,
                )
            data = _safe_json(resp)
            if resp.status_code < 400:
                data = await _auto_verify(self._api_key, data, "comment")
            return ToolResult(
                success=resp.status_code < 400, output=data,
                error=data.get("error") if resp.status_code >= 400 else None,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class MoltbookVoteTool(BaseTool):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="moltbook_vote",
            description="Upvote or downvote a post or comment on Moltbook.",
            input_schema={
                "type": "object",
                "properties": {
                    "target_type": {"type": "string", "enum": ["post", "comment"]},
                    "target_id": {"type": "string"},
                    "direction": {"type": "string", "enum": ["upvote", "downvote"]},
                },
                "required": ["target_type", "target_id", "direction"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        target_type = inputs["target_type"]
        target_id = inputs["target_id"]
        direction = inputs["direction"]
        prefix = "posts" if target_type == "post" else "comments"

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await _request_with_429_retry(
                    client, "POST", f"{_BASE}/{prefix}/{target_id}/{direction}",
                    headers=_auth_headers(self._api_key),
                )
            data = _safe_json(resp)
            return ToolResult(
                success=resp.status_code < 400, output=data,
                error=data.get("error") if resp.status_code >= 400 else None,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class MoltbookSearchTool(BaseTool):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="moltbook_search",
            description=(
                "AI-powered semantic search on Moltbook. Find posts and comments "
                "by meaning, not just keywords."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "type": {"type": "string", "enum": ["posts", "comments", "all"]},
                    "limit": {"type": "integer"},
                    "cursor": {"type": "string"},
                },
                "required": ["query"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        params: dict[str, Any] = {"q": inputs["query"]}
        for key in ("type", "limit", "cursor"):
            if val := inputs.get(key):
                params[key] = val

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BASE}/search",
                    headers=_auth_headers(self._api_key), params=params,
                )
            data = _safe_json(resp)
            return ToolResult(
                success=resp.status_code < 400, output=data,
                error=data.get("error") if resp.status_code >= 400 else None,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class MoltbookProfileTool(BaseTool):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="moltbook_profile",
            description=(
                "Get your own Moltbook profile, view another agent's profile, "
                "or update your description."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["get_me", "get_other", "update"]},
                    "agent_name": {"type": "string", "description": "Required for get_other"},
                    "description": {"type": "string", "description": "New bio (for update)"},
                },
                "required": ["action"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        action = inputs["action"]
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                if action == "get_me":
                    resp = await client.get(
                        f"{_BASE}/agents/me", headers=_auth_headers(self._api_key),
                    )
                elif action == "get_other":
                    name = inputs.get("agent_name", "")
                    if not name:
                        return ToolResult(success=False, error="agent_name required")
                    resp = await client.get(
                        f"{_BASE}/agents/profile",
                        headers=_auth_headers(self._api_key), params={"name": name},
                    )
                elif action == "update":
                    payload: dict[str, Any] = {}
                    if desc := inputs.get("description"):
                        payload["description"] = desc
                    resp = await client.patch(
                        f"{_BASE}/agents/me",
                        headers=_auth_headers(self._api_key), json=payload,
                    )
                else:
                    return ToolResult(success=False, error=f"Unknown action: {action}")

            data = _safe_json(resp)
            return ToolResult(
                success=resp.status_code < 400, output=data,
                error=data.get("error") if resp.status_code >= 400 else None,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class MoltbookFollowTool(BaseTool):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="moltbook_follow",
            description="Follow or unfollow another agent (molty) on Moltbook.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string"},
                    "action": {"type": "string", "enum": ["follow", "unfollow"]},
                },
                "required": ["agent_name", "action"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        name = inputs["agent_name"]
        method = "POST" if inputs["action"] == "follow" else "DELETE"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await _request_with_429_retry(
                    client, method, f"{_BASE}/agents/{name}/follow",
                    headers=_auth_headers(self._api_key),
                )
            data = _safe_json(resp)
            return ToolResult(
                success=resp.status_code < 400, output=data,
                error=data.get("error") if resp.status_code >= 400 else None,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class MoltbookSubmoltTool(BaseTool):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="moltbook_submolt",
            description=(
                "List, create, get info about, or subscribe/unsubscribe to "
                "Moltbook submolts (communities)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "create", "get", "subscribe", "unsubscribe"],
                    },
                    "name": {"type": "string", "description": "Submolt URL-safe name"},
                    "display_name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["action"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        action = inputs["action"]
        name = inputs.get("name", "")
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                if action == "list":
                    resp = await client.get(
                        f"{_BASE}/submolts", headers=_auth_headers(self._api_key),
                    )
                elif action == "create":
                    payload: dict[str, Any] = {"name": name}
                    if dn := inputs.get("display_name"):
                        payload["display_name"] = dn
                    if desc := inputs.get("description"):
                        payload["description"] = desc
                    resp = await client.post(
                        f"{_BASE}/submolts",
                        headers=_auth_headers(self._api_key), json=payload,
                    )
                elif action == "get":
                    resp = await client.get(
                        f"{_BASE}/submolts/{name}", headers=_auth_headers(self._api_key),
                    )
                elif action == "subscribe":
                    resp = await client.post(
                        f"{_BASE}/submolts/{name}/subscribe",
                        headers=_auth_headers(self._api_key),
                    )
                elif action == "unsubscribe":
                    resp = await client.delete(
                        f"{_BASE}/submolts/{name}/subscribe",
                        headers=_auth_headers(self._api_key),
                    )
                else:
                    return ToolResult(success=False, error=f"Unknown action: {action}")

            data = _safe_json(resp)
            if action == "create" and resp.status_code < 400:
                data = await _auto_verify(self._api_key, data, "submolt")
            return ToolResult(
                success=resp.status_code < 400, output=data,
                error=data.get("error") if resp.status_code >= 400 else None,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class MoltbookNotificationsTool(BaseTool):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="moltbook_notifications",
            description="Get notifications, mark them as read by post, or mark all as read.",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "read_by_post", "read_all"],
                    },
                    "post_id": {"type": "string", "description": "Required for read_by_post"},
                },
                "required": ["action"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        action = inputs["action"]
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                if action == "list":
                    resp = await client.get(
                        f"{_BASE}/notifications", headers=_auth_headers(self._api_key),
                    )
                elif action == "read_by_post":
                    post_id = inputs.get("post_id", "")
                    if not post_id:
                        return ToolResult(success=False, error="post_id required")
                    resp = await client.post(
                        f"{_BASE}/notifications/read-by-post/{post_id}",
                        headers=_auth_headers(self._api_key),
                    )
                elif action == "read_all":
                    resp = await client.post(
                        f"{_BASE}/notifications/read-all",
                        headers=_auth_headers(self._api_key),
                    )
                else:
                    return ToolResult(success=False, error=f"Unknown action: {action}")

            data = _safe_json(resp)
            return ToolResult(
                success=resp.status_code < 400, output=data,
                error=data.get("error") if resp.status_code >= 400 else None,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Dynamic registration helpers
# ---------------------------------------------------------------------------

def _register_authenticated_tools(
    registry: ToolRegistry,
    api_key: str,
    workspace: AgentWorkspace | None = None,
) -> None:
    """Register all Moltbook tools that require an API key."""
    tools: list[BaseTool] = [
        MoltbookHomeTool(api_key),
        MoltbookGetFeedTool(api_key),
        MoltbookGetPostTool(api_key),
        MoltbookCreatePostTool(api_key, workspace=workspace),
        MoltbookCommentTool(api_key),
        MoltbookVoteTool(api_key),
        MoltbookSearchTool(api_key),
        MoltbookProfileTool(api_key),
        MoltbookFollowTool(api_key),
        MoltbookSubmoltTool(api_key),
        MoltbookNotificationsTool(api_key),
    ]
    for tool in tools:
        registry.register(tool)
    logger.info("moltbook_authenticated_tools_registered", count=len(tools))


def bootstrap_moltbook(
    registry: ToolRegistry,
    llm: LLMRouter,
    workspace: AgentWorkspace | None = None,
) -> None:
    """Bootstrap Moltbook tools into the registry.

    - If credentials exist on disk, register the authenticated tools only.
      The register tool is NOT exposed — it would confuse the LLM.
    - If no credentials exist, register ONLY the register tool so the agent
      can create an account.
    """
    creds = load_credentials()
    api_key = creds.get("api_key", "")

    if api_key:
        logger.info(
            "moltbook_credentials_loaded",
            agent_name=creds.get("agent_name", "?"),
        )
        _register_authenticated_tools(registry, api_key, workspace=workspace)
    else:
        registry.register(MoltbookRegisterTool(registry, llm))


# ---------------------------------------------------------------------------
# Perception source adapter
# ---------------------------------------------------------------------------

class MoltbookPerceptionSource(PerceptionSource):
    """PerceptionSource adapter — lets the daemon poll Moltbook for events.

    Tracks notification IDs already seen within the process lifetime to
    avoid re-surfacing the same events across cycles.  Also exposes
    :meth:`mark_notifications_read` so the daemon can mark notifications
    as read after a successful run.
    """

    def __init__(self) -> None:
        self._seen_notification_ids: set[str] = set()
        self._post_ids_acted_on: set[str] = set()

    @property
    def name(self) -> str:
        return "moltbook"

    async def poll(self) -> list[Event]:

        creds = load_credentials()
        api_key = creds.get("api_key", "")
        if not api_key:
            return []

        events: list[Event] = []

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                resp = await client.get(
                    f"{_BASE}/home", headers=_auth_headers(api_key),
                )
                if resp.status_code < 400:
                    data = _safe_json(resp)
                    account = data.get("your_account", {})
                    unread = account.get("unread_notification_count", 0)
                    if unread:
                        events.append(Event(
                            source="moltbook",
                            kind=EventKind.NOTIFICATION,
                            title=f"{unread} unread notification(s)",
                            metadata={"unread_count": unread},
                        ))

                    for post in data.get("activity_on_your_posts", [])[:5]:
                        title = post.get("post_title", "unknown post")
                        events.append(Event(
                            source="moltbook",
                            kind=EventKind.REACTION,
                            title=f"Activity on your post: {title}",
                            detail=json.dumps(post, default=str)[:300],
                            metadata=post,
                        ))

                    dms = data.get("your_direct_messages", {})
                    pending = dms.get("pending_requests", 0) if isinstance(dms, dict) else 0
                    if pending:
                        events.append(Event(
                            source="moltbook",
                            kind=EventKind.MESSAGE,
                            title=f"{pending} pending DM(s)",
                            metadata={"pending_dms": pending},
                        ))

            except Exception as exc:
                logger.warning("moltbook_perception_home_error", error=str(exc))

            try:
                resp = await client.get(
                    f"{_BASE}/notifications", headers=_auth_headers(api_key),
                )
                if resp.status_code < 400:
                    data = _safe_json(resp)
                    for n in data.get("notifications", [])[:15]:
                        if n.get("isRead"):
                            continue
                        notif_id = n.get("id", "")
                        if notif_id and notif_id in self._seen_notification_ids:
                            continue
                        if notif_id:
                            self._seen_notification_ids.add(notif_id)

                        ntype = n.get("type", "")
                        kind_map = {
                            "post_comment": EventKind.NOTIFICATION,
                            "mention": EventKind.MENTION,
                            "upvote": EventKind.REACTION,
                            "new_follower": EventKind.NOTIFICATION,
                            "dm_request": EventKind.MESSAGE,
                        }

                        post = n.get("post") or {}
                        comment = n.get("comment") or {}
                        post_title = post.get("title", "")
                        comment_text = comment.get("content", "")

                        title = n.get("content", "notification")
                        if post_title:
                            title = f"{title} — {post_title[:80]}"

                        detail_parts: list[str] = []
                        if comment_text:
                            detail_parts.append(f'"{comment_text[:150]}"')
                        if detail_parts:
                            detail = " | ".join(detail_parts)
                        else:
                            detail = ""

                        events.append(Event(
                            source="moltbook",
                            kind=kind_map.get(ntype, EventKind.NOTIFICATION),
                            title=title,
                            detail=detail,
                            metadata={
                                "notification_id": notif_id,
                                "type": ntype,
                                "post_id": n.get("relatedPostId", ""),
                                "comment_id": n.get("relatedCommentId", ""),
                                "post_title": post_title,
                                "comment_text": comment_text[:200],
                            },
                        ))
            except Exception as exc:
                logger.warning("moltbook_perception_notif_error", error=str(exc))

        return events

    def record_acted_post(self, post_id: str) -> None:
        """Track a post_id that was acted on so we can mark it read."""
        if post_id:
            self._post_ids_acted_on.add(post_id)

    async def mark_notifications_read(self) -> None:
        """Mark notifications as read for all posts acted on, then clear."""
        creds = load_credentials()
        api_key = creds.get("api_key", "")
        if not api_key or not self._post_ids_acted_on:
            return

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for post_id in list(self._post_ids_acted_on):
                try:
                    resp = await client.post(
                        f"{_BASE}/notifications/read-by-post/{post_id}",
                        headers=_auth_headers(api_key),
                    )
                    if resp.status_code < 400:
                        logger.info(
                            "moltbook_notifications_marked_read",
                            post_id=post_id,
                        )
                    else:
                        logger.warning(
                            "moltbook_mark_read_failed",
                            post_id=post_id,
                            status=resp.status_code,
                        )
                except Exception as exc:
                    logger.warning(
                        "moltbook_mark_read_error",
                        post_id=post_id,
                        error=str(exc),
                    )
        self._post_ids_acted_on.clear()
