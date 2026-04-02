from __future__ import annotations

from typing import TYPE_CHECKING, Any

from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec

if TYPE_CHECKING:
    from klovis_agent.tools.builtin.skills import SkillStore

MAX_RESPONSE_BYTES = 100_000
DEFAULT_TIMEOUT = 30


class HttpRequestTool(BaseTool):
    """Make HTTP requests to external URLs.

    When a ``SkillStore`` is provided, authentication headers are injected
    automatically for URLs that match a loaded skill's ``api_base``.
    """

    def __init__(self, skill_store: SkillStore | None = None) -> None:
        self._skill_store = skill_store

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="http_request",
            description=(
                "Make an HTTP request to a URL and return the response. "
                "Supports GET, POST, PUT, PATCH, DELETE methods. "
                "Use for calling APIs, fetching data, or checking endpoints. "
                "Authentication headers are injected automatically for URLs "
                "covered by a loaded skill (e.g. Moltbook API) — you do NOT "
                "need to provide Authorization headers for those APIs."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    },
                    "url": {"type": "string"},
                    "headers": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "body": {"type": "object"},
                    "timeout": {"type": "integer"},
                },
                "required": ["method", "url"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "status_code": {"type": "integer"},
                    "headers": {"type": "object"},
                    "body": {"type": "string"},
                },
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        import httpx

        method = inputs.get("method", "GET")
        url = inputs.get("url", "")
        headers: dict[str, str] = dict(inputs.get("headers") or {})
        body = inputs.get("body")
        timeout = inputs.get("timeout", DEFAULT_TIMEOUT)

        if not url:
            return ToolResult(success=False, error="Missing required field: 'url'")

        if self._skill_store and "Authorization" not in headers:
            skill_auth = self._skill_store.get_auth_for_url(url)
            if skill_auth:
                headers.update(skill_auth)
                if "Content-Type" not in headers:
                    headers["Content-Type"] = "application/json"

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.request(
                    method,
                    url,
                    headers=headers,
                    json=body if body else None,
                )
                response_body = resp.text[:MAX_RESPONSE_BYTES]
                truncated = len(resp.text) > MAX_RESPONSE_BYTES

                return ToolResult(
                    success=200 <= resp.status_code < 400,
                    output={
                        "status_code": resp.status_code,
                        "headers": dict(resp.headers),
                        "body": response_body,
                        "truncated": truncated,
                    },
                )
        except httpx.TimeoutException:
            return ToolResult(success=False, error=f"Request timed out after {timeout}s")
        except Exception as exc:
            return ToolResult(success=False, error=f"HTTP error: {exc}")


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_search",
            description=(
                "Search the web for information. Returns a list of results "
                "with titles, URLs, and snippets. Use when you need to find "
                "documentation, answers, or current information."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                    },
                },
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "snippet": {"type": "string"},
                            },
                        },
                    },
                },
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        import asyncio

        query = inputs.get("query", "")
        max_results = inputs.get("max_results", 5)

        if not query:
            return ToolResult(success=False, error="Missing required field: 'query'")

        try:
            from ddgs import DDGS

            def _search() -> list[dict[str, Any]]:
                with DDGS() as ddgs:
                    return ddgs.text(query, max_results=max_results)

            raw = await asyncio.to_thread(_search)

            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in raw
            ]
            return ToolResult(
                success=True,
                output={"results": results, "query": query},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Search error: {exc}")
