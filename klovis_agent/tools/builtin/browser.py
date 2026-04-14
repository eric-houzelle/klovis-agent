"""Advanced headless-browser tool powered by Playwright.

Provides the agent with the ability to navigate web pages, interact with
DOM elements, extract structured content, and take screenshots — all
through a single multi-action ``BrowserTool``.

Playwright is an **optional** dependency.  Install it with::

    pip install klovis-agent[browser]
    playwright install chromium
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

_ACTIONS = [
    "navigate",
    "click",
    "type_text",
    "select_option",
    "check",
    "press_key",
    "scroll",
    "back",
    "forward",
    "wait_for",
    "extract_text",
    "extract_html",
    "extract_links",
    "extract_table",
    "accessibility_tree",
    "screenshot",
    "evaluate_js",
    "fill_form",
    "get_cookies",
    "set_cookies",
    "close",
]

MAX_TEXT_LENGTH = 80_000
DEFAULT_TIMEOUT_MS = 30_000
SCREENSHOT_MAX_WIDTH = 1280
SCREENSHOT_MAX_HEIGHT = 720


class BrowserTool(BaseTool):
    """Headless browser automation via Playwright (async API).

    The browser instance is created lazily on first use and reuses a single
    ``BrowserContext`` (isolated cookies / storage) across actions within the
    same agent run.  Call the ``close`` action — or ``await cleanup()`` — to
    release resources.
    """

    requires_confirmation = True

    def __init__(
        self,
        *,
        headless: bool = True,
        default_timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        super().__init__(requires_confirmation=True)
        self._headless = headless
        self._default_timeout_ms = default_timeout_ms
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    # ------------------------------------------------------------------
    # Spec
    # ------------------------------------------------------------------

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser",
            description=(
                "Control a headless browser to navigate web pages, interact "
                "with elements (click, type, select), extract content (text, "
                "HTML, links, tables, accessibility tree), take screenshots, "
                "and execute JavaScript. Useful for pages that require "
                "JavaScript rendering, form filling, or multi-step web "
                "workflows that go beyond simple HTTP requests."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": _ACTIONS,
                        "description": "The browser action to perform.",
                    },
                    "url": {
                        "type": "string",
                        "description": "Target URL (for 'navigate').",
                    },
                    "selector": {
                        "type": "string",
                        "description": (
                            "CSS selector targeting a DOM element "
                            "(for click, type_text, select_option, check, "
                            "wait_for, extract_text, extract_html, extract_table)."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "Text to type (for 'type_text'), option value "
                            "(for 'select_option'), or key name (for 'press_key')."
                        ),
                    },
                    "script": {
                        "type": "string",
                        "description": "JavaScript expression to evaluate (for 'evaluate_js').",
                    },
                    "form_data": {
                        "type": "object",
                        "description": (
                            "Mapping of CSS selector → value for 'fill_form'. "
                            "Each key is a selector and each value is the text to fill."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "cookies": {
                        "type": "array",
                        "description": (
                            "List of cookie objects for 'set_cookies'. "
                            "Each must have 'name', 'value', and 'url' or 'domain'."
                        ),
                        "items": {"type": "object"},
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Scroll direction (for 'scroll'). Default: 'down'.",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Scroll amount in pixels (for 'scroll'). Default: 500.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in milliseconds for this action.",
                    },
                },
                "required": ["action"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "links": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "href": {"type": "string"},
                            },
                        },
                    },
                    "table": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                    "screenshot_base64": {"type": "string"},
                    "cookies": {"type": "array", "items": {"type": "object"}},
                    "result": {},
                },
            },
        )

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def _ensure_browser(self) -> None:
        if self._page is not None:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for the browser tool. "
                "Install it with: pip install klovis-agent[browser] && playwright install chromium"
            ) from exc

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        self._context = await self._browser.new_context(
            viewport={"width": SCREENSHOT_MAX_WIDTH, "height": SCREENSHOT_MAX_HEIGHT},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        )
        self._context.set_default_timeout(self._default_timeout_ms)
        self._page = await self._context.new_page()

    async def cleanup(self) -> None:
        """Release all browser resources."""
        if self._context:
            with _suppress():
                await self._context.close()
            self._context = None
        if self._browser:
            with _suppress():
                await self._browser.close()
            self._browser = None
        if self._playwright:
            with _suppress():
                await self._playwright.stop()
            self._playwright = None
        self._page = None

    # ------------------------------------------------------------------
    # Execute dispatcher
    # ------------------------------------------------------------------

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        action: str = inputs.get("action", "")
        if action not in _ACTIONS:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Must be one of: {', '.join(_ACTIONS)}",
            )

        if action == "close":
            await self.cleanup()
            return ToolResult(success=True, output={"message": "Browser closed."})

        try:
            await self._ensure_browser()
        except RuntimeError as exc:
            return ToolResult(success=False, error=str(exc))

        handler = getattr(self, f"_action_{action}", None)
        if handler is None:
            return ToolResult(success=False, error=f"Action '{action}' not implemented.")

        try:
            return await handler(inputs)
        except Exception as exc:
            logger.warning("browser_action_error", extra={"action": action, "error": str(exc)})
            return ToolResult(success=False, error=f"Browser error ({action}): {exc}")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    async def _action_navigate(self, inputs: dict[str, Any]) -> ToolResult:
        url = inputs.get("url", "")
        if not url:
            return ToolResult(success=False, error="Missing required field: 'url'")
        timeout = inputs.get("timeout", self._default_timeout_ms)
        await self._page.goto(url, timeout=timeout, wait_until="domcontentloaded")
        return self._page_info()

    async def _action_back(self, _inputs: dict[str, Any]) -> ToolResult:
        await self._page.go_back(wait_until="domcontentloaded")
        return self._page_info()

    async def _action_forward(self, _inputs: dict[str, Any]) -> ToolResult:
        await self._page.go_forward(wait_until="domcontentloaded")
        return self._page_info()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    async def _action_click(self, inputs: dict[str, Any]) -> ToolResult:
        selector = inputs.get("selector", "")
        if not selector:
            return ToolResult(success=False, error="Missing required field: 'selector'")
        timeout = inputs.get("timeout", self._default_timeout_ms)
        await self._page.click(selector, timeout=timeout)
        await self._page.wait_for_load_state("domcontentloaded")
        return self._page_info()

    async def _action_type_text(self, inputs: dict[str, Any]) -> ToolResult:
        selector = inputs.get("selector", "")
        text = inputs.get("text", "")
        if not selector:
            return ToolResult(success=False, error="Missing required field: 'selector'")
        timeout = inputs.get("timeout", self._default_timeout_ms)
        await self._page.fill(selector, text, timeout=timeout)
        return self._page_info()

    async def _action_select_option(self, inputs: dict[str, Any]) -> ToolResult:
        selector = inputs.get("selector", "")
        value = inputs.get("text", "")
        if not selector:
            return ToolResult(success=False, error="Missing required field: 'selector'")
        timeout = inputs.get("timeout", self._default_timeout_ms)
        selected = await self._page.select_option(selector, value, timeout=timeout)
        return ToolResult(success=True, output={**self._page_snapshot(), "selected": selected})

    async def _action_check(self, inputs: dict[str, Any]) -> ToolResult:
        selector = inputs.get("selector", "")
        if not selector:
            return ToolResult(success=False, error="Missing required field: 'selector'")
        timeout = inputs.get("timeout", self._default_timeout_ms)
        await self._page.check(selector, timeout=timeout)
        return self._page_info()

    async def _action_press_key(self, inputs: dict[str, Any]) -> ToolResult:
        key = inputs.get("text", "")
        if not key:
            return ToolResult(success=False, error="Missing required field: 'text' (key name)")
        await self._page.keyboard.press(key)
        return self._page_info()

    async def _action_scroll(self, inputs: dict[str, Any]) -> ToolResult:
        direction = inputs.get("direction", "down")
        amount = inputs.get("amount", 500)
        delta = amount if direction == "down" else -amount
        await self._page.mouse.wheel(0, delta)
        await asyncio.sleep(0.3)
        return self._page_info()

    async def _action_fill_form(self, inputs: dict[str, Any]) -> ToolResult:
        form_data: dict[str, str] = inputs.get("form_data", {})
        if not form_data:
            return ToolResult(success=False, error="Missing required field: 'form_data'")
        timeout = inputs.get("timeout", self._default_timeout_ms)
        for selector, value in form_data.items():
            await self._page.fill(selector, value, timeout=timeout)
        return self._page_info()

    async def _action_wait_for(self, inputs: dict[str, Any]) -> ToolResult:
        selector = inputs.get("selector", "")
        if not selector:
            return ToolResult(success=False, error="Missing required field: 'selector'")
        timeout = inputs.get("timeout", self._default_timeout_ms)
        await self._page.wait_for_selector(selector, timeout=timeout)
        return self._page_info()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    async def _action_extract_text(self, inputs: dict[str, Any]) -> ToolResult:
        selector = inputs.get("selector")
        if selector:
            el = await self._page.query_selector(selector)
            if el is None:
                return ToolResult(success=False, error=f"Element not found: {selector}")
            text = (await el.inner_text()) or ""
        else:
            text = (await self._page.inner_text("body")) or ""
        text = _truncate(text)
        return ToolResult(success=True, output={**self._page_snapshot(), "content": text})

    async def _action_extract_html(self, inputs: dict[str, Any]) -> ToolResult:
        selector = inputs.get("selector")
        if selector:
            el = await self._page.query_selector(selector)
            if el is None:
                return ToolResult(success=False, error=f"Element not found: {selector}")
            html = (await el.inner_html()) or ""
        else:
            html = (await self._page.content()) or ""
        html = _truncate(html)
        return ToolResult(success=True, output={**self._page_snapshot(), "content": html})

    async def _action_extract_links(self, _inputs: dict[str, Any]) -> ToolResult:
        links = await self._page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                text: a.innerText.trim().substring(0, 120),
                href: a.href,
            }))
        """)
        return ToolResult(success=True, output={**self._page_snapshot(), "links": links})

    async def _action_extract_table(self, inputs: dict[str, Any]) -> ToolResult:
        selector = inputs.get("selector", "table")
        table_data: list[list[str]] = await self._page.evaluate("""
            (sel) => {
                const table = document.querySelector(sel);
                if (!table) return [];
                return Array.from(table.querySelectorAll('tr')).map(row =>
                    Array.from(row.querySelectorAll('th, td')).map(cell =>
                        cell.innerText.trim()
                    )
                );
            }
        """, selector)
        return ToolResult(success=True, output={**self._page_snapshot(), "table": table_data})

    async def _action_accessibility_tree(self, _inputs: dict[str, Any]) -> ToolResult:
        snapshot = await self._page.accessibility.snapshot()  # type: ignore[union-attr]
        tree_text = _format_a11y_node(snapshot) if snapshot else "(empty page)"
        tree_text = _truncate(tree_text)
        return ToolResult(success=True, output={**self._page_snapshot(), "content": tree_text})

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    async def _action_screenshot(self, _inputs: dict[str, Any]) -> ToolResult:
        raw = await self._page.screenshot(type="png", full_page=False)
        encoded = base64.b64encode(raw).decode("ascii")
        return ToolResult(
            success=True,
            output={**self._page_snapshot(), "screenshot_base64": encoded},
        )

    # ------------------------------------------------------------------
    # JavaScript
    # ------------------------------------------------------------------

    async def _action_evaluate_js(self, inputs: dict[str, Any]) -> ToolResult:
        script = inputs.get("script", "")
        if not script:
            return ToolResult(success=False, error="Missing required field: 'script'")
        result = await self._page.evaluate(script)
        return ToolResult(success=True, output={**self._page_snapshot(), "result": result})

    # ------------------------------------------------------------------
    # Cookies
    # ------------------------------------------------------------------

    async def _action_get_cookies(self, _inputs: dict[str, Any]) -> ToolResult:
        cookies = await self._context.cookies()
        return ToolResult(success=True, output={"cookies": cookies})

    async def _action_set_cookies(self, inputs: dict[str, Any]) -> ToolResult:
        cookies = inputs.get("cookies", [])
        if not cookies:
            return ToolResult(success=False, error="Missing required field: 'cookies'")
        await self._context.add_cookies(cookies)
        return ToolResult(success=True, output={"message": f"Set {len(cookies)} cookie(s)."})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _page_snapshot(self) -> dict[str, str]:
        return {
            "url": self._page.url,
            "title": self._page.url,  # title requires await; use url as fallback
        }

    def _page_info(self) -> ToolResult:
        return ToolResult(success=True, output=self._page_snapshot())


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _truncate(text: str) -> str:
    if len(text) > MAX_TEXT_LENGTH:
        return text[:MAX_TEXT_LENGTH] + "\n\n... (truncated)"
    return text


def _format_a11y_node(node: dict[str, Any], indent: int = 0) -> str:
    """Recursively format an accessibility tree snapshot into readable text."""
    prefix = "  " * indent
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    parts = [role]
    if name:
        parts.append(f'"{name}"')
    if value:
        parts.append(f"[{value}]")

    line = f"{prefix}{' '.join(parts)}"
    children = node.get("children", [])
    child_lines = [_format_a11y_node(c, indent + 1) for c in children]
    return "\n".join([line, *child_lines])


class _suppress:
    """Async-friendly context manager that suppresses all exceptions."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> bool:
        return True
