from __future__ import annotations

import json
import re
import time
from typing import Protocol, runtime_checkable

import structlog

from klovis_agent.llm.types import ModelRequest, ModelResponse

logger = structlog.get_logger(__name__)


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json(text: str) -> dict[str, object] | None:
    """Best-effort JSON extraction from LLM output.

    Handles: raw JSON, markdown-fenced JSON (with or without newlines after
    the fence opener), reasoning text before/after JSON, and truncated output
    where the outermost braces are unbalanced.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start != -1:
        depth = 0
        end = -1
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

    return None


@runtime_checkable
class ModelGateway(Protocol):
    """Required abstraction for any LLM provider."""

    async def invoke(self, request: ModelRequest) -> ModelResponse: ...


class OpenAIGateway:
    """ModelGateway implementation for OpenAI-compatible APIs."""

    DEFAULT_TIMEOUT = 60
    MAX_RETRIES = 3
    RETRY_BACKOFF = (2, 5, 10)

    def __init__(
        self,
        api_key: str,
        default_model: str = "gpt-4o",
        base_url: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        import httpx
        from openai import AsyncOpenAI

        effective_timeout = timeout or self.DEFAULT_TIMEOUT
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(effective_timeout, connect=10),
        )
        self._default_model = default_model
        self._max_retries = max_retries if max_retries is not None else self.MAX_RETRIES

    async def _call_with_fallbacks(self, kwargs: dict[str, object]) -> object:
        """Call the API with timeout, retries on transient errors, and param fallback."""
        import asyncio

        from openai import APIConnectionError, APITimeoutError, BadRequestError

        last_exc: BaseException | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]

            except BadRequestError as exc:
                error_msg = str(exc).lower()
                dropped: list[str] = []

                if "reasoning_effort" in error_msg and "reasoning_effort" in kwargs:
                    del kwargs["reasoning_effort"]
                    dropped.append("reasoning_effort")

                if "json_schema" in error_msg and "response_format" in kwargs:
                    kwargs["response_format"] = {"type": "json_object"}  # type: ignore[assignment]
                    dropped.append("response_format(json_schema->json_object)")

                if not dropped:
                    raise

                logger.warning(
                    "llm_param_fallback",
                    model=kwargs.get("model"),
                    dropped=dropped,
                )
                return await self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]

            except (APITimeoutError, APIConnectionError) as exc:
                last_exc = exc
                backoff = self.RETRY_BACKOFF[min(attempt - 1, len(self.RETRY_BACKOFF) - 1)]
                logger.warning(
                    "llm_retry",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    error=type(exc).__name__,
                    backoff_s=backoff,
                    model=kwargs.get("model"),
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(backoff)

        raise last_exc  # type: ignore[misc]

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        start = time.monotonic()

        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]

        kwargs: dict[str, object] = {
            "model": self._default_model,
            "messages": messages,
            "temperature": request.temperature if request.temperature is not None else 0.2,
            "max_tokens": request.max_tokens or 4096,
        }

        if request.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.reasoning_effort

        if request.structured_output_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": request.structured_output_schema,
                },
            }

        response = await self._call_with_fallbacks(kwargs)
        choice = response.choices[0]
        raw = choice.message.content or ""
        finish_reason = choice.finish_reason

        if finish_reason == "length":
            logger.warning(
                "llm_output_truncated",
                max_tokens=request.max_tokens,
                purpose=request.purpose,
                raw_tail=raw[-500:] if raw else "(empty)",
            )

        structured: dict[str, object] | None = None
        if request.structured_output_schema is not None:
            structured = _extract_json(raw)
            if structured is None:
                logger.warning(
                    "llm_structured_parse_failed",
                    raw_head=raw[:500] if raw else "(empty)",
                    raw_tail=raw[-500:] if raw else "(empty)",
                    raw_len=len(raw),
                    finish_reason=finish_reason,
                )

        duration_ms = int((time.monotonic() - start) * 1000)
        usage_data = {}
        if response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        logger.info(
            "llm_call",
            purpose=request.purpose,
            model=self._default_model,
            duration_ms=duration_ms,
            tokens=usage_data.get("total_tokens"),
        )

        return ModelResponse(
            raw_text=raw if structured is None else None,
            structured_output=structured,
            usage=usage_data,
            model_name=response.model,
        )
