from __future__ import annotations

from collections.abc import AsyncIterator

import structlog

from klovis_agent.llm.gateway import ModelGateway, OpenAIGateway
from klovis_agent.llm.types import ModelRequest, ModelResponse, ModelRoutingPolicy

logger = structlog.get_logger(__name__)


class LLMRouter:
    """Routes LLM requests to the appropriate model based on the routing policy."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        default_model: str = "gpt-4o",
        default_max_tokens: int = 4096,
        default_temperature: float = 0.2,
        policy: ModelRoutingPolicy | None = None,
    ) -> None:
        self._policy = policy or ModelRoutingPolicy()
        self._gateways: dict[tuple[str, str | None], ModelGateway] = {}
        self._api_key = api_key
        self._base_url = base_url
        self._default_model = default_model
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature

    def _get_gateway(self, model: str, base_url: str | None = None) -> ModelGateway:
        effective_url = base_url or self._base_url
        key = (model, effective_url)
        if key not in self._gateways:
            self._gateways[key] = OpenAIGateway(
                api_key=self._api_key,
                default_model=model,
                base_url=effective_url,
            )
        return self._gateways[key]

    def effective_max_tokens(self, purpose: str) -> int:
        """Return the effective max_tokens for a given purpose."""
        phase_max = self._policy.max_tokens_for_purpose(purpose)
        return phase_max or self._default_max_tokens

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        model = self._policy.model_for_purpose(request.purpose, self._default_model)
        phase_base_url = self._policy.base_url_for_purpose(request.purpose)
        gateway = self._get_gateway(model, phase_base_url)

        phase_max = self._policy.max_tokens_for_purpose(request.purpose)
        effective_max = request.max_tokens or phase_max or self._default_max_tokens

        reasoning = (
            request.reasoning_effort
            or self._policy.reasoning_effort_for_purpose(request.purpose)
        )

        temp = (
            request.temperature
            if request.temperature is not None
            else self._default_temperature
        )
        request = request.model_copy(update={
            "max_tokens": effective_max,
            "temperature": temp,
            "reasoning_effort": reasoning,
        })

        logger.info("llm_route", purpose=request.purpose, model=model)
        return await gateway.invoke(request)

    async def invoke_stream(self, request: ModelRequest) -> AsyncIterator[str]:
        """Stream text chunks for lightweight narration calls."""
        model = self._policy.model_for_purpose(request.purpose, self._default_model)
        phase_base_url = self._policy.base_url_for_purpose(request.purpose)
        gateway = self._get_gateway(model, phase_base_url)

        temp = (
            request.temperature
            if request.temperature is not None
            else self._default_temperature
        )
        request = request.model_copy(update={"temperature": temp})

        logger.info("llm_route_stream", purpose=request.purpose, model=model)
        async for chunk in gateway.invoke_stream(request):
            yield chunk
