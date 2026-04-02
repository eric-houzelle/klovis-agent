from __future__ import annotations

import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger(__name__)


class EmbeddingClient:
    """Async embedding client using an OpenAI-compatible API (e.g. OVH AI Endpoints)."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "BGE-M3",
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text."""
        if not texts:
            return []

        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )

        vectors = [item.embedding for item in response.data]
        logger.info(
            "embeddings_created",
            model=self._model,
            count=len(vectors),
            dim=len(vectors[0]) if vectors else 0,
        )
        return vectors

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        result = await self.embed([text])
        return result[0]
