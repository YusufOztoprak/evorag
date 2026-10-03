import logging
import time

from openai import AsyncOpenAI

from evorag.observability.context import add_embedding_tokens

logger = logging.getLogger(__name__)

class OpenAIEmbeddingProvider:
    def __init__(self, client: AsyncOpenAI, model: str, dimensions: int) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        started = time.perf_counter()
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        tokens = response.usage.total_tokens
        add_embedding_tokens(tokens)
        logger.info(
            "embedding.created",
            extra={"fields": {
                "model": self._model,
                "inputs": len(texts),
                "tokens": tokens,
                "latency_ms": latency_ms,
            }}
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]