import math

from evorag.domain.chunk import Chunk

class FakeEmbeddingProvider:
    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.vectors.get(text, [1.0, 0.0, 0.0]) for text in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)

class InMemoryChunkStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    async def save_all(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    async def find_similar(self, embedding: list[float], limit: int) -> list[tuple[Chunk, float]]:
        scored = [
            (chunk, cosine_similarity(embedding, chunk.embedding))
            for chunk in self.chunks
            if chunk.embedding is not None
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]