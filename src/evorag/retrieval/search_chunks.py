from dataclasses import dataclass
from uuid import UUID

from evorag.retrieval.ports import ChunkSearcher
from evorag.shared.ports import EmbeddingProvider


@dataclass(frozen=True)
class SearchHit:
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float


class SearchChunks:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        searcher: ChunkSearcher,
        min_score: float = 0.75,
    ) -> None:
        self._embedder = embedder
        self._searcher = searcher
        self._min_score = min_score

    async def execute(self, query: str, limit: int = 5) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("query is empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        [query_vector] = await self._embedder.embed([query])
        candidates = await self._searcher.find_similar(query_vector, limit)

        return [
            SearchHit(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                score=score,
            )
            for chunk, score in candidates
            if score >= self._min_score
        ]