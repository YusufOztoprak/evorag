from typing import Protocol

from evorag.domain.chunk import Chunk


class ChunkSearcher(Protocol):
    async def find_similar(self, embedding: list[float], limit: int) -> list[tuple[Chunk, float]]: ...
