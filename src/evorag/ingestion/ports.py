from typing import Protocol

from evorag.domain.chunk import Chunk

class ChunkWriter(Protocol):
    async def save_all(self, chunks: list[Chunk]) -> None: ...