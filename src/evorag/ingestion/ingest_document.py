from dataclasses import dataclass,replace
from uuid import UUID,uuid4

from evorag.ingestion.chunking import split_into_chunks
from evorag.ingestion.ports import ChunkWriter
from evorag.shared.ports import EmbeddingProvider

@dataclass(frozen=True)
class IngestResult:
    document_id: UUID
    chunk_count: int

class IngestDocument:
    def __init__(self, embedder: EmbeddingProvider, writer: ChunkWriter) -> None:
        self.embedder = embedder
        self.writer = writer

    async def execute(self, text: str) -> IngestResult:
        document_id = uuid4()
        chunks = split_into_chunks(text, document_id)
        if not chunks:
            raise ValueError("document is empty")

        vectors = await self.embedder.embed([chunk.content for chunk in chunks])
        embedded = [
            replace(chunk, embedding=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        await self.writer.save_all(embedded)
        return IngestResult(document_id=document_id, chunk_count=len(chunks))