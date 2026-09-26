from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from evorag.domain.chunk import Chunk

EMBEDDING_DIMENSIONS = 1536


class Base(DeclarativeBase):
    pass


class ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    document_id: Mapped[UUID] = mapped_column(index=True)
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))


def to_row(chunk: Chunk) -> ChunkRow:
    if chunk.embedding is None:
        raise ValueError(f"chunk {chunk.id} has no embedding and cannot be stored")
    return ChunkRow(
        id=chunk.id,
        document_id=chunk.document_id,
        chunk_index=chunk.index,
        content=chunk.content,
        embedding=chunk.embedding,
    )


def to_domain(row: ChunkRow) -> Chunk:
    return Chunk(
        id=row.id,
        document_id=row.document_id,
        index=row.chunk_index,
        content=row.content,
        embedding=[float(value) for value in row.embedding],
    )


class PgVectorChunkStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def save_all(self, chunks: list[Chunk]) -> None:
        async with self._session_factory() as session, session.begin():
            session.add_all([to_row(chunk) for chunk in chunks])

    async def find_similar(self, embedding: list[float], limit: int) -> list[tuple[Chunk, float]]:
        distance = ChunkRow.embedding.cosine_distance(embedding)
        statement = select(ChunkRow, distance.label("distance")).order_by(distance).limit(limit)

        async with self._session_factory() as session:
            result = await session.execute(statement)
            return [(to_domain(row), 1.0 - dist) for row, dist in result.all()]