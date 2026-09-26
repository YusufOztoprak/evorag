from uuid import uuid4

import pytest

from evorag.domain.chunk import Chunk
from evorag.retrieval.search_chunks import SearchChunks
from tests.fakes import FakeEmbeddingProvider, InMemoryChunkStore


def make_chunk(content: str, embedding: list[float]) -> Chunk:
    return Chunk(document_id=uuid4(), index=0, content=content, embedding=embedding)


@pytest.mark.asyncio
async def test_returns_only_chunks_above_threshold():
    store = InMemoryChunkStore()
    await store.save_all([
        make_chunk("relevant", [1.0, 0.0, 0.0]),
        make_chunk("somewhat related", [1.0, 1.0, 0.0]),
        make_chunk("unrelated", [0.0, 1.0, 0.0]),
    ])
    embedder = FakeEmbeddingProvider({"my question": [1.0, 0.0, 0.0]})
    use_case = SearchChunks(embedder=embedder, searcher=store)

    hits = await use_case.execute("my question")

    assert [hit.content for hit in hits] == ["relevant"]
    assert hits[0].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_respects_limit():
    store = InMemoryChunkStore()
    await store.save_all([make_chunk(f"chunk {i}", [1.0, 0.0, 0.0]) for i in range(3)])
    use_case = SearchChunks(embedder=FakeEmbeddingProvider(), searcher=store)

    hits = await use_case.execute("any question", limit=2)

    assert len(hits) == 2


@pytest.mark.asyncio
async def test_empty_query_raises():
    use_case = SearchChunks(embedder=FakeEmbeddingProvider(), searcher=InMemoryChunkStore())

    with pytest.raises(ValueError):
        await use_case.execute("   ")