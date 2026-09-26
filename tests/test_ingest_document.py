import pytest

from evorag.domain.chunk import Chunk
from evorag.ingestion.ingest_document import IngestDocument
from tests.fakes import FakeEmbeddingProvider, InMemoryChunkStore

LONG_TEXT = "Hello World. " * 200

@pytest.mark.asyncio
async def test_stores_all_chunks_with_embeddings():
    store = InMemoryChunkStore()
    use_case = IngestDocument(embedder=FakeEmbeddingProvider(), writer=store)

    result = await use_case.execute(LONG_TEXT)

    assert result.chunk_count == len(store.chunks)
    assert all(chunk.embedding is not None for chunk in store.chunks)
    assert all(chunk.document_id == result.document_id for chunk in store.chunks)


@pytest.mark.asyncio
async def test_embeds_all_chunks_in_a_single_batch():
    embedder = FakeEmbeddingProvider()
    use_case = IngestDocument(embedder=embedder, writer=InMemoryChunkStore())

    await use_case.execute(LONG_TEXT)

    assert len(embedder.calls) == 1

@pytest.mark.asyncio
async def test_sends_chunk_contents_not_the_whole_document():
    embedder = FakeEmbeddingProvider()
    store = InMemoryChunkStore()
    use_case = IngestDocument(embedder=embedder, writer=store)

    await use_case.execute(LONG_TEXT)

    assert embedder.calls[0] == [chunk.content for chunk in store.chunks]

@pytest.mark.asyncio
async def test_empty_text_raises_and_saves_nothing():
    store = InMemoryChunkStore()
    use_case = IngestDocument(embedder=FakeEmbeddingProvider(), writer=store)

    with pytest.raises(ValueError):
        await use_case.execute("")

    assert store.chunks == []
