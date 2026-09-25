from uuid import uuid4

import pytest
import tiktoken

from evorag.ingestion.chunking import split_into_chunks

enc = tiktoken.get_encoding("cl100k_base")


def test_empty_text_returns_no_chunks():
    assert split_into_chunks("", uuid4()) == []


def test_chunks_have_sequential_indices_and_same_document():
    doc_id = uuid4()
    chunks = split_into_chunks("Hello World. " * 200, doc_id, chunk_size=50, overlap=10)

    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.document_id == doc_id for c in chunks)


def test_overlap_not_smaller_than_chunk_size_raises():
    with pytest.raises(ValueError):
        split_into_chunks("some text", uuid4(), chunk_size=10, overlap=10)


def test_short_text_produces_single_chunk():
    text = "A short text."
    chunks = split_into_chunks(text, uuid4(), chunk_size=500, overlap=50)

    assert len(chunks) == 1
    assert chunks[0].content == text