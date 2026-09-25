from uuid import UUID

import tiktoken

from evorag.domain.chunk import Chunk

_ENCODING = tiktoken.get_encoding("cl100k_base")

def split_into_chunks(
        text: str,
        document_id: UUID,
        chunk_size: int = 500,
        overlap: int = 50,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError(f"Overlap must be less than {chunk_size}")

    tokens = _ENCODING.encode(text)
    step = chunk_size - overlap
    chunks: list[Chunk] = []

    for index, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start: start + chunk_size]
        content = _ENCODING.decode(window)
        chunks.append(Chunk(document_id=document_id,index=index, content=content))

    return chunks