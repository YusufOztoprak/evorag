from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from evorag.adapters.http.schemas import (
    IngestRequest,
    IngestResponse,
    SearchHitResponse,
    SearchRequest,
    SearchResponse,
)
from evorag.ingestion.ingest_document import IngestDocument
from evorag.retrieval.search_chunks import SearchChunks

router = APIRouter()


def get_ingest_document(request: Request) -> IngestDocument:
    return request.app.state.ingest_document


def get_search_chunks(request: Request) -> SearchChunks:
    return request.app.state.search_chunks


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/documents", status_code=201)
async def ingest_document(
    body: IngestRequest,
    use_case: Annotated[IngestDocument, Depends(get_ingest_document)],
) -> IngestResponse:
    try:
        result = await use_case.execute(body.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return IngestResponse(document_id=result.document_id, chunk_count=result.chunk_count)


@router.post("/search")
async def search(
    body: SearchRequest,
    use_case: Annotated[SearchChunks, Depends(get_search_chunks)],
) -> SearchResponse:
    try:
        hits = await use_case.execute(body.query, body.limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SearchResponse(
        hits=[
            SearchHitResponse(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                content=hit.content,
                score=hit.score,
            )
            for hit in hits
        ]
    )