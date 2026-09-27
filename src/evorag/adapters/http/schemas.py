from uuid import UUID

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)


class IngestResponse(BaseModel):
    document_id: UUID
    chunk_count: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


class SearchHitResponse(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHitResponse]