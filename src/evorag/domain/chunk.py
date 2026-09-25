from dataclasses import dataclass, field
from uuid import UUID, uuid4

@dataclass(frozen=True)
class Chunk:
    document_id: UUID
    index: int
    content: str
    embedding: list[float] | None = None
    id: UUID = field(default_factory=uuid4)
