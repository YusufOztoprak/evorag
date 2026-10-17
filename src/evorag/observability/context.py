from contextvars import ContextVar
from dataclasses import dataclass

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


@dataclass
class RequestMetrics:
    embedding_tokens: int = 0

_metrics: ContextVar[RequestMetrics | None] = ContextVar("metrics", default=None)

def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)

def get_request_id() -> str | None:
    return _request_id.get()

def start_metrics() -> RequestMetrics:
    metrics = RequestMetrics()
    _metrics.set(metrics)
    return metrics

def add_embedding_tokens(tokens: int) -> None:
    metrics = _metrics.get()
    if metrics is None:
        return
    metrics.embedding_tokens += tokens
