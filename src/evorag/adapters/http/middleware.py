import logging
import time
import uuid

from fastapi import Request, Response

from evorag.observability.context import set_request_id, start_metrics

logger = logging.getLogger(__name__)

async def log_requests(request: Request, call_next) -> Response:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    set_request_id(request_id)
    metrics = start_metrics()

    status_code = 500
    started = time.perf_counter()
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "request.completed",
            extra={
                "fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "latency_ms": latency_ms,
                    "embedding_tokens": metrics.embedding_tokens,
                    "embedding_cost_usd": metrics.embedding_cost_usd,
                }
            },

        )