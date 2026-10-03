import logging
import time
import uuid

from fastapi import Request, Response

from evorag.observability.context import set_request_id

logger = logging.getLogger(__name__)

async def log_requests(request: Request, call_next) -> Response:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    set_request_id(request_id)

    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    response.headers["x-request-id"] = request_id
    logger.info(
        "request.completed",
        extra={
            "fields": {
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
            }
        },
    )
    return response