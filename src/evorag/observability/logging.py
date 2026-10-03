import json
from datetime import UTC, datetime
import logging
import sys
from evorag.observability.context import get_request_id

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": get_request_id(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)

def configure_logging(level: str = 'INFO') -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("evorag")
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
