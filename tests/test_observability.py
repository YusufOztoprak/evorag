import logging
import json

from fastapi.testclient import TestClient
from fastapi import FastAPI

from evorag.adapters.http.middleware import log_requests
from evorag.observability.context import add_embedding_tokens
from evorag.observability.logging import JsonFormatter

def make_app():
    app = FastAPI()
    app.middleware("http")(log_requests)

    @app.get("/fake")
    async def fake() -> dict[str, str]:
        add_embedding_tokens(30)
        add_embedding_tokens(12)
        return {"ok": "yes"}

    return app

def test_middleware_generates_request_id():
    app = make_app()
    client = TestClient(app)
    response = client.get("/fake")
    assert response.status_code == 200
    assert len(response.headers["X-Request-Id"]) == 32

def test_formatter_adds_fields_to_json():
    record = logging.makeLogRecord(
        {
            "name": "evorag.test",
            "levelname": "INFO",
            "msg": "request.completed",
            "fields": {"status": 200},
        }
    )

    output = JsonFormatter().format(record)

    payload = json.loads(output)
    assert payload["event"] == "request.completed"
    assert payload["status"] == 200

def test_formatter_works_without_fields():
    record = logging.makeLogRecord(
        {
            "name": "evorag.test",
            "msg": "hello",
        }
    )

    output = JsonFormatter().format(record)
    payload = json.loads(output)
    assert payload["event"] == "hello"

def test_middleware_reuses_incoming_request_id():
    client = TestClient(make_app())

    response = client.get("/fake", headers={"X-Request-ID": "from-rag-service"})

    assert response.headers["x-request-id"] == "from-rag-service"

def test_middleware_sums_embedding_tokens(caplog):
    caplog.set_level(logging.INFO, logger="evorag")
    client = TestClient(make_app())

    client.get("/fake")

    [record] = [r for r in caplog.records if r.getMessage() == "request.completed"]

    assert record.fields["embedding_tokens"] == 42
