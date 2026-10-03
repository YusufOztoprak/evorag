from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from evorag.adapters.http.middleware import log_requests
from evorag.adapters.http.routes import router
from evorag.adapters.openai_embeddings import OpenAIEmbeddingProvider
from evorag.adapters.pgvector_store import PgVectorChunkStore
from evorag.config import Settings
from evorag.ingestion.ingest_document import IngestDocument
from evorag.observability.logging import configure_logging
from evorag.retrieval.search_chunks import SearchChunks


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    configure_logging()

    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    embedder = OpenAIEmbeddingProvider(
        openai_client, settings.embedding_model, settings.embedding_dimensions
    )
    store = PgVectorChunkStore(session_factory)

    app.state.ingest_document = IngestDocument(embedder=embedder, writer=store)
    app.state.search_chunks = SearchChunks(
        embedder=embedder, searcher=store, min_score=settings.min_score
    )

    yield

    await openai_client.close()
    await engine.dispose()


app = FastAPI(title="evorag", lifespan=lifespan)
app.include_router(router)
app.middleware("http")(log_requests)