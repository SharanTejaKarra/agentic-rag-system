"""FastAPI application for the Agentic RAG system."""

from contextlib import asynccontextmanager
from uuid import uuid4

from pathlib import Path

from fastapi import FastAPI, HTTPException
from neo4j import GraphDatabase
from pydantic import BaseModel

from config.prompts import SYSTEM_PROMPT
from config.settings import settings
from src.graph.builder import build_graph
from src.ingestion.pipeline import run_ingestion
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Build the LangGraph pipeline once at module level
graph = build_graph()


class QueryRequest(BaseModel):
    question: str
    thread_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    confidence: str | None


class IngestRequest(BaseModel):
    directory: str
    collection_name: str = "legal_docs"


class IngestResponse(BaseModel):
    files_found: int
    documents_parsed: int
    chunks: int
    loaded: int
    graph: dict


class HealthResponse(BaseModel):
    status: str
    chroma: str
    neo4j: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify connections
    logger.info("Starting Agentic RAG service")
    yield
    # Shutdown: cleanup
    logger.info("Shutting down Agentic RAG service")


app = FastAPI(title="Agentic RAG", lifespan=lifespan)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Run a question through the LangGraph RAG pipeline."""
    thread_id = req.thread_id or uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(
            {"original_query": req.question, "messages": []},
            config=config,
        )
    except Exception as exc:
        logger.exception("Pipeline error for query: %s", req.question)
        raise HTTPException(status_code=500, detail=str(exc))

    citations = []
    for c in result.get("citations", []):
        if hasattr(c, "model_dump"):
            citations.append(c.model_dump())
        else:
            citations.append(c)

    confidence = result.get("confidence")
    if confidence and hasattr(confidence, "value"):
        confidence = confidence.value

    return QueryResponse(
        answer=result.get("synthesis", ""),
        citations=citations,
        confidence=confidence,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(req: IngestRequest):
    """Run the ingestion pipeline on a directory of documents."""
    try:
        stats = run_ingestion(req.directory, req.collection_name)
    except Exception as exc:
        logger.exception("Ingestion error for dir: %s", req.directory)
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(**stats)


@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Check ChromaDB persist directory and Neo4j connectivity."""
    chroma_status = "unknown"
    neo4j_status = "unknown"

    # Check ChromaDB
    chroma_path = Path(settings.chroma_persist_dir)
    if chroma_path.exists():
        chroma_status = "ok"
    else:
        chroma_status = f"error: persist dir not found ({settings.chroma_persist_dir})"

    # Check Neo4j
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        driver.verify_connectivity()
        driver.close()
        neo4j_status = "ok"
    except Exception as exc:
        neo4j_status = f"error: {exc}"

    overall = "healthy" if chroma_status == "ok" and neo4j_status == "ok" else "degraded"
    return HealthResponse(status=overall, chroma=chroma_status, neo4j=neo4j_status)
