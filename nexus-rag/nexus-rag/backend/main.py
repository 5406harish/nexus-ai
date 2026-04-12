"""
main.py — Nexus FastAPI backend

Endpoints:
  POST /api/ingest          Ingest documents
  POST /api/search          Hybrid search
  POST /api/chat            Streaming RAG chat (SSE)
  GET  /api/index/stats     Index statistics
  DELETE /api/index/reset   Reset index
  GET  /api/health          Health check
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

load_dotenv()   # must come before other imports that read env

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from endee_client    import EndeeClient
from embeddings      import EmbeddingPipeline
from ingestion       import IngestionPipeline
from rag_pipeline    import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# App lifecycle — initialise heavy objects once at startup
# ──────────────────────────────────────────────────────────────────────

endee_client: EndeeClient
emb_pipeline: EmbeddingPipeline
ingest:       IngestionPipeline
rag:          RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global endee_client, emb_pipeline, ingest, rag
    logger.info("🚀 Nexus starting …")

    endee_client = EndeeClient()
    endee_client.ensure_index()

    emb_pipeline = EmbeddingPipeline.get()
    ingest       = IngestionPipeline(endee_client, emb_pipeline)
    rag          = RAGPipeline(endee_client, emb_pipeline)

    logger.info("✅ Nexus ready")
    yield
    logger.info("👋 Nexus shutting down")


app = FastAPI(
    title="Nexus — AI Knowledge Base API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────

class DocumentIn(BaseModel):
    text:     str
    title:    str
    category: str = "general"
    source:   str = "manual"
    author:   str = ""
    doc_id:   Optional[str] = None

class IngestRequest(BaseModel):
    documents: list[DocumentIn]

class SearchRequest(BaseModel):
    query:        str
    top_k:        int            = Field(5,   ge=1, le=50)
    filters:      Optional[dict] = None
    hybrid_alpha: float          = Field(0.7, ge=0.0, le=1.0)

class ChatMessage(BaseModel):
    role:    str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message:              str
    conversation_history: list[ChatMessage] = []
    filters:              Optional[dict]    = None
    top_k:                int               = Field(5, ge=1, le=20)
    hybrid_alpha:         float             = Field(0.7, ge=0.0, le=1.0)


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "nexus"}


# ── Ingest ────────────────────────────────────────────────────────────

@app.post("/api/ingest")
def ingest_documents(req: IngestRequest):
    """Ingest one or more documents into Endee."""
    docs = [d.model_dump() for d in req.documents]
    result = ingest.ingest_documents(docs)
    return {
        "status":         "success",
        "total_chunks":   result["total_chunks"],
        "documents_count": len(result["doc_ids"]),
        "doc_ids":        result["doc_ids"],
    }


# ── Search ────────────────────────────────────────────────────────────

@app.post("/api/search")
def search(req: SearchRequest):
    """Hybrid semantic + BM25 search."""
    results = rag.search(
        query   = req.query,
        top_k   = req.top_k,
        filters = req.filters,
    )
    hits = []
    for r in results:
        meta = r.get("meta", {})
        hits.append({
            "id":         r.get("id"),
            "similarity": round(r.get("similarity", 0), 4),
            "title":      meta.get("title", ""),
            "category":   meta.get("category", ""),
            "source":     meta.get("source", ""),
            "author":     meta.get("author", ""),
            "excerpt":    (meta.get("text", "")[:300] + "…")
                          if len(meta.get("text", "")) > 300 else meta.get("text", ""),
            "chunk_idx":  meta.get("chunk_idx", 0),
            "timestamp":  meta.get("timestamp", ""),
        })
    return {"query": req.query, "results": hits, "count": len(hits)}


# ── Chat (streaming SSE) ──────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Streaming RAG chat.
    Returns Server-Sent Events (SSE) — text/event-stream.
    """
    history = [{"role": m.role, "content": m.content} for m in req.conversation_history]

    async def event_generator():
        async for chunk in rag.chat_stream(
            message              = req.message,
            conversation_history = history,
            filters              = req.filters,
            top_k                = req.top_k,
            hybrid_alpha         = req.hybrid_alpha,
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Index management ──────────────────────────────────────────────────

@app.get("/api/index/stats")
def index_stats():
    """Return index metadata."""
    info = endee_client.describe()
    return {"index": info, "status": "ok"}


@app.delete("/api/index/reset")
def reset_index():
    """
    Drop and recreate the Endee index.
    ⚠️  Destroys all indexed data — use with caution.
    """
    endee_client.delete_index()
    endee_client.ensure_index(recreate=False)
    return {"status": "reset", "message": "Index recreated. Re-ingest your documents."}
