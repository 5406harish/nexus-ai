"""
ingestion.py — Document ingestion pipeline

Handles:
  • Text chunking (sentence-boundary aware, configurable size/overlap)
  • Metadata extraction
  • Batch embedding (dense + BM25 sparse)
  • Upsert into Endee
"""

import hashlib
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))


# ──────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter (handles ., !, ?, newline-based paragraphs)."""
    # Split on sentence-ending punctuation followed by whitespace
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int   = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Sentence-boundary aware chunker.

    Builds chunks of ≈ chunk_size characters, then adds the last
    chunk_overlap characters of the previous chunk as prefix.
    Never cuts mid-sentence.
    """
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current:
            chunk_text_val = " ".join(current)
            chunks.append(chunk_text_val)
            # Overlap: keep last sentences that fit in chunk_overlap chars
            overlap_buf: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap_buf.insert(0, s)
                overlap_len += len(s)
            current     = overlap_buf
            current_len = overlap_len

        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


# ──────────────────────────────────────────────────────────────────────
# Ingestion pipeline
# ──────────────────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Orchestrates document → chunks → embeddings → Endee upsert.

    Usage:
        pipeline = IngestionPipeline(endee_client, embedding_pipeline)
        result   = pipeline.ingest_text(
            text="…", title="My Doc", category="tech", source="manual"
        )
    """

    UPSERT_BATCH = 200   # vectors per upsert call (max 1000)

    def __init__(self, endee_client, embedding_pipeline):
        self.db  = endee_client
        self.emb = embedding_pipeline

    # ------------------------------------------------------------------

    def ingest_text(
        self,
        text: str,
        title: str,
        category: str     = "general",
        source: str       = "manual",
        author: str       = "",
        doc_id: Optional[str] = None,
        extra_meta: Optional[dict] = None,
    ) -> dict:
        """
        Chunk, embed, and ingest a plain-text document.
        Returns {"chunks_ingested": N, "doc_id": "…"}
        """
        doc_id    = doc_id or str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        chunks = chunk_text(text)
        if not chunks:
            return {"chunks_ingested": 0, "doc_id": doc_id}

        logger.info(
            "Ingesting '%s' → %d chunks (doc_id=%s)", title, len(chunks), doc_id
        )

        dense_vecs, sparse_vecs = self.emb.embed_for_ingestion(chunks)

        points: list[dict] = []
        for i, (chunk, dvec, svec) in enumerate(
            zip(chunks, dense_vecs, sparse_vecs)
        ):
            chunk_id = _stable_id(doc_id, i)

            point: dict = {
                "id":     chunk_id,
                "vector": dvec.tolist(),
                "meta": {
                    "doc_id":    doc_id,
                    "title":     title,
                    "source":    source,
                    "author":    author,
                    "category":  category,
                    "chunk_idx": i,
                    "chunk_total": len(chunks),
                    "text":      chunk,
                    "timestamp": timestamp,
                    **(extra_meta or {}),
                },
                "filter": {
                    "category": category,
                    "source":   source,
                },
            }

            if svec["indices"]:
                point["sparse_indices"] = svec["indices"]
                point["sparse_values"]  = svec["values"]

            points.append(point)

        # Batch upsert
        self.db.upsert(points)

        return {"chunks_ingested": len(points), "doc_id": doc_id}

    def ingest_documents(self, documents: list[dict]) -> dict:
        """
        Ingest a batch of documents in one call.

        Each document dict:
          text     : str  (required)
          title    : str
          category : str
          source   : str
          author   : str
        """
        total_chunks = 0
        doc_ids      = []

        for doc in documents:
            result = self.ingest_text(
                text     = doc.get("text", ""),
                title    = doc.get("title", "Untitled"),
                category = doc.get("category", "general"),
                source   = doc.get("source", "batch"),
                author   = doc.get("author", ""),
                doc_id   = doc.get("doc_id"),
            )
            total_chunks += result["chunks_ingested"]
            doc_ids.append(result["doc_id"])

        return {"total_chunks": total_chunks, "doc_ids": doc_ids}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _stable_id(doc_id: str, chunk_idx: int) -> str:
    """Deterministic chunk ID so re-ingestion is idempotent."""
    raw = f"{doc_id}::{chunk_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]
