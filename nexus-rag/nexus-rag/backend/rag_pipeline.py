"""
rag_pipeline.py — RAG orchestration with Ollama (local LLM)

Pipeline:
  1. Embed user query (dense + BM25 sparse)
  2. Hybrid search in Endee
  3. Build grounded prompt with retrieved chunks
  4. Stream answer from Ollama
"""

import json
import logging
import os
from typing import AsyncGenerator, Optional

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
MAX_CONTEXT_CHUNKS = 6


class RAGPipeline:
    def __init__(self, endee_client, embedding_pipeline):
        self.db  = endee_client
        self.emb = embedding_pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        message:              str,
        conversation_history: list[dict],
        filters:              Optional[dict] = None,
        top_k:                int   = 5,
        hybrid_alpha:         float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Yields SSE-formatted data strings."""
        try:
            # 1. Search the knowledge base
            yield self._sse({"type": "search_start", "query": message})

            chunks = self._retrieve(
                query   = message,
                top_k   = min(top_k, MAX_CONTEXT_CHUNKS),
                filters = self._build_filters("", filters),
            )

            if chunks:
                yield self._sse({
                    "type":   "chunk_found",
                    "chunks": [self._chunk_preview(c) for c in chunks],
                })

            # 2. Build prompt
            context = self._format_chunks(chunks)
            messages = self._build_messages(message, conversation_history, context)

            # 3. Stream from Ollama
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model":    OLLAMA_MODEL,
                        "messages": messages,
                        "stream":   True,
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        delta = data.get("message", {}).get("content", "")
                        if delta:
                            yield self._sse({"type": "text", "delta": delta})
                        if data.get("done"):
                            break

            yield self._sse({"type": "done"})

        except Exception as exc:
            logger.exception("RAG pipeline error")
            yield self._sse({"type": "error", "message": str(exc)})

    # ------------------------------------------------------------------
    # Non-streaming search (used by /api/search endpoint)
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[dict]:
        return self._retrieve(query, top_k, self._build_filters("", filters))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, top_k: int, filters: Optional[list[dict]]) -> list[dict]:
        dense_vec, sparse_vec = self.emb.embed_for_query(query)
        results = self.db.query(
            vector         = dense_vec,
            top_k          = top_k,
            ef             = 200,
            filters        = filters,
            sparse_indices = sparse_vec["indices"] or None,
            sparse_values  = sparse_vec["values"]  or None,
        )
        return results or []

    def _build_filters(self, category: str, extra: Optional[dict]) -> Optional[list[dict]]:
        parts: list[dict] = []
        if category:
            parts.append({"category": {"$eq": category}})
        if extra:
            for k, v in extra.items():
                if v:
                    parts.append({k: {"$eq": v}})
        return parts if parts else None

    def _format_chunks(self, chunks: list[dict]) -> str:
        if not chunks:
            return ""
        lines = ["Relevant knowledge base excerpts:\n"]
        for i, c in enumerate(chunks, 1):
            meta  = c.get("meta", {})
            score = c.get("similarity", 0)
            lines.append(
                f"[{i}] {meta.get('title', 'Unknown')} "
                f"(category: {meta.get('category', '?')}, score: {score:.3f})\n"
                f"{meta.get('text', '')}\n"
            )
        return "\n".join(lines)

    def _chunk_preview(self, chunk: dict) -> dict:
        meta = chunk.get("meta", {})
        return {
            "id":         chunk.get("id"),
            "title":      meta.get("title", ""),
            "category":   meta.get("category", ""),
            "source":     meta.get("source", ""),
            "similarity": round(chunk.get("similarity", 0), 4),
            "excerpt":    (meta.get("text", "")[:200] + "…")
                          if len(meta.get("text", "")) > 200 else meta.get("text", ""),
        }

    def _build_messages(self, message: str, history: list[dict], context: str) -> list[dict]:
        system = (
            "You are Nexus, an expert AI assistant with access to a curated knowledge base. "
            "Answer questions accurately and helpfully. Use markdown formatting for clarity. "
            "When you use information from the provided context, mention the source title."
        )
        if context:
            system += f"\n\nCONTEXT FROM KNOWLEDGE BASE:\n{context}"

        messages = [{"role": "system", "content": system}]
        for turn in history[-10:]:
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        messages.append({"role": "user", "content": message})
        return messages

    @staticmethod
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"
