"""
embeddings.py — Dual embedding pipeline (dense + BM25 sparse)

Dense:  all-MiniLM-L6-v2  (384-dim, via sentence-transformers)
Sparse: endee/bm25         (BM25 token weights, via endee-model)
"""

import logging
import os
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DENSE_MODEL_NAME  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SPARSE_MODEL_NAME = "endee/bm25"


class EmbeddingPipeline:
    """
    Lazy-loaded dual embedding pipeline.
    Dense model is loaded on first use (heavy); sparse model is faster.
    """

    _instance: Optional["EmbeddingPipeline"] = None

    def __init__(self):
        self._dense_model  = None
        self._sparse_model = None

    # Singleton pattern — reuse across requests
    @classmethod
    def get(cls) -> "EmbeddingPipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Dense embeddings
    # ------------------------------------------------------------------

    def _load_dense(self):
        if self._dense_model is None:
            logger.info("Loading dense model '%s' …", DENSE_MODEL_NAME)
            self._dense_model = SentenceTransformer(DENSE_MODEL_NAME)
            logger.info("Dense model loaded (dim=%d)", self._dense_model.get_sentence_embedding_dimension())
        return self._dense_model

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of document texts into 384-dim float32 vectors."""
        model = self._load_dense()
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 64,
            normalize_embeddings=True,
        )
        return vecs  # shape (N, 384)

    def embed_query(self, text: str) -> list[float]:
        """Encode a single query string into a 384-dim float32 vector."""
        model = self._load_dense()
        vec = model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()

    # ------------------------------------------------------------------
    # BM25 sparse embeddings
    # ------------------------------------------------------------------

    def _load_sparse(self):
        if self._sparse_model is None:
            try:
                from endee_model import SparseModel  # type: ignore
                logger.info("Loading sparse model '%s' …", SPARSE_MODEL_NAME)
                self._sparse_model = SparseModel(model_name=SPARSE_MODEL_NAME)
                logger.info("Sparse model ready")
            except ImportError:
                logger.warning("endee-model not installed — BM25 disabled. pip install endee-model")
                self._sparse_model = "UNAVAILABLE"
        return self._sparse_model

    def sparse_embed_documents(self, texts: list[str]) -> list[dict]:
        """
        BM25-encode a list of document texts.
        Returns list of {"indices": [...], "values": [...]} dicts.
        Uses .embed() which applies full BM25 (TF × IDF, length-normalised).
        """
        model = self._load_sparse()
        if model == "UNAVAILABLE":
            return [{"indices": [], "values": []} for _ in texts]

        results = []
        BATCH = 256
        for start in range(0, len(texts), BATCH):
            batch = texts[start : start + BATCH]
            for sv in model.embed(batch, batch_size=BATCH):
                if sv is None or not sv.indices.tolist():
                    results.append({"indices": [], "values": []})
                else:
                    results.append({
                        "indices": sv.indices.tolist(),
                        "values":  [float(v) for v in sv.values.tolist()],
                    })
        return results

    def sparse_embed_query(self, text: str) -> dict:
        """
        BM25-encode a single query string.
        Uses .query_embed() which applies IDF-only (no length penalty).
        """
        model = self._load_sparse()
        if model == "UNAVAILABLE":
            return {"indices": [], "values": []}

        sv = next(model.query_embed(text))
        if sv is None or not sv.indices.tolist():
            return {"indices": [], "values": []}
        return {
            "indices": sv.indices.tolist(),
            "values":  [float(v) for v in sv.values.tolist()],
        }

    # ------------------------------------------------------------------
    # Convenience: embed texts + sparse in one call
    # ------------------------------------------------------------------

    def embed_for_ingestion(
        self, texts: list[str]
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Returns (dense_matrix, sparse_list) for a batch of texts.
        Dense: (N, 384) float32 numpy array
        Sparse: list of {"indices", "values"} dicts
        """
        dense  = self.embed_documents(texts)
        sparse = self.sparse_embed_documents(texts)
        return dense, sparse

    def embed_for_query(
        self, text: str
    ) -> tuple[list[float], dict]:
        """
        Returns (dense_vector, sparse_dict) for a single query string.
        """
        dense  = self.embed_query(text)
        sparse = self.sparse_embed_query(text)
        return dense, sparse
