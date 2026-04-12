"""
endee_client.py — Endee vector database client wrapper
Handles index management, upsert, and hybrid (dense + BM25) queries.
"""

import os
import logging
from typing import Optional
from endee import Endee, Precision

logger = logging.getLogger(__name__)

INDEX_NAME  = os.getenv("ENDEE_INDEX_NAME", "nexus_knowledge_base")
BASE_URL    = os.getenv("ENDEE_BASE_URL",   "http://localhost:8080/api/v1")
AUTH_TOKEN  = os.getenv("ENDEE_AUTH_TOKEN", "")
DENSE_DIM   = 384   # all-MiniLM-L6-v2 output dimension


class EndeeClient:
    """
    Thin wrapper around the Endee Python SDK.

    Exposes:
      • ensure_index()  — create index if it doesn't exist
      • upsert()        — batch upsert vectors (dense + optional sparse)
      • query()         — hybrid nearest-neighbour search
      • delete_index()  — drop and recreate the index
      • describe()      — return index metadata / doc count
    """

    def __init__(self):
        if AUTH_TOKEN:
            self._client = Endee(AUTH_TOKEN)
            self._client.set_base_url(BASE_URL)
        else:
            self._client = Endee()
            self._client.set_base_url(BASE_URL)

        self._index = None
        logger.info("EndeeClient initialised — base_url=%s", BASE_URL)

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def ensure_index(self, recreate: bool = False) -> None:
        """Create the hybrid index (dense + BM25) if it does not exist."""
        if recreate:
            try:
                self._client.delete_index(INDEX_NAME)
                logger.info("Dropped existing index '%s'", INDEX_NAME)
            except Exception:
                pass

        try:
            self._client.create_index(
                name=INDEX_NAME,
                dimension=DENSE_DIM,
                space_type="cosine",
                precision=Precision.INT8,
                sparse_model="endee_bm25",   # enable hybrid search
            )
            logger.info("Created index '%s'", INDEX_NAME)
        except Exception as e:
            # Already exists → that's fine
            if "already exists" in str(e).lower() or "conflict" in str(e).lower():
                logger.info("Index '%s' already exists — reusing", INDEX_NAME)
            else:
                raise

        self._index = self._client.get_index(INDEX_NAME)

    def _get_index(self):
        if self._index is None:
            self.ensure_index()
        return self._index

    def delete_index(self) -> None:
        """Drop the index entirely."""
        try:
            self._client.delete_index(INDEX_NAME)
            self._index = None
            logger.info("Deleted index '%s'", INDEX_NAME)
        except Exception as e:
            logger.warning("delete_index failed: %s", e)

    def describe(self) -> dict:
        """Return index metadata from the server."""
        try:
            idx = self._get_index()
            info = self._client.describe_index(INDEX_NAME)
            return info if isinstance(info, dict) else {"name": INDEX_NAME}
        except Exception as e:
            return {"name": INDEX_NAME, "error": str(e)}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, points: list[dict]) -> int:
        """
        Upsert a list of vector points.

        Each point must have:
          id             : str
          vector         : list[float]  — 384-dim dense vector
          meta           : dict         — arbitrary metadata (returned in results)
          filter         : dict         — filterable key/value pairs
          sparse_indices : list[int]    — BM25 token IDs   (optional)
          sparse_values  : list[float]  — BM25 TF weights  (optional)
        """
        idx = self._get_index()
        # Endee SDK accepts up to 1 000 vectors per call
        BATCH = 1000
        total = 0
        for start in range(0, len(points), BATCH):
            batch = points[start : start + BATCH]
            idx.upsert(batch)
            total += len(batch)
            logger.debug("Upserted batch of %d (total=%d)", len(batch), total)
        return total

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        ef: int = 128,
        filters: Optional[list[dict]] = None,
        sparse_indices: Optional[list[int]] = None,
        sparse_values: Optional[list[float]] = None,
        include_vectors: bool = False,
    ) -> list[dict]:
        """
        Run a hybrid (dense + BM25) nearest-neighbour query.

        Returns a list of dicts:
          id         : str
          similarity : float
          meta       : dict
        """
        idx = self._get_index()

        kwargs: dict = dict(
            vector=vector,
            top_k=top_k,
            ef=ef,
            include_vectors=include_vectors,
        )

        if filters:
            kwargs["filter"] = filters

        # Hybrid search — only if sparse vectors were supplied
        if sparse_indices and sparse_values:
            kwargs["sparse_indices"] = sparse_indices
            kwargs["sparse_values"]  = sparse_values

        results = idx.query(**kwargs)
        return results if results else []

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def list_indexes(self) -> list[str]:
        try:
            return self._client.list_indexes() or []
        except Exception:
            return []
