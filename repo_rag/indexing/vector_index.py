"""Dense vector index (PRD §10).

Flat inner-product search over normalised embeddings.  For a repository the
size of vLLM (~10^5 chunks x 384 dims = ~150 MB) an exact numpy matmul answers
in tens of milliseconds, which keeps the index exactly reproducible — no ANN
graph parameters, no training step.  FAISS is used automatically when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorIndex:
    """Flat inner-product search over normalised embeddings.

    Exact rather than approximate: for a repository the size of vLLM
    (~10^5 chunks x 384 dims, ~150 MB) a numpy matmul answers in tens of
    milliseconds, and an exact index has no graph parameters and no training
    step to reproduce. FAISS is used automatically when installed, but only
    for unfiltered queries -- a masked search falls back to numpy, which is
    still exact.

    Attributes:
      vectors: Row-major float32 embeddings, one row per chunk.
      chunk_ids: Chunk id per row.
      model: Embedding model id, recorded so a stale index is detectable.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        chunk_ids: list[str],
        *,
        model: str = "",
        use_faiss: bool = True,
    ) -> None:
        """Store the vectors and optionally build a FAISS index over them.

        Args:
          vectors: Embeddings, one row per chunk. Assumed L2-normalised, so
            inner product is cosine similarity.
          chunk_ids: Chunk id per row.
          model: Embedding model id, for the index metadata.
          use_faiss: Use FAISS when it is importable.

        Raises:
          ValueError: `vectors` and `chunk_ids` disagree in length.
        """
        if vectors.shape[0] != len(chunk_ids):
            raise ValueError("vectors and chunk_ids must have the same length")
        self.vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.chunk_ids = chunk_ids
        self.model = model
        self._faiss = None
        if use_faiss and len(chunk_ids):
            self._faiss = self._try_faiss()

    def _try_faiss(self):
        """Build a FAISS flat inner-product index if FAISS is available.

        Returns:
          The FAISS index, or `None` when FAISS is missing or fails to
          initialise. FAISS is an optional accelerator, never a requirement.
        """
        try:
            import faiss  # type: ignore
        except Exception:
            return None
        index = faiss.IndexFlatIP(self.vectors.shape[1])
        index.add(self.vectors)
        return index

    # ------------------------------------------------------------------
    @property
    def dimension(self) -> int:
        """int: Embedding width, or 0 for an empty index."""
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def __len__(self) -> int:
        """Return the number of indexed vectors."""
        return len(self.chunk_ids)

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        *,
        allowed: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        """Return the top-k rows by inner product.

        Args:
          query_vector: Query embedding, normalised to match the corpus.
          k: Maximum results.
          allowed: Boolean mask over rows. A masked search bypasses FAISS and
            scores with numpy, since masking a FAISS index would need a
            separate ID selector per query.

        Returns:
          `(chunk_id, score)` pairs, highest first.
        """
        if len(self) == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if self._faiss is not None and allowed is None:
            scores, idx = self._faiss.search(q.reshape(1, -1), min(k, len(self)))
            return [
                (self.chunk_ids[int(i)], float(s))
                for i, s in zip(idx[0], scores[0])
                if i >= 0
            ]
        scores = self.vectors @ q
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        k = min(k, int(np.sum(allowed)) if allowed is not None else len(self))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunk_ids[int(i)], float(scores[int(i)])) for i in top
                if np.isfinite(scores[int(i)])]

    # ------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        """Write the index to a directory.

        Args:
          directory: Destination, created if absent. Written as
            `embeddings.npy` and `vector_meta.json`. The FAISS index is not
            persisted; it is rebuilt on load, which takes about a second.
        """
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.vectors)
        with open(directory / "vector_meta.json", "w", encoding="utf-8") as fh:
            json.dump({"chunk_ids": self.chunk_ids, "model": self.model}, fh)

    @classmethod
    def load(cls, directory: Path, *, use_faiss: bool = True) -> "VectorIndex":
        """Read an index back from a directory written by `save`.

        Args:
          directory: Directory holding the index files.
          use_faiss: Use FAISS when it is importable.

        Returns:
          The loaded index.
        """
        vectors = np.load(directory / "embeddings.npy")
        with open(directory / "vector_meta.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return cls(
            vectors, meta["chunk_ids"], model=meta.get("model", ""), use_faiss=use_faiss
        )
