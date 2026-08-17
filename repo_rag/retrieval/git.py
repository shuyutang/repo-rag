"""History retrieval over commit chunks (PRD M7)."""

from __future__ import annotations

import numpy as np

from ..schema import RetrievedChunk
from .base import BaseRetriever


class GitRetriever(BaseRetriever):
    """Hybrid lexical and dense search restricted to commit chunks.

    This is the retriever behind "why was this introduced?". It fuses its own
    two sources internally with a fixed RRF offset of 60, because a BM25
    score and a cosine similarity are not comparable and the caller only ever
    sees one merged commit ranking.
    """

    name = "git"

    def retrieve(
        self,
        query: str,
        k: int = 10,
        *,
        touching_path: str | None = None,
        **_: object,
    ) -> list[RetrievedChunk]:
        """Retrieve commits matching a query.

        Args:
          query: Query text.
          k: Maximum results.
          touching_path: Restrict to commits that changed a matching path,
            which is what turns "why does this file look like this?" into a
            history question rather than a text search.
          **_: Options for other retrievers, ignored.

        Returns:
          Commit chunks ranked by internal RRF over BM25 and dense hits.
          Empty when history was never ingested, or when `touching_path`
          matches nothing.
        """
        mask = self.mask_for(["commit"])
        if mask is None or not mask.any():
            return []
        if touching_path:
            touch = self.kb.row_mask(
                lambda c: c.artifact_type == "commit"
                and any(touching_path in f for f in c.files_changed)
            )
            mask = np.logical_and(mask, touch)
            if not mask.any():
                return []

        ranked: dict[str, float] = {}
        if self.kb.bm25_index is not None:
            for rank, (chunk_id, _score) in enumerate(
                self.kb.bm25_index.search(query, k * 2, allowed=mask), start=1
            ):
                ranked[chunk_id] = ranked.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        if self.kb.vector_index is not None:
            vector = self.kb.embedder.encode_queries([query])[0]
            for rank, (chunk_id, _score) in enumerate(
                self.kb.vector_index.search(vector, k * 2, allowed=mask), start=1
            ):
                ranked[chunk_id] = ranked.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        hits = sorted(ranked.items(), key=lambda kv: -kv[1])[:k]
        return self.wrap(hits, query=query)
