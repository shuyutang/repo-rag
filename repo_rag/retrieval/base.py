"""Retriever interface (PRD §19).

Every retrieval strategy implements the same protocol, so strategies can be
swapped, composed and ablated without touching the rest of the system.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from ..indexing.knowledge_base import KnowledgeBase
from ..schema import Chunk, RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    """The one interface every retrieval strategy implements.

    Keeping this surface to a single method is what lets the ablation swap
    strategies without touching anything downstream of retrieval.

    Attributes:
      name: Short identifier recorded on every result, e.g. "bm25".
    """

    name: str

    def retrieve(self, query: str, k: int, **kwargs) -> list[RetrievedChunk]:
        """Retrieve chunks for a query.

        Args:
          query: Query text.
          k: Maximum results.
          **kwargs: Strategy-specific options, ignored where unsupported.

        Returns:
          Results ranked best first, at most `k` of them.
        """
        ...


class BaseRetriever:
    """Shared plumbing: knowledge base access, filters and result wrapping.

    Attributes:
      name: Short identifier recorded on every result.
      kb: Knowledge base the retriever reads from.
    """

    name = "base"

    def __init__(self, kb: KnowledgeBase) -> None:
        """Bind the retriever to a knowledge base.

        Args:
          kb: Loaded knowledge base.
        """
        self.kb = kb
        self._mask_cache: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    def chunk(self, chunk_id: str) -> Chunk | None:
        """Resolve a chunk id against the store, or `None` if it is absent."""
        return self.kb.store.get(chunk_id)

    def mask_for(
        self,
        artifact_types: Iterable[str] | None = None,
        path_prefix: str | None = None,
    ) -> np.ndarray | None:
        """Build a metadata filter as a boolean row mask.

        Masks are cached per retriever instance, because the same filter is
        typically reused across a whole benchmark run and each one costs a
        full pass over the corpus.

        Args:
          artifact_types: Artifact types to keep; `None` keeps all.
          path_prefix: Keep only chunks whose path starts with this.

        Returns:
          A boolean array over index rows, or `None` when no filter was
          requested -- which callers treat as "search everything" and is
          faster than an all-True mask.
        """
        if not artifact_types and not path_prefix:
            return None
        key = (tuple(sorted(artifact_types)) if artifact_types else None, path_prefix)
        if key not in self._mask_cache:
            allowed = set(artifact_types) if artifact_types else None
            self._mask_cache[key] = self.kb.row_mask(
                lambda c: (allowed is None or c.artifact_type in allowed)
                and (path_prefix is None or c.path.startswith(path_prefix))
            )
        return self._mask_cache[key]

    def wrap(
        self,
        hits: list[tuple[str, float]],
        *,
        query: str,
        component: str | None = None,
    ) -> list[RetrievedChunk]:
        """Turn raw `(chunk_id, score)` hits into ranked results.

        Args:
          hits: Scored chunk ids, best first.
          query: Query that produced them, recorded for debugging.
          component: Key to record the score under in `component_scores`;
            defaults to the retriever name.

        Returns:
          Results with 1-based ranks. Ids the store does not know are
          dropped, which can only happen if an index outlived its chunk file.
        """
        out: list[RetrievedChunk] = []
        for rank, (chunk_id, score) in enumerate(hits, start=1):
            chunk = self.chunk(chunk_id)
            if chunk is None:
                continue
            out.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    retriever=self.name,
                    rank=rank,
                    component_scores={component or self.name: float(score)},
                    query=query,
                )
            )
        return out

    def retrieve(self, query: str, k: int, **kwargs) -> list[RetrievedChunk]:
        """Retrieve chunks for a query.

        Args:
          query: Query text.
          k: Maximum results.
          **kwargs: Strategy-specific options.

        Returns:
          Results ranked best first.

        Raises:
          NotImplementedError: Always; subclasses must override this.
        """
        raise NotImplementedError
