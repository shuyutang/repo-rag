"""BM25 lexical retrieval."""

from __future__ import annotations

from ..schema import RetrievedChunk
from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """Lexical retrieval over the BM25 index.

    The strongest single retriever on this corpus, by a wide margin. The
    reason is the code-aware tokenizer rather than BM25 itself: splitting
    identifiers into subtokens lets a natural-language question match an
    identifier it never spells out.
    """

    name = "bm25"

    def retrieve(
        self,
        query: str,
        k: int = 20,
        *,
        artifact_types=None,
        path_prefix: str | None = None,
        **_: object,
    ) -> list[RetrievedChunk]:
        """Retrieve by BM25 score.

        Args:
          query: Query text.
          k: Maximum results.
          artifact_types: Artifact types to restrict to.
          path_prefix: Path prefix to restrict to.
          **_: Options for other retrievers, ignored.

        Returns:
          Results ranked by BM25 score, only those scoring above zero.

        Raises:
          RuntimeError: The knowledge base has no BM25 index.
        """
        if self.kb.bm25_index is None:
            raise RuntimeError("knowledge base has no BM25 index; run `rag index`")
        hits = self.kb.bm25_index.search(
            query, k, allowed=self.mask_for(artifact_types, path_prefix)
        )
        return self.wrap(hits, query=query)
