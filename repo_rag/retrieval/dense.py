"""Dense (embedding) retrieval."""

from __future__ import annotations

from ..schema import RetrievedChunk
from .base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Embedding retrieval: encode the query, search the vector index.

    Strong on paraphrase and on questions that share no vocabulary with the
    code, and measurably weaker than BM25 everywhere else on this corpus,
    because a wordpiece tokenizer shreds the identifiers that carry the
    meaning.
    """

    name = "dense"

    def retrieve(
        self,
        query: str,
        k: int = 20,
        *,
        artifact_types=None,
        path_prefix: str | None = None,
        **_: object,
    ) -> list[RetrievedChunk]:
        """Retrieve by embedding similarity.

        Args:
          query: Query text.
          k: Maximum results.
          artifact_types: Artifact types to restrict to.
          path_prefix: Path prefix to restrict to.
          **_: Options for other retrievers, ignored.

        Returns:
          Results ranked by cosine similarity.

        Raises:
          RuntimeError: The knowledge base has no vector index.
        """
        if self.kb.vector_index is None:
            raise RuntimeError("knowledge base has no vector index; run `rag index`")
        vector = self.kb.embedder.encode_queries([query])[0]
        hits = self.kb.vector_index.search(
            vector, k, allowed=self.mask_for(artifact_types, path_prefix)
        )
        return self.wrap(hits, query=query)
