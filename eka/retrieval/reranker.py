"""Cross-encoder reranking (PRD §14).

Hybrid retrieval optimises recall over a large candidate set; the reranker
optimises precision over the small set that actually reaches the LLM.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from ..config import RerankerConfig
from ..schema import RetrievedChunk


class Reranker(Protocol):
    """Interface every reranker implements.

    Attributes:
      name: Model or strategy identifier, recorded in traces.
    """

    name: str

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]:
        """Reorder candidates by relevance to the query.

        Args:
          query: Query text.
          candidates: Fused candidates to reorder.
          k: Maximum results to keep.

        Returns:
          The top `k` candidates, re-ranked from 1.
        """
        ...


def _document_text(item: RetrievedChunk, max_chars: int) -> str:
    """Render a candidate as the document side of a cross-encoder pair.

    The chunk title leads, so the model sees the path and symbol name even
    when truncation removes most of the body.

    Args:
      item: Candidate to render.
      max_chars: Truncation point, applied after the title is prepended.

    Returns:
      Title and content, truncated to `max_chars`.
    """
    chunk = item.chunk
    header = chunk.title
    return f"{header}\n{chunk.content}"[:max_chars]


class CrossEncoderReranker:
    """Scores each query/document pair jointly with a cross-encoder.

    Unlike the bi-encoder behind dense retrieval, this model sees query and
    document in one forward pass and can attend across them, which is what
    buys the precision. It costs one forward pass per candidate rather than
    one per corpus, hence its position at the end of the funnel.

    The default model is trained on MS MARCO web passages, not on code, and
    it is used unchanged -- no fine-tuning on this corpus happens anywhere.

    Attributes:
      config: Reranker configuration.
      name: Model identifier.
      model: The loaded cross-encoder.
    """

    def __init__(self, config: RerankerConfig) -> None:
        """Load the cross-encoder onto the configured device.

        Args:
          config: Reranker configuration naming the model and device.
        """
        from sentence_transformers import CrossEncoder

        device = config.device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # pragma: no cover
                device = "cpu"
        self.config = config
        self.name = config.model
        self.model = CrossEncoder(config.model, max_length=config.max_length, device=device)

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]:
        """Rescore and reorder candidates with the cross-encoder.

        Args:
          query: Query text.
          candidates: Fused candidates to reorder.
          k: Maximum results to keep.

        Returns:
          The top `k` candidates. Scores are the model's raw logits, not
          probabilities, so they are comparable within one query and not
          across queries. Each result keeps its `pre_rerank_score` and
          `pre_rerank_rank`, which is what makes a reranker regression
          visible in a trace rather than merely suspected.
        """
        if not candidates:
            return []
        pairs = [
            (query, _document_text(c, self.config.max_doc_chars)) for c in candidates
        ]
        scores = self.model.predict(
            pairs, batch_size=self.config.batch_size, show_progress_bar=False
        )
        ordered = sorted(zip(candidates, scores), key=lambda kv: -float(kv[1]))
        out: list[RetrievedChunk] = []
        for rank, (item, score) in enumerate(ordered[:k], start=1):
            out.append(
                RetrievedChunk(
                    chunk=item.chunk,
                    score=float(score),
                    retriever=f"{item.retriever}+rerank",
                    rank=rank,
                    component_scores={
                        **item.component_scores,
                        "pre_rerank_score": round(item.score, 6),
                        "pre_rerank_rank": item.rank,
                        "rerank": round(float(score), 6),
                    },
                    query=item.query,
                )
            )
        return out


class IdentityReranker:
    """No-op reranker that keeps fusion order.

    Used by the no-reranker ablation and by CI, where downloading a
    cross-encoder is not an option.

    Attributes:
      name: Always "identity".
    """

    name = "identity"

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]:
        """Return the first `k` candidates unchanged.

        Args:
          query: Query text, ignored.
          candidates: Fused candidates.
          k: Maximum results to keep.

        Returns:
          The first `k` candidates, order and scores untouched.
        """
        return list(candidates[:k])


def build_reranker(config: RerankerConfig) -> Reranker:
    """Construct the reranker a config asks for.

    Args:
      config: Reranker configuration.

    Returns:
      An `IdentityReranker` when reranking is disabled or the model is a
      sentinel name, otherwise a `CrossEncoderReranker`.
    """
    if not config.enabled or config.model in {"identity", "none"}:
        return IdentityReranker()
    return CrossEncoderReranker(config)
