"""Hybrid retrieval pipeline: dense + BM25 + symbol (+ git) -> fusion -> rerank.

This is the single entry point every consumer (CLI, API, agent, benchmark)
uses, and the object the ablation study reconfigures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import Config
from ..indexing.knowledge_base import KnowledgeBase
from ..observability.tracing import Trace
from ..schema import RetrievedChunk
from .base import BaseRetriever
from .dense import DenseRetriever
from .fusion import fuse
from .git import GitRetriever
from .reranker import Reranker, build_reranker
from .sparse import BM25Retriever
from .symbol import SymbolRetriever, extract_symbols

SOURCE_NAMES = ("dense", "bm25", "symbol", "git")


@dataclass
class RetrievalRequest:
    """One retrieval call's parameters.

    Bundled into an object so the CLI, the API and the agent can pass an
    identical request through the same code path.

    Attributes:
      query: Query text.
      k: Final result count; falls back to `retrieval.final_k`.
      sources: Retriever names to use; `None` uses the configured set.
      artifact_types: Artifact types to restrict to.
      path_prefix: Path prefix to restrict to.
      use_reranker: Override the configured reranker setting.
      symbols: Explicit identifiers for the symbol retriever.
    """

    query: str
    k: int | None = None
    sources: Sequence[str] | None = None
    artifact_types: Sequence[str] | None = None
    path_prefix: str | None = None
    use_reranker: bool | None = None
    symbols: list[str] | None = None


class HybridRetriever(BaseRetriever):
    """Runs several retrievers, fuses their rankings, then reranks.

    The single entry point the CLI, API, agent and benchmark all use, and the
    object the ablation study reconfigures -- which is what makes the
    ablation an honest comparison rather than five separate code paths.

    Attributes:
      name: Always "hybrid".
      config: Configuration supplying depths, fusion and cut-offs.
      sources: Active retriever names.
    """

    name = "hybrid"

    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        sources: Sequence[str] = ("dense", "bm25", "symbol"),
        reranker: Reranker | None = None,
    ) -> None:
        """Construct the requested retrievers over one knowledge base.

        Args:
          kb: Loaded knowledge base.
          sources: Retriever names to construct. Unknown names are ignored.
          reranker: Reranker to use. When omitted, one is built from the
            config on first use, so a retrieval-only run never loads the
            cross-encoder at all.
        """
        super().__init__(kb)
        self.config: Config = kb.config
        self.sources = list(sources)
        self._retrievers: dict[str, BaseRetriever] = {}
        if "dense" in self.sources:
            self._retrievers["dense"] = DenseRetriever(kb)
        if "bm25" in self.sources:
            self._retrievers["bm25"] = BM25Retriever(kb)
        if "symbol" in self.sources:
            self._retrievers["symbol"] = SymbolRetriever(kb)
        if "git" in self.sources:
            self._retrievers["git"] = GitRetriever(kb)
        self._reranker = reranker
        self._reranker_loaded = reranker is not None

    # ------------------------------------------------------------------
    @property
    def reranker(self) -> Reranker:
        """Reranker: The reranker, built from config on first access."""
        if not self._reranker_loaded:
            self._reranker = build_reranker(self.config.reranker)
            self._reranker_loaded = True
        return self._reranker  # type: ignore[return-value]

    def source_k(self, source: str) -> int:
        """Return how many candidates one source should contribute.

        Args:
          source: Retriever name.

        Returns:
          The configured depth for that source, defaulting to `dense_k` for
          an unrecognised name. Symbol and git are configured far shallower
          than dense and BM25, because a symbol hit is either right or
          useless -- there is no long tail worth paying for.
        """
        rc = self.config.retrieval
        return {
            "dense": rc.dense_k,
            "bm25": rc.bm25_k,
            "symbol": rc.symbol_k,
            "git": rc.git_k,
        }.get(source, rc.dense_k)

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        k: int | None = None,
        *,
        trace: Trace | None = None,
        request: RetrievalRequest | None = None,
        **kwargs,
    ) -> list[RetrievedChunk]:
        """Retrieve, fuse and rerank.

        A single-source configuration skips fusion entirely, so an ablation
        of "BM25 alone" measures BM25 and not BM25 passed through a
        degenerate RRF.

        Args:
          query: Query text; ignored when `request` is given.
          k: Final result count; ignored when `request` is given.
          trace: Trace to record each stage into.
          request: Full request object, overriding `query`, `k` and `kwargs`.
          **kwargs: Fields of `RetrievalRequest`, used when `request` is not.

        Returns:
          Final results ranked best first, at most `final_k` of them.
        """
        req = request or RetrievalRequest(query=query, k=k, **kwargs)
        rc = self.config.retrieval
        final_k = req.k or rc.final_k
        sources = list(req.sources or self.sources)

        result_sets: dict[str, list[RetrievedChunk]] = {}
        for source in sources:
            retriever = self._retrievers.get(source)
            if retriever is None:
                continue
            step = None
            if trace:
                ctx = trace.step(f"{source} search", kind="retrieval", query=req.query)
                step = ctx.__enter__()
            results = retriever.retrieve(
                req.query,
                self.source_k(source),
                artifact_types=req.artifact_types,
                path_prefix=req.path_prefix,
                symbols=req.symbols,
            )
            if trace and step is not None:
                trace.record_results(step, results)
                ctx.__exit__(None, None, None)
            result_sets[source] = results

        # single-source configurations skip fusion entirely (clean ablations)
        if len(result_sets) == 1:
            only = next(iter(result_sets.values()))
            candidates = only[: max(rc.candidate_k, final_k)]
        else:
            with trace.step("fusion", kind="fusion", method=rc.fusion) if trace else _null() as step:
                candidates = fuse(
                    result_sets,
                    method=rc.fusion,
                    k=max(rc.candidate_k, final_k),
                    rrf_k=rc.rrf_k,
                    weights=rc.fusion_weights,
                )
                if trace and step is not None:
                    trace.record_results(step, candidates)

        use_rerank = (
            self.config.reranker.enabled if req.use_reranker is None else req.use_reranker
        )
        if use_rerank and candidates:
            with trace.step(
                "rerank", kind="rerank", model=self.config.reranker.model,
                n_candidates=len(candidates),
            ) if trace else _null() as step:
                candidates = self.reranker.rerank(req.query, candidates, final_k)
                if trace and step is not None:
                    trace.record_results(step, candidates)
        else:
            candidates = candidates[:final_k]
            for rank, item in enumerate(candidates, start=1):
                item.rank = rank
        return candidates

    # ------------------------------------------------------------------
    def diagnostics(self, query: str) -> dict:
        """Report what each source retrieved and how much they agree.

        Backs `rag diagnose`. Agreement between sources is the quickest read
        on whether fusion has anything to work with on a given query.

        Args:
          query: Query text.

        Returns:
          The detected symbols, per-source result counts, pairwise overlap
          counts, and each source's top five results.
        """
        sets: dict[str, list[RetrievedChunk]] = {}
        for source, retriever in self._retrievers.items():
            sets[source] = retriever.retrieve(query, self.source_k(source))
        ids = {s: {r.chunk_id for r in v} for s, v in sets.items()}
        overlap = {
            f"{a}&{b}": len(ids[a] & ids[b])
            for i, a in enumerate(sorted(ids))
            for b in sorted(ids)[i + 1 :]
        }
        return {
            "query": query,
            "symbols_detected": extract_symbols(query),
            "counts": {s: len(v) for s, v in sets.items()},
            "overlap": overlap,
            "top": {
                s: [
                    {"location": r.chunk.location, "score": round(r.score, 4),
                     "symbol": r.chunk.qualified_name}
                    for r in v[:5]
                ]
                for s, v in sets.items()
            },
        }


class _null:
    """Null context manager used in place of a trace step when untraced."""

    def __enter__(self):
        """Return `None` in place of a trace step."""
        return None

    def __exit__(self, *exc):
        """Propagate any exception unchanged."""
        return False


def build_retriever(
    kb: KnowledgeBase,
    *,
    sources: Sequence[str] | None = None,
    reranker: Reranker | None = None,
) -> HybridRetriever:
    """Build a hybrid retriever with sensible sources for a knowledge base.

    Args:
      kb: Loaded knowledge base.
      sources: Explicit retriever names, bypassing the default choice.
      reranker: Reranker to use; built lazily from config when omitted.

    Returns:
      The retriever. By default it uses dense, BM25 and symbol retrieval, and
      adds git only when history was actually ingested -- an empty commit
      corpus would otherwise contribute an empty list to every fusion.
    """
    if sources is None:
        sources = ["dense", "bm25", "symbol"]
        if kb.store.stats().get("commit"):
            sources.append("git")
    return HybridRetriever(kb, sources=sources, reranker=reranker)
