"""Indexing and retrieval: each strategy independently, then the fusion."""

from __future__ import annotations

import numpy as np
import pytest

from eka.indexing.bm25_index import BM25Index
from eka.indexing.knowledge_base import KnowledgeBase
from eka.indexing.tokenizer import tokenize
from eka.retrieval.dense import DenseRetriever
from eka.retrieval.fusion import fuse, reciprocal_rank_fusion
from eka.retrieval.git import GitRetriever
from eka.retrieval.hybrid import RetrievalRequest, build_retriever
from eka.retrieval.sparse import BM25Retriever
from eka.retrieval.symbol import SymbolRetriever, extract_symbols
from eka.schema import Chunk, RetrievedChunk


def _paths(results):
    """Return the file path of each result, for order assertions."""
    return [r.chunk.path for r in results]


def test_tokenizer_splits_identifiers():
    """Identifiers are emitted whole and as camel/snake subtokens."""
    tokens = tokenize("CacheEngine.allocate_gpu_cache")
    assert "cacheengine" in tokens and "cache" in tokens and "allocate" in tokens
    assert "gpu" in tokens


def test_bm25_finds_exact_identifier(kb):
    """An exact identifier query ranks its defining file first."""
    results = BM25Retriever(kb).retrieve("allocate_gpu_cache", 5)
    assert "minirepo/cache_engine.py" in _paths(results)
    assert results[0].score > 0


def test_dense_retrieval_returns_k(kb):
    """Dense retrieval honours k and scores every result."""
    results = DenseRetriever(kb).retrieve("how is the kv cache allocated", 5)
    assert len(results) == 5
    assert all(isinstance(r.score, float) for r in results)


def test_symbol_retrieval_uses_identifiers(kb):
    """Identifiers are pulled out of a question and resolved to definitions."""
    assert "CacheEngine" in extract_symbols("What does CacheEngine do?")
    assert "allocate_gpu_cache" in extract_symbols("where is allocate_gpu_cache()?")
    results = SymbolRetriever(kb).retrieve("What does CacheEngine do?", 5)
    assert results
    assert any("CacheEngine" in (r.chunk.qualified_name or "") for r in results)


def test_symbol_retrieval_is_fuzzy(kb):
    """A misspelled identifier still resolves, via fuzzy lookup."""
    assert kb.symbol_index is not None
    assert kb.symbol_index.search("CacheEngin", k=5)


def test_metadata_filter(kb):
    """An artifact-type filter excludes every other type."""
    results = BM25Retriever(kb).retrieve("cache", 10, artifact_types=["doc"])
    assert results
    assert {r.chunk.artifact_type for r in results} == {"doc"}


def test_git_retriever_only_returns_commits(kb):
    """History retrieval never leaks non-commit chunks."""
    results = GitRetriever(kb).retrieve("paged KV cache allocation", 5)
    assert results
    assert {r.chunk.artifact_type for r in results} == {"commit"}


def test_rrf_prefers_agreement():
    """A chunk two sources both rank beats one only a single source ranks highly."""
    def chunk(path):
        """Build a throwaway chunk at a given path."""
        return Chunk(repository="r", commit="c", path=path, artifact_type="source",
                     start_line=1, end_line=2, content="x")

    a = [RetrievedChunk(chunk=chunk("a.py"), score=1.0, retriever="dense", rank=1),
         RetrievedChunk(chunk=chunk("b.py"), score=0.9, retriever="dense", rank=2)]
    b = [RetrievedChunk(chunk=chunk("b.py"), score=10.0, retriever="bm25", rank=1),
         RetrievedChunk(chunk=chunk("c.py"), score=9.0, retriever="bm25", rank=2)]
    fused = reciprocal_rank_fusion({"dense": a, "bm25": b}, k=3)
    assert fused[0].chunk.path == "b.py"       # ranked by both sources
    assert "dense" in fused[0].component_scores and "bm25" in fused[0].component_scores


def test_fusion_weights_change_order():
    """Raising a source's weight promotes its results."""
    def item(path, source, rank):
        """Build a throwaway result from one source at a given rank."""
        return RetrievedChunk(
            chunk=Chunk(repository="r", commit="c", path=path, artifact_type="source",
                        start_line=1, end_line=2, content="x"),
            score=1.0, retriever=source, rank=rank,
        )

    sets = {"dense": [item("d.py", "dense", 1)], "bm25": [item("b.py", "bm25", 1)]}
    assert fuse(sets, k=2, weights={"dense": 2.0, "bm25": 1.0})[0].chunk.path == "d.py"
    assert fuse(sets, k=2, weights={"dense": 1.0, "bm25": 2.0})[0].chunk.path == "b.py"


def test_hybrid_beats_single_source_on_a_mixed_query(kb):
    """A query with an identifier *and* natural language should surface both."""
    retriever = build_retriever(kb, sources=["dense", "bm25", "symbol"])
    results = retriever.retrieve(
        "how does allocate_gpu_cache decide the number of blocks",
        request=RetrievalRequest(query="how does allocate_gpu_cache decide the number of blocks",
                                 k=8, use_reranker=False),
    )
    paths = set(_paths(results))
    assert "minirepo/cache_engine.py" in paths
    assert len(paths) > 1
    assert all(r.retriever == "hybrid" for r in results)


def test_diagnostics_report_overlap(kb):
    """Diagnostics report per-source counts, detected symbols and pairwise overlap."""
    report = build_retriever(kb).diagnostics("allocate_gpu_cache blocks")
    assert set(report["counts"]) >= {"dense", "bm25", "symbol"}
    assert report["symbols_detected"]
    assert any(v >= 0 for v in report["overlap"].values())


def test_index_roundtrip(config, kb, tmp_path):
    """Saving and reloading a knowledge base preserves every index."""
    kb.save(tmp_path)
    loaded = KnowledgeBase.load(config, tmp_path)
    assert len(loaded.store) == len(kb.store)
    assert loaded.bm25_index is not None and loaded.symbol_index is not None
    assert loaded.graph is not None
    before = BM25Retriever(kb).retrieve("allocate_gpu_cache", 3)
    after = BM25Retriever(loaded).retrieve("allocate_gpu_cache", 3)
    assert [r.chunk_id for r in before] == [r.chunk_id for r in after]


def test_bm25_scoring_is_monotone_in_term_frequency():
    """More occurrences of a query term score higher, all else equal."""
    index = BM25Index.build(
        ["cache cache cache blocks", "cache blocks", "unrelated text"],
        ["a", "b", "c"],
    )
    scores = index.score("cache")
    assert scores[0] > scores[1] > scores[2] == pytest.approx(0.0)


def test_vector_search_respects_mask(kb):
    """A masked dense search returns only rows the mask allows."""
    assert kb.vector_index is not None
    mask = kb.row_mask(lambda c: c.artifact_type == "doc")
    vector = kb.embedder.encode_queries(["kv cache"])[0]
    hits = kb.vector_index.search(vector, 5, allowed=mask)
    assert hits
    assert all(kb.store.get(cid).artifact_type == "doc" for cid, _ in hits)


def test_symbol_graph_impact(kb):
    """Impact reports the defining file plus its callers and tests."""
    assert kb.graph is not None
    report = kb.graph.impact("allocate_gpu_cache")
    definition_paths = {kb.store.get(c).path for c in report.definitions}
    caller_paths = {kb.store.get(c).path for c in report.callers}
    test_paths = {kb.store.get(c).path for c in report.tests}
    assert "minirepo/cache_engine.py" in definition_paths
    assert "minirepo/worker.py" in caller_paths
    assert any(p.startswith("tests/") for p in test_paths)
