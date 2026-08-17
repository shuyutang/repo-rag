"""Context construction, citation handling and the end-to-end pipeline."""

from __future__ import annotations

from eka.generation.answer_generator import (
    AnswerGenerator,
    parse_citations,
    validate_citations,
)
from eka.generation.context_builder import ContextBuilder, estimate_tokens
from eka.generation.llm import EchoClient, extract_json
from eka.pipeline import Pipeline
from eka.retrieval.hybrid import build_retriever
from eka.schema import Chunk, RetrievedChunk


def _item(path, start, end, text="body text", artifact="source"):
    """Build a throwaway retrieval result over a synthetic chunk."""
    return RetrievedChunk(
        chunk=Chunk(repository="r", commit="c", path=path, artifact_type=artifact,
                    start_line=start, end_line=end, content=text,
                    qualified_name=f"{path}:{start}"),
        score=1.0, retriever="test",
    )


def test_context_deduplicates_overlapping_chunks(config):
    """Two chunks covering overlapping lines of one file yield a single block."""
    builder = ContextBuilder(config.generation)
    context = builder.build([_item("a.py", 1, 50), _item("a.py", 20, 60), _item("b.py", 1, 5)])
    kept = [b.chunk.location for b in context.blocks]
    assert "a.py:1-50" in kept
    assert "a.py:20-60" not in kept
    assert any("overlaps" in d for d in context.dropped)


def test_context_respects_per_file_cap(config):
    """No more than per_file_chunk_cap blocks come from any one file."""
    config.generation.per_file_chunk_cap = 2
    builder = ContextBuilder(config.generation)
    items = [_item("a.py", i * 100, i * 100 + 10) for i in range(1, 6)]
    context = builder.build(items)
    assert sum(1 for b in context.blocks if b.chunk.path == "a.py") == 2


def test_context_respects_token_budget(config):
    """Assembly stops at the budget and records what it dropped."""
    builder = ContextBuilder(config.generation)
    big = "x " * 4000
    context = builder.build([_item(f"f{i}.py", 1, 10, big) for i in range(20)], budget=500)
    assert context.total_tokens <= 500
    assert context.dropped


def test_context_render_carries_provenance(config):
    """Each rendered block states the exact citation string to use."""
    builder = ContextBuilder(config.generation)
    rendered = builder.build([_item("a.py", 5, 9)]).render()
    assert "a.py" in rendered and "lines 5-9" in rendered
    assert "cite as [a.py:5-9]" in rendered


def test_citation_parsing_and_validation(config):
    """Citations parse out of the answer and split into supported and not."""
    builder = ContextBuilder(config.generation)
    context = builder.build([_item("a.py", 10, 20)])
    citations = parse_citations(
        "Allocation happens here [a.py:10-20] and maybe here [ghost.py:1-2]."
    )
    assert len(citations) == 2
    valid, invalid = validate_citations(citations, context)
    assert [c.path for c in valid] == ["a.py"]
    assert [c.path for c in invalid] == ["ghost.py"]
    assert valid[0].chunk_id


def test_citation_inside_a_retrieved_range_counts_as_supported(config):
    """A citation just inside a retrieved block's range is accepted."""
    builder = ContextBuilder(config.generation)
    context = builder.build([_item("a.py", 10, 40)])
    valid, invalid = validate_citations(parse_citations("see [a.py:15-20]"), context)
    assert valid and not invalid


def test_answer_generator_reports_unsupported_citations(config, kb):
    """An invented citation is reported in usage, not silently dropped."""
    class Fake(EchoClient):
        def complete(self, prompt, *, system=None, max_tokens=None):
            response = super().complete(prompt, system=system, max_tokens=max_tokens)
            response.text = "Claim one [minirepo/nonexistent.py:1-2]."
            return response

    generator = AnswerGenerator(config, llm=Fake())
    results = build_retriever(kb, sources=["bm25"]).retrieve(
        "allocate_gpu_cache", request=None, k=3, use_reranker=False
    )
    answer = generator.generate("where is the cache allocated?", results)
    assert answer.usage["unsupported_citations"]
    assert not answer.citations


def test_pipeline_end_to_end(config, kb):
    """Ask returns an answer, evidence and a trace with per-stage timings."""
    pipeline = Pipeline(config, kb, llm=EchoClient())
    result = pipeline.ask("Where is the KV cache allocated?", use_reranker=False)
    assert result.evidence
    assert result.trace.steps
    assert {s.kind for s in result.trace.steps} >= {"retrieval", "generation"}
    assert result.trace.total_ms > 0
    saved = pipeline.trace_store.get(result.trace.trace_id)
    assert saved and saved["question"]


def test_estimate_tokens_and_json_extraction():
    """Token estimation is positive and JSON survives fences and prose."""
    assert estimate_tokens("a" * 400) == 100
    assert extract_json('noise ```json\n{"a": 1}\n``` more') == {"a": 1}
    assert extract_json('{"a": [1,2,],}') == {"a": [1, 2]}
    assert extract_json("no json here") is None
