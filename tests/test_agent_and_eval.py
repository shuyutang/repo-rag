"""Query planning, the iterative agent, metrics and the benchmark harness."""

from __future__ import annotations

import json

from eka.agent.planner import QueryPlanner, classify
from eka.agent.retrieval_agent import RetrievalAgent
from eka.agent.tools import ToolBox
from eka.evaluation.answer_metrics import citation_metrics, judge_agreement
from eka.evaluation.benchmark import BenchmarkRunner, RunSpec, render_ablation
from eka.evaluation.curated import curated_questions
from eka.evaluation.dataset import (
    BenchmarkQuestion,
    assign_splits,
    dataset_summary,
    save_dataset,
    validate_dataset,
)
from eka.evaluation.retrieval_metrics import aggregate, evaluate_question, relevance_grade
from eka.generation.llm import EchoClient, LLMResponse
from eka.observability.tracing import Trace
from eka.retrieval.hybrid import build_retriever
from eka.schema import Answer, Chunk, Citation, RetrievedChunk


# ---------------------------------------------------------------- planner
def test_rule_based_classification():
    """Each rule pattern routes its question to the intended category."""
    assert classify("Why was batching introduced?") == "historical"
    assert classify("If I change PagedAttention what breaks?") == "change_impact"
    assert classify("CUDA out of memory during decode, what causes it?") == "debugging"
    assert classify("Where is CacheEngine implemented?") == "code_lookup"


def test_planner_without_llm_still_plans():
    """With no LLM the planner returns a usable rule-based plan."""
    plan = QueryPlanner().plan("Where is CacheEngine.allocate_gpu_cache implemented?")
    assert plan.category == "code_lookup"
    assert "CacheEngine.allocate_gpu_cache" in plan.symbols
    assert plan.subqueries and plan.source == "rules"


def test_planner_uses_llm_json():
    """A parseable LLM response refines the plan and is marked source=llm."""
    class Planner(EchoClient):
        def complete(self, prompt, *, system=None, max_tokens=None):
            return LLMResponse(
                text=json.dumps(
                    {"category": "architecture",
                     "subqueries": ["kv cache allocation", "block size configuration"],
                     "symbols": ["CacheConfig"], "rationale": "needs both"}
                )
            )

    plan = QueryPlanner(llm=Planner()).plan("How is the KV cache sized?")
    assert plan.category == "architecture"
    assert "kv cache allocation" in plan.subqueries
    assert "CacheConfig" in plan.symbols
    assert plan.source == "llm"


# ---------------------------------------------------------------- tools/agent
def test_toolbox_dispatch(kb):
    """Each tool dispatches, and an unknown tool is reported rather than raised."""
    tools = ToolBox(kb, build_retriever(kb, sources=["bm25", "symbol"]))
    results, call = tools.dispatch("search", {"query": "allocate_gpu_cache", "k": 3})
    assert results and call.n_results == len(results)
    results, call = tools.dispatch("impact", {"symbol": "allocate_gpu_cache"})
    assert results and any(r.retriever.startswith("graph:") for r in results)
    _results, call = tools.dispatch("nonexistent", {})
    assert "unknown tool" in call.note


def test_agent_stops_when_controller_is_satisfied(config, kb):
    """The loop ends as soon as the controller declares the evidence sufficient."""
    class Controller(EchoClient):
        calls = 0

        def complete(self, prompt, *, system=None, max_tokens=None):
            Controller.calls += 1
            if "retrieval planner" in prompt:
                return LLMResponse(text=json.dumps(
                    {"category": "code_lookup", "subqueries": ["kv cache blocks"],
                     "symbols": ["CacheEngine"], "rationale": "r"}))
            return LLMResponse(text=json.dumps(
                {"sufficient": True, "reason": "evidence covers the question"}))

    agent = RetrievalAgent(kb, build_retriever(kb, sources=["bm25", "symbol"]),
                           config=config, llm=Controller())
    trace = Trace("q")
    result = agent.run("How is the KV cache sized?", trace=trace)
    assert result.evidence
    assert result.steps[-1].sufficient
    assert any(s.kind == "agent" for s in trace.steps)


def test_agent_respects_iteration_limit(config, kb):
    """A controller that never stops is bounded by max_iterations."""
    class NeverSatisfied(EchoClient):
        def complete(self, prompt, *, system=None, max_tokens=None):
            if "retrieval planner" in prompt:
                return LLMResponse(text=json.dumps(
                    {"category": "architecture", "subqueries": ["a"], "symbols": []}))
            return LLMResponse(text=json.dumps(
                {"sufficient": False, "reason": "need more",
                 "next": {"tool": "search", "args": {"query": "another query"}}}))

    config.agent.max_iterations = 3
    agent = RetrievalAgent(kb, build_retriever(kb, sources=["bm25"]),
                           config=config, llm=NeverSatisfied())
    result = agent.run("How does the worker allocate the cache?")
    assert len(result.steps) <= config.agent.max_iterations


def test_agent_does_not_repeat_a_query(config, kb):
    """Re-proposing an already-issued query ends the loop instead of looping."""
    class Repeater(EchoClient):
        def complete(self, prompt, *, system=None, max_tokens=None):
            if "retrieval planner" in prompt:
                return LLMResponse(text=json.dumps(
                    {"category": "code_lookup", "subqueries": ["kv cache"], "symbols": []}))
            return LLMResponse(text=json.dumps(
                {"sufficient": False, "reason": "again",
                 "next": {"tool": "search", "args": {"query": "kv cache"}}}))

    agent = RetrievalAgent(kb, build_retriever(kb, sources=["bm25"]),
                           config=config, llm=Repeater())
    result = agent.run("kv cache")
    assert result.steps[-1].sufficient
    assert "repeated query" in result.steps[-1].reason


# ---------------------------------------------------------------- metrics
def _question(**kwargs):
    """Build a benchmark question with defaults for the fields under test."""
    base = dict(id="q1", question="q", category="code_lookup", difficulty="single_hop")
    base.update(kwargs)
    return BenchmarkQuestion(**base)


def _hit(path, symbol=None, artifact="source"):
    """Build a throwaway retrieval result at a given path."""
    return RetrievedChunk(
        chunk=Chunk(repository="r", commit="c", path=path, artifact_type=artifact,
                    start_line=1, end_line=9, content="x", qualified_name=symbol),
        score=1.0, retriever="test",
    )


def test_relevance_grades():
    """Exact chunk and symbol hits grade 2, right-file hits 1, misses 0."""
    question = _question(relevant_files=["a.py"], relevant_symbols=["pkg.mod.Thing"])
    assert relevance_grade(_hit("a.py"), question) == 1
    assert relevance_grade(_hit("b.py", "pkg.mod.Thing"), question) == 2
    assert relevance_grade(_hit("z.py"), question) == 0


def test_recall_mrr_ndcg():
    """Recall, MRR and nDCG match hand-computed values for a known ranking."""
    question = _question(relevant_files=["a.py", "b.py"])
    results = [_hit("x.py"), _hit("a.py"), _hit("b.py")]
    metrics = evaluate_question(question, results, ks=(1, 5))
    assert metrics.recall_at[1] == 0.0
    assert metrics.recall_at[5] == 1.0
    assert metrics.mrr == 0.5
    assert 0 < metrics.ndcg_at[5] <= 1.0
    perfect = evaluate_question(question, [_hit("a.py"), _hit("b.py")], ks=(5,))
    assert perfect.ndcg_at[5] == 1.0


def test_aggregate_reports_all_metrics():
    """Aggregation emits every metric at every requested cut-off."""
    question = _question(relevant_files=["a.py"])
    summary = aggregate([evaluate_question(question, [_hit("a.py")])])
    for key in ("recall@5", "recall@10", "mrr", "ndcg@10", "hit_rate@10"):
        assert key in summary


def test_citation_metrics():
    """Validity, precision and completeness follow from citations and gold files."""
    answer = Answer(
        question="q", text="t",
        citations=[Citation(path="a.py", start_line=1, end_line=2)],
        usage={"unsupported_citations": ["[ghost.py:1-2]"]},
    )
    question = _question(relevant_files=["a.py", "b.py"])
    n, validity, precision, completeness = citation_metrics(answer, question)
    assert n == 2 and validity == 0.5
    assert precision == 1.0 and completeness == 0.5


# ---------------------------------------------------------------- dataset
def test_curated_questions_are_wellformed():
    """Every hand-written question has an id, gold evidence and a reference answer."""
    curated = curated_questions()
    assert len(curated) >= 25
    ids = [c.id for c in curated]
    assert len(ids) == len(set(ids))
    for record in curated:
        assert record.question.endswith("?")
        assert record.expected_answer
        assert record.relevant_files
        assert record.reviewed
    multi_hop = sum(1 for c in curated if c.is_multi_hop)
    assert multi_hop / len(curated) > 0.5


def test_split_assignment_is_deterministic_and_balanced():
    """Splits are stable across runs and roughly match dev_fraction."""
    records = [_question(id=f"q{i}") for i in range(400)]
    first = [r.split for r in assign_splits(records)]
    second = [r.split for r in assign_splits(records)]
    assert first == second
    dev_fraction = first.count("dev") / len(first)
    assert 0.2 < dev_fraction < 0.4


def test_dataset_validation_catches_missing_gold(kb, tmp_path):
    """A gold path absent from the index is reported as a problem."""
    records = [
        _question(id="ok", relevant_files=["minirepo/cache_engine.py"]),
        _question(id="bad", relevant_files=["does/not/exist.py"]),
        _question(id="empty"),
    ]
    problems = validate_dataset(records, kb)
    assert any("unknown file" in p for p in problems)
    assert any("no gold evidence" in p for p in problems)
    assert not any(p.startswith("ok:") for p in problems)
    save_dataset(records, tmp_path / "d.jsonl")
    assert dataset_summary(records)["n_questions"] == 3


# ---------------------------------------------------------------- benchmark
def test_benchmark_runs_end_to_end(config, kb, tmp_path):
    """A full run produces metrics, per-question rows and a saved report."""
    from eka.pipeline import Pipeline

    records = [
        _question(id="q1", relevant_files=["minirepo/cache_engine.py"],
                  question="Where is allocate_gpu_cache implemented?", split="test"),
        _question(id="q2", relevant_files=["minirepo/worker.py"], category="architecture",
                  question="How does the worker initialise the cache?", split="test"),
    ]
    dataset_path = tmp_path / "bench.jsonl"
    save_dataset(records, dataset_path)

    pipeline = Pipeline(config, kb, llm=EchoClient())
    runner = BenchmarkRunner(config, dataset_path=dataset_path, pipeline=pipeline)
    report = runner.run(RunSpec(name="unit", retriever="hybrid", reranker=False,
                                evaluate_generation=True))
    assert report.retrieval["n_questions"] == 2
    assert report.retrieval["recall@10"] > 0
    assert report.generation["n_answers"] == 2
    assert report.latency["retrieval_ms"]["p50"] >= 0
    assert report.fingerprint["dataset_version"]
    assert not report.failures

    path = runner.save(report, tmp_path / "results")
    assert path.exists()
    markdown = render_ablation([json.loads(path.read_text())])
    assert "| unit |" in markdown and "Recall@10" in markdown


def test_judge_agreement_reports_when_unlabelled(tmp_path):
    """An unlabelled audit sheet reports zero labelled rather than perfect agreement."""
    path = tmp_path / "audit.jsonl"
    path.write_text(json.dumps({"question_id": "q", "judge_correctness": 1.0,
                                "human_correctness": None}) + "\n")
    assert judge_agreement(path)["labelled"] == 0
    path.write_text(
        json.dumps({"question_id": "q", "judge_correctness": 1.0, "human_correctness": 1.0,
                    "judge_faithfulness": 0.5, "human_faithfulness": 1.0}) + "\n"
    )
    report = judge_agreement(path)
    assert report["correctness"]["exact_agreement"] == 1.0
    assert report["faithfulness"]["mae"] == 0.5
