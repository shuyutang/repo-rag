"""Reproducible benchmark runner (PRD §21-23, §29).

A run pins the repository commit, the models, the chunking and retrieval
configuration and the dataset version into its result file, so any number in
the ablation table can be traced back to the exact configuration that produced
it.
"""

from __future__ import annotations

import hashlib
import json
import random
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import Config
from ..indexing.knowledge_base import KnowledgeBase
from ..pipeline import Pipeline
from ..schema import RetrievedChunk
from .answer_metrics import (
    AnswerJudge,
    AnswerScores,
    aggregate_answer_scores,
    citation_metrics,
)
from .dataset import BenchmarkQuestion, load_dataset
from .retrieval_metrics import (
    QuestionMetrics,
    aggregate,
    aggregate_by,
    evaluate_question,
)

RETRIEVER_SOURCES: dict[str, list[str] | None] = {
    "dense": ["dense"],
    "bm25": ["bm25"],
    "symbol": ["symbol"],
    "git": ["git"],
    "hybrid": None,          # all available sources
}


@dataclass
class RunSpec:
    """What one benchmark run measures.

    Attributes:
      name: Run name, also the result filename stem.
      retriever: A key of `RETRIEVER_SOURCES`.
      reranker: Enable the cross-encoder.
      agentic: Use iterative agentic retrieval.
      evaluate_generation: Also generate and judge answers, which needs a
        live LLM and is far slower than retrieval alone.
      limit: Score only the first N questions.
      sample: Score a deterministic random subset of N questions.
      sample_seed: Seed for that subset.
      split: "dev", "test" or "all". Reported numbers use "test".
      k: Retrieval depth handed to the metrics.
    """

    name: str
    retriever: str = "hybrid"
    reranker: bool = False
    agentic: bool = False
    evaluate_generation: bool = False
    limit: int | None = None
    sample: int | None = None
    sample_seed: int = 0
    split: str = "test"
    k: int = 20

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as a JSON-serialisable dict."""
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Everything one benchmark run produced.

    Attributes:
      spec: The run specification.
      fingerprint: Commit, models, configuration, dataset hash and seed --
        what makes the run reproducible and comparable.
      retrieval: Aggregate retrieval metrics.
      retrieval_by_category: Retrieval metrics per question category.
      retrieval_by_difficulty: Retrieval metrics per difficulty.
      generation: Aggregate answer metrics; empty for a retrieval-only run.
      latency: Per-stage latency statistics.
      per_question: Per-question rows, for drilling into a failure.
      failures: Questions that raised, with their error.
      started_at: ISO-8601 start time.
      duration_s: Total run time.
    """

    spec: RunSpec
    fingerprint: dict[str, Any]
    retrieval: dict[str, Any] = field(default_factory=dict)
    retrieval_by_category: dict[str, Any] = field(default_factory=dict)
    retrieval_by_difficulty: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    per_question: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the report as the JSON written to `results/*.json`."""
        return {
            "spec": self.spec.to_dict(),
            "fingerprint": self.fingerprint,
            "retrieval": self.retrieval,
            "retrieval_by_category": self.retrieval_by_category,
            "retrieval_by_difficulty": self.retrieval_by_difficulty,
            "generation": self.generation,
            "latency": self.latency,
            "per_question": self.per_question,
            "failures": self.failures,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
        }

    def render(self) -> str:
        """Render the report as plain text for the CLI.

        Returns:
          Retrieval, generation and latency figures, then a per-category
          breakdown of recall@10 and MRR.
        """
        lines = [f"run: {self.spec.name}", ""]
        lines.append("retrieval:")
        for key in ("recall@1", "recall@5", "recall@10", "recall@20", "mrr",
                    "ndcg@10", "hit_rate@10"):
            if key in self.retrieval:
                lines.append(f"  {key:<12} {self.retrieval[key]:.4f}")
        if self.generation:
            lines.append("generation:")
            for key, value in self.generation.items():
                lines.append(f"  {key:<24} {value}")
        if self.latency:
            lines.append("latency (ms):")
            for key, value in self.latency.items():
                lines.append(f"  {key:<24} {value}")
        lines.append("")
        lines.append("by category (recall@10 / mrr):")
        for category, values in self.retrieval_by_category.items():
            lines.append(
                f"  {category:<16} {values.get('recall@10', 0):.3f}  "
                f"{values.get('mrr', 0):.3f}  (n={values.get('n_questions')})"
            )
        return "\n".join(lines)


class BenchmarkRunner:
    """Runs the benchmark and writes reproducible reports.

    The runner drives the same `Pipeline` the CLI and API use, so a reported
    number comes from the code path the demo runs rather than from an
    evaluation-only reimplementation of it.

    Attributes:
      config: Configuration under test.
      dataset_path: Benchmark file.
      questions: Loaded questions.
      pipeline: Pipeline under test, built on first run.
      dataset_version: Short hash of the dataset file, recorded in every
        fingerprint so two runs over different datasets are never compared.
    """

    def __init__(
        self,
        config: Config,
        *,
        dataset_path: Path | None = None,
        pipeline: Pipeline | None = None,
    ) -> None:
        """Load the benchmark dataset.

        Args:
          config: Configuration under test.
          dataset_path: Benchmark file; defaults to
            `evaluation_data/benchmark.jsonl` under the config root.
          pipeline: Pipeline to reuse; one is built on first run otherwise.
        """
        self.config = config
        self.dataset_path = dataset_path or config.resolve("evaluation_data/benchmark.jsonl")
        self.questions = load_dataset(self.dataset_path)
        self.pipeline = pipeline
        self.dataset_version = _file_hash(self.dataset_path)

    # ------------------------------------------------------------------
    def _get_pipeline(self, spec: RunSpec) -> Pipeline:
        """Return the pipeline, loading the knowledge base on first use.

        Args:
          spec: The run specification, for symmetry with future per-spec
            pipelines.

        Returns:
          The pipeline, cached across runs so an ablation loads the index
          once rather than once per configuration.
        """
        if self.pipeline is None:
            kb = KnowledgeBase.load(self.config)
            self.pipeline = Pipeline(self.config, kb)
        return self.pipeline

    def run(
        self, spec: RunSpec, *, progress: Callable[[str], None] | None = None
    ) -> BenchmarkReport:
        """Execute one benchmark run.

        Args:
          spec: What to measure.
          progress: Called with a short status line as questions complete.

        Returns:
          The report. A question that raises is recorded in `failures`
          rather than aborting the run.
        """
        pipeline = self._get_pipeline(spec)
        questions = self.questions
        if spec.split and spec.split != "all":
            questions = [q for q in questions if q.split == spec.split]
        if spec.sample and spec.sample < len(questions):
            questions = random.Random(spec.sample_seed).sample(questions, spec.sample)
            questions.sort(key=lambda q: q.id)
        if spec.limit:
            questions = questions[: spec.limit]
        sources = RETRIEVER_SOURCES.get(spec.retriever, None)

        judge = None
        if spec.evaluate_generation:
            from ..generation.llm import build_llm

            judge_cfg = self.config.judge_llm or self.config.llm
            judge = AnswerJudge(build_llm(judge_cfg))

        metrics: list[QuestionMetrics] = []
        answer_scores: list[AnswerScores] = []
        per_question: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        latencies: dict[str, list[float]] = {"retrieval_ms": [], "total_ms": [],
                                             "generation_ms": []}
        started = time.time()

        # warm the models so the first question does not absorb model load time
        pipeline.retriever.retrieve("warmup query", k=1)
        if spec.reranker:
            _ = pipeline.retriever.reranker

        for i, question in enumerate(questions, start=1):
            if progress and i % 10 == 0:
                progress(f"{i}/{len(questions)} questions")
            try:
                record = self._run_question(pipeline, spec, question, judge, latencies)
            except Exception as exc:  # keep the run alive, record the failure
                failures.append({"question_id": question.id, "error": repr(exc)})
                continue
            metrics.append(record["metrics_obj"])
            if record.get("scores_obj") is not None:
                answer_scores.append(record["scores_obj"])
            per_question.append(
                {k: v for k, v in record.items() if not k.endswith("_obj")}
            )

        report = BenchmarkReport(
            spec=spec,
            fingerprint=self._fingerprint(spec),
            retrieval=aggregate(metrics),
            retrieval_by_category=aggregate_by(metrics, "category"),
            retrieval_by_difficulty=aggregate_by(metrics, "difficulty"),
            generation=aggregate_answer_scores(answer_scores) if answer_scores else {},
            latency={
                key: _latency_stats(values) for key, values in latencies.items() if values
            },
            per_question=per_question,
            failures=failures,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
            duration_s=round(time.time() - started, 1),
        )
        return report

    # ------------------------------------------------------------------
    def _run_question(
        self,
        pipeline: Pipeline,
        spec: RunSpec,
        question: BenchmarkQuestion,
        judge: AnswerJudge | None,
        latencies: dict[str, list[float]],
    ) -> dict[str, Any]:
        """Retrieve for one question, optionally answer it, and score both.

        Args:
          pipeline: Pipeline under test.
          spec: The run specification.
          question: The question with its gold labels.
          judge: Answer judge, or `None` for retrieval-only runs.
          latencies: Per-stage latency lists, appended to in place so the
            run's latency table covers every question.

        Returns:
          The per-question row: retrieval metrics, and the answer with its
          scores when generation was evaluated.
        """
        sources = RETRIEVER_SOURCES.get(spec.retriever)
        t0 = time.perf_counter()
        if spec.evaluate_generation:
            result = pipeline.ask(
                question.question,
                sources=sources,
                k=spec.k if not spec.agentic else None,
                use_reranker=spec.reranker,
                use_agent=spec.agentic,
                save_trace=False,
            )
            evidence: list[RetrievedChunk] = result.evidence
            retrieval_ms = sum(
                s.duration_ms for s in result.trace.steps if s.kind != "generation"
            )
            generation_ms = sum(
                s.duration_ms for s in result.trace.steps if s.kind == "generation"
            )
            total_ms = (time.perf_counter() - t0) * 1000
            answer = result.answer
        else:
            evidence, _agent = pipeline.retrieve(
                question.question,
                sources=sources,
                k=spec.k,
                use_reranker=spec.reranker,
                use_agent=spec.agentic,
            )
            retrieval_ms = total_ms = (time.perf_counter() - t0) * 1000
            generation_ms = 0.0
            answer = None

        latencies["retrieval_ms"].append(retrieval_ms)
        latencies["total_ms"].append(total_ms)
        if generation_ms:
            latencies["generation_ms"].append(generation_ms)

        question_metrics = evaluate_question(question, evidence)
        record: dict[str, Any] = {
            "question_id": question.id,
            "question": question.question,
            "category": question.category,
            "difficulty": question.difficulty,
            "metrics": question_metrics.to_dict(),
            "metrics_obj": question_metrics,
            "retrieved": [
                {"location": e.chunk.location, "symbol": e.chunk.qualified_name,
                 "score": round(e.score, 5), "via": e.retriever}
                for e in evidence[:10]
            ],
            "gold_files": question.relevant_files,
            "retrieval_ms": round(retrieval_ms, 1),
        }
        if answer is not None:
            record["answer"] = answer.text
            record["citations"] = [c.render() for c in answer.citations]
            if judge is not None:
                scores = judge.score(question, answer, evidence)
            else:
                n, validity, precision, completeness = citation_metrics(answer, question)
                scores = AnswerScores(
                    question_id=question.id, n_citations=n,
                    citation_validity=validity, citation_precision=precision,
                    citation_completeness=completeness,
                )
            record["scores"] = scores.to_dict()
            record["scores_obj"] = scores
        return record

    # ------------------------------------------------------------------
    def _fingerprint(self, spec: RunSpec) -> dict[str, Any]:
        """Collect everything needed to reproduce this run.

        Args:
          spec: The run specification.

        Returns:
          Config fingerprint plus index commit, dataset hash, question count
          and the platform, as stored under `fingerprint` in the result file.
        """
        kb_meta = {}
        meta_path = self.config.index_path / "meta.json"
        if meta_path.exists():
            kb_meta = json.loads(meta_path.read_text())
        return {
            **self.config.fingerprint(),
            "repository_commit": kb_meta.get("commit", ""),
            "n_chunks": kb_meta.get("n_chunks", 0),
            "dataset": _portable_path(self.dataset_path, self.config),
            "dataset_version": self.dataset_version,
            "n_questions": len(self.questions),
            "spec": spec.to_dict(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }

    def save(self, report: BenchmarkReport, directory: Path | None = None) -> Path:
        """Write a report to `<directory>/<run name>.json`.

        Args:
          report: The report to write.
          directory: Destination; defaults to `results/` under the config
            root.

        Returns:
          The path written.
        """
        directory = directory or self.config.resolve("results")
        directory.mkdir(parents=True, exist_ok=True)
        safe = report.spec.name.replace("/", "_").replace(" ", "_")
        path = directory / f"{safe}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
            fh.write("\n")  # Result files are committed; keep them diff-clean.
        return path


# ----------------------------------------------------------------------
def _latency_stats(values: Sequence[float]) -> dict[str, float]:
    """Summarise a set of latencies.

    Args:
      values: Latencies in milliseconds; must not be empty.

    Returns:
      Mean, median, 95th percentile and maximum, rounded.
    """
    ordered = sorted(values)
    n = len(ordered)
    return {
        "mean": round(sum(ordered) / n, 1),
        "p50": round(ordered[n // 2], 1),
        "p95": round(ordered[min(int(n * 0.95), n - 1)], 1),
        "max": round(ordered[-1], 1),
    }


def _portable_path(path: Path, config: Config) -> str:
    """Express a path relative to the config root, when it lives under it.

    Result files are committed, so an absolute path would bake one machine's
    directory layout into the reproducibility record and produce a spurious
    diff on every other machine. The dataset is identified by
    `dataset_version` (its content hash) regardless, so only the name matters
    here.

    Args:
      path: Path to record.
      config: Config whose root the path is expressed against.

    Returns:
      The path relative to the config root, or unchanged when it lies outside.
    """
    try:
        return str(Path(path).relative_to(config.resolve(".")))
    except ValueError:
        return str(path)


def _file_hash(path: Path) -> str:
    """Hash a file's contents.

    Args:
      path: File to hash.

    Returns:
      The first 12 hex characters of its SHA-1, or "" if it does not exist.
    """
    if not Path(path).exists():
        return ""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def load_reports(directory: Path) -> list[dict[str, Any]]:
    """Read every benchmark report in a directory.

    Args:
      directory: Directory of result files.

    Returns:
      Reports sorted by filename. Unreadable files and JSON that is not a
      report are skipped.
    """
    reports = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if "retrieval" in data and "spec" in data:
            reports.append(data)
    return reports


_ABLATION_ORDER = ["dense", "bm25", "symbol", "hybrid", "hybrid+rerank", "agentic"]


def _system_name(run_name: str) -> str:
    """Strip the generation-run prefix from a run name.

    "gen-hybrid+rerank" and "hybrid+rerank" are the same system measured two
    ways, and belong on one row of the ablation table.

    Args:
      run_name: Run name from a report.

    Returns:
      The system name.
    """
    return run_name[4:] if run_name.startswith("gen-") else run_name


def merge_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge reports into one row per system.

    Retrieval figures come from the retrieval run and generation figures from
    the generation run, which is scored on a sample of the same split --
    generation is too slow to run over every question.

    Args:
      reports: Reports loaded from `results/`.

    Returns:
      One row per system, each carrying its retrieval, generation and latency
      figures, its fingerprint and the specs that contributed.
    """
    merged: dict[str, dict[str, Any]] = {}
    for report in reports:
        name = _system_name(report["spec"]["name"])
        row = merged.setdefault(
            name, {"name": name, "retrieval": {}, "generation": {}, "latency": {},
                   "fingerprint": report.get("fingerprint", {}), "specs": []}
        )
        row["specs"].append(report["spec"])
        generation = report.get("generation") or {}
        if generation:
            row["generation"] = generation
            row["generation_n"] = report.get("retrieval", {}).get("n_questions")
        # prefer the retrieval-only run: it covers the whole split
        if not generation or not row["retrieval"]:
            row["retrieval"] = report.get("retrieval", {})
            row["latency"] = report.get("latency", {})
            row["retrieval_n"] = report.get("retrieval", {}).get("n_questions")
    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        name = row["name"]
        return (_ABLATION_ORDER.index(name) if name in _ABLATION_ORDER else 99, name)

    return sorted(merged.values(), key=sort_key)


def render_ablation(reports: list[dict[str, Any]]) -> str:
    """The PRD §23 table, built only from recorded runs."""
    rows_data = merge_reports(reports)
    header = (
        "| System | Recall@5 | Recall@10 | MRR | nDCG@10 | Answer accuracy | "
        "Faithfulness | Unsupported claims | Citation validity | Citation completeness | "
        "Retrieval p50 (ms) |"
    )
    sep = "| --- |" + " ---: |" * 10
    rows = [header, sep]
    for row in rows_data:
        retrieval = row["retrieval"]
        generation = row["generation"]
        latency = (row.get("latency") or {}).get("retrieval_ms", {})
        rows.append(
            "| {name} | {r5} | {r10} | {mrr} | {ndcg} | {acc} | {faith} | {unsup} | "
            "{cite} | {comp} | {lat} |".format(
                name=row["name"],
                r5=_fmt(retrieval.get("recall@5"), 3),
                r10=_fmt(retrieval.get("recall@10"), 3),
                mrr=_fmt(retrieval.get("mrr"), 3),
                ndcg=_fmt(retrieval.get("ndcg@10"), 3),
                acc=_fmt(generation.get("answer_accuracy"), 3),
                faith=_fmt(generation.get("faithfulness"), 3),
                unsup=_fmt(generation.get("unsupported_claim_rate"), 3),
                cite=_fmt(generation.get("citation_validity"), 3),
                comp=_fmt(generation.get("citation_completeness"), 3),
                lat=_fmt(latency.get("p50"), 0),
            )
        )
    meta = rows_data[0].get("fingerprint", {}) if rows_data else {}
    n_retrieval = rows_data[0].get("retrieval_n") if rows_data else None
    n_generation = next(
        (r.get("generation_n") for r in rows_data if r.get("generation_n")), None
    )
    footer = [
        "",
        f"Retrieval metrics: {n_retrieval} questions ({meta.get('spec', {}).get('split', 'test')} "
        f"split). Generation metrics: {n_generation} sampled questions from the same split.",
        "",
        f"repository commit `{str(meta.get('repository_commit', ''))[:10]}` · "
        f"{meta.get('n_chunks')} chunks · "
        f"embedding `{meta.get('embedding_model')}` · "
        f"reranker `{meta.get('reranker')}` · "
        f"LLM `{meta.get('llm')}` · "
        f"dataset `{Path(str(meta.get('dataset', ''))).name}@{meta.get('dataset_version')}` "
        f"({meta.get('n_questions')} questions)",
    ]
    return "\n".join(rows + footer)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if digits else f"{value:.0f}"
    return str(value)
