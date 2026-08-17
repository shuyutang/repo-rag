"""Command line interface (PRD §26).

Example:
  rag ingest ./data/vllm
  rag index
  rag ask "Where is KV cache allocated?"
  rag benchmark --retriever hybrid --reranker
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import REPO_ROOT, Config
from .indexing.knowledge_base import KnowledgeBase
from .ingestion.scanner import RepositoryScanner, read_chunks, write_chunks

app = typer.Typer(add_completion=False, help="Engineering Knowledge Agent")
dataset_app = typer.Typer(help="Evaluation dataset commands")
app.add_typer(dataset_app, name="dataset")
console = Console()

CONFIG_OPTION = typer.Option(None, "--config", "-c", help="path to a YAML config")


def _load_config(config: Optional[str], **overrides) -> Config:
    """Load a config file, or the default config when none is named."""
    path = Path(config) if config else REPO_ROOT / "configs" / "default.yaml"
    cfg = Config.load(path if Path(path).exists() else None)
    for key, value in overrides.items():
        if value is not None:
            parts = key.split(".")
            target = cfg
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], value)
    return cfg


# ----------------------------------------------------------------------
@app.command()
def ingest(
    repo: Optional[str] = typer.Argument(None, help="path to the repository checkout"),
    config: Optional[str] = CONFIG_OPTION,
    no_git: bool = typer.Option(False, "--no-git", help="skip commit ingestion"),
    max_commits: Optional[int] = typer.Option(None, help="override git history depth"),
) -> None:
    """Parse a repository into structure-aware chunks."""
    cfg = _load_config(config, **{"ingestion.git_max_commits": max_commits})
    repo_path = Path(repo).resolve() if repo else cfg.repo_dir
    scanner = RepositoryScanner(cfg)
    start = time.time()
    chunks, stats = scanner.scan(
        repo_path, include_git=not no_git, progress=lambda m: console.print(f"  {m}", style="dim")
    )
    out = cfg.index_path / "chunks.jsonl"
    write_chunks(chunks, out)
    (cfg.index_path / "ingest_stats.json").write_text(json.dumps(stats.to_dict(), indent=2))

    table = Table(title=f"ingested {repo_path.name} @ {stats.commit[:10]}")
    table.add_column("artifact")
    table.add_column("chunks", justify="right")
    for key, value in sorted(stats.by_artifact.items(), key=lambda kv: -kv[1]):
        table.add_row(key, str(value))
    table.add_row("[bold]total", f"[bold]{stats.chunks}")
    console.print(table)
    console.print(
        f"symbol types: {stats.by_symbol_type} | files: {stats.files_scanned} "
        f"| {time.time() - start:.1f}s -> {out}"
    )


@app.command()
def index(
    config: Optional[str] = CONFIG_OPTION,
    no_embeddings: bool = typer.Option(False, "--no-embeddings", help="lexical indexes only"),
    embedding_model: Optional[str] = typer.Option(None, help="override the embedding model"),
) -> None:
    """Build vector, BM25, symbol and graph indexes over ingested chunks."""
    cfg = _load_config(config, **{"embedding.model": embedding_model})
    chunks_file = cfg.index_path / "chunks.jsonl"
    if not chunks_file.exists():
        raise typer.BadParameter(f"no chunks at {chunks_file}; run `rag ingest` first")
    chunks = read_chunks(chunks_file)
    kb = KnowledgeBase.build(
        cfg, chunks,
        progress=lambda m: console.print(f"  {m}", style="dim"),
        with_embeddings=not no_embeddings,
    )
    path = kb.save()
    console.print(json.dumps(kb.meta, indent=2))
    console.print(f"[green]index written to {path}")


# ----------------------------------------------------------------------
def _pipeline(cfg: Config):
    """Build a pipeline from a config, loading its knowledge base."""
    from .pipeline import Pipeline

    return Pipeline.load(cfg)


@app.command()
def ask(
    question: str,
    config: Optional[str] = CONFIG_OPTION,
    retriever: str = typer.Option("hybrid", help="dense|bm25|symbol|hybrid"),
    reranker: Optional[bool] = typer.Option(None, "--reranker/--no-reranker"),
    agentic: bool = typer.Option(False, "--agentic", help="iterative retrieval"),
    k: Optional[int] = typer.Option(None, help="evidence chunks handed to the LLM"),
    show_trace: bool = typer.Option(False, "--trace", help="print the retrieval trace"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Answer a question with citations."""
    cfg = _load_config(config)
    if reranker is not None:
        cfg.reranker.enabled = reranker
    cfg.agent.enabled = agentic
    pipeline = _pipeline(cfg)
    sources = None if retriever == "hybrid" else [retriever]
    result = pipeline.ask(
        question, sources=sources, k=k, use_reranker=reranker, use_agent=agentic
    )
    if as_json:
        console.print_json(json.dumps(result.to_dict(), default=str))
        return
    console.rule("ANSWER")
    # markup=False: citations look like rich markup tags and would be swallowed
    console.print(result.answer.text, markup=False, highlight=False)
    console.rule("EVIDENCE")
    table = Table(show_header=True)
    table.add_column("#", justify="right")
    table.add_column("location")
    table.add_column("symbol")
    table.add_column("score", justify="right")
    table.add_column("via")
    for i, item in enumerate(result.evidence, start=1):
        table.add_row(
            str(i), item.chunk.location, (item.chunk.qualified_name or "")[:60],
            f"{item.score:.4f}", item.retriever,
        )
    console.print(table)
    if show_trace:
        console.rule("TRACE")
        console.print(result.trace.render(), markup=False, highlight=False)
    console.print(
        f"[dim]trace {result.trace.trace_id} | {result.trace.total_ms:.0f} ms | "
        f"{result.trace.usage['prompt_tokens']}+{result.trace.usage['completion_tokens']} tokens"
    )


@app.command()
def search(
    query: str,
    config: Optional[str] = CONFIG_OPTION,
    retriever: str = typer.Option("hybrid", help="dense|bm25|symbol|git|hybrid"),
    k: int = typer.Option(10),
    reranker: Optional[bool] = typer.Option(None, "--reranker/--no-reranker"),
    show: bool = typer.Option(False, "--show", help="print chunk contents"),
) -> None:
    """Retrieval only — no LLM call."""
    cfg = _load_config(config)
    if reranker is not None:
        cfg.reranker.enabled = reranker
    kb = KnowledgeBase.load(cfg)
    from .retrieval.hybrid import RetrievalRequest, build_retriever

    sources = None if retriever == "hybrid" else [retriever]
    hybrid = build_retriever(kb, sources=sources)
    start = time.perf_counter()
    results = hybrid.retrieve(
        query, request=RetrievalRequest(query=query, k=k, use_reranker=reranker)
    )
    elapsed = (time.perf_counter() - start) * 1000
    table = Table(title=f"{retriever} · {elapsed:.0f} ms")
    table.add_column("#", justify="right")
    table.add_column("location")
    table.add_column("symbol")
    table.add_column("type")
    table.add_column("score", justify="right")
    for i, item in enumerate(results, start=1):
        table.add_row(
            str(i), item.chunk.location, (item.chunk.qualified_name or "")[:60],
            item.chunk.artifact_type, f"{item.score:.4f}",
        )
    console.print(table)
    if show:
        for i, item in enumerate(results, start=1):
            console.rule(f"{i}. {item.chunk.location}")
            console.print(item.chunk.content[:1500], markup=False, highlight=False)


@app.command()
def diagnose(query: str, config: Optional[str] = CONFIG_OPTION) -> None:
    """Per-retriever candidates and overlap for one query (PRD M3)."""
    cfg = _load_config(config)
    kb = KnowledgeBase.load(cfg)
    from .retrieval.hybrid import build_retriever

    console.print_json(json.dumps(build_retriever(kb).diagnostics(query), indent=2))


@app.command()
def impact(symbol: str, config: Optional[str] = CONFIG_OPTION, k: int = 15) -> None:
    """Callers, tests and importers of a symbol (PRD M8)."""
    cfg = _load_config(config)
    kb = KnowledgeBase.load(cfg, with_vectors=False)
    if kb.graph is None:
        raise typer.BadParameter("index has no symbol graph")
    report = kb.graph.impact(symbol, limit=k)
    for label in ("definitions", "callers", "tests"):
        table = Table(title=f"{label} of {symbol}")
        table.add_column("location")
        table.add_column("symbol")
        for chunk_id in getattr(report, label):
            chunk = kb.store.get(chunk_id)
            if chunk:
                table.add_row(chunk.location, chunk.qualified_name or "")
        console.print(table)
    console.print(f"importers: {', '.join(report.importers[:20]) or '(none)'}")


@app.command()
def stats(config: Optional[str] = CONFIG_OPTION) -> None:
    """Index statistics."""
    cfg = _load_config(config)
    kb = KnowledgeBase.load(cfg, with_vectors=False)
    console.print_json(json.dumps(kb.meta, indent=2))


# ----------------------------------------------------------------------
@app.command()
def benchmark(
    config: Optional[str] = CONFIG_OPTION,
    dataset: Optional[str] = typer.Option(None, help="path to the benchmark JSONL"),
    retriever: str = typer.Option("hybrid", help="dense|bm25|symbol|hybrid"),
    reranker: bool = typer.Option(False, "--reranker/--no-reranker"),
    agentic: bool = typer.Option(False, "--agentic"),
    generation: bool = typer.Option(False, "--generation", help="also evaluate answers"),
    limit: Optional[int] = typer.Option(None, help="evaluate the first N questions"),
    sample: Optional[int] = typer.Option(None, help="evaluate a seeded random sample of N"),
    split: str = typer.Option("test", help="dev|test|all — tune on dev, report on test"),
    name: Optional[str] = typer.Option(None, help="run name"),
    out: Optional[str] = typer.Option(None, help="results directory"),
) -> None:
    """Run the reproducible benchmark (PRD §21-23)."""
    from .evaluation.benchmark import BenchmarkRunner, RunSpec

    cfg = _load_config(config)
    # the agent reads the reranker setting from the config, so keep both in sync
    cfg.reranker.enabled = reranker
    cfg.agent.enabled = agentic
    runner = BenchmarkRunner(cfg, dataset_path=Path(dataset) if dataset else None)
    spec = RunSpec(
        name=name or f"{retriever}{'+rerank' if reranker else ''}{'+agent' if agentic else ''}",
        retriever=retriever,
        reranker=reranker,
        agentic=agentic,
        evaluate_generation=generation,
        limit=limit,
        sample=sample,
        sample_seed=cfg.seed,
        split=split,
    )
    report = runner.run(spec, progress=lambda m: console.print(f"  {m}", style="dim"))
    path = runner.save(report, Path(out) if out else None)
    console.print(report.render())
    console.print(f"[green]results written to {path}")


@app.command()
def report(
    config: Optional[str] = CONFIG_OPTION,
    results_dir: Optional[str] = typer.Option(None, help="directory with benchmark runs"),
    out: Optional[str] = typer.Option(None, help="write a markdown report here"),
) -> None:
    """Aggregate benchmark runs into the ablation table (PRD §23)."""
    from .evaluation.benchmark import load_reports, render_ablation

    cfg = _load_config(config)
    directory = Path(results_dir) if results_dir else cfg.resolve("results")
    reports = load_reports(directory)
    if not reports:
        raise typer.BadParameter(f"no benchmark results in {directory}")
    markdown = render_ablation(reports)
    console.print(markdown)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(markdown, encoding="utf-8")
        console.print(f"[green]written to {out}")


# ----------------------------------------------------------------------
@dataset_app.command("build")
def dataset_build(
    config: Optional[str] = CONFIG_OPTION,
    out: Optional[str] = typer.Option(None, help="output JSONL"),
    n_auto: int = typer.Option(120, help="auto-derived questions to generate"),
    seed: int = typer.Option(0),
    no_llm: bool = typer.Option(False, "--no-llm", help="template questions only"),
) -> None:
    """Build the evaluation dataset from the indexed repository (PRD §20)."""
    from .evaluation.dataset_builder import DatasetBuilder

    cfg = _load_config(config)
    kb = KnowledgeBase.load(cfg, with_vectors=False)
    builder = DatasetBuilder(cfg, kb, use_llm=not no_llm, seed=seed)
    path = Path(out) if out else cfg.resolve("evaluation_data/benchmark.jsonl")
    records = builder.build(n_auto=n_auto, progress=lambda m: console.print(f"  {m}", style="dim"))
    builder.save(records, path)
    console.print(f"[green]{len(records)} questions -> {path}")
    console.print_json(json.dumps(builder.summary(records), indent=2))


@dataset_app.command("validate")
def dataset_validate(
    config: Optional[str] = CONFIG_OPTION,
    path: Optional[str] = typer.Option(None, help="dataset JSONL"),
) -> None:
    """Check that every gold file/symbol still exists in the index."""
    from .evaluation.dataset import load_dataset, validate_dataset

    cfg = _load_config(config)
    kb = KnowledgeBase.load(cfg, with_vectors=False)
    dataset_path = Path(path) if path else cfg.resolve("evaluation_data/benchmark.jsonl")
    records = load_dataset(dataset_path)
    problems = validate_dataset(records, kb)
    console.print(f"{len(records)} questions, {len(problems)} problems")
    for problem in problems[:40]:
        console.print(f"  [yellow]{problem}")


@dataset_app.command("stats")
def dataset_stats(
    config: Optional[str] = CONFIG_OPTION,
    path: Optional[str] = typer.Option(None),
) -> None:
    """Print the benchmark's composition: counts by category, source, split.

    Args:
      config: Path to a config file.
      path: Dataset file; defaults to the configured benchmark.
    """
    from .evaluation.dataset import dataset_summary, load_dataset

    cfg = _load_config(config)
    dataset_path = Path(path) if path else cfg.resolve("evaluation_data/benchmark.jsonl")
    console.print_json(json.dumps(dataset_summary(load_dataset(dataset_path)), indent=2))


@app.command()
def audit(
    config: Optional[str] = CONFIG_OPTION,
    run: str = typer.Option("gen-hybrid+rerank", help="benchmark run name to audit"),
    n: int = typer.Option(20, help="answers to sample for manual review"),
    path: Optional[str] = typer.Option(None, help="audit sheet (jsonl)"),
    score: bool = typer.Option(False, "--score", help="report judge/human agreement"),
) -> None:
    """Sample judged answers for manual review, or score the filled-in sheet.

    An LLM judge that is never checked against a human is not a measurement
    (PRD §22).
    """
    from .evaluation.answer_metrics import judge_agreement, sample_for_audit

    cfg = _load_config(config)
    sheet = Path(path) if path else cfg.resolve("evaluation_data/judge_audit.jsonl")
    if score:
        console.print_json(json.dumps(judge_agreement(sheet), indent=2))
        return
    results = cfg.resolve("results") / f"{run}.json"
    if not results.exists():
        raise typer.BadParameter(f"no benchmark run at {results}")
    rows = json.loads(results.read_text()).get("per_question", [])
    rows = [r for r in rows if r.get("answer")]
    if not rows:
        raise typer.BadParameter(f"{results} has no generated answers; run with --generation")
    sample_for_audit(rows, sheet, n=n, seed=cfg.seed)
    console.print(f"[green]{min(n, len(rows))} answers -> {sheet}")
    console.print("fill in human_correctness / human_faithfulness, then: rag audit --score")


# ----------------------------------------------------------------------
@app.command()
def serve(
    config: Optional[str] = CONFIG_OPTION,
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8100),
) -> None:
    """Serve the API and the demo UI (PRD §25)."""
    import uvicorn

    from .api.server import create_app

    cfg = _load_config(config)
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")


def main() -> None:
    """Entry point for the `rag` console script."""
    app()


if __name__ == "__main__":
    main()
