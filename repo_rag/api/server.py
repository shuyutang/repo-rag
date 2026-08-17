"""FastAPI service and demo UI host (PRD §24-25)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from ..config import REPO_ROOT, Config, default_config
from ..indexing.knowledge_base import KnowledgeBase
from ..pipeline import Pipeline

UI_DIR = REPO_ROOT / "ui"


class AskRequest(BaseModel):
    """Body of `POST /api/ask`.

    Attributes:
      question: The question to answer.
      retriever: Retrieval strategy: "dense", "bm25", "symbol" or "hybrid".
      reranker: Override the configured reranker setting.
      agentic: Use iterative agentic retrieval.
      k: Evidence chunks to retrieve.
    """

    question: str
    retriever: str = Field("hybrid", description="dense|bm25|symbol|hybrid")
    reranker: Optional[bool] = None
    agentic: bool = False
    k: Optional[int] = None


class SearchRequest(BaseModel):
    """Body of `POST /api/search`, which retrieves without generating.

    Attributes:
      query: The search query.
      retriever: Retrieval strategy.
      k: Results to return.
      reranker: Override the configured reranker setting.
    """

    query: str
    retriever: str = "hybrid"
    k: int = 10
    reranker: Optional[bool] = None


class EvaluateRequest(BaseModel):
    """Run one benchmark question live and score it against its gold labels."""

    question_id: str
    retriever: str = "hybrid"
    reranker: Optional[bool] = None
    agentic: bool = False
    k: Optional[int] = None
    generate: bool = True


def create_app(config: Config | None = None) -> FastAPI:
    """Build the FastAPI application serving the API and demo UI.

    The knowledge base is loaded on the first request that needs it, not at
    import time, so the server starts instantly and a missing index reports
    as a clean error rather than a crash on boot. One consequence: after
    re-indexing, the process must be restarted to pick up the new index.

    Args:
      config: Configuration to serve; the default config is loaded when
        omitted.

    Returns:
      The configured application.
    """
    config = config or default_config()
    app = FastAPI(title="repo-rag", version="1.0.0")
    state: dict[str, Any] = {}
    lock = threading.Lock()

    def pipeline() -> Pipeline:
        # uvicorn runs these sync endpoints in a threadpool, so two concurrent
        # first-requests would otherwise both pay the (heavy) index load.
        """Return the shared pipeline, loading the knowledge base on first use.

        uvicorn runs these sync endpoints in a threadpool, so without the lock
        two concurrent first-requests would both pay the heavy index load.

        Returns:
          The pipeline.

        Raises:
          HTTPException: 503 when no index has been built yet.
        """
        with lock:
            if "pipeline" not in state:
                try:
                    kb = KnowledgeBase.load(config)
                except FileNotFoundError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "no index found — run `rag ingest` then `rag index` "
                            f"({exc})"
                        ),
                    ) from exc
                state["kb"] = kb
                state["pipeline"] = Pipeline(config, kb)
        return state["pipeline"]

    def _dataset() -> list:
        """The benchmark, re-read when the file changes.

        `rag dataset build` can rewrite it while the server is up; keying the
        cache on mtime means the UI sees the new questions without a restart.
        """
        from ..evaluation.dataset import load_dataset

        path = config.resolve("evaluation_data/benchmark.jsonl")
        stamp = path.stat().st_mtime_ns if path.exists() else None
        if "dataset" not in state or state["dataset_stamp"] != stamp:
            state["dataset"] = load_dataset(path) if stamp is not None else []
            state["dataset_stamp"] = stamp
        return state["dataset"]

    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the demo UI."""
        html = UI_DIR / "index.html"
        if not html.exists():
            return "<h1>repo-rag</h1><p>UI not installed.</p>"
        return html.read_text(encoding="utf-8")

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        """Report the repository, commit, chunk counts and the models in use."""
        kb = pipeline().kb
        return {
            "repository": config.repository,
            "commit": kb.commit,
            "n_chunks": len(kb.store),
            "by_artifact": kb.store.stats(),
            "embedding_model": config.embedding.model,
            "reranker": config.reranker.model if config.reranker.enabled else None,
            "llm": f"{config.llm.provider}:{config.llm.model}",
            "index_meta": kb.meta,
        }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Is the index loaded and is the generation backend reachable?"""
        report: dict[str, Any] = {"index": "missing", "llm": "unknown"}
        try:
            report["index"] = "ready" if len(pipeline().kb.store) else "empty"
        except HTTPException as exc:
            report["index_detail"] = exc.detail
        try:
            reply = pipeline().llm.complete("Reply with the single word: ok", max_tokens=8)
            report["llm"] = "ready" if reply.text else "empty response"
        except Exception as exc:
            report["llm"] = "unreachable"
            report["llm_detail"] = (
                f"{type(exc).__name__}: {exc}"[:300]
                + " — start it with ./scripts/serve_llm.sh, or set llm.provider "
                "to `echo` for retrieval-only testing"
            )
        report["llm_endpoint"] = f"{config.llm.provider}:{config.llm.model}"
        return report

    def _evidence_rows(result_evidence: list, gold: set[str] | None = None) -> list:
        """Render evidence for the UI.

        Args:
          result_evidence: Retrieval results to render.
          gold: Gold file paths; when given, each row is marked `is_gold`,
            which is what colours hits and misses in the live evaluation view.

        Returns:
          One JSON-serialisable row per result, content truncated.
        """
        rows = []
        for item in result_evidence:
            row = {
                **item.to_dict(),
                "title": item.chunk.title,
                "content": item.chunk.content[:4000],
                "artifact_type": item.chunk.artifact_type,
                "reason": item.retriever,
            }
            if gold is not None:
                row["is_gold"] = item.chunk.path in gold
            rows.append(row)
        return rows

    @app.post("/api/ask")
    def ask(request: AskRequest) -> JSONResponse:
        """Answer a question, returning the answer, trace and evidence."""
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="empty question")
        sources = None if request.retriever == "hybrid" else [request.retriever]
        try:
            result = pipeline().ask(
                request.question,
                sources=sources,
                k=request.k,
                use_reranker=request.reranker,
                use_agent=request.agentic,
            )
        except RuntimeError as exc:  # the generation backend is down
            raise HTTPException(
                status_code=503,
                detail=(
                    f"generation backend unavailable ({exc}). Start it with "
                    "./scripts/serve_llm.sh, or use the Search tab for "
                    "retrieval-only testing."
                ),
            ) from exc
        payload = {
            "answer": result.answer.text,
            "citations": [c.render() for c in result.answer.citations],
            "unsupported_citations": result.answer.usage.get("unsupported_citations", []),
            "evidence": _evidence_rows(result.evidence),
            "trace": result.trace.to_dict(),
            "agent": result.agent.to_dict() if result.agent else None,
            "usage": result.answer.usage,
        }
        return JSONResponse(payload)

    @app.post("/api/evaluate")
    def evaluate(request: EvaluateRequest) -> JSONResponse:
        """Run one benchmark question live and score it against its gold labels.

        This is the interactive counterpart to `rag benchmark`: same retrieval
        code path, same metric functions, one question at a time so a failure
        can be inspected rather than only counted.
        """
        from ..evaluation.answer_metrics import citation_metrics
        from ..evaluation.retrieval_metrics import evaluate_question

        record = next((r for r in _dataset() if r.id == request.question_id), None)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown question id")

        sources = None if request.retriever == "hybrid" else [request.retriever]
        pipe = pipeline()
        answer_payload: dict[str, Any] = {}
        if request.generate:
            try:
                result = pipe.ask(
                    record.question,
                    sources=sources,
                    k=request.k,
                    use_reranker=request.reranker,
                    use_agent=request.agentic,
                )
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=503, detail=f"generation backend unavailable ({exc})"
                ) from exc
            evidence, trace = result.evidence, result.trace
            answer_payload = {
                "answer": result.answer.text,
                "citations": [c.render() for c in result.answer.citations],
                "unsupported_citations": result.answer.usage.get(
                    "unsupported_citations", []
                ),
                "citation_metrics": dict(
                    zip(
                        ("n_citations", "validity", "precision", "completeness"),
                        citation_metrics(result.answer, record),
                    )
                ),
            }
        else:
            from ..observability.tracing import Trace

            trace = Trace(record.question, config_fingerprint=config.fingerprint())
            evidence, _agent = pipe.retrieve(
                record.question,
                trace=trace,
                sources=sources,
                k=request.k,
                use_reranker=request.reranker,
                use_agent=request.agentic,
            )

        metrics = evaluate_question(record, evidence)
        gold = set(record.relevant_files)
        found = {item.chunk.path for item in evidence} & gold
        return JSONResponse(
            {
                "question": record.to_dict(),
                "retrieval": metrics.to_dict(),
                "gold_found": sorted(found),
                "gold_missed": sorted(gold - found),
                "evidence": _evidence_rows(evidence, gold=gold),
                "trace": trace.to_dict(),
                **answer_payload,
            }
        )

    @app.post("/api/search")
    def search(request: SearchRequest) -> JSONResponse:
        """Retrieve without generating: milliseconds, and no LLM involved."""
        from ..retrieval.hybrid import RetrievalRequest, build_retriever

        kb = pipeline().kb
        sources = None if request.retriever == "hybrid" else [request.retriever]
        retriever = build_retriever(kb, sources=sources)
        results = retriever.retrieve(
            request.query,
            request=RetrievalRequest(
                query=request.query, k=request.k, use_reranker=request.reranker
            ),
        )
        return JSONResponse(
            {
                "results": [
                    {**item.to_dict(), "title": item.chunk.title,
                     "content": item.chunk.content[:2000]}
                    for item in results
                ]
            }
        )

    @app.get("/api/chunk/{chunk_id}")
    def chunk(chunk_id: str) -> dict[str, Any]:
        """Return one chunk by id."""
        found = pipeline().kb.store.get(chunk_id)
        if found is None:
            raise HTTPException(status_code=404, detail="unknown chunk")
        return found.to_dict()

    @app.get("/api/impact/{symbol}")
    def impact(symbol: str, limit: int = 15) -> dict[str, Any]:
        """Report the definitions, callers, tests and importers of a symbol."""
        kb = pipeline().kb
        if kb.graph is None:
            raise HTTPException(status_code=404, detail="no symbol graph in index")
        report = kb.graph.impact(symbol, limit=limit)

        def render(ids: list[str]) -> list[dict[str, str]]:
            """Resolve chunk ids to their locations and symbol names."""
            out = []
            for chunk_id in ids:
                chunk = kb.store.get(chunk_id)
                if chunk:
                    out.append(
                        {"chunk_id": chunk_id, "location": chunk.location,
                         "symbol": chunk.qualified_name or ""}
                    )
            return out

        return {
            "symbol": symbol,
            "definitions": render(report.definitions),
            "callers": render(report.callers),
            "tests": render(report.tests),
            "importers": report.importers,
        }

    @app.get("/api/traces")
    def traces(limit: int = 50) -> dict[str, Any]:
        """List the most recent traces, newest first."""
        return {"traces": pipeline().trace_store.list(limit=limit)}

    @app.get("/api/traces/{trace_id}")
    def trace(trace_id: str) -> dict[str, Any]:
        """Return one full trace by id."""
        found = pipeline().trace_store.get(trace_id)
        if found is None:
            raise HTTPException(status_code=404, detail="unknown trace")
        return found

    @app.get("/api/benchmark")
    def benchmark() -> dict[str, Any]:
        """The ablation table, read from recorded benchmark runs."""
        from ..evaluation.benchmark import load_reports, merge_reports, render_ablation

        directory = config.resolve("results")
        reports = load_reports(directory)
        by_name = {r["spec"]["name"]: r for r in reports}
        merged = merge_reports(reports)
        for row in merged:
            source = by_name.get(row["name"]) or by_name.get(f"gen-{row['name']}")
            row["by_category"] = (source or {}).get("retrieval_by_category", {})
            row["by_difficulty"] = (source or {}).get("retrieval_by_difficulty", {})
        return {
            "runs": merged,
            "markdown": render_ablation(reports) if reports else "",
            "fingerprint": reports[0].get("fingerprint", {}) if reports else {},
        }

    @app.get("/api/dataset")
    def dataset(
        limit: int = 200,
        category: str = "",
        split: str = "",
        source: str = "",
        difficulty: str = "",
        q: str = "",
    ) -> dict[str, Any]:
        """Browse benchmark questions, filtered by category, split or source."""
        from ..evaluation.dataset import dataset_summary

        records = _dataset()
        if not records:
            return {"n_questions": 0, "questions": []}
        summary = dataset_summary(records)
        needle = q.strip().lower()
        selected = [
            r
            for r in records
            if (not category or r.category == category)
            and (not split or r.split == split)
            # `source` is a family prefix: "auto" matches "auto:llm_phrased"
            and (not source or r.source == source or r.source.startswith(source + ":"))
            and (not difficulty or r.difficulty == difficulty)
            and (not needle or needle in r.question.lower())
        ]
        return {
            **summary,
            "n_matching": len(selected),
            "categories": sorted({r.category for r in records}),
            "sources": sorted({r.source.split(":")[0] for r in records}),
            "questions": [r.to_dict() for r in selected[:limit]],
            "examples": [r.to_dict() for r in records[:20]],
        }

    return app


def main() -> None:  # pragma: no cover - convenience entry point
    """Serve the default configuration on 127.0.0.1:8100."""
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8100)


if __name__ == "__main__":  # pragma: no cover
    main()
