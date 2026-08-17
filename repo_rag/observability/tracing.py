"""Query tracing (PRD §24).

Every stage of a query appends a step to the trace: what it was asked, what it
returned, how long it took, what it cost.  Traces are written to disk as JSON so
the UI, the benchmark and a post-mortem can all read the same record.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceStep:
    """One measured stage of a query.

    Attributes:
      name: Stage name, e.g. "bm25 search".
      kind: Stage class: "retrieval", "fusion", "rerank", "agent",
        "generation" or "other".
      started_at: Wall-clock start, as a Unix timestamp.
      duration_ms: Measured duration.
      detail: Stage-specific fields, e.g. the model or candidate count.
      results: What the stage produced, capped when recorded.
    """

    name: str
    kind: str
    started_at: float
    duration_ms: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the step as a JSON-serialisable dict."""
        return {
            "name": self.name,
            "kind": self.kind,
            "duration_ms": round(self.duration_ms, 2),
            "detail": self.detail,
            "results": self.results,
        }


class Trace:
    """The full record of one query: stages, results, tokens, cost.

    Written to disk as JSON so the UI, the benchmark and a post-mortem all
    read the same record rather than three approximations of it.

    Attributes:
      trace_id: Short unique id, also the filename stem.
      question: The question that was asked.
      created_at: Wall-clock creation time, as a Unix timestamp.
      steps: Stages in the order they started.
      config: Config fingerprint, so a trace stays interpretable after the
        configuration moves on.
      usage: Running totals for LLM calls, tokens and estimated cost.
      answer: The generated answer, once generation has run.
    """

    def __init__(self, question: str, *, config_fingerprint: dict | None = None) -> None:
        """Start a trace.

        Args:
          question: The question being traced.
          config_fingerprint: Config fingerprint to embed in the record.
        """
        self.trace_id = uuid.uuid4().hex[:12]
        self.question = question
        self.created_at = time.time()
        self.steps: list[TraceStep] = []
        self.config = config_fingerprint or {}
        self.usage: dict[str, Any] = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
        self.answer: dict[str, Any] | None = None
        self._open: list[tuple[TraceStep, float]] = []

    # ------------------------------------------------------------------
    class _StepContext:
        """Context manager that times one step.

        The clock starts at `start`, not at `__enter__`, so a caller may
        create the context and enter it separately without losing the
        interval between the two.

        Attributes:
          trace: The owning trace.
          step: The step being timed.
        """

        def __init__(self, trace: "Trace", step: TraceStep) -> None:
            """Bind the context to a trace and its step."""
            self.trace = trace
            self.step = step

        def __enter__(self) -> TraceStep:
            """Return the step being timed."""
            return self.step

        def __exit__(self, *exc: Any) -> None:
            """Record the elapsed time, whether or not the body raised."""
            self.step.duration_ms = (time.perf_counter() - self._t0) * 1000

        def start(self) -> "Trace._StepContext":
            """Start the clock and return self, ready to be entered."""
            self._t0 = time.perf_counter()
            return self

    def step(self, name: str, kind: str = "other", **detail: Any) -> "_StepContextAlias":
        """Open a timed step.

        Args:
          name: Stage name.
          kind: Stage class.
          **detail: Stage-specific fields recorded on the step.

        Returns:
          A started context manager yielding the step. The step is appended
          to the trace immediately, so a stage that raises still appears.
        """
        step = TraceStep(name=name, kind=kind, started_at=time.time(), detail=detail)
        self.steps.append(step)
        ctx = Trace._StepContext(self, step)
        return ctx.start()

    # ------------------------------------------------------------------
    def record_results(self, step: TraceStep, results: list[Any], limit: int = 25) -> None:
        """Record what a step returned.

        Args:
          step: Step to record onto.
          results: Objects with a `to_dict`, or plain dicts.
          limit: How many to store. The full count is kept in
            `detail["n_results"]`, so truncation never hides how much was
            actually retrieved.
        """
        step.results = [
            r.to_dict() if hasattr(r, "to_dict") else r for r in results[:limit]
        ]
        step.detail["n_results"] = len(results)

    def add_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
        calls: int = 1,
    ) -> None:
        """Add one LLM call's usage to the running totals.

        Args:
          prompt_tokens: Input tokens.
          completion_tokens: Output tokens.
          cost: Estimated cost in USD.
          calls: Number of calls this represents.
        """
        self.usage["llm_calls"] += calls
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["estimated_cost_usd"] = round(
            self.usage["estimated_cost_usd"] + cost, 6
        )

    @property
    def total_ms(self) -> float:
        """float: Sum of all step durations.

        A sum rather than an end-to-end wall clock, so it excludes time spent
        outside any instrumented stage.
        """
        return round(sum(s.duration_ms for s in self.steps), 2)

    def stage_latencies(self) -> dict[str, float]:
        """Total the durations of steps sharing a name.

        Returns:
          A mapping of stage name to summed duration, which is what the
          benchmark's per-stage latency table is built from.
        """
        out: dict[str, float] = {}
        for step in self.steps:
            out[step.name] = round(out.get(step.name, 0.0) + step.duration_ms, 2)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return the whole trace as a JSON-serialisable dict."""
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "created_at": self.created_at,
            "total_ms": self.total_ms,
            "stage_latencies": self.stage_latencies(),
            "usage": self.usage,
            "steps": [s.to_dict() for s in self.steps],
            "answer": self.answer,
            "config": self.config,
        }

    def save(self, directory: Path) -> Path:
        """Write the trace to `<directory>/<trace_id>.json`.

        Args:
          directory: Destination, created if absent.

        Returns:
          The path written.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.trace_id}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return path

    # ------------------------------------------------------------------
    def render(self) -> str:
        """Render the trace as an ASCII tree for the CLI.

        Returns:
          One line per step with its duration and scalar details, up to five
          results each, and a final line of token and cost totals.
        """
        lines = [f"Query: {self.question}", f"trace {self.trace_id}  {self.total_ms} ms"]
        for step in self.steps:
            head = f" ├─ {step.name} ({step.duration_ms:.0f} ms)"
            extras = ", ".join(
                f"{k}={v}" for k, v in step.detail.items() if not isinstance(v, (list, dict))
            )
            lines.append(f"{head}  {extras}" if extras else head)
            for result in step.results[:5]:
                loc = result.get("location") or result.get("path") or ""
                score = result.get("score")
                score_text = f"{score:.4f}" if isinstance(score, (int, float)) else ""
                lines.append(f" │    {loc} {score_text}")
        usage = self.usage
        lines.append(
            f" └─ llm_calls={usage['llm_calls']} "
            f"tokens={usage['prompt_tokens']}+{usage['completion_tokens']} "
            f"cost=${usage['estimated_cost_usd']:.4f}"
        )
        return "\n".join(lines)


# `Trace.step` returns a context manager; alias kept for typing clarity
_StepContextAlias = Trace._StepContext


class TraceStore:
    """Directory-backed trace store used by the API and UI.

    Attributes:
      directory: Where traces are read from and written to.
    """

    def __init__(self, directory: Path) -> None:
        """Bind the store to a directory, creating it if absent.

        Args:
          directory: Trace directory.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, trace: Trace) -> Path:
        """Write a trace into the store and return its path."""
        return trace.save(self.directory)

    def get(self, trace_id: str) -> dict[str, Any] | None:
        """Read one trace by id.

        Args:
          trace_id: Id of the trace.

        Returns:
          The trace as a dict, or `None` if no such file exists.
        """
        path = self.directory / f"{trace_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        """Summarise the most recent traces.

        Args:
          limit: Maximum traces to return.

        Returns:
          One summary per trace, newest first. Files that are unreadable or
          malformed are skipped rather than raising, since a trace written by
          a crashed run should not break the listing.
        """
        files = sorted(
            self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        out = []
        for path in files[:limit]:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            out.append(
                {
                    "trace_id": data.get("trace_id"),
                    "question": data.get("question"),
                    "total_ms": data.get("total_ms"),
                    "created_at": data.get("created_at"),
                    "n_steps": len(data.get("steps") or []),
                }
            )
        return out
