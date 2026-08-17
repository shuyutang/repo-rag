"""Iterative retrieval agent (PRD §17).

    plan -> retrieve -> inspect evidence -> "is anything still missing?" -> ...

The loop is bounded by ``agent.max_iterations``.  Every step is written to the
trace, including the model's own stated reason for continuing, so that the
ablation can answer the project's central question: do the extra retrieval
steps actually buy anything?
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..generation.llm import LLMClient, extract_json
from ..indexing.knowledge_base import KnowledgeBase
from ..observability.tracing import Trace
from ..retrieval.hybrid import HybridRetriever
from ..schema import RetrievedChunk
from .planner import QueryPlan, QueryPlanner
from .tools import ToolBox, ToolCall

INSPECT_PROMPT = """\
You are the retrieval controller for a code question-answering system on the \
`{repository}` repository. Decide whether the evidence collected so far is \
sufficient, or which single follow-up retrieval would most improve the answer.

Question: {question}
Question category: {category}
Iteration {iteration} of {max_iterations}.

Evidence collected so far (title + first lines):
{evidence}

Queries already issued: {issued}

Available tools:
{tools}

Respond with a JSON object only:
{{
  "sufficient": true | false,
  "reason": "one sentence",
  "next": {{"tool": "<tool name>", "args": {{...}}}}   // omit when sufficient
}}

Ask for a follow-up only when a *specific* missing artifact would change the \
answer — for example a configuration object, a caller, a test, or the commit \
that introduced the behaviour. Do not repeat a query that was already issued.
"""


@dataclass
class AgentStep:
    """One iteration of the agent loop.

    Attributes:
      iteration: 1-based iteration number.
      duration_ms: Wall-clock time for the iteration.
      tool_calls: Tools invoked, in order.
      reason: The model's stated reason for continuing or stopping.
      sufficient: Whether the model judged the evidence complete.
      new_evidence: Chunks this iteration added that were not already held.
    """

    iteration: int
    duration_ms: float = 0.0
    tool_calls: list[ToolCall] = field(default_factory=list)
    reason: str = ""
    sufficient: bool = False
    new_evidence: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the step as a JSON-serialisable dict for the trace."""
        return {
            "iteration": self.iteration,
            "duration_ms": round(self.duration_ms, 2),
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "reason": self.reason,
            "sufficient": self.sufficient,
            "new_evidence": self.new_evidence,
        }


@dataclass
class AgentResult:
    """What one agent run produced.

    Attributes:
      plan: The plan the run started from.
      evidence: Final ranked evidence.
      steps: Every iteration, in order.
    """

    plan: QueryPlan
    evidence: list[RetrievedChunk]
    steps: list[AgentStep]

    def to_dict(self) -> dict[str, Any]:
        """Return the run as a JSON-serialisable dict, without chunk bodies."""
        return {
            "plan": self.plan.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "n_evidence": len(self.evidence),
        }


class RetrievalAgent:
    """Iterative retrieval: plan, retrieve, inspect, repeat.

    Iteration 1 is deterministic -- it executes the plan and makes no LLM
    call -- and inspection begins at iteration 2. The loop has four
    independent stop conditions: the model declares the evidence sufficient,
    it proposes no next call, it repeats a query already issued, or an
    iteration adds nothing new. Every step is written to the trace, including
    the model's own stated reason, so the ablation can answer whether the
    extra retrieval steps actually buy anything.

    Attributes:
      kb: Knowledge base being searched.
      config: Full system configuration.
      retriever: Hybrid retriever backing the search tools.
      tools: Tool surface the agent calls.
      llm: Client used for planning and inspection; `None` reduces the agent
        to a single deterministic iteration.
      planner: Query planner.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        retriever: HybridRetriever,
        *,
        config: Config | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        """Wire the agent to a knowledge base and retriever.

        Args:
          kb: Loaded knowledge base.
          retriever: Hybrid retriever backing the search tools.
          config: Full system configuration; defaults to the knowledge
            base's own.
          llm: Client for planning and inspection.
        """
        self.kb = kb
        self.config = config or kb.config
        self.retriever = retriever
        self.tools = ToolBox(kb, retriever)
        self.llm = llm
        self.planner = QueryPlanner(
            llm=llm,
            repository=self.config.repository,
            max_subqueries=self.config.agent.planner_max_subqueries,
        )

    # ------------------------------------------------------------------
    def run(self, question: str, *, trace: Trace | None = None) -> AgentResult:
        """Run the agent loop until it stops or hits the iteration cap.

        Args:
          question: The user's question.
          trace: Trace to record the plan, each iteration and the final
            ranking into.

        Returns:
          The plan, the final ranked evidence and every step taken.
        """
        cfg = self.config.agent
        with _step(trace, "planner", kind="agent") as tstep:
            plan = self.planner.plan(question)
            if tstep is not None:
                tstep.detail.update(plan.to_dict())
        if trace and plan.usage:
            trace.add_usage(
                prompt_tokens=plan.usage.get("prompt_tokens", 0),
                completion_tokens=plan.usage.get("completion_tokens", 0),
                cost=plan.usage.get("cost_usd", 0.0),
            )

        evidence: dict[str, RetrievedChunk] = {}
        fusion: dict[str, float] = {}
        steps: list[AgentStep] = []
        issued: list[str] = []

        # --- iteration 1: execute the plan ------------------------------
        started = time.perf_counter()
        first = AgentStep(iteration=1, reason=plan.rationale or "initial plan")
        for subquery in plan.subqueries[: cfg.planner_max_subqueries + 1]:
            results, call = self.tools.dispatch(
                "search", {"query": subquery, "k": cfg.per_step_k}
            )
            issued.append(subquery)
            first.tool_calls.append(call)
            first.new_evidence += self._merge(evidence, results, fusion)
        for symbol in plan.symbols[:3]:
            results, call = self.tools.dispatch("symbol_search", {"name": symbol})
            first.tool_calls.append(call)
            first.new_evidence += self._merge(evidence, results, fusion)
        if "impact" in plan.tools and plan.symbols:
            results, call = self.tools.dispatch("impact", {"symbol": plan.symbols[0]})
            first.tool_calls.append(call)
            first.new_evidence += self._merge(evidence, results, fusion)
        if "history" in plan.tools:
            results, call = self.tools.dispatch(
                "history", {"query": question, "k": cfg.per_step_k}
            )
            first.tool_calls.append(call)
            first.new_evidence += self._merge(evidence, results, fusion)
        first.duration_ms = (time.perf_counter() - started) * 1000
        steps.append(first)
        self._record(trace, first, evidence)

        # --- iterations 2..N: inspect and refine ------------------------
        for iteration in range(2, cfg.max_iterations + 1):
            started = time.perf_counter()
            decision = self._inspect(question, plan, evidence, issued, iteration, trace)
            step = AgentStep(
                iteration=iteration,
                reason=decision.get("reason", ""),
                sufficient=bool(decision.get("sufficient", True)),
            )
            if step.sufficient or not decision.get("next"):
                step.duration_ms = (time.perf_counter() - started) * 1000
                steps.append(step)
                self._record(trace, step, evidence)
                break
            call_spec = decision["next"]
            tool = str(call_spec.get("tool", "search"))
            args = call_spec.get("args") or {}
            if tool == "search" and "k" not in args:
                args["k"] = cfg.per_step_k
            if isinstance(args.get("query"), str):
                if args["query"] in issued:      # stop the agent looping
                    step.sufficient = True
                    step.reason += " (repeated query; stopping)"
                    step.duration_ms = (time.perf_counter() - started) * 1000
                    steps.append(step)
                    self._record(trace, step, evidence)
                    break
                issued.append(args["query"])
            results, call = self.tools.dispatch(tool, args)
            step.tool_calls.append(call)
            step.new_evidence = self._merge(evidence, results, fusion)
            step.duration_ms = (time.perf_counter() - started) * 1000
            steps.append(step)
            self._record(trace, step, evidence)
            if step.new_evidence == 0:
                break
            if len(evidence) >= cfg.max_evidence * 2:
                break

        ranked = self._finalize(question, list(evidence.values()), trace, fusion)
        return AgentResult(plan=plan, evidence=ranked, steps=steps)

    # ------------------------------------------------------------------
    def _merge(
        self,
        evidence: dict[str, RetrievedChunk],
        results: list[RetrievedChunk],
        fusion: dict[str, float] | None = None,
    ) -> int:
        """Merge one tool's results, accumulating a rank-based fusion score.

        Scores from different tools are not comparable -- RRF is around 0.03,
        symbol around 1.0, graph around 0.7 -- so the agent ranks its own
        evidence by reciprocal rank across tool calls rather than by raw
        score.

        Args:
          evidence: Accumulated evidence by chunk id, updated in place.
          results: One tool call's results, best first.
          fusion: Accumulated reciprocal-rank scores, updated in place.

        Returns:
          How many chunks were new. This is what the "no new evidence" stop
          condition tests, so a tool that only re-finds known chunks ends the
          loop.
        """
        added = 0
        for rank, item in enumerate(results, start=1):
            if fusion is not None:
                fusion[item.chunk_id] = fusion.get(item.chunk_id, 0.0) + 1.0 / (60 + rank)
            existing = evidence.get(item.chunk_id)
            if existing is None:
                evidence[item.chunk_id] = item
                added += 1
            elif item.score > existing.score:
                existing.score = item.score
        return added

    def _record(
        self, trace: Trace | None, step: AgentStep, evidence: dict[str, RetrievedChunk]
    ) -> None:
        """Write one agent iteration to the trace.

        Args:
          trace: Trace to write to; a no-op when `None`.
          step: The completed iteration.
          evidence: Accumulated evidence, for the running total.
        """
        if not trace:
            return
        with trace.step(
            f"agent iteration #{step.iteration}", kind="agent", **{
                "reason": step.reason,
                "sufficient": step.sufficient,
                "new_evidence": step.new_evidence,
                "total_evidence": len(evidence),
                "tools": [c.tool for c in step.tool_calls],
            }
        ) as tstep:
            if tstep is not None:
                tstep.results = [c.to_dict() for c in step.tool_calls]
        # the work happened before this context manager opened; report it
        if trace.steps:
            trace.steps[-1].duration_ms = step.duration_ms

    def _inspect(
        self,
        question: str,
        plan: QueryPlan,
        evidence: dict[str, RetrievedChunk],
        issued: list[str],
        iteration: int,
        trace: Trace | None,
    ) -> dict[str, Any]:
        """Ask the model whether the evidence so far is sufficient.

        Args:
          question: The user's question.
          plan: The plan the run started from.
          evidence: Accumulated evidence by chunk id.
          issued: Queries already issued, shown to discourage repeats.
          iteration: Current iteration number.
          trace: Trace to record token usage into.

        Returns:
          The model's decision: `sufficient`, `reason` and optionally `next`.
          Every failure -- no client, an unreachable server, an unparseable
          response -- returns `sufficient: True` with the cause as the
          reason, so a broken inspection ends the loop cleanly instead of
          burning the remaining iterations.
        """
        if self.llm is None:
            return {"sufficient": True, "reason": "no LLM configured for inspection"}
        top = sorted(evidence.values(), key=lambda r: -r.score)[: self.config.agent.max_evidence]
        rendered = "\n".join(
            f"- {item.chunk.title} [{item.chunk.location}] :: "
            + " ".join(item.chunk.content.split())[:160]
            for item in top
        )
        prompt = INSPECT_PROMPT.format(
            repository=self.config.repository,
            question=question,
            category=plan.category,
            iteration=iteration,
            max_iterations=self.config.agent.max_iterations,
            evidence=rendered or "(nothing retrieved yet)",
            issued=json.dumps(issued[-8:]),
            tools=self.tools.describe(),
        )
        try:
            response = self.llm.complete(prompt, max_tokens=400)
        except Exception as exc:
            return {"sufficient": True, "reason": f"inspection failed: {exc}"}
        if trace:
            trace.add_usage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cost=response.cost_usd,
            )
        data = extract_json(response.text)
        if not data:
            return {"sufficient": True, "reason": "unparseable controller response"}
        return data

    def _finalize(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        trace: Trace | None,
        fusion: dict[str, float] | None = None,
    ) -> list[RetrievedChunk]:
        """Rank everything the agent gathered down to the final context set.

        This is where the cross-encoder earns its place: the agent's evidence
        arrives from four tools whose scores are incomparable, and the
        reranker is the only stage that puts them on one scale.

        Args:
          question: The user's question, the reranking query.
          evidence: Everything gathered, unordered.
          trace: Trace to record the reranking step into.
          fusion: Accumulated reciprocal-rank scores, used to order when
            reranking is disabled.

        Returns:
          At most `agent.max_evidence` results, ranked from 1.
        """
        cfg = self.config
        if not evidence:
            return []
        if cfg.reranker.enabled:
            with _step(
                trace, "agent rerank", kind="rerank", n_candidates=len(evidence)
            ) as tstep:
                ranked = self.retriever.reranker.rerank(
                    question, evidence, cfg.agent.max_evidence
                )
                if trace and tstep is not None:
                    trace.record_results(tstep, ranked)
        else:
            key = (
                (lambda r: -fusion.get(r.chunk_id, 0.0)) if fusion
                else (lambda r: -r.score)
            )
            ranked = sorted(evidence, key=key)[: cfg.agent.max_evidence]
            for rank, item in enumerate(ranked, start=1):
                item.rank = rank
                if fusion:
                    item.component_scores["agent_rrf"] = round(
                        fusion.get(item.chunk_id, 0.0), 6
                    )
        return ranked


class _NullStep:
    """Null context manager used in place of a trace step when untraced."""

    def __enter__(self):
        """Return `None` in place of a trace step."""
        return None

    def __exit__(self, *exc):
        """Propagate any exception unchanged."""
        return False


def _step(trace: Trace | None, name: str, **kwargs):
    """Open a trace step, or a null context when there is no trace.

    Args:
      trace: Trace to open a step on, or `None`.
      name: Stage name.
      **kwargs: Stage detail fields.

    Returns:
      A context manager yielding the step, or `None` when untraced.
    """
    return trace.step(name, **kwargs) if trace else _NullStep()
