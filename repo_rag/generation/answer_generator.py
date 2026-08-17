"""Grounded answer generation (PRD §16)."""

from __future__ import annotations

import re
from typing import Sequence

from ..config import Config
from ..observability.tracing import Trace
from ..schema import Answer, Citation, RetrievedChunk
from .context_builder import BuiltContext, ContextBuilder
from .llm import LLMClient, build_llm

SYSTEM_PROMPT = """\
You are an engineering knowledge agent answering questions about a specific \
software repository. You answer only from the evidence you are given.

Rules:
1. Ground every substantive technical claim in the evidence blocks.
2. Cite with the exact bracket form shown in the evidence header, e.g. \
[vllm/worker/cache_engine.py:95-142]. Put the citation immediately after the \
claim it supports. Never invent a path, symbol or line range.
3. Separate the three epistemic levels explicitly:
   - state facts supported by evidence directly;
   - prefix reasoning that goes beyond the evidence with "Inference:";
   - say plainly what the retrieved evidence does not establish, in a final \
"Not established from retrieved evidence:" line when anything material is missing.
4. Prefer naming concrete symbols, functions and configuration fields over \
general description.
5. Be concise: a short explanation, then the mechanism, then citations. Do not \
repeat the evidence verbatim at length.
"""

USER_TEMPLATE = """\
Repository: {repository} (commit {commit})

Question:
{question}

Evidence:
{context}

Write the answer now. Cite evidence inline using the bracket form given in each \
evidence header.
"""

_CITATION = re.compile(r"\[([^\]\s]+?):(\d+)-(\d+)\]")


def parse_citations(text: str) -> list[Citation]:
    """Extract bracketed citations from an answer.

    Args:
      text: Generated answer text.

    Returns:
      Citations in order of first appearance, deduplicated by path and line
      range. Symbol and chunk id are left unset; validation fills them in.
    """
    out: list[Citation] = []
    seen: set[tuple[str, int, int]] = set()
    for path, start, end in _CITATION.findall(text):
        key = (path, int(start), int(end))
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(path=path, start_line=int(start), end_line=int(end)))
    return out


def validate_citations(
    citations: Sequence[Citation], context: BuiltContext
) -> tuple[list[Citation], list[Citation]]:
    """Split citations into those backed by evidence and those that are not.

    An exact location match is tried first, then a +/-5 line tolerance
    against any block from the same file -- a model that cites the body of a
    function it was shown is being accurate, not inventive, even if it
    trims the decorator line.

    Args:
      citations: Citations parsed from the answer.
      context: The context the answer was generated from.

    Returns:
      A `(valid, invalid)` pair. Valid citations are mutated in place to
      carry the symbol and chunk id they matched. Invalid ones are returned
      rather than dropped, so an unsupported citation is reported in the
      answer's usage instead of quietly disappearing.
    """
    by_location = context.citation_map()
    by_path: dict[str, list[RetrievedChunk]] = {}
    for item in by_location.values():
        by_path.setdefault(item.chunk.path, []).append(item)

    valid: list[Citation] = []
    invalid: list[Citation] = []
    for citation in citations:
        location = f"{citation.path}:{citation.start_line}-{citation.end_line}"
        item = by_location.get(location)
        if item is None:
            # accept a citation whose line range sits inside a retrieved block
            for candidate in by_path.get(citation.path, []):
                chunk = candidate.chunk
                if (
                    citation.start_line >= chunk.start_line - 5
                    and citation.end_line <= chunk.end_line + 5
                ):
                    item = candidate
                    break
        if item is None:
            invalid.append(citation)
        else:
            citation.symbol = item.chunk.qualified_name
            citation.chunk_id = item.chunk.chunk_id
            valid.append(citation)
    return valid, invalid


class AnswerGenerator:
    """Prompts the LLM with evidence and validates what comes back.

    The system prompt demands three separated epistemic levels -- supported
    fact, explicit "Inference:", and an explicit statement of what the
    evidence does not establish -- so that a thin retrieval produces a
    visibly thin answer rather than a confident one.

    Attributes:
      config: Full system configuration.
      context_builder: Packs evidence into the context window.
    """

    def __init__(
        self,
        config: Config,
        *,
        llm: LLMClient | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        """Configure the generator.

        Args:
          config: Full system configuration.
          llm: Generation client; built on first use when omitted.
          context_builder: Context builder; a config-only one is built when
            omitted, which then cannot attach related symbols.
        """
        self.config = config
        self._llm = llm
        self.context_builder = context_builder or ContextBuilder(config.generation)

    @property
    def llm(self) -> LLMClient:
        """LLMClient: Generation client, connected on first access."""
        if self._llm is None:
            self._llm = build_llm(self.config.llm)
        return self._llm

    # ------------------------------------------------------------------
    def generate(
        self,
        question: str,
        results: Sequence[RetrievedChunk],
        *,
        trace: Trace | None = None,
        commit: str = "",
    ) -> Answer:
        """Generate a grounded answer from retrieved evidence.

        Args:
          question: The user's question.
          results: Retrieved evidence, best first.
          trace: Trace to record the generation step and token usage into.
          commit: Repository commit, shown to the model for context.

        Returns:
          The answer, carrying only the citations that validated. Rejected
          ones are recorded under `usage["unsupported_citations"]`, alongside
          the evidence that was dropped for budget -- which together are what
          make a bad answer diagnosable.
        """
        context = self.context_builder.build(results)
        prompt = USER_TEMPLATE.format(
            repository=self.config.repository,
            commit=(commit or "")[:10],
            question=question,
            context=context.render() or "(no evidence retrieved)",
        )
        step = None
        ctx = None
        if trace:
            ctx = trace.step(
                "generation",
                kind="generation",
                model=f"{self.config.llm.provider}:{self.config.llm.model}",
                context_tokens=context.total_tokens,
                evidence_blocks=len(context.blocks),
                dropped=len(context.dropped),
            )
            step = ctx.__enter__()

        response = self.llm.complete(prompt, system=SYSTEM_PROMPT)

        citations = parse_citations(response.text)
        valid, invalid = validate_citations(citations, context)
        if trace and step is not None:
            step.detail.update(
                {
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "citations": len(citations),
                    "unsupported_citations": len(invalid),
                }
            )
            step.results = [
                {"location": c.render(), "supported": True} for c in valid
            ] + [{"location": c.render(), "supported": False} for c in invalid]
            trace.add_usage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cost=response.cost_usd,
            )
            ctx.__exit__(None, None, None)

        answer = Answer(
            question=question,
            text=response.text,
            citations=valid,
            evidence=[b.item for b in context.blocks],
            trace_id=trace.trace_id if trace else None,
            usage={
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "cost_usd": response.cost_usd,
                "latency_ms": round(response.latency_ms, 1),
                "context_tokens": context.total_tokens,
                "unsupported_citations": [c.render() for c in invalid],
                "dropped_evidence": context.dropped[:20],
            },
        )
        if trace:
            trace.answer = answer.to_dict()
        return answer
