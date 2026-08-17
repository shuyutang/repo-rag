"""End-to-end pipeline: question -> retrieval (-> agent) -> context -> answer.

One object the CLI, the API and the benchmark all share, so that a number
reported by the benchmark comes from exactly the code path the demo runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .agent.retrieval_agent import AgentResult, RetrievalAgent
from .config import Config
from .generation.answer_generator import AnswerGenerator
from .generation.context_builder import ContextBuilder
from .generation.llm import LLMClient, build_llm
from .indexing.knowledge_base import KnowledgeBase
from .observability.tracing import Trace, TraceStore
from .retrieval.hybrid import HybridRetriever, RetrievalRequest, build_retriever
from .retrieval.reranker import Reranker
from .schema import Answer, RetrievedChunk


@dataclass
class QueryResult:
    """Everything one `Pipeline.ask` call produced.

    Attributes:
      answer: The generated answer with its validated citations.
      evidence: Chunks actually placed in the context window.
      trace: Per-stage timings, results and token usage for this query.
      agent: Agent transcript when agentic retrieval ran, otherwise `None`.
    """

    answer: Answer
    evidence: list[RetrievedChunk]
    trace: Trace
    agent: AgentResult | None = None

    def to_dict(self) -> dict:
        """Return the result as the JSON payload the API serves."""
        return {
            "answer": self.answer.to_dict(),
            "trace": self.trace.to_dict(),
            "agent": self.agent.to_dict() if self.agent else None,
        }


class Pipeline:
    """Question in, grounded answer out.

    The CLI, the API and the benchmark all drive this one object, so a number
    reported by the benchmark comes from exactly the code path the demo runs.
    The LLM client and the agent are built lazily, which is what keeps
    retrieval-only work -- `rag search`, the retrieval benchmark -- free of
    any dependency on a running generation backend.

    Attributes:
      config: The configuration this pipeline was built from.
      kb: Loaded knowledge base: chunks plus all four indexes.
      retriever: Hybrid retriever used for single-pass retrieval.
      trace_store: Where traces are persisted.
      context_builder: Packs evidence into the context window.
      generator: Prompts the LLM and validates the citations it returns.
    """

    def __init__(
        self,
        config: Config,
        kb: KnowledgeBase,
        *,
        retriever: HybridRetriever | None = None,
        llm: LLMClient | None = None,
        reranker: Reranker | None = None,
        trace_store: TraceStore | None = None,
    ) -> None:
        """Wire the pipeline together.

        Args:
          config: Full system configuration.
          kb: Loaded knowledge base.
          retriever: Retriever to use; built from `kb` when omitted.
          llm: Generation client; built on first use when omitted.
          reranker: Reranker passed to the default retriever. Ignored when
            `retriever` is given, since that retriever carries its own.
          trace_store: Trace sink; defaults to `config.trace_dir`.
        """
        self.config = config
        self.kb = kb
        self.retriever = retriever or build_retriever(kb, reranker=reranker)
        self._llm = llm
        self.trace_store = trace_store or TraceStore(config.resolve(config.trace_dir))
        self.context_builder = ContextBuilder(
            config.generation, store=kb.store, graph=kb.graph
        )
        self.generator = AnswerGenerator(
            config, llm=self._llm, context_builder=self.context_builder
        )
        self._agent: RetrievalAgent | None = None

    # ------------------------------------------------------------------
    @property
    def llm(self) -> LLMClient:
        """LLMClient: Generation client, connected on first access."""
        if self._llm is None:
            self._llm = build_llm(self.config.llm)
            self.generator._llm = self._llm
        return self._llm

    @property
    def agent(self) -> RetrievalAgent:
        """RetrievalAgent: Agentic retriever, constructed on first access."""
        if self._agent is None:
            self._agent = RetrievalAgent(
                self.kb, self.retriever, config=self.config, llm=self.llm
            )
        return self._agent

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, config: Config, **kwargs) -> "Pipeline":
        """Load the knowledge base named by a config and build a pipeline.

        Args:
          config: Configuration naming the index directory to load.
          **kwargs: Forwarded to `__init__`.

        Returns:
          A ready pipeline.

        Raises:
          FileNotFoundError: The index directory has not been built yet.
        """
        kb = KnowledgeBase.load(config)
        return cls(config, kb, **kwargs)

    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        *,
        trace: Trace | None = None,
        sources: Sequence[str] | None = None,
        k: int | None = None,
        use_reranker: bool | None = None,
        use_agent: bool | None = None,
    ) -> tuple[list[RetrievedChunk], AgentResult | None]:
        """Retrieve evidence without generating an answer.

        This is the path the retrieval benchmark measures, and it makes no
        LLM call unless agentic retrieval is on.

        Args:
          question: The user's question.
          trace: Trace to record stage timings into, if any.
          sources: Retriever names to restrict to, e.g. `("bm25",)`. `None`
            uses every configured source.
          k: Number of results to return; falls back to the configured
            `final_k`, or `max_evidence` in agentic mode.
          use_reranker: Override the configured reranker setting.
          use_agent: Override the configured `agent.enabled` setting.

        Returns:
          A `(evidence, agent_result)` pair. `agent_result` is `None` for
          single-pass retrieval.
        """
        agentic = self.config.agent.enabled if use_agent is None else use_agent
        if agentic:
            result = self.agent.run(question, trace=trace)
            evidence = result.evidence[: k or self.config.agent.max_evidence]
            return evidence, result
        evidence = self.retriever.retrieve(
            question,
            trace=trace,
            request=RetrievalRequest(
                query=question, k=k, sources=sources, use_reranker=use_reranker
            ),
        )
        return evidence, None

    def ask(
        self,
        question: str,
        *,
        sources: Sequence[str] | None = None,
        k: int | None = None,
        use_reranker: bool | None = None,
        use_agent: bool | None = None,
        save_trace: bool = True,
    ) -> QueryResult:
        """Answer a question end to end.

        Args:
          question: The user's question.
          sources: Retriever names to restrict to; `None` uses all of them.
          k: Evidence chunks to retrieve.
          use_reranker: Override the configured reranker setting.
          use_agent: Override the configured `agent.enabled` setting.
          save_trace: Persist the trace to `trace_dir`. The benchmark turns
            this off to avoid writing thousands of files.

        Returns:
          The answer, the evidence behind it, the trace, and the agent
          transcript when the agent ran.
        """
        trace = Trace(question, config_fingerprint=self.config.fingerprint())
        evidence, agent_result = self.retrieve(
            question,
            trace=trace,
            sources=sources,
            k=k,
            use_reranker=use_reranker,
            use_agent=use_agent,
        )
        answer = self.generator.generate(
            question, evidence, trace=trace, commit=self.kb.commit
        )
        if save_trace:
            self.trace_store.save(trace)
        return QueryResult(
            answer=answer, evidence=evidence, trace=trace, agent=agent_result
        )
