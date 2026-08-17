"""Retrieval tools the agent can call (PRD §17, §19).

Each tool is a small, independently testable wrapper over the retrieval layer.
Tools return ``RetrievedChunk`` lists so every downstream stage — fusion,
context building, citation validation — works unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..indexing.knowledge_base import KnowledgeBase
from ..retrieval.hybrid import HybridRetriever, RetrievalRequest
from ..schema import RetrievedChunk


@dataclass
class ToolCall:
    """A record of one tool invocation, for the trace.

    Attributes:
      tool: Tool name.
      args: Arguments as passed.
      n_results: Results returned.
      note: Failure explanation; empty when the call succeeded.
    """

    tool: str
    args: dict[str, Any]
    n_results: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the call record as a JSON-serialisable dict."""
        return {
            "tool": self.tool,
            "args": self.args,
            "n_results": self.n_results,
            "note": self.note,
        }


class ToolBox:
    """The tool surface exposed to the retrieval agent.

    Every tool returns `RetrievedChunk` lists, so fusion, context building
    and citation validation all work downstream without knowing which tool
    produced what.

    Attributes:
      kb: Knowledge base backing the graph-based tools.
      retriever: Hybrid retriever backing the search-based tools.
    """

    def __init__(self, kb: KnowledgeBase, retriever: HybridRetriever) -> None:
        """Bind the toolbox to a knowledge base and retriever.

        Args:
          kb: Loaded knowledge base.
          retriever: Hybrid retriever to search through.
        """
        self.kb = kb
        self.retriever = retriever

    # -- descriptions handed to the planner LLM -------------------------
    def describe(self) -> str:
        """Describe the tools for the agent's inspection prompt.

        Returns:
          A plain-text listing of each tool's name, arguments and purpose.
        """
        return (
            "- search(query, artifact_types?): hybrid dense+lexical search over code, "
            "tests, docs and commits. artifact_types is an optional subset of "
            '["source", "test", "doc", "commit", "config"].\n'
            "- symbol_search(name): exact/fuzzy lookup of a code identifier "
            "(class, function, method, module).\n"
            "- history(query, path?): search commit messages and diffs.\n"
            "- impact(symbol): callers, tests and importers of a symbol.\n"
            "- expand(chunk_id): other chunks defined in the same file."
        )

    # -- tools -----------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 8,
        artifact_types: list[str] | None = None,
        use_reranker: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Search code, tests, docs and commits with the hybrid retriever.

        Args:
          query: Query text.
          k: Maximum results.
          artifact_types: Artifact types to restrict to.
          use_reranker: Override the reranker setting. Defaults to off,
            because the agent reranks once at the end over the merged
            evidence rather than per tool call.

        Returns:
          Hybrid retrieval results.
        """
        return self.retriever.retrieve(
            query,
            request=RetrievalRequest(
                query=query, k=k, artifact_types=artifact_types,
                use_reranker=False if use_reranker is None else use_reranker,
            ),
        )

    def symbol_search(self, name: str, k: int = 6) -> list[RetrievedChunk]:
        """Look up a code identifier exactly or fuzzily.

        Args:
          name: Identifier to look up.
          k: Maximum results.

        Returns:
          Symbol-index results, or an empty list when no symbol index exists.
        """
        retriever = self.retriever._retrievers.get("symbol")
        if retriever is None:
            return []
        return retriever.retrieve(name, k, symbols=[name])

    def history(self, query: str, k: int = 6, path: str | None = None) -> list[RetrievedChunk]:
        """Search commit messages and diffs.

        Args:
          query: Query text.
          k: Maximum results.
          path: Restrict to commits touching a matching path.

        Returns:
          Commit results, or an empty list when history was not ingested.
        """
        retriever = self.retriever._retrievers.get("git")
        if retriever is None:
            return []
        return retriever.retrieve(query, k, touching_path=path)

    def impact(self, symbol: str, k: int = 10) -> list[RetrievedChunk]:
        """Return the structural neighbourhood of a symbol.

        Args:
          symbol: Symbol name, bare or dotted.
          k: Cap per category.

        Returns:
          Definitions, then callers, then tests, scored 1.0, 0.7 and 0.6
          respectively and decayed by rank within each group -- so a
          definition always precedes its callers. Empty when the knowledge
          base has no symbol graph.
        """
        if self.kb.graph is None:
            return []
        report = self.kb.graph.impact(symbol, limit=k)
        out: list[RetrievedChunk] = []
        groups = (
            ("definition", report.definitions, 1.0),
            ("caller", report.callers, 0.7),
            ("test", report.tests, 0.6),
        )
        for label, chunk_ids, weight in groups:
            for rank, chunk_id in enumerate(chunk_ids[:k], start=1):
                chunk = self.kb.store.get(chunk_id)
                if chunk is None:
                    continue
                out.append(
                    RetrievedChunk(
                        chunk=chunk,
                        score=weight / (1 + 0.1 * rank),
                        retriever=f"graph:{label}",
                        rank=len(out) + 1,
                        component_scores={"graph": weight},
                        query=symbol,
                    )
                )
        return out

    def expand(self, chunk_id: str, k: int = 5) -> list[RetrievedChunk]:
        """Return other chunks defined in the same file as a chunk.

        Args:
          chunk_id: Chunk to expand around.
          k: Maximum results.

        Returns:
          Same-file neighbours with rank-decayed scores, or an empty list
          when the knowledge base has no symbol graph.
        """
        if self.kb.graph is None:
            return []
        out: list[RetrievedChunk] = []
        for rank, neighbor in enumerate(self.kb.graph.neighbors(chunk_id)[:k], start=1):
            chunk = self.kb.store.get(neighbor)
            if chunk is not None:
                out.append(
                    RetrievedChunk(
                        chunk=chunk,
                        score=0.5 / rank,
                        retriever="graph:same_file",
                        rank=rank,
                        query=chunk_id,
                    )
                )
        return out

    # ------------------------------------------------------------------
    def dispatch(self, tool: str, args: dict[str, Any]) -> tuple[list[RetrievedChunk], ToolCall]:
        """Invoke a tool by name with model-supplied arguments.

        Every failure mode -- unknown tool, wrong arguments, an exception
        inside the tool -- is caught and recorded on the call rather than
        raised, because the arguments come from an LLM and one bad tool call
        must not kill the query.

        Args:
          tool: Tool name.
          args: Keyword arguments for the tool.

        Returns:
          A `(results, call_record)` pair. Results are empty on failure, and
          `call_record.note` says what went wrong.
        """
        handlers: dict[str, Callable[..., list[RetrievedChunk]]] = {
            "search": self.search,
            "symbol_search": self.symbol_search,
            "history": self.history,
            "impact": self.impact,
            "expand": self.expand,
        }
        handler = handlers.get(tool)
        call = ToolCall(tool=tool, args=dict(args))
        if handler is None:
            call.note = f"unknown tool {tool!r}"
            return [], call
        try:
            results = handler(**args)
        except TypeError as exc:
            call.note = f"bad arguments: {exc}"
            return [], call
        except Exception as exc:  # a failing tool must not kill the query
            call.note = f"error: {exc}"
            return [], call
        call.n_results = len(results)
        return results, call
