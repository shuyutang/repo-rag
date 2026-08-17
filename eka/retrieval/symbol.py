"""Symbol retrieval (PRD §13).

Pulls identifier-looking tokens out of a natural-language question
("what does CacheEngine.allocate_gpu_cache do?") and resolves them against the
symbol index — exactly the case where dense retrieval is weakest.
"""

from __future__ import annotations

import re

from ..schema import RetrievedChunk
from .base import BaseRetriever

_BACKTICKED = re.compile(r"`([^`]+)`")
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*\(\s*\)")
_DOTTED = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b")
_CAMEL = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+)\b")
_SNAKE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")

_STOP_SYMBOLS = {"self", "true", "false", "none"}


def extract_symbols(query: str) -> list[str]:
    """Pull identifier candidates out of a natural-language question.

    Patterns are tried most-specific first -- backticked, called, dotted,
    camelCase, snake_case -- and that order becomes the ranking bonus in
    `SymbolRetriever`.

    Args:
      query: Question text.

    Returns:
      Candidate identifiers in order of specificity, deduplicated. Names
      under three characters and Python literals are dropped.
    """
    found: list[str] = []
    for pattern in (_BACKTICKED, _CALL, _DOTTED, _CAMEL, _SNAKE):
        for match in pattern.findall(query):
            name = match.strip().strip("`.,()")
            if not name or name.lower() in _STOP_SYMBOLS or len(name) < 3:
                continue
            if name not in found:
                found.append(name)
    return found


class SymbolRetriever(BaseRetriever):
    """Resolves identifiers named in a question against the symbol index.

    Exactly the case where dense retrieval is weakest: "what does
    `CacheEngine.allocate_gpu_cache` do?" should return that definition
    first, not a paraphrase of it.
    """

    name = "symbol"

    def retrieve(
        self,
        query: str,
        k: int = 10,
        *,
        symbols: list[str] | None = None,
        **_: object,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks defining or matching identifiers in the query.

        Args:
          query: Question text; identifiers are extracted from it.
          k: Maximum results.
          symbols: Explicit identifiers to look up, bypassing extraction.
            The agent uses this to look up a name it decided on itself.
          **_: Options for other retrievers, ignored.

        Returns:
          Results ranked by symbol-match score, with earlier (more specific)
          candidates weighted slightly higher. Empty when the query contains
          no identifier-shaped token at all, which is the common case for
          conceptual questions.

        Raises:
          RuntimeError: The knowledge base has no symbol index.
        """
        if self.kb.symbol_index is None:
            raise RuntimeError("knowledge base has no symbol index; run `rag index`")
        names = symbols if symbols is not None else extract_symbols(query)
        if not names:
            return []
        scored: dict[str, float] = {}
        for position, name in enumerate(names):
            # earlier (more specific) candidates get a small priority bonus
            weight = 1.0 / (1.0 + 0.25 * position)
            for chunk_id, score in self.kb.symbol_index.search(name, k=k):
                scored[chunk_id] = max(scored.get(chunk_id, 0.0), score * weight)
        hits = sorted(scored.items(), key=lambda kv: -kv[1])[:k]
        return self.wrap(hits, query=query)
