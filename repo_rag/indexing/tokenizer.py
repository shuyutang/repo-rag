"""Code-aware tokenisation shared by BM25 and the symbol index.

`CacheEngine.allocate_gpu_cache` has to match a query that says "allocate gpu
cache", so identifiers are emitted *both* whole and split into their camelCase
and snake_case components. This one detail is most of why BM25 outperforms
dense retrieval on this corpus.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+|\d+")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to", "in",
    "for", "on", "with", "as", "by", "at", "from", "that", "this", "it", "its",
    "and", "or", "not", "how", "what", "where", "which", "when", "why", "does",
    "do", "did", "can", "could", "should", "would", "i", "we", "you", "if",
    "into", "about", "there", "here", "then", "than", "so", "such", "any",
    "self", "def", "class", "return", "import", "none", "true", "false",
}


def subtokens(word: str) -> list[str]:
    """Split an identifier into its camelCase and snake_case parts.

    Args:
      word: A single identifier, e.g. "allocate_gpu_cache" or "KVCache".

    Returns:
      Lowercased parts of more than one character, e.g. `["kv", "cache"]`.
      Single characters are dropped as noise.
    """
    parts = [p.lower() for p in _CAMEL.findall(word) if p]
    return [p for p in parts if len(p) > 1]


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Tokenise text for lexical matching.

    Each identifier is emitted whole *and* as its parts, so "allocate gpu
    cache" matches `allocate_gpu_cache` without the query needing the exact
    identifier.

    Args:
      text: Query or document text.
      keep_stopwords: Retain stopwords. The stoplist includes Python keywords
        such as "class" and "return", which are near-universal in this corpus
        and therefore carry no signal.

    Returns:
      Lowercased tokens in order, with duplicates preserved so that term
      frequency is meaningful.
    """
    tokens: list[str] = []
    for match in _WORD.findall(text):
        low = match.lower()
        if len(low) > 1 and (keep_stopwords or low not in STOPWORDS):
            tokens.append(low)
        for part in subtokens(match):
            if part != low and (keep_stopwords or part not in STOPWORDS):
                tokens.append(part)
    return tokens


def normalize_symbol(name: str) -> str:
    """Normalise a symbol name for lookup: strip whitespace, "()" and case."""
    return name.strip().strip("()").lower()


def symbol_variants(name: str) -> set[str]:
    """Enumerate the keys a symbol should be findable under.

    Args:
      name: Symbol name, bare or dotted.

    Returns:
      A set containing the lowercased name, its last dotted component, and
      its subtokens joined with and without underscores -- so that
      `KVCache.allocate_slots` is reachable as "allocate_slots",
      "allocateslots" and "allocate_slots" alike. Empty strings are dropped.
    """
    name = name.strip()
    out = {name.lower()}
    if "." in name:
        out.add(name.rsplit(".", 1)[-1].lower())
    out.add("".join(subtokens(name)))
    out.add("_".join(subtokens(name)))
    return {o for o in out if o}
