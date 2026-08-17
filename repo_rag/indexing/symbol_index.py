"""Symbol index (PRD §13): exact and fuzzy lookup of code identifiers."""

from __future__ import annotations

import difflib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..schema import Chunk
from .tokenizer import symbol_variants

# definition-bearing chunks outrank the module overview that merely mentions them
_TYPE_PRIORITY = {"class": 0, "function": 1, "method": 2, "module": 3, "file": 4,
                  "section": 5, "commit": 6}


class SymbolIndex:
    """Exact and fuzzy lookup of code identifiers.

    This is the retriever a developer actually wants when they type a name:
    neither BM25 nor a dense model can promise that `allocate_slots` outranks
    every chunk that merely mentions it, and this index can, because it knows
    which chunk *defines* the symbol.

    Attributes:
      exact: Lookup key to chunk ids, definitions first.
      definitions: Qualified name to the chunk id that defines it.
      names: Every known display name, the candidate pool for fuzzy matching.
    """

    def __init__(
        self,
        exact: dict[str, list[str]],
        definitions: dict[str, str],
        names: list[str],
    ) -> None:
        """Store the prebuilt lookup tables.

        Args:
          exact: Lookup key to chunk ids, already ordered and truncated.
          definitions: Qualified name to defining chunk id.
          names: Every known display name.
        """
        self.exact = exact
        self.definitions = definitions   # qualified name -> chunk_id
        self.names = names               # all known display names (for fuzzy)
        self._lower_names = {n.lower(): n for n in names}

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, chunks: Iterable[Chunk]) -> "SymbolIndex":
        """Build the index from chunks.

        Every chunk is registered under each of its name variants, and the
        posting list for a key is sorted so that definition-bearing chunks
        (class, function, method) outrank the module overview that merely
        mentions the name.

        Args:
          chunks: Chunks to index. Commit chunks and chunks with no symbol
            type are skipped, having no identifiers to look up.

        Returns:
          The built index, each posting list capped at 50 entries.
        """
        exact: dict[str, list[tuple[int, str]]] = defaultdict(list)
        definitions: dict[str, str] = {}
        names: set[str] = set()
        for chunk in chunks:
            if chunk.symbol_type in (None, "commit"):
                continue
            keys: set[str] = set()
            if chunk.symbol:
                keys |= symbol_variants(chunk.symbol)
                names.add(chunk.symbol)
            if chunk.qualified_name:
                keys |= symbol_variants(chunk.qualified_name)
                names.add(chunk.qualified_name)
                if chunk.symbol_type in ("class", "function", "method", "module"):
                    definitions.setdefault(chunk.qualified_name, chunk.chunk_id)
            priority = _TYPE_PRIORITY.get(chunk.symbol_type or "", 9)
            for key in keys:
                exact[key].append((priority, chunk.chunk_id))
        ordered = {
            key: [cid for _, cid in sorted(vals, key=lambda kv: kv[0])][:50]
            for key, vals in exact.items()
        }
        return cls(ordered, definitions, sorted(names))

    # ------------------------------------------------------------------
    def lookup(self, name: str, k: int = 10) -> list[tuple[str, float]]:
        """Look a symbol up by exact name or name variant.

        Args:
          name: Symbol name, bare or dotted.
          k: Maximum results.

        Returns:
          `(chunk_id, score)` pairs, scoring 1.0 for an exact case-insensitive
          match and 0.9 for a variant match such as a subtoken join.
        """
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        for score, key in ((1.0, name.lower()), *[(0.9, v) for v in symbol_variants(name)]):
            for chunk_id in self.exact.get(key, []):
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    out.append((chunk_id, score))
            if len(out) >= k:
                break
        return out[:k]

    def fuzzy(self, name: str, k: int = 10, cutoff: float = 0.72) -> list[tuple[str, float]]:
        """Look a symbol up by approximate string similarity.

        This is what catches a misremembered or misspelled identifier. It
        compares against every known name, so its cost grows linearly with
        the corpus -- the first thing that would need replacing at a scale
        well beyond this one.

        Args:
          name: Approximate symbol name.
          k: Maximum results.
          cutoff: Minimum similarity ratio for a candidate name to qualify.

        Returns:
          `(chunk_id, score)` pairs, the exact-lookup score scaled by the
          similarity ratio so that a fuzzy hit can never outrank an exact one.
        """
        matches = difflib.get_close_matches(name, self.names, n=k, cutoff=cutoff)
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        for match in matches:
            ratio = difflib.SequenceMatcher(None, name.lower(), match.lower()).ratio()
            for chunk_id, base in self.lookup(match, k=3):
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    out.append((chunk_id, round(base * ratio, 4)))
        return out[:k]

    def search(self, name: str, k: int = 10) -> list[tuple[str, float]]:
        """Look a symbol up, falling back to fuzzy matching to fill out `k`.

        Args:
          name: Symbol name.
          k: Maximum results.

        Returns:
          `(chunk_id, score)` pairs, exact matches first and fuzzy matches
          only where exact lookup came up short.
        """
        out = self.lookup(name, k=k)
        if len(out) < k:
            have = {cid for cid, _ in out}
            out.extend(
                (cid, s) for cid, s in self.fuzzy(name, k=k) if cid not in have
            )
        return out[:k]

    def definition_of(self, qualified_name: str) -> str | None:
        """Find the chunk that defines a qualified name.

        A suffix match is allowed, so "CacheEngine.allocate" resolves to
        "vllm.worker.cache_engine.CacheEngine.allocate" -- callers rarely have
        the fully qualified name to hand.

        Args:
          qualified_name: Dotted name, fully qualified or a suffix of one.

        Returns:
          The defining chunk id, or `None` if nothing matches.
        """
        if qualified_name in self.definitions:
            return self.definitions[qualified_name]
        # allow a suffix match: "CacheEngine.allocate" -> "vllm.worker...allocate"
        tail = qualified_name.lower()
        for name, chunk_id in self.definitions.items():
            if name.lower().endswith(tail):
                return chunk_id
        return None

    # ------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        """Write the index to `symbols.json` in a directory.

        Args:
          directory: Destination, created if absent.
        """
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "symbols.json", "w", encoding="utf-8") as fh:
            json.dump(
                {"exact": self.exact, "definitions": self.definitions, "names": self.names},
                fh,
            )

    @classmethod
    def load(cls, directory: Path) -> "SymbolIndex":
        """Read an index back from a directory written by `save`.

        Args:
          directory: Directory holding `symbols.json`.

        Returns:
          The loaded index.
        """
        with open(directory / "symbols.json", "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data["exact"], data["definitions"], data["names"])
