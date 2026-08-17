"""Lightweight symbol graph for change-impact reasoning (PRD M8).

This is deliberately *not* a compiler-grade call graph (explicit non-goal, §4).
It is a name-resolution heuristic built from the AST facts the parser already
extracted:

    symbol -> callers      (chunks whose body calls that name)
    symbol -> tests        (test chunks that reference the name)
    module -> importers    (modules importing it)
    symbol -> definition   (defining chunk)

The edges are recall-oriented: a call to ``allocate`` matches every ``allocate``
in the repository.  Precision is recovered downstream by reranking and by the
answer generator, which sees the actual source.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..schema import Chunk

# names so common that "who calls this" is meaningless
_NOISE = {
    "append", "get", "len", "str", "int", "list", "dict", "set", "print", "range",
    "format", "join", "add", "update", "items", "keys", "values", "isinstance",
    "super", "type", "min", "max", "sum", "sorted", "enumerate", "zip", "open",
    "getattr", "setattr", "hasattr", "extend", "pop", "copy", "next", "iter",
    "tuple", "float", "bool", "abs", "round", "any", "all", "map", "filter",
}


@dataclass
class ImpactReport:
    """What a change to one symbol could plausibly touch.

    Attributes:
      symbol: The symbol asked about.
      definitions: Chunk ids that define it.
      callers: Chunk ids of non-test code referencing it.
      tests: Chunk ids of test code referencing it.
      importers: Module names importing it.
      same_file: Other chunk ids from the defining files.
    """

    symbol: str
    definitions: list[str] = field(default_factory=list)   # chunk ids
    callers: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    importers: list[str] = field(default_factory=list)     # module names
    same_file: list[str] = field(default_factory=list)

    def all_chunk_ids(self) -> list[str]:
        """Flatten every chunk id in the report.

        Returns:
          Definitions, callers, tests and same-file chunks in that order,
          deduplicated with the first occurrence kept -- so the ordering
          already encodes relevance.
        """
        seen: list[str] = []
        for group in (self.definitions, self.callers, self.tests, self.same_file):
            for cid in group:
                if cid not in seen:
                    seen.append(cid)
        return seen

    def to_dict(self) -> dict:
        """Return the report as a JSON-serialisable dict for the API."""
        return {
            "symbol": self.symbol,
            "definitions": self.definitions,
            "callers": self.callers,
            "tests": self.tests,
            "importers": self.importers,
            "same_file": self.same_file,
        }


class SymbolGraph:
    """Name-resolution graph over symbols, callers, tests and imports.

    Deliberately not a compiler-grade call graph. Edges are built from AST
    facts the parser already extracted and matched by *name*, so a call to
    `allocate` matches every `allocate` in the repository. The edges are
    recall-oriented on purpose: precision is recovered downstream by reranking
    and by the answer generator, which sees the actual source.

    Attributes:
      callers: Symbol name to chunk ids of non-test code referencing it.
      tests: Symbol name to chunk ids of test code referencing it.
      importers: Module name to the module names importing it.
      definitions: Symbol name to chunk ids defining it.
      file_of: Chunk id to the path it came from.
    """

    def __init__(
        self,
        callers: dict[str, list[str]],
        tests: dict[str, list[str]],
        importers: dict[str, list[str]],
        definitions: dict[str, list[str]],
        file_of: dict[str, str],
    ) -> None:
        """Store the prebuilt edge maps.

        Args:
          callers: Symbol name to referencing non-test chunk ids.
          tests: Symbol name to referencing test chunk ids.
          importers: Module name to importing module names.
          definitions: Symbol name to defining chunk ids.
          file_of: Chunk id to source path.
        """
        self.callers = callers
        self.tests = tests
        self.importers = importers
        self.definitions = definitions
        self.file_of = file_of
        self._chunks_in_file: dict[str, list[str]] | None = None

    def _file_map(self) -> dict[str, list[str]]:
        """Invert `file_of` into path to chunk ids, computed once and cached.

        Returns:
          A mapping of path to the chunk ids from that file.
        """
        if self._chunks_in_file is None:
            grouped: dict[str, list[str]] = defaultdict(list)
            for cid, path in self.file_of.items():
                grouped[path].append(cid)
            self._chunks_in_file = dict(grouped)
        return self._chunks_in_file

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, chunks: Iterable[Chunk], *, max_edges: int = 60) -> "SymbolGraph":
        """Build the graph from chunks.

        References from test chunks become test edges rather than caller
        edges, which is what makes "which tests cover this?" answerable.
        Very common names and names under three characters are dropped: "who
        calls `get`" has no useful answer.

        Args:
          chunks: Chunks to build from. Commit chunks are skipped.
          max_edges: Cap on edges per symbol, keeping a heavily used name
            from dominating the graph file.

        Returns:
          The built graph.
        """
        callers: dict[str, set[str]] = defaultdict(set)
        tests: dict[str, set[str]] = defaultdict(set)
        importers: dict[str, set[str]] = defaultdict(set)
        definitions: dict[str, set[str]] = defaultdict(set)
        file_of: dict[str, str] = {}

        for chunk in chunks:
            if chunk.artifact_type == "commit":
                continue
            file_of[chunk.chunk_id] = chunk.path
            if chunk.symbol and chunk.symbol_type in ("class", "function", "method"):
                definitions[chunk.symbol].add(chunk.chunk_id)
            for ref in chunk.references:
                if ref in _NOISE or len(ref) < 3:
                    continue
                if chunk.artifact_type == "test":
                    tests[ref].add(chunk.chunk_id)
                else:
                    callers[ref].add(chunk.chunk_id)
            for imported in chunk.imports:
                leaf = imported.rsplit(".", 1)[-1]
                if chunk.artifact_type == "test":
                    tests[leaf].add(chunk.chunk_id)
                module = chunk.parent_symbol or chunk.qualified_name or chunk.path
                importers[imported].add(module)
                importers[leaf].add(module)

        def trim(d: dict[str, set[str]]) -> dict[str, list[str]]:
            return {k: sorted(v)[:max_edges] for k, v in d.items() if v}

        return cls(trim(callers), trim(tests), trim(importers), trim(definitions), file_of)

    # ------------------------------------------------------------------
    def impact(self, symbol: str, *, limit: int = 25) -> ImpactReport:
        """Report what a change to a symbol could reach.

        Lookup is by the last dotted component, so a qualified name and a
        bare name behave alike.

        Args:
          symbol: Symbol name, bare or dotted.
          limit: Cap per category.

        Returns:
          The impact report. Definitions are excluded from callers so a
          symbol is never reported as calling itself.
        """
        leaf = symbol.rsplit(".", 1)[-1]
        defs = self.definitions.get(leaf, [])
        report = ImpactReport(symbol=symbol, definitions=defs[:limit])
        report.callers = [c for c in self.callers.get(leaf, []) if c not in defs][:limit]
        report.tests = self.tests.get(leaf, [])[:limit]
        report.importers = self.importers.get(leaf, [])[:limit]
        file_map = self._file_map()
        same_file: list[str] = []
        for cid in defs:
            path = self.file_of.get(cid)
            same_file.extend(c for c in file_map.get(path or "", []) if c not in defs)
        report.same_file = same_file[:limit]
        return report

    def neighbors(self, chunk_id: str) -> list[str]:
        """Return chunks structurally adjacent to a chunk.

        Args:
          chunk_id: Chunk to find neighbours of.

        Returns:
          Other chunk ids from the same file, which is the cheap proxy for
          adjacency the context builder uses to attach a related symbol.
        """
        path = self.file_of.get(chunk_id)
        if not path:
            return []
        return [cid for cid in self._file_map().get(path, []) if cid != chunk_id]

    # ------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        """Write the graph to `graph.json` in a directory.

        Args:
          directory: Destination, created if absent.
        """
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "graph.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "callers": self.callers,
                    "tests": self.tests,
                    "importers": self.importers,
                    "definitions": self.definitions,
                    "file_of": self.file_of,
                },
                fh,
            )

    @classmethod
    def load(cls, directory: Path) -> "SymbolGraph":
        """Read a graph back from a directory written by `save`.

        Args:
          directory: Directory holding `graph.json`.

        Returns:
          The loaded graph.
        """
        with open(directory / "graph.json", "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            data["callers"], data["tests"], data["importers"],
            data["definitions"], data["file_of"],
        )
