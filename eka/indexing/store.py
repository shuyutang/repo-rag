"""Chunk store: the single source of truth an index row maps back to."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from ..schema import Chunk


class ChunkStore:
    """In-memory chunk collection with the lookups retrievers need.

    Every index stores row positions into this list, so the store is the one
    place a retrieval result becomes a citable chunk again. All four
    secondary maps are built eagerly at construction: it costs one pass over
    the corpus, and it makes every lookup below O(1).

    Holding the whole corpus in memory is what caps the system at roughly
    10^5-10^6 chunks on a single machine.

    Attributes:
      chunks: The chunks, in index-row order.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        """Take ownership of `chunks` and build the lookup maps.

        Args:
          chunks: Chunks in the order the indexes were built from.
        """
        self.chunks = chunks
        self._by_id: dict[str, int] = {c.chunk_id: i for i, c in enumerate(chunks)}
        self._by_path: dict[str, list[int]] = defaultdict(list)
        self._by_qualified: dict[str, list[int]] = defaultdict(list)
        self._by_symbol: dict[str, list[int]] = defaultdict(list)
        for i, chunk in enumerate(chunks):
            self._by_path[chunk.path].append(i)
            if chunk.qualified_name:
                self._by_qualified[chunk.qualified_name].append(i)
            if chunk.symbol:
                self._by_symbol[chunk.symbol].append(i)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of chunks."""
        return len(self.chunks)

    def __iter__(self) -> Iterator[Chunk]:
        """Iterate the chunks in row order."""
        return iter(self.chunks)

    def __getitem__(self, index: int) -> Chunk:
        """Return the chunk at an index-row position."""
        return self.chunks[index]

    def get(self, chunk_id: str) -> Chunk | None:
        """Look a chunk up by id.

        Args:
          chunk_id: Stable chunk identity.

        Returns:
          The chunk, or `None` if this store does not hold it.
        """
        idx = self._by_id.get(chunk_id)
        return self.chunks[idx] if idx is not None else None

    def index_of(self, chunk_id: str) -> int | None:
        """Return a chunk's index row, or `None` if it is not in this store."""
        return self._by_id.get(chunk_id)

    def by_path(self, path: str) -> list[Chunk]:
        """Return every chunk from one file, in source order."""
        return [self.chunks[i] for i in self._by_path.get(path, [])]

    def by_qualified_name(self, name: str) -> list[Chunk]:
        """Return chunks whose qualified name matches exactly.

        Args:
          name: Dotted name, e.g. "vllm.core.block.BlockTable.allocate".

        Returns:
          Matching chunks; more than one when an oversized symbol was split.
        """
        return [self.chunks[i] for i in self._by_qualified.get(name, [])]

    def by_symbol(self, name: str) -> list[Chunk]:
        """Return chunks whose bare symbol name matches exactly.

        Args:
          name: Bare name, e.g. "allocate".

        Returns:
          Matching chunks across every file, since bare names collide freely.
        """
        return [self.chunks[i] for i in self._by_symbol.get(name, [])]

    @property
    def paths(self) -> list[str]:
        """list[str]: Every file path with at least one chunk."""
        return list(self._by_path.keys())

    def stats(self) -> dict[str, int]:
        """Count chunks per artifact type.

        Returns:
          A mapping of artifact type to chunk count.
        """
        out: dict[str, int] = defaultdict(int)
        for chunk in self.chunks:
            out[chunk.artifact_type] += 1
        return dict(out)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "ChunkStore":
        """Read a store from a JSON Lines file.

        Read line by line rather than into one buffer, so peak memory stays
        close to the size of the corpus itself.

        Args:
          path: `chunks.jsonl` written by `save`.

        Returns:
          The populated store.
        """
        chunks: list[Chunk] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    chunks.append(Chunk.from_dict(json.loads(line)))
        return cls(chunks)

    def save(self, path: Path) -> None:
        """Write the store to `path` as JSON Lines, creating parent dirs.

        Line order is index-row order, which is what makes the file
        round-trip through `load` with row positions intact.

        Args:
          path: Destination file, overwritten if present.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def from_chunks(cls, chunks: Iterable[Chunk]) -> "ChunkStore":
        """Build a store from any iterable of chunks, preserving order."""
        return cls(list(chunks))
