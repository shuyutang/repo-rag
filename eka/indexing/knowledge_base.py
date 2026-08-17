"""The knowledge base: builds and loads every index for one repository."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import Config
from ..schema import Chunk
from .bm25_index import BM25Index
from .embeddings import Embedder, build_embedder
from .graph_index import SymbolGraph
from .store import ChunkStore
from .symbol_index import SymbolIndex
from .vector_index import VectorIndex

CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"


@dataclass
class KnowledgeBase:
    """Every index for one repository, plus the chunks they point back to.

    This is the object retrieval is built on: the four indexes are
    independent, and each retriever uses exactly one of them.

    Attributes:
      config: Configuration this knowledge base was built or loaded with.
      store: The chunks, in index-row order.
      vector_index: Dense index; `None` when built without embeddings.
      bm25_index: Lexical index.
      symbol_index: Identifier lookup index.
      graph: Symbol graph for change-impact reasoning.
      meta: Index metadata: repository, commit, counts, models, timings.
    """

    config: Config
    store: ChunkStore
    vector_index: VectorIndex | None = None
    bm25_index: BM25Index | None = None
    symbol_index: SymbolIndex | None = None
    graph: SymbolGraph | None = None
    meta: dict | None = None
    _embedder: Embedder | None = None

    # ------------------------------------------------------------------
    @property
    def embedder(self) -> Embedder:
        """Embedder: Query encoder, loaded on first access.

        Lazy because BM25, symbol and graph retrieval never need it, and
        loading a sentence-transformers model costs seconds.
        """
        if self._embedder is None:
            self._embedder = build_embedder(self.config.embedding)
        return self._embedder

    @property
    def commit(self) -> str:
        """str: Repository HEAD this index was built from, or "" if unknown."""
        return (self.meta or {}).get("commit", "")

    def row_mask(self, predicate: Callable[[Chunk], bool]) -> np.ndarray:
        """Build a boolean mask over index rows from a per-chunk predicate.

        Index rows follow store order, so one mask filters every index. The
        predicate runs in Python once per chunk, which is why filtered
        retrieval costs milliseconds at 10^5 chunks and would need a
        different design well beyond that.

        Args:
          predicate: Called with each chunk; True keeps the row.

        Returns:
          A boolean array of length `len(store)`.
        """
        return np.fromiter(
            (predicate(c) for c in self.store.chunks), dtype=bool, count=len(self.store)
        )

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        config: Config,
        chunks: list[Chunk],
        *,
        progress: Callable[[str], None] | None = None,
        with_embeddings: bool = True,
    ) -> "KnowledgeBase":
        """Build every index over a set of chunks.

        Args:
          config: Configuration supplying the embedding model and repository
            name.
          chunks: Chunks to index. Their order fixes the index row order.
          progress: Called with a short status line as each index is built.
          with_embeddings: Build the dense index. Turning this off makes an
            index that BM25, symbol and graph retrieval can still use, at a
            fraction of the build time.

        Returns:
          The built knowledge base, its `meta` recording per-stage timings.
        """
        def say(msg: str) -> None:
            if progress:
                progress(msg)

        store = ChunkStore.from_chunks(chunks)
        timings: dict[str, float] = {}

        t0 = time.time()
        say(f"BM25 index over {len(store)} chunks ...")
        bm25 = BM25Index.build(
            [c.indexed_text() for c in store.chunks], [c.chunk_id for c in store.chunks]
        )
        timings["bm25_s"] = round(time.time() - t0, 2)

        t0 = time.time()
        say("symbol index ...")
        symbols = SymbolIndex.build(store.chunks)
        timings["symbol_s"] = round(time.time() - t0, 2)

        t0 = time.time()
        say("symbol graph ...")
        graph = SymbolGraph.build(store.chunks)
        timings["graph_s"] = round(time.time() - t0, 2)

        vector_index = None
        embedder = None
        if with_embeddings:
            t0 = time.time()
            embedder = build_embedder(config.embedding)
            say(f"embedding {len(store)} chunks with {embedder.name} ...")
            vectors = embedder.encode_documents(
                [c.indexed_text() for c in store.chunks], progress=True
            )
            vector_index = VectorIndex(
                vectors, [c.chunk_id for c in store.chunks], model=embedder.name
            )
            timings["embed_s"] = round(time.time() - t0, 2)

        meta = {
            "repository": config.repository,
            "commit": chunks[0].commit if chunks else "",
            "n_chunks": len(store),
            "by_artifact": store.stats(),
            "embedding_model": config.embedding.model if with_embeddings else None,
            "embedding_dim": vector_index.dimension if vector_index else 0,
            "config": config.fingerprint(),
            "timings": timings,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        kb = cls(
            config=config,
            store=store,
            vector_index=vector_index,
            bm25_index=bm25,
            symbol_index=symbols,
            graph=graph,
            meta=meta,
        )
        kb._embedder = embedder
        return kb

    # ------------------------------------------------------------------
    def save(self, directory: Path | None = None) -> Path:
        """Write the chunks, every present index and the metadata to disk.

        Args:
          directory: Destination; defaults to `config.index_path`.

        Returns:
          The directory written to.
        """
        directory = Path(directory or self.config.index_path)
        directory.mkdir(parents=True, exist_ok=True)
        self.store.save(directory / CHUNKS_FILE)
        if self.vector_index is not None:
            self.vector_index.save(directory)
        if self.bm25_index is not None:
            self.bm25_index.save(directory)
        if self.symbol_index is not None:
            self.symbol_index.save(directory)
        if self.graph is not None:
            self.graph.save(directory)
        with open(directory / META_FILE, "w", encoding="utf-8") as fh:
            json.dump(self.meta or {}, fh, indent=2)
        return directory

    @classmethod
    def load(
        cls, config: Config, directory: Path | None = None, *, with_vectors: bool = True
    ) -> "KnowledgeBase":
        """Load a knowledge base from disk.

        Each index is optional and loaded only if its files are present, so
        an index built without embeddings loads and serves.

        Args:
          config: Configuration naming the index directory.
          directory: Override for `config.index_path`.
          with_vectors: Load the dense index. Skipping it saves both the load
            time and the memory of the embedding matrix.

        Returns:
          The loaded knowledge base.

        Raises:
          FileNotFoundError: The directory holds no chunk file.
        """
        directory = Path(directory or config.index_path)
        if not (directory / CHUNKS_FILE).exists():
            raise FileNotFoundError(
                f"no index at {directory}; run `rag ingest` and `rag index` first"
            )
        store = ChunkStore.load(directory / CHUNKS_FILE)
        vector_index = None
        if with_vectors and (directory / "embeddings.npy").exists():
            vector_index = VectorIndex.load(directory)
        bm25 = BM25Index.load(directory) if (directory / "bm25_meta.json").exists() else None
        symbols = SymbolIndex.load(directory) if (directory / "symbols.json").exists() else None
        graph = SymbolGraph.load(directory) if (directory / "graph.json").exists() else None
        meta = {}
        if (directory / META_FILE).exists():
            with open(directory / META_FILE, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        return cls(
            config=config,
            store=store,
            vector_index=vector_index,
            bm25_index=bm25,
            symbol_index=symbols,
            graph=graph,
            meta=meta,
        )
