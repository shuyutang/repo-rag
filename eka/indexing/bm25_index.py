"""Sparse lexical index — BM25-Okapi over a code-aware tokenizer (PRD §11).

Implemented directly on scipy sparse matrices rather than pulled from a library
so that the scoring, the tokenizer and the index format are all inspectable and
version-controlled with the rest of the system.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse

from .tokenizer import tokenize


class BM25Index:
    """BM25-Okapi over a sparse term-document matrix.

    Implemented directly on scipy sparse matrices rather than pulled from a
    library, so scoring, tokenisation and the on-disk format are all
    inspectable and version-controlled with the rest of the system.

    The matrix is held column-major (CSC) because scoring iterates over query
    *terms*, and a term is a column: each term slices one contiguous run of
    `data`, which is what keeps a query at sub-millisecond cost.

    Attributes:
      matrix: Term-document counts, documents as rows, terms as columns.
      vocabulary: Term to column index.
      doc_lengths: Token count per document.
      chunk_ids: Chunk id per row.
      k1: Term-frequency saturation parameter.
      b: Length-normalisation parameter.
      avg_length: Mean document length, the BM25 length baseline.
      idf: Inverse document frequency per term.
    """

    def __init__(
        self,
        matrix: sparse.csc_matrix,
        vocabulary: dict[str, int],
        doc_lengths: np.ndarray,
        chunk_ids: list[str],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        """Store the matrix and precompute the corpus statistics.

        Args:
          matrix: Term-document count matrix.
          vocabulary: Term to column index.
          doc_lengths: Token count per document.
          chunk_ids: Chunk id per row, positionally aligned with `matrix`.
          k1: Term-frequency saturation parameter.
          b: Length-normalisation parameter.
        """
        self.matrix = matrix.tocsc()
        self.vocabulary = vocabulary
        self.doc_lengths = doc_lengths.astype(np.float32)
        self.chunk_ids = chunk_ids
        self.k1 = k1
        self.b = b
        self.avg_length = float(self.doc_lengths.mean()) if len(doc_lengths) else 0.0
        n_docs = len(chunk_ids)
        df = np.diff(self.matrix.indptr).astype(np.float32)
        # Robertson/Sparck-Jones idf with the +1 smoothing used by Lucene
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of indexed documents."""
        return len(self.chunk_ids)

    @classmethod
    def build(cls, texts: Sequence[str], chunk_ids: list[str], **kwargs) -> "BM25Index":
        """Build an index from raw texts in a single pass.

        The vocabulary grows as terms are first seen, so no separate
        vocabulary pass over the corpus is needed.

        Args:
          texts: Document texts, tokenised with the code-aware tokenizer.
          chunk_ids: Chunk id per document, positionally aligned.
          **kwargs: Forwarded to `__init__`, i.e. `k1` and `b`.

        Returns:
          The built index.
        """
        vocabulary: dict[str, int] = {}
        indptr = [0]
        indices: list[int] = []
        data: list[int] = []
        doc_lengths = np.zeros(len(texts), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, int] = {}
            tokens = tokenize(text)
            doc_lengths[row] = len(tokens)
            for token in tokens:
                term_id = vocabulary.get(token)
                if term_id is None:
                    term_id = len(vocabulary)
                    vocabulary[token] = term_id
                counts[term_id] = counts.get(term_id, 0) + 1
            indices.extend(counts.keys())
            data.extend(counts.values())
            indptr.append(len(indices))
        matrix = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
            shape=(len(texts), max(len(vocabulary), 1)),
        )
        return cls(matrix.tocsc(), vocabulary, doc_lengths, chunk_ids, **kwargs)

    # ------------------------------------------------------------------
    def score(self, query: str, *, allowed: np.ndarray | None = None) -> np.ndarray:
        """Score every document against a query.

        A term repeated in the query gets a mild boost rather than a linear
        one, since a doubled query term signals emphasis, not twice the need.

        Args:
          query: Raw query text.
          allowed: Boolean mask over rows; disallowed rows score zero. Used
            for artifact-type and path filtering.

        Returns:
          One BM25 score per document, in row order. Terms outside the
          vocabulary contribute nothing.
        """
        scores = np.zeros(len(self.chunk_ids), dtype=np.float32)
        terms = tokenize(query)
        if not terms:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self.doc_lengths /
                          (self.avg_length or 1.0))
        seen: dict[int, int] = {}
        for term in terms:
            term_id = self.vocabulary.get(term)
            if term_id is None:
                continue
            seen[term_id] = seen.get(term_id, 0) + 1
        for term_id, qtf in seen.items():
            start, end = self.matrix.indptr[term_id], self.matrix.indptr[term_id + 1]
            rows = self.matrix.indices[start:end]
            tf = self.matrix.data[start:end]
            contrib = self.idf[term_id] * (tf * (self.k1 + 1)) / (tf + norm[rows])
            scores[rows] += contrib * (1.0 + 0.25 * (qtf - 1))
        if allowed is not None:
            scores = np.where(allowed, scores, 0.0)
        return scores

    def search(
        self, query: str, k: int, *, allowed: np.ndarray | None = None
    ) -> list[tuple[str, float]]:
        """Return the top-k documents for a query.

        Args:
          query: Raw query text.
          k: Maximum results.
          allowed: Boolean mask over rows, as for `score`.

        Returns:
          `(chunk_id, score)` pairs, highest first, containing only documents
          that scored above zero -- so fewer than `k` results is normal and
          means nothing else matched at all.
        """
        scores = self.score(query, allowed=allowed)
        if not np.any(scores > 0):
            return []
        k = min(k, int(np.sum(scores > 0)))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunk_ids[int(i)], float(scores[int(i)])) for i in top]

    # ------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        """Write the index to a directory as four files.

        Args:
          directory: Destination, created if absent. Written as
            `bm25_matrix.npz`, `bm25_doclen.npy`, `bm25_vocab.pkl` and
            `bm25_meta.json`.
        """
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / "bm25_matrix.npz", self.matrix.tocsc())
        np.save(directory / "bm25_doclen.npy", self.doc_lengths)
        with open(directory / "bm25_vocab.pkl", "wb") as fh:
            pickle.dump(self.vocabulary, fh, protocol=4)
        with open(directory / "bm25_meta.json", "w", encoding="utf-8") as fh:
            json.dump({"chunk_ids": self.chunk_ids, "k1": self.k1, "b": self.b}, fh)

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        """Read an index back from a directory written by `save`.

        Args:
          directory: Directory holding the four index files.

        Returns:
          The loaded index. Corpus statistics are recomputed rather than
          stored, so the file format stays independent of the scoring code.
        """
        matrix = sparse.load_npz(directory / "bm25_matrix.npz").tocsc()
        doc_lengths = np.load(directory / "bm25_doclen.npy")
        with open(directory / "bm25_vocab.pkl", "rb") as fh:
            vocabulary = pickle.load(fh)
        with open(directory / "bm25_meta.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return cls(
            matrix, vocabulary, doc_lengths, meta["chunk_ids"],
            k1=meta.get("k1", 1.2), b=meta.get("b", 0.75),
        )


def build_bm25(chunks: Iterable) -> BM25Index:
    """Build a BM25 index over chunks, using each chunk's indexed text.

    Args:
      chunks: Chunks to index; consumed once, order preserved.

    Returns:
      The built index, its rows aligned with the input order.
    """
    chunk_list = list(chunks)
    return BM25Index.build(
        [c.indexed_text() for c in chunk_list], [c.chunk_id for c in chunk_list]
    )
