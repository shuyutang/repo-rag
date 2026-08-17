"""Embedding backend (PRD §10).

A thin interface so the embedding model is swappable; the default is a local
sentence-transformers model so that indexing needs no network or API key.
"""

from __future__ import annotations

import zlib
from typing import Protocol, Sequence

import numpy as np

from ..config import EmbeddingConfig


class Embedder(Protocol):
    """Interface every embedding backend implements.

    Documents and queries are encoded through separate methods because some
    models require an asymmetric instruction prefix on each side.

    Attributes:
      dimension: Width of the produced vectors.
      name: Model identifier, recorded in the index metadata.
    """

    dimension: int
    name: str

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Encode documents.

        Args:
          texts: Document texts.

        Returns:
          A `(len(texts), dimension)` float32 array.
        """
        ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Encode queries.

        Args:
          texts: Query texts.

        Returns:
          A `(len(texts), dimension)` float32 array.
        """
        ...


def _resolve_device(requested: str) -> str:
    """Resolve a configured device string to a concrete torch device.

    Args:
      requested: "auto", "cuda" or "cpu". Anything but "auto" is returned
        unchanged, so an explicit choice is never overridden.

    Returns:
      "cuda" when auto-detection finds a GPU, otherwise "cpu".
    """
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


class SentenceTransformerEmbedder:
    """Local sentence-transformers encoder.

    The default model is small and local, so indexing needs neither network
    access nor an API key. Note that it is a general-purpose text model used
    unchanged: no fine-tuning on this corpus happens anywhere in the system.

    Attributes:
      config: Embedding configuration.
      name: Model identifier.
      device: Resolved torch device.
      model: The loaded sentence-transformers model.
      dimension: Width of the produced vectors.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Load the model onto the configured device.

        Imported lazily so that retrieval-only work never pays the
        sentence-transformers import cost.

        Args:
          config: Embedding configuration naming the model and device.
        """
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.name = config.model
        self.device = _resolve_device(config.device)
        self.model = SentenceTransformer(
            config.model, device=self.device, trust_remote_code=config.trust_remote_code
        )
        self.model.max_seq_length = config.max_seq_length
        # sentence-transformers renamed this in 5.x; support both
        dimension_of = getattr(
            self.model, "get_embedding_dimension", None
        ) or self.model.get_sentence_embedding_dimension
        self.dimension = int(dimension_of())

    def _encode(self, texts: Sequence[str], prefix: str, progress: bool) -> np.ndarray:
        """Encode texts with an optional instruction prefix.

        Args:
          texts: Texts to encode.
          prefix: Instruction prefix, or the empty string for none.
          progress: Show a progress bar.

        Returns:
          A `(len(texts), dimension)` float32 array, L2-normalised when the
          config asks for it.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        payload = [prefix + t for t in texts] if prefix else list(texts)
        vectors = self.model.encode(
            payload,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        """Encode documents with the configured document prefix.

        Args:
          texts: Document texts.
          progress: Show a progress bar; used during indexing.

        Returns:
          A `(len(texts), dimension)` float32 array.
        """
        return self._encode(texts, self.config.document_prefix, progress)

    def encode_queries(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        """Encode queries with the configured query prefix.

        Args:
          texts: Query texts.
          progress: Show a progress bar.

        Returns:
          A `(len(texts), dimension)` float32 array.
        """
        return self._encode(texts, self.config.query_prefix, progress)


class HashingEmbedder:
    """Deterministic, dependency-free encoder used by tests and CI.

    Character n-gram hashing into a fixed-size vector. Not competitive with a
    trained model, but it keeps the whole pipeline runnable with no model
    download, which is what lets the test suite exercise the dense path
    offline.

    Attributes:
      config: Embedding configuration, defaulted when not supplied.
      dimension: Width of the produced vectors.
      name: Model identifier, e.g. "hashing-256".
    """

    def __init__(self, config: EmbeddingConfig | None = None, dimension: int = 256) -> None:
        """Configure the hashing encoder.

        Args:
          config: Embedding configuration; a "hashing" default is used when
            omitted.
          dimension: Width of the produced vectors.
        """
        self.config = config or EmbeddingConfig(model="hashing")
        self.dimension = dimension
        self.name = f"hashing-{dimension}"

    def _vector(self, text: str) -> np.ndarray:
        """Hash one text into a normalised vector.

        Args:
          text: Text to encode.

        Returns:
          An L2-normalised float32 vector of 3- and 4-gram counts. Hashing
          uses crc32 rather than `hash()`, whose string seed is salted per
          process and would make an index unusable in the next one.
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().split()
        for token in tokens:
            for n in (3, 4):
                for i in range(max(len(token) - n + 1, 1)):
                    gram = token[i : i + n]
                    # crc32, not hash(): str hashing is salted per process
                    vec[zlib.crc32(gram.encode("utf-8")) % self.dimension] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode_documents(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        """Encode documents by n-gram hashing.

        Args:
          texts: Document texts.
          progress: Accepted for interface compatibility and ignored.

        Returns:
          A `(len(texts), dimension)` float32 array.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    def encode_queries(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        """Encode queries; identical to `encode_documents`, as this is symmetric.

        Args:
          texts: Query texts.
          progress: Accepted for interface compatibility and ignored.

        Returns:
          A `(len(texts), dimension)` float32 array.
        """
        return self.encode_documents(texts)


def build_embedder(config: EmbeddingConfig) -> Embedder:
    """Construct the embedder a config asks for.

    Args:
      config: Embedding configuration.

    Returns:
      A `HashingEmbedder` for the sentinel model names "hashing", "test" and
      "none", otherwise a `SentenceTransformerEmbedder`.
    """
    if config.model in {"hashing", "test", "none"}:
        return HashingEmbedder(config)
    return SentenceTransformerEmbedder(config)
