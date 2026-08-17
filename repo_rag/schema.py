"""Core data model shared by ingestion, indexing, retrieval and generation.

Everything that can be retrieved is a `Chunk`. A chunk always carries enough
provenance (repository, commit, path, line range, symbol) to build a citation
that a human can click through to.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ArtifactType = Literal["source", "test", "doc", "commit", "config"]
SymbolType = Literal[
    "module", "class", "function", "method", "section", "commit", "file"
]


def _stable_id(*parts: Any) -> str:
    """Hash identifying parts into a short, deterministic chunk id.

    The id must be stable across re-indexing runs so that an unchanged symbol
    keeps its identity when the repository moves forward. Note that the repo
    HEAD is deliberately *not* an input: only `commit_sha`, which is set for
    commit chunks alone.

    Args:
      *parts: Identifying values. `None` is encoded as the empty string so
        that an absent field and an empty field hash alike.

    Returns:
      The first 16 hex characters of the SHA-1 of the joined parts.
    """
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class Chunk:
    """A single retrievable unit of evidence.

    One chunk is one *symbol* (function, method, class overview or module) for
    code, one section for documents, and one commit for history -- never a
    fixed-size window. That is what makes line-level citations, symbol lookup
    and the change-impact graph possible.

    Attributes:
      repository: Name of the repository the chunk was ingested from.
      commit: Repository HEAD at ingestion time. Not part of `chunk_id`.
      path: Repository-relative path of the source file.
      artifact_type: Which corpus the chunk belongs to.
      start_line: First line of the chunk, 1-based and inclusive.
      end_line: Last line of the chunk, inclusive.
      content: Raw text of the chunk.
      language: Source language, or "unknown" for unparsed files.
      symbol: Bare symbol name, e.g. "allocate_slots".
      qualified_name: Dotted name including parents, e.g. "KVCache.allocate".
      symbol_type: Structural kind of the symbol.
      parent_symbol: Enclosing class or module, if any.
      imports: Modules imported by the chunk, used by the symbol graph.
      references: Names referenced in the body, used by the symbol graph.
      decorators: Decorator names applied to the symbol.
      docstring: The symbol's own docstring, if it has one.
      signature: Rendered call signature for callables.
      heading_path: Section headings above a document chunk, outermost first.
      commit_sha: SHA of the commit, for commit chunks only.
      author: Commit author, for commit chunks only.
      timestamp: ISO-8601 commit timestamp, for commit chunks only.
      files_changed: Paths touched by the commit, for commit chunks only.
      extra: Parser-specific fields that do not warrant a column.
      chunk_id: Stable identity; derived in `__post_init__` when empty.
    """

    # --- provenance -----------------------------------------------------
    repository: str
    commit: str
    path: str
    artifact_type: ArtifactType
    start_line: int
    end_line: int
    content: str

    language: str = "unknown"
    symbol: str | None = None
    qualified_name: str | None = None
    symbol_type: SymbolType | None = None
    parent_symbol: str | None = None

    # --- structural relations (populated by the code parser) ------------
    imports: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None
    signature: str | None = None

    # --- documents ------------------------------------------------------
    heading_path: list[str] = field(default_factory=list)

    # --- git commits ----------------------------------------------------
    commit_sha: str | None = None
    author: str | None = None
    timestamp: str | None = None
    files_changed: list[str] = field(default_factory=list)

    extra: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        """Derive `chunk_id` when the caller did not supply one."""
        if not self.chunk_id:
            self.chunk_id = _stable_id(
                self.repository,
                self.path,
                self.qualified_name or self.symbol,
                self.symbol_type,
                self.start_line,
                self.end_line,
                self.commit_sha,
            )

    # --- derived views --------------------------------------------------
    @property
    def location(self) -> str:
        """str: Citation target, e.g. "vllm/config.py:718-760"."""
        if self.artifact_type == "commit":
            return f"commit {(self.commit_sha or '')[:10]}"
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @property
    def title(self) -> str:
        """str: Display heading, the most specific name the chunk can offer."""
        if self.artifact_type == "commit":
            return f"commit {(self.commit_sha or '')[:10]}"
        if self.heading_path:
            return f"{self.path} § {' > '.join(self.heading_path)}"
        if self.qualified_name:
            return f"{self.path}::{self.qualified_name}"
        return self.path

    def indexed_text(self) -> str:
        """Build the text handed to the embedding model and BM25 tokenizer.

        A short natural-language header is prepended so that a dense model can
        latch onto the file path and symbol name even when the body is dense
        code.

        Returns:
          The header line followed by the chunk body.
        """
        header_bits = [self.path]
        if self.qualified_name:
            header_bits.append(self.qualified_name)
        if self.heading_path:
            header_bits.append(" > ".join(self.heading_path))
        if self.symbol_type:
            header_bits.append(self.symbol_type)
        header = " | ".join(header_bits)
        return f"{header}\n{self.content}"

    def to_dict(self) -> dict[str, Any]:
        """Return the chunk as a JSON-serialisable dict, fields verbatim."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Rebuild a chunk from `to_dict` output.

        Args:
          data: Mapping whose keys are exactly the dataclass field names.

        Returns:
          The reconstructed chunk, with `chunk_id` preserved rather than
          recomputed.
        """
        return cls(**data)


@dataclass
class RetrievedChunk:
    """A chunk plus the retrieval bookkeeping needed for debugging and eval.

    Attributes:
      chunk: The retrieved chunk itself.
      score: Score under whichever retriever or fusion produced this result.
      retriever: Producing retriever, e.g. "bm25" or "hybrid+rerank".
      rank: Zero-based position in the result list.
      component_scores: Per-source contributions, e.g.
        `{"dense": 0.81, "bm25": 12.4, "rrf": 0.032}`. Scores from different
        sources are not comparable; this exists to explain a ranking, not to
        be arithmetic on.
      query: The query that surfaced the chunk, which for agentic retrieval is
        a sub-query rather than the user's question.
    """

    chunk: Chunk
    score: float
    retriever: str
    rank: int = 0
    # per-source contributions, e.g. {"dense": 0.81, "bm25": 12.4, "rrf": 0.032}
    component_scores: dict[str, float] = field(default_factory=dict)
    # why the retriever surfaced this chunk (query it matched, agent step, ...)
    query: str | None = None

    @property
    def chunk_id(self) -> str:
        """str: Identity of the wrapped chunk."""
        return self.chunk.chunk_id

    def to_dict(self) -> dict[str, Any]:
        """Flatten into the shape the API, traces and result files record.

        Returns:
          A JSON-serialisable dict of the scoring fields plus the provenance
          needed to render a citation, without the chunk body.
        """
        return {
            "chunk_id": self.chunk.chunk_id,
            "score": self.score,
            "retriever": self.retriever,
            "rank": self.rank,
            "component_scores": self.component_scores,
            "query": self.query,
            "path": self.chunk.path,
            "location": self.chunk.location,
            "symbol": self.chunk.qualified_name or self.chunk.symbol,
            "artifact_type": self.chunk.artifact_type,
        }


@dataclass
class Citation:
    """A parsed `[path:start-end (symbol)]` reference from a generated answer.

    Attributes:
      path: Repository-relative path named by the citation.
      start_line: First cited line, inclusive.
      end_line: Last cited line, inclusive.
      symbol: Symbol name, when the model included one.
      chunk_id: Evidence chunk this citation was matched to, set by
        validation; `None` means the citation was never matched.
    """

    path: str
    start_line: int
    end_line: int
    symbol: str | None = None
    chunk_id: str | None = None

    def render(self) -> str:
        """Render back to the bracketed citation form used in answers."""
        if self.symbol:
            return f"[{self.path}:{self.start_line}-{self.end_line} ({self.symbol})]"
        return f"[{self.path}:{self.start_line}-{self.end_line}]"


@dataclass
class Answer:
    """A generated answer together with everything needed to audit it.

    Attributes:
      question: The question as asked.
      text: Generated answer text, citations left inline.
      citations: Citations parsed out of `text`.
      evidence: Chunks that were actually placed in the context window.
      trace_id: Id of the recorded trace, for looking the run up later.
      usage: Token counts, per-stage latencies, LLM call count and any
        unsupported citations found during validation.
    """

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    evidence: list[RetrievedChunk] = field(default_factory=list)
    trace_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the answer as a JSON-serialisable dict for the API layer."""
        return {
            "question": self.question,
            "text": self.text,
            "citations": [asdict(c) for c in self.citations],
            "evidence": [e.to_dict() for e in self.evidence],
            "trace_id": self.trace_id,
            "usage": self.usage,
        }
