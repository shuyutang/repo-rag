"""Repository scanner: walks a checkout and produces chunks (PRD §7, M1)."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from ..config import Config
from ..schema import Chunk
from .code_parser import PythonCodeParser, is_test_path
from .document_parser import DocumentParser
from .git_parser import GitParser, head_commit

_CODE_SUFFIXES = {".py"}
_DOC_SUFFIXES = {".md", ".rst"}
_CONFIG_SUFFIXES = {".yaml", ".yml", ".toml", ".cfg", ".ini", ".json"}


@dataclass
class IngestionStats:
    """Counts describing one scan, recorded in the index metadata.

    Attributes:
      files_scanned: Files that produced at least one parse attempt.
      files_skipped: Files rejected for size, encoding or unknown suffix.
      chunks: Chunks surviving deduplication.
      by_artifact: Chunk count per artifact type.
      by_symbol_type: Chunk count per symbol type; "none" for chunks with no
        symbol, such as commits.
      commit: Repository HEAD at scan time.
    """

    files_scanned: int = 0
    files_skipped: int = 0
    chunks: int = 0
    by_artifact: dict[str, int] | None = None
    by_symbol_type: dict[str, int] | None = None
    commit: str = ""

    def to_dict(self) -> dict:
        """Return the stats as a JSON-serialisable dict, empty maps for None."""
        return {
            "files_scanned": self.files_scanned,
            "files_skipped": self.files_skipped,
            "chunks": self.chunks,
            "by_artifact": self.by_artifact or {},
            "by_symbol_type": self.by_symbol_type or {},
            "commit": self.commit,
        }


class RepositoryScanner:
    """Walks a checkout and turns it into chunks.

    Dispatch is by file suffix: Python goes through the AST parser, markdown
    and rst through the document parser, common config formats are cut into
    fixed line windows, and anything else is skipped rather than chunked.
    Skipping is deliberate -- a windowed chunk of C++ would retrieve but could
    not support symbol lookup, the change-impact graph or a structural
    citation, so it would degrade precision while looking like coverage.

    Attributes:
      config: Configuration supplying the include globs and chunking limits.
      code_parser: Python AST parser.
      doc_parser: Markdown and reStructuredText parser.
      git_parser: Commit history parser.
    """

    def __init__(self, config: Config) -> None:
        """Build the scanner and its three parsers from `config`."""
        self.config = config
        self.code_parser = PythonCodeParser(config.ingestion)
        self.doc_parser = DocumentParser(config.ingestion)
        self.git_parser = GitParser(config.ingestion)

    # ------------------------------------------------------------------
    def iter_files(self, repo_path: Path) -> Iterator[Path]:
        """Yield candidate files under a checkout in sorted order.

        Directories named in `ingestion.exclude_dirs` are pruned, as are paths
        matching an exclude pattern. Sorting keeps ingestion deterministic,
        which matters because chunk order reaches the index.

        Args:
          repo_path: Root of the checkout.

        Yields:
          Absolute paths to files that survived exclusion.
        """
        excluded = set(self.config.ingestion.exclude_dirs)
        for path in sorted(repo_path.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(repo_path)
            parts = set(rel.parts[:-1])
            if parts & excluded:
                continue
            if any(fnmatch.fnmatch(str(rel), pat) for pat in excluded):
                continue
            yield path

    def _matches(self, rel: str) -> bool:
        """Report whether a repo-relative path matches any include glob.

        Args:
          rel: Repository-relative path.

        Returns:
          True if `rel` matches a pattern, with and without a leading slash so
          that both "**/*.py" and "/vllm/*.py" styles work.
        """
        return any(
            fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch("/" + rel, "/" + pat)
            for pat in self.config.ingestion.include_globs
        )

    # ------------------------------------------------------------------
    def scan(
        self,
        repo_path: Path | None = None,
        *,
        include_git: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[list[Chunk], IngestionStats]:
        """Scan a checkout into chunks.

        Args:
          repo_path: Checkout to scan; defaults to `config.repo_dir`.
          include_git: Ingest commit history as well as files.
          progress: Called with a short status line every 500 files and once
            before git history is read.

        Returns:
          A `(chunks, stats)` pair. Chunks are deduplicated by `chunk_id`.

        Raises:
          FileNotFoundError: `repo_path` does not exist.
        """
        repo_path = Path(repo_path or self.config.repo_dir)
        if not repo_path.exists():
            raise FileNotFoundError(f"repository not found: {repo_path}")
        commit = head_commit(repo_path)
        stats = IngestionStats(by_artifact={}, by_symbol_type={}, commit=commit)
        chunks: list[Chunk] = []

        for path in self.iter_files(repo_path):
            rel = str(path.relative_to(repo_path))
            if not self._matches(rel):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > self.config.ingestion.max_file_bytes or size == 0:
                stats.files_skipped += 1
                continue
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                stats.files_skipped += 1
                continue

            suffix = path.suffix.lower()
            if suffix in _CODE_SUFFIXES:
                new = self.code_parser.parse(
                    repository=self.config.repository,
                    commit=commit,
                    rel_path=rel,
                    source=source,
                )
            elif suffix in _DOC_SUFFIXES:
                new = self.doc_parser.parse(
                    repository=self.config.repository,
                    commit=commit,
                    rel_path=rel,
                    source=source,
                )
            elif suffix in _CONFIG_SUFFIXES:
                new = self._config_chunks(rel, source, commit)
            else:
                stats.files_skipped += 1
                continue

            stats.files_scanned += 1
            chunks.extend(new)
            if progress and stats.files_scanned % 500 == 0:
                progress(f"{stats.files_scanned} files, {len(chunks)} chunks")

        if include_git:
            if progress:
                progress("reading git history ...")
            chunks.extend(
                self.git_parser.parse(
                    repository=self.config.repository,
                    repo_path=repo_path,
                    commit=commit,
                )
            )

        chunks = _deduplicate(chunks)
        stats.chunks = len(chunks)
        for chunk in chunks:
            stats.by_artifact[chunk.artifact_type] = (
                stats.by_artifact.get(chunk.artifact_type, 0) + 1
            )
            key = chunk.symbol_type or "none"
            stats.by_symbol_type[key] = stats.by_symbol_type.get(key, 0) + 1
        return chunks, stats

    # ------------------------------------------------------------------
    def _config_chunks(self, rel: str, source: str, commit: str) -> list[Chunk]:
        """Cut a config file into fixed line windows.

        Config formats have no symbol structure to parse, so these are the one
        place windowed chunking is used.

        Args:
          rel: Repository-relative path of the file.
          source: File contents.
          commit: Repository HEAD, recorded on each chunk.

        Returns:
          Chunks of at most `max_chunk_lines` lines, skipping windows shorter
          than `min_chunk_chars`.
        """
        lines = source.splitlines()
        limit = self.config.ingestion.max_chunk_lines
        out: list[Chunk] = []
        for offset in range(0, len(lines), limit):
            piece = lines[offset : offset + limit]
            text = "\n".join(piece)
            if len(text.strip()) < self.config.ingestion.min_chunk_chars:
                continue
            out.append(
                Chunk(
                    repository=self.config.repository,
                    commit=commit,
                    path=rel,
                    artifact_type="config",
                    language=Path(rel).suffix.lstrip("."),
                    start_line=offset + 1,
                    end_line=offset + len(piece),
                    content=text,
                    symbol=Path(rel).name,
                    qualified_name=rel,
                    symbol_type="file",
                )
            )
        return out


def _deduplicate(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Drop repeated `chunk_id`s, keeping first occurrence and input order.

    Args:
      chunks: Chunks in ingestion order.

    Returns:
      The deduplicated list.
    """
    seen: set[str] = set()
    out: list[Chunk] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        out.append(chunk)
    return out


def write_chunks(chunks: Iterable[Chunk], path: Path) -> None:
    """Write chunks to `path` as JSON Lines, creating parent directories.

    Args:
      chunks: Chunks to serialise, one JSON object per line.
      path: Destination file, overwritten if present.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    """Read chunks back from a JSON Lines file written by `write_chunks`.

    Args:
      path: File to read. Blank lines are ignored.

    Returns:
      The chunks, in file order.
    """
    out: list[Chunk] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Chunk.from_dict(json.loads(line)))
    return out


__all__ = [
    "RepositoryScanner",
    "IngestionStats",
    "write_chunks",
    "read_chunks",
    "is_test_path",
]
