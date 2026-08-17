"""Section-aware documentation chunking (PRD §9).

Markdown / reStructuredText are split on their heading hierarchy so that each
chunk keeps the path of headings that led to it.  Oversized sections are split
on paragraph boundaries rather than mid-sentence.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import IngestionConfig
from ..schema import Chunk

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_FENCE = re.compile(r"^\s*(```|~~~)")
_RST_UNDERLINE = re.compile(r"^([=\-`:'\"~^_*+#])\1{2,}\s*$")
_RST_LEVELS = "#*=-^\"~'`:+_"


def _md_sections(lines: list[str]) -> list[tuple[int, int, list[str]]]:
    """Split markdown lines on their ATX heading hierarchy.

    Headings inside fenced code blocks are ignored, so a `# comment` in a
    shell example does not open a spurious section.

    Args:
      lines: The document's lines.

    Returns:
      One `(start_line, end_line, heading_path)` triple per section, with
      1-based inclusive lines and the heading path running outermost first.
      A document with no headings yields a single whole-file section.
    """
    boundaries: list[tuple[int, int, str]] = []  # (line_idx, level, title)
    in_fence = False
    for idx, line in enumerate(lines):
        if _MD_FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _MD_HEADING.match(line)
        if m:
            boundaries.append((idx, len(m.group(1)), m.group(2).strip()))
    if not boundaries:
        return [(1, len(lines), [])]

    sections: list[tuple[int, int, list[str]]] = []
    if boundaries[0][0] > 0:
        sections.append((1, boundaries[0][0], []))
    stack: list[tuple[int, str]] = []
    for i, (idx, level, title) in enumerate(boundaries):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
        sections.append((idx + 1, end, [t for _, t in stack]))
    return sections


def _rst_sections(lines: list[str]) -> list[tuple[int, int, list[str]]]:
    """Split reStructuredText lines on their underline hierarchy.

    reST assigns no fixed meaning to underline characters, so nesting depth
    is inferred from the order in which characters first appear.

    Args:
      lines: The document's lines.

    Returns:
      One `(start_line, end_line, heading_path)` triple per section, in the
      same shape as `_md_sections`.
    """
    boundaries: list[tuple[int, str, str]] = []  # (line_idx_of_title, char, title)
    for idx in range(len(lines) - 1):
        title, underline = lines[idx], lines[idx + 1]
        if (
            title.strip()
            and _RST_UNDERLINE.match(underline)
            and len(underline.strip()) >= len(title.strip()) - 2
        ):
            boundaries.append((idx, underline.strip()[0], title.strip()))
    if not boundaries:
        return [(1, len(lines), [])]

    order: list[str] = []
    for _, char, _ in boundaries:
        if char not in order:
            order.append(char)
    sections: list[tuple[int, int, list[str]]] = []
    if boundaries[0][0] > 0:
        sections.append((1, boundaries[0][0], []))
    stack: list[tuple[int, str]] = []
    for i, (idx, char, title) in enumerate(boundaries):
        level = order.index(char)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
        sections.append((idx + 1, end, [t for _, t in stack]))
    return sections


class DocumentParser:
    """Parses markdown and reStructuredText into section chunks.

    Each chunk keeps the path of headings that led to it, which is what makes
    a document citation point at a section rather than at a file.

    Attributes:
      config: Ingestion config supplying the document size limits.
    """

    def __init__(self, config: IngestionConfig) -> None:
        """Store the ingestion config supplying the document size limits."""
        self.config = config

    def parse(
        self, *, repository: str, commit: str, rel_path: str, source: str
    ) -> list[Chunk]:
        """Parse one document into section chunks.

        Args:
          repository: Repository name recorded on every chunk.
          commit: Repository HEAD recorded on every chunk.
          rel_path: Repository-relative path; the suffix selects the syntax.
          source: File contents.

        Returns:
          Section chunks in document order, skipping sections shorter than
          `doc_min_chars`.
        """
        lines = source.splitlines()
        suffix = Path(rel_path).suffix.lower()
        if suffix == ".rst":
            sections = _rst_sections(lines)
            language = "rst"
        else:
            sections = _md_sections(lines)
            language = "markdown"

        chunks: list[Chunk] = []
        for start, end, heading_path in sections:
            body = "\n".join(lines[start - 1 : end]).strip("\n")
            if len(body.strip()) < self.config.doc_min_chars:
                continue
            for piece_start, piece_end, piece in self._split(body, start):
                chunks.append(
                    Chunk(
                        repository=repository,
                        commit=commit,
                        path=rel_path,
                        artifact_type="doc",
                        language=language,
                        start_line=piece_start,
                        end_line=piece_end,
                        content=piece,
                        symbol=heading_path[-1] if heading_path else Path(rel_path).stem,
                        qualified_name=(
                            f"{rel_path}#{' > '.join(heading_path)}"
                            if heading_path
                            else rel_path
                        ),
                        symbol_type="section",
                        heading_path=heading_path,
                    )
                )
        return chunks

    def _split(self, body: str, start_line: int) -> list[tuple[int, int, str]]:
        """Split an oversized section on blank lines.

        Line numbers stay exact through the split, because they are what the
        citation points at.

        Args:
          body: Section text.
          start_line: Line number `body` starts at, 1-based.

        Returns:
          `(start_line, end_line, text)` pieces. Prose is cut at paragraph
          breaks; a section with no blank lines at all -- a long table, say --
          is hard-split once it exceeds twice the limit.
        """
        limit = self.config.doc_max_chars
        lines = body.splitlines()
        if len(body) <= limit:
            return [(start_line, start_line + max(len(lines) - 1, 0), body)]

        pieces: list[tuple[int, int, str]] = []
        buf: list[str] = []
        buf_start = start_line
        size = 0
        for offset, line in enumerate(lines):
            buf.append(line)
            size += len(line) + 1
            at_break = not line.strip()
            if size >= limit and at_break:
                pieces.append((buf_start, start_line + offset, "\n".join(buf)))
                buf, size = [], 0
                buf_start = start_line + offset + 1
        if buf and "".join(buf).strip():
            pieces.append((buf_start, start_line + len(lines) - 1, "\n".join(buf)))
        # a section with no blank lines can still exceed the limit: hard-split
        out: list[tuple[int, int, str]] = []
        for s, e, text in pieces:
            if len(text) <= limit * 2:
                out.append((s, e, text))
                continue
            piece_lines = text.splitlines()
            step = max(len(piece_lines) // (len(text) // limit + 1), 1)
            for off in range(0, len(piece_lines), step):
                sub = piece_lines[off : off + step]
                out.append((s + off, s + off + len(sub) - 1, "\n".join(sub)))
        return out
