"""Git history ingestion (PRD §9, M7).

Each commit becomes one retrievable chunk holding the subject, body, the list
of touched files and a truncated diff.  That is what lets the system answer
"why was this introduced?" instead of only "where is it implemented?".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import IngestionConfig
from ..schema import Chunk

_SEP = "\x1e"      # record separator
_FIELD = "\x1f"    # field separator
# the separator must lead each record: git appends the file list *after*
# the formatted header, so a trailing separator would misalign every record
_FORMAT = _SEP + _FIELD.join(["%H", "%an", "%aI", "%s", "%b"])


def git(repo: Path, *args: str, check: bool = True) -> str:
    """Run a git command in a repository and return its stdout.

    Args:
      repo: Checkout to run in.
      *args: Arguments after the implicit `git -C <repo>`.
      check: Raise on a non-zero exit status.

    Returns:
      Captured stdout, with undecodable bytes replaced.

    Raises:
      RuntimeError: The command failed and `check` is True.
    """
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def head_commit(repo: Path) -> str:
    """Resolve a checkout's HEAD.

    Args:
      repo: Checkout to inspect.

    Returns:
      The full SHA, or "unknown" if the path is not a repository. Ingestion
      of a plain directory is allowed to proceed without history.
    """
    try:
        return git(repo, "rev-parse", "HEAD").strip()
    except Exception:
        return "unknown"


class GitParser:
    """Turns commit history into retrievable chunks.

    One commit becomes one chunk holding subject, body, touched files and a
    truncated diff -- the evidence behind "why was this introduced?" as
    opposed to "where is it implemented?".

    Attributes:
      config: Ingestion config supplying the history depth and diff limits.
    """

    def __init__(self, config: IngestionConfig) -> None:
        """Store the ingestion config supplying the history and diff limits."""
        self.config = config

    # ------------------------------------------------------------------
    def parse(self, *, repository: str, repo_path: Path, commit: str) -> list[Chunk]:
        """Read the whole history in one pass.

        A single streaming `git log` subprocess covers every commit; the
        record separator *leads* each record, because git appends the file
        list after the formatted header and a trailing separator would
        misalign every record.

        Args:
          repository: Repository name recorded on every chunk.
          repo_path: Checkout to read history from.
          commit: Repository HEAD recorded on every chunk.

        Returns:
          One chunk per commit, newest first, capped at `git_max_commits`.
          A directory that is not a git repository yields an empty list.
        """
        if not (repo_path / ".git").exists():
            return []
        args = [
            "git", "-C", str(repo_path), "log",
            f"-n{self.config.git_max_commits}",
            f"--pretty=format:{_FORMAT}",
            "--raw",
            "--no-color",
            "--unified=2",
        ]
        if self.config.git_max_diff_chars > 0:
            args.append("--patch")
        if not self.config.git_include_merges:
            args.append("--no-merges")

        chunks: list[Chunk] = []
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, errors="replace", bufsize=1 << 20,
        )
        assert proc.stdout is not None
        buffer = ""
        records: list[str] = []
        while True:
            block = proc.stdout.read(1 << 20)
            if not block:
                break
            buffer += block
            while _SEP in buffer:
                record, _, buffer = buffer.partition(_SEP)
                records.append(record)
        proc.wait()
        if buffer.strip():
            records.append(buffer)

        for record in records:
            record = record.strip("\n")
            if not record.strip():
                continue
            head, _, tail = record.partition("\n")
            fields = head.split(_FIELD)
            if len(fields) < 4:
                continue
            sha, author, timestamp, subject = fields[0].strip(), fields[1], fields[2], fields[3]
            body = fields[4] if len(fields) > 4 else ""
            if not sha:
                continue
            files, diff = self._split_body(tail)
            content = self._render(sha, author, timestamp, subject, body, files, diff)
            chunks.append(
                Chunk(
                    repository=repository,
                    commit=commit,
                    path=f"git/{sha[:10]}",
                    artifact_type="commit",
                    language="diff",
                    start_line=0,
                    end_line=0,
                    content=content,
                    symbol=subject[:120],
                    qualified_name=f"commit:{sha[:10]}",
                    symbol_type="commit",
                    commit_sha=sha,
                    author=author,
                    timestamp=timestamp,
                    files_changed=files,
                    extra={"subject": subject, "body": body.strip()},
                )
            )
        return chunks

    # ------------------------------------------------------------------
    def _split_body(self, tail: str) -> tuple[list[str], str]:
        """Separate the `--raw` file list from the patch that follows it.

        Args:
          tail: Everything in the record after the formatted header line.

        Returns:
          A `(files_changed, diff)` pair. The diff is truncated to
          `git_max_diff_chars` with a marker, or empty when diffs are off.
        """
        files: list[str] = []
        diff_lines: list[str] = []
        in_diff = False
        for line in tail.splitlines():
            if line.startswith("diff --git "):
                in_diff = True
            if in_diff:
                diff_lines.append(line)
                continue
            if not line.strip():
                continue
            if line.startswith(":"):          # --raw entry
                parts = line.split("\t")
                if len(parts) >= 2:
                    files.append(parts[-1])
        diff = "\n".join(diff_lines)
        limit = self.config.git_max_diff_chars
        if limit <= 0:
            return files, ""
        if len(diff) > limit:
            diff = diff[:limit] + "\n... [diff truncated]"
        return files, diff

    @staticmethod
    def _render(
        sha: str,
        author: str,
        timestamp: str,
        subject: str,
        body: str,
        files: list[str],
        diff: str,
    ) -> str:
        """Render a commit as the indexed text of its chunk.

        Args:
          sha: Full commit SHA.
          author: Author name.
          timestamp: ISO-8601 author date.
          subject: Commit subject line.
          body: Commit message body; omitted when blank.
          files: Paths touched, listed up to the first 40.
          diff: Already-truncated patch text; omitted when blank.

        Returns:
          A plain-text rendering, labelled field by field so that both BM25
          and the embedder see the structure.
        """
        parts = [
            f"commit {sha}",
            f"author: {author}",
            f"date: {timestamp}",
            f"subject: {subject}",
        ]
        if body.strip():
            parts.append(body.strip())
        if files:
            parts.append("files changed: " + ", ".join(files[:40]))
        if diff.strip():
            parts.append("diff:\n" + diff)
        return "\n".join(parts)
