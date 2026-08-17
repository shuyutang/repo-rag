"""Benchmark dataset schema and IO (PRD §20)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

DIFFICULTIES = ("single_hop", "multi_hop")


@dataclass
class BenchmarkQuestion:
    """One benchmark question and its gold evidence.

    Attributes:
      id: Stable question id, also the seed for split assignment.
      question: The question text.
      category: Question category, e.g. "architecture" or "historical".
      difficulty: "single_hop" or "multi_hop".
      expected_answer: Reference answer, possibly partial or empty.
      relevant_files: Gold file paths.
      relevant_symbols: Gold qualified symbol names.
      relevant_chunks: Gold chunk ids, when known exactly.
      relevant_commits: Gold commit SHAs.
      source: "auto" for generated questions, "curated" for hand-written
        ones. Generated questions inherit vocabulary from their source chunk,
        which flatters lexical retrieval, so the two are reported separately.
      split: "dev" for tuning, "test" for reported numbers.
      provenance: How the question was produced.
      reviewed: Whether a human has checked it.
    """

    id: str
    question: str
    category: str
    difficulty: str = "single_hop"
    expected_answer: str = ""
    relevant_files: list[str] = field(default_factory=list)
    relevant_symbols: list[str] = field(default_factory=list)
    relevant_chunks: list[str] = field(default_factory=list)
    relevant_commits: list[str] = field(default_factory=list)
    source: str = "auto"
    split: str = "test"
    provenance: dict[str, Any] = field(default_factory=dict)
    reviewed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the question as a JSON-serialisable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkQuestion":
        """Build a question from a dict, ignoring unknown keys.

        Tolerant of extra keys so that a dataset written by a later version
        still loads.

        Args:
          data: Mapping of field names to values.

        Returns:
          The question.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def is_multi_hop(self) -> bool:
        """bool: Whether answering needs more than one source.

        True when the question is labelled multi-hop *or* its gold evidence
        spans several files, so a mislabelled question is still counted by
        what it actually requires.
        """
        return self.difficulty == "multi_hop" or len(set(self.relevant_files)) > 1


def assign_splits(
    records: list["BenchmarkQuestion"], *, dev_fraction: float = 0.3
) -> list["BenchmarkQuestion"]:
    """Assign each question to the dev or test split, deterministically.

    Retrieval hyper-parameters -- fusion weights, `k` -- are tuned on dev
    only; every number in the ablation table comes from test, so the reported
    result is not the result of fitting the benchmark. Assignment hashes the
    question id, so it is stable as the dataset grows.

    Args:
      records: Questions to assign, mutated in place.
      dev_fraction: Share of questions going to dev.

    Returns:
      The same list, for chaining.
    """
    import hashlib

    cut = int(dev_fraction * 1000)
    for record in records:
        digest = hashlib.sha1(record.id.encode("utf-8")).hexdigest()
        record.split = "dev" if int(digest[:6], 16) % 1000 < cut else "test"
    return records


def load_dataset(path: Path) -> list[BenchmarkQuestion]:
    """Read a benchmark from a JSON Lines file.

    Args:
      path: Dataset file.

    Returns:
      The questions, in file order.
    """
    records: list[BenchmarkQuestion] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(BenchmarkQuestion.from_dict(json.loads(line)))
    return records


def save_dataset(records: Iterable[BenchmarkQuestion], path: Path) -> None:
    """Write a benchmark to a JSON Lines file, creating parent directories.

    Args:
      records: Questions to write.
      path: Destination, overwritten if present.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def dataset_summary(records: list[BenchmarkQuestion]) -> dict[str, Any]:
    """Summarise a benchmark's composition.

    Args:
      records: The questions.

    Returns:
      Counts by category, source and split, the multi-hop fraction, how many
      have a reference answer, and the mean gold-file count per question.
    """
    categories = Counter(r.category for r in records)
    multi_hop = sum(1 for r in records if r.is_multi_hop)
    return {
        "n_questions": len(records),
        "by_category": dict(categories),
        "by_source": dict(Counter(r.source for r in records)),
        "multi_hop": multi_hop,
        "multi_hop_fraction": round(multi_hop / max(len(records), 1), 3),
        "with_expected_answer": sum(1 for r in records if r.expected_answer.strip()),
        "reviewed": sum(1 for r in records if r.reviewed),
        "by_split": dict(Counter(r.split for r in records)),
        "avg_gold_files": round(
            sum(len(set(r.relevant_files)) for r in records) / max(len(records), 1), 2
        ),
    }


def validate_dataset(records: list[BenchmarkQuestion], kb) -> list[str]:
    """Check that every gold artifact still exists in the index.

    Worth running after any re-index: gold labels are paths, symbols and line
    ranges, and a repository update quietly invalidates some of them. A
    recall figure computed against unreachable gold is not a low score, it is
    a wrong one.

    Args:
      records: The questions to check.
      kb: Knowledge base to check against.

    Returns:
      One human-readable problem string per issue found: duplicate ids, empty
      questions, questions with no gold evidence, and unknown paths, chunks,
      symbols or commits. Empty means the dataset is sound.
    """
    problems: list[str] = []
    known_paths = set(kb.store.paths)
    known_chunks = {c.chunk_id for c in kb.store.chunks}
    known_symbols = {
        c.qualified_name for c in kb.store.chunks if c.qualified_name
    } | {c.symbol for c in kb.store.chunks if c.symbol}
    known_commits = {
        c.commit_sha for c in kb.store.chunks if c.commit_sha
    }
    seen_ids: set[str] = set()
    for record in records:
        if record.id in seen_ids:
            problems.append(f"{record.id}: duplicate id")
        seen_ids.add(record.id)
        if not record.question.strip():
            problems.append(f"{record.id}: empty question")
        if not (record.relevant_files or record.relevant_chunks or record.relevant_commits):
            problems.append(f"{record.id}: no gold evidence")
        for path in record.relevant_files:
            if path not in known_paths:
                problems.append(f"{record.id}: unknown file {path}")
        for chunk_id in record.relevant_chunks:
            if chunk_id not in known_chunks:
                problems.append(f"{record.id}: unknown chunk {chunk_id}")
        for symbol in record.relevant_symbols:
            if symbol not in known_symbols and symbol.rsplit(".", 1)[-1] not in known_symbols:
                problems.append(f"{record.id}: unknown symbol {symbol}")
        for sha in record.relevant_commits:
            if sha not in known_commits:
                problems.append(f"{record.id}: unknown commit {sha[:10]}")
    return problems
