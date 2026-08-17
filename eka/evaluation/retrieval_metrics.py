"""Retrieval metrics (PRD §21): Recall@K, MRR, nDCG@K, precision.

Relevance is judged at *file* granularity by default: a retrieved chunk counts
as relevant when its file is one of the question's gold files (or its commit is
a gold commit).  Chunk-level gold labels, when present, are also honoured — a
chunk-level hit is graded higher than a file-level one in nDCG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..schema import RetrievedChunk
from .dataset import BenchmarkQuestion


def relevance_grade(item: RetrievedChunk, question: BenchmarkQuestion) -> int:
    """Grade one retrieved result against a question's gold evidence.

    Args:
      item: The retrieved result.
      question: The question with its gold labels.

    Returns:
      2 for an exact gold chunk, commit or symbol; 1 for the right file; 0
      otherwise. Grading at file granularity by default is what keeps the
      metric fair to a system that retrieves the right code in a differently
      chunked form.
    """
    chunk = item.chunk
    if chunk.chunk_id in question.relevant_chunks:
        return 2
    if chunk.commit_sha and chunk.commit_sha in question.relevant_commits:
        return 2
    symbol = chunk.qualified_name or ""
    if symbol and question.relevant_symbols:
        leaf = symbol.rsplit(".", 1)[-1]
        for gold in question.relevant_symbols:
            if symbol == gold or (leaf and leaf == gold.rsplit(".", 1)[-1]):
                return 2
    if chunk.path in question.relevant_files:
        return 1
    return 0


def _gold_units(question: BenchmarkQuestion) -> set[str]:
    """Determine the units recall is computed over.

    Args:
      question: The question with its gold labels.

    Returns:
      Gold file paths plus prefixed gold commits, falling back to gold chunk
      ids when a question names neither.
    """
    units = set(question.relevant_files)
    units |= {f"commit:{sha}" for sha in question.relevant_commits}
    if not units:
        units = set(question.relevant_chunks)
    return units


def _unit_of(item: RetrievedChunk) -> str:
    """Return the recall unit a result belongs to: its commit or its path."""
    if item.chunk.artifact_type == "commit" and item.chunk.commit_sha:
        return f"commit:{item.chunk.commit_sha}"
    return item.chunk.path


@dataclass
class QuestionMetrics:
    """Retrieval metrics for one question.

    Attributes:
      question_id: The question's id.
      category: Question category, for per-category aggregation.
      difficulty: "single_hop" or "multi_hop".
      recall_at: Cut-off to recall.
      precision_at: Cut-off to precision.
      mrr: Reciprocal rank of the first relevant result.
      ndcg_at: Cut-off to nDCG.
      first_hit_rank: 1-based rank of the first relevant result, or `None`.
      n_gold: Size of the gold set, the recall denominator.
    """

    question_id: str
    category: str
    difficulty: str
    recall_at: dict[int, float] = field(default_factory=dict)
    precision_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at: dict[int, float] = field(default_factory=dict)
    first_hit_rank: int | None = None
    n_gold: int = 0

    def to_dict(self) -> dict:
        """Return the metrics as a JSON-serialisable dict, keys stringified."""
        return {
            "question_id": self.question_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "recall_at": {str(k): v for k, v in self.recall_at.items()},
            "precision_at": {str(k): v for k, v in self.precision_at.items()},
            "mrr": self.mrr,
            "ndcg_at": {str(k): v for k, v in self.ndcg_at.items()},
            "first_hit_rank": self.first_hit_rank,
            "n_gold": self.n_gold,
        }


def evaluate_question(
    question: BenchmarkQuestion,
    results: Sequence[RetrievedChunk],
    *,
    ks: Sequence[int] = (1, 5, 10, 20),
) -> QuestionMetrics:
    """Score one question's retrieval results.

    Args:
      question: The question with its gold labels.
      results: Retrieval results, best first.
      ks: Cut-offs to report at.

    Returns:
      The per-question metrics.
    """
    gold = _gold_units(question)
    grades = [relevance_grade(item, question) for item in results]
    units = [_unit_of(item) for item in results]

    metrics = QuestionMetrics(
        question_id=question.id,
        category=question.category,
        difficulty=question.difficulty,
        n_gold=len(gold),
    )
    for k in ks:
        top_units = {u for u, g in zip(units[:k], grades[:k]) if g > 0}
        hits = len(top_units & gold) if gold else 0
        metrics.recall_at[k] = hits / len(gold) if gold else 0.0
        metrics.precision_at[k] = (
            sum(1 for g in grades[:k] if g > 0) / k if k else 0.0
        )
        metrics.ndcg_at[k] = _ndcg(
            grades[:k], n_missed=len(gold - top_units) if gold else 0
        )

    for rank, grade in enumerate(grades, start=1):
        if grade > 0:
            metrics.mrr = 1.0 / rank
            metrics.first_hit_rank = rank
            break
    return metrics


def _ndcg(grades: Sequence[int], *, n_missed: int) -> float:
    """Compute nDCG against the best ranking achievable at this depth.

    The ideal ranking is the retrieved grades sorted descending, plus one
    grade-2 slot for every gold unit missed entirely. A run is therefore
    penalised both for ranking evidence low and for not finding it at all,
    and the score can never exceed 1 -- which it could if the ideal were
    built from the retrieved grades alone.

    Args:
      grades: Relevance grades in rank order, already cut at `k`.
      n_missed: Gold units that were not retrieved at all.

    Returns:
      nDCG in [0, 1], or 0.0 when nothing relevant was achievable.
    """
    dcg = sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(grades, start=1))
    ideal_grades = sorted(list(grades) + [2] * n_missed, reverse=True)[: len(grades)]
    idcg = sum(
        (2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(ideal_grades, start=1)
    )
    return round(dcg / idcg, 6) if idcg else 0.0


def aggregate(
    metrics: Iterable[QuestionMetrics], *, ks: Sequence[int] = (1, 5, 10, 20)
) -> dict:
    """Average per-question metrics into one report.

    Args:
      metrics: Per-question metrics.
      ks: Cut-offs to report at.

    Returns:
      Macro-averaged recall, precision and nDCG at each cut-off, plus MRR and
      hit rate at 10. Empty for an empty input.
    """
    metrics = list(metrics)
    if not metrics:
        return {}
    out: dict = {"n_questions": len(metrics)}
    for k in ks:
        out[f"recall@{k}"] = round(
            sum(m.recall_at.get(k, 0.0) for m in metrics) / len(metrics), 4
        )
        out[f"precision@{k}"] = round(
            sum(m.precision_at.get(k, 0.0) for m in metrics) / len(metrics), 4
        )
        out[f"ndcg@{k}"] = round(
            sum(m.ndcg_at.get(k, 0.0) for m in metrics) / len(metrics), 4
        )
    out["mrr"] = round(sum(m.mrr for m in metrics) / len(metrics), 4)
    out["hit_rate@10"] = round(
        sum(1 for m in metrics if m.first_hit_rank and m.first_hit_rank <= 10)
        / len(metrics),
        4,
    )
    return out


def aggregate_by(
    metrics: Iterable[QuestionMetrics],
    attribute: str = "category",
    *,
    ks: Sequence[int] = (5, 10),
) -> dict[str, dict]:
    """Aggregate metrics grouped by one attribute.

    Args:
      metrics: Per-question metrics.
      attribute: Attribute to group by, e.g. "category" or "difficulty".
      ks: Cut-offs to report at.

    Returns:
      Group value to its aggregate report, sorted by group.
    """
    grouped: dict[str, list[QuestionMetrics]] = {}
    for metric in metrics:
        grouped.setdefault(getattr(metric, attribute), []).append(metric)
    return {key: aggregate(values, ks=ks) for key, values in sorted(grouped.items())}
