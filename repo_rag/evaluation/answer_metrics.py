"""Answer evaluation (PRD §22).

Two independent families of metrics:

*Deterministic citation metrics* — computed from the answer text and the
retrieved evidence, no model involved: are the cited locations real, do they
match the gold files, how much of the gold evidence was cited.

*LLM-judged metrics* — correctness against the reference answer, faithfulness
to the evidence, and unsupported-claim counting.  ``judge_audit`` samples judged
answers for manual review so the judge itself can be validated.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..generation.llm import LLMClient, extract_json
from ..schema import Answer, RetrievedChunk
from .dataset import BenchmarkQuestion

JUDGE_PROMPT = """\
You are grading an answer produced by a repository question-answering system.

Question:
{question}

Reference answer (may be partial; treat it as ground truth where it speaks):
{reference}

Gold evidence locations for this question:
{gold}

System answer:
{answer}

Evidence that was actually retrieved and shown to the system:
{evidence}

Grade strictly and respond with a JSON object only:
{{
  "correctness": 0 | 1 | 2,        // 0 wrong/irrelevant, 1 partially correct, 2 correct
  "faithfulness": 0 | 1 | 2,       // 0 contradicts evidence, 1 partly unsupported, 2 fully supported
  "n_claims": <int>,               // substantive technical claims in the answer
  "n_unsupported_claims": <int>,   // claims not supported by the evidence shown
  "cites_correct_location": true | false,
  "justification": "<one sentence>"
}}
"""


@dataclass
class AnswerScores:
    """Deterministic and judged scores for one answer.

    Attributes:
      question_id: The question's id.
      n_citations: Citations emitted, valid and invalid alike.
      citation_validity: Share of cited locations that exist in the evidence.
      citation_precision: Share of cited files that are gold files.
      citation_completeness: Share of gold files that were cited.
      correctness: Judged correctness in [0, 1], or `None` if not judged.
      faithfulness: Judged faithfulness in [0, 1], or `None`.
      unsupported_claim_rate: Share of claims the judge found unsupported.
      cites_correct_location: Judge's verdict on citation targeting.
      judge_raw: The judge's raw response, or an error record.
    """

    question_id: str
    n_citations: int = 0
    citation_validity: float = 0.0
    citation_precision: float = 0.0
    citation_completeness: float = 0.0
    correctness: float | None = None
    faithfulness: float | None = None
    unsupported_claim_rate: float | None = None
    cites_correct_location: bool | None = None
    judge_raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the scores as a JSON-serialisable dict."""
        return {
            "question_id": self.question_id,
            "n_citations": self.n_citations,
            "citation_validity": self.citation_validity,
            "citation_precision": self.citation_precision,
            "citation_completeness": self.citation_completeness,
            "correctness": self.correctness,
            "faithfulness": self.faithfulness,
            "unsupported_claim_rate": self.unsupported_claim_rate,
            "cites_correct_location": self.cites_correct_location,
            "judge": self.judge_raw,
        }


def citation_metrics(
    answer: Answer, question: BenchmarkQuestion
) -> tuple[int, float, float, float]:
    """Compute citation metrics without involving a model.

    Args:
      answer: The generated answer, whose usage records the citations that
        failed validation.
      question: The question with its gold files.

    Returns:
      A `(n_citations, validity, precision, completeness)` tuple. Precision
      and completeness are 0.0 when either side is empty, since there is
      nothing to be right about.
    """
    unsupported = answer.usage.get("unsupported_citations", []) or []
    n_valid = len(answer.citations)
    n_total = n_valid + len(unsupported)
    validity = n_valid / n_total if n_total else 0.0

    cited_files = {c.path for c in answer.citations}
    gold_files = set(question.relevant_files)
    if gold_files and cited_files:
        precision = len(cited_files & gold_files) / len(cited_files)
        completeness = len(cited_files & gold_files) / len(gold_files)
    else:
        precision = 0.0
        completeness = 0.0
    return n_total, round(validity, 4), round(precision, 4), round(completeness, 4)


class AnswerJudge:
    """Grades answers with an LLM, alongside the deterministic metrics.

    The judge sees the reference answer, the gold locations *and* the
    evidence actually retrieved, so it can distinguish an answer that is
    wrong from one that is faithful to evidence that was itself incomplete.

    Attributes:
      llm: Judging client.
      max_evidence_chars: Cap on rendered evidence in the prompt.
    """

    def __init__(self, llm: LLMClient, *, max_evidence_chars: int = 6000) -> None:
        """Configure the judge.

        Args:
          llm: Judging client. Configure `judge_llm` separately from `llm` to
            avoid a model grading its own output.
          max_evidence_chars: Cap on rendered evidence in the prompt.
        """
        self.llm = llm
        self.max_evidence_chars = max_evidence_chars

    def score(
        self,
        question: BenchmarkQuestion,
        answer: Answer,
        evidence: Sequence[RetrievedChunk],
    ) -> AnswerScores:
        """Score one answer, deterministically and by judgement.

        Args:
          question: The question with its gold labels.
          answer: The generated answer.
          evidence: The evidence the answer was generated from.

        Returns:
          The scores. A judge failure leaves the judged fields `None` and
          records the error in `judge_raw`, so an unreachable judge shows up
          as missing data rather than as a zero.
        """
        n_cit, validity, precision, completeness = citation_metrics(answer, question)
        scores = AnswerScores(
            question_id=question.id,
            n_citations=n_cit,
            citation_validity=validity,
            citation_precision=precision,
            citation_completeness=completeness,
        )
        rendered = self._render_evidence(evidence)
        prompt = JUDGE_PROMPT.format(
            question=question.question,
            reference=question.expected_answer or "(no reference answer available)",
            gold=", ".join(question.relevant_files + question.relevant_symbols) or "(none)",
            answer=answer.text[:6000] or "(empty)",
            evidence=rendered,
        )
        try:
            response = self.llm.complete(prompt, max_tokens=500)
            data = extract_json(response.text) or {}
        except Exception as exc:
            scores.judge_raw = {"error": str(exc)}
            return scores

        scores.judge_raw = data
        correctness = _as_int(data.get("correctness"))
        faithfulness = _as_int(data.get("faithfulness"))
        if correctness is not None:
            scores.correctness = round(min(max(correctness, 0), 2) / 2, 4)
        if faithfulness is not None:
            scores.faithfulness = round(min(max(faithfulness, 0), 2) / 2, 4)
        n_claims = _as_int(data.get("n_claims")) or 0
        n_unsupported = _as_int(data.get("n_unsupported_claims")) or 0
        if n_claims > 0:
            scores.unsupported_claim_rate = round(min(n_unsupported / n_claims, 1.0), 4)
        elif n_unsupported == 0:
            scores.unsupported_claim_rate = 0.0
        cites = data.get("cites_correct_location")
        if isinstance(cites, bool):
            scores.cites_correct_location = cites
        return scores

    def _render_evidence(self, evidence: Sequence[RetrievedChunk]) -> str:
        """Render evidence for the judge prompt within the character budget.

        Args:
          evidence: Evidence blocks to render.

        Returns:
          Delimited blocks sharing the budget evenly, each keeping at least
          400 characters so no block is reduced to nothing.
        """
        parts: list[str] = []
        budget = self.max_evidence_chars
        for item in evidence:
            chunk = item.chunk
            block = f"[{chunk.location}] {chunk.qualified_name or ''}\n{chunk.content}"
            block = block[: max(budget // max(len(evidence), 1), 400)]
            parts.append(block)
        return "\n---\n".join(parts)[: self.max_evidence_chars] or "(no evidence)"


def _as_int(value: Any) -> int | None:
    """Coerce a judge-supplied value to an int.

    Args:
      value: Whatever the model returned in that field.

    Returns:
      The integer, or `None` when the value cannot be read as one -- which
      keeps a malformed field from silently scoring as zero.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def aggregate_answer_scores(scores: Sequence[AnswerScores]) -> dict[str, Any]:
    """Average answer scores into one report.

    Args:
      scores: Per-answer scores.

    Returns:
      Mean correctness, faithfulness, unsupported-claim rate and citation
      metrics, plus the judge failure count. Means skip `None` values, so a
      partially failed judging run still reports the answers it did grade.
    """
    if not scores:
        return {}

    def mean(values: list[float]) -> float | None:
        """Average the non-`None` values, or return `None` if there are none."""
        clean = [v for v in values if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else None

    return {
        "n_answers": len(scores),
        "answer_accuracy": mean([s.correctness for s in scores]),
        "faithfulness": mean([s.faithfulness for s in scores]),
        "unsupported_claim_rate": mean([s.unsupported_claim_rate for s in scores]),
        "citation_validity": mean([s.citation_validity for s in scores]),
        "citation_precision": mean([s.citation_precision for s in scores]),
        "citation_completeness": mean([s.citation_completeness for s in scores]),
        "citations_per_answer": mean([float(s.n_citations) for s in scores]),
        "judge_failures": sum(1 for s in scores if "error" in (s.judge_raw or {})),
    }


# ----------------------------------------------------------------------
def sample_for_audit(
    records: list[dict[str, Any]], out_path: Path, *, n: int = 20, seed: int = 0
) -> Path:
    """Write a manual-review sheet so the LLM judge can itself be validated.

    Reporting a judge/human agreement number without doing this would be
    reporting a number nobody checked.

    Args:
      records: Judged answer records.
      out_path: Destination JSON Lines file.
      n: How many to sample.
      seed: Sampling seed, so the sheet is reproducible.

    Returns:
      The path written. Each row carries the judge's scores and blank
      `human_*` fields for a reviewer to fill in.
    """
    rng = random.Random(seed)
    sample = rng.sample(records, min(n, len(records)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in sample:
            fh.write(
                json.dumps(
                    {
                        "question_id": record.get("question_id"),
                        "question": record.get("question"),
                        "answer": record.get("answer"),
                        "judge_correctness": (record.get("scores") or {}).get("correctness"),
                        "judge_faithfulness": (record.get("scores") or {}).get("faithfulness"),
                        "human_correctness": None,   # fill in: 0 | 0.5 | 1
                        "human_faithfulness": None,  # fill in: 0 | 0.5 | 1
                        "note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out_path


def judge_agreement(audit_path: Path) -> dict[str, Any]:
    """Compare the LLM judge against human labels in an audit file.

    Args:
      audit_path: Audit sheet written by `sample_for_audit` and since
        filled in.

    Returns:
      Exact agreement and mean absolute error for correctness and
      faithfulness. When no row has been labelled, a note saying so rather
      than a vacuous agreement of 1.0.
    """
    rows = [
        json.loads(line)
        for line in Path(audit_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labelled = [r for r in rows if r.get("human_correctness") is not None]
    if not labelled:
        return {"labelled": 0, "note": "fill in human_* fields to compute agreement"}

    def agreement(field_name: str) -> dict[str, Any]:
        """Compare judge and human labels for one field.

        Args:
          field_name: Field stem, e.g. "correctness".

        Returns:
          The pair count, exact agreement and mean absolute error; empty
          when no row has both labels.
        """
        pairs = [
            (r.get(f"judge_{field_name}"), r.get(f"human_{field_name}"))
            for r in labelled
            if r.get(f"human_{field_name}") is not None
            and r.get(f"judge_{field_name}") is not None
        ]
        if not pairs:
            return {}
        exact = sum(1 for j, h in pairs if abs(float(j) - float(h)) < 1e-6) / len(pairs)
        mae = sum(abs(float(j) - float(h)) for j, h in pairs) / len(pairs)
        return {"n": len(pairs), "exact_agreement": round(exact, 4), "mae": round(mae, 4)}

    return {
        "labelled": len(labelled),
        "correctness": agreement("correctness"),
        "faithfulness": agreement("faithfulness"),
    }
