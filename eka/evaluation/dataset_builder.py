"""Benchmark construction (PRD §20).

The dataset is built *from the repository*, not from the model's imagination:
every question is generated out of a concrete artifact — a symbol, a test
relation, a caller relation, a doc section, a commit — and its gold labels are
exactly the artifacts it was generated from.  The LLM only phrases the question;
it never chooses the answer key.

Two consequences worth stating plainly:

* the labels are correct by construction, but they are *minimal* — other files
  may also be reasonable evidence, so recall numbers are a lower bound;
* questions are **not** filtered by whether this system can retrieve them,
  which would make the benchmark self-congratulatory.

Curated multi-hop questions (architecture / debugging) are hand-written in
``curated.py`` and validated against the index like everything else.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from ..config import Config
from ..generation.llm import LLMClient, build_llm, extract_json
from ..indexing.knowledge_base import KnowledgeBase
from ..schema import Chunk
from .curated import curated_questions
from .dataset import (
    BenchmarkQuestion,
    assign_splits,
    dataset_summary,
    save_dataset,
)

QUESTION_PROMPT = """\
You are building an evaluation benchmark for a code question-answering system \
on the `{repository}` repository. Write ONE question that an engineer \
unfamiliar with the codebase would realistically ask, whose correct answer is \
the artifact below.

Artifact ({kind}) from `{path}`:
```
{content}
```

Requirements:
- The question must be answerable from this artifact{extra}.
- The question must stand on its own: a reader who cannot see the artifact must
  still know what is being asked. Never write "this method", "this commit",
  "the above" or similar — name the symbol, the behaviour or the error instead.
- Do NOT mention the file path, and do not quote the code.
- {style}
- Write it as one sentence, ending with a question mark.

Respond with a JSON object only:
{{"question": "...", "expected_answer": "<one or two sentences stating the \
answer, naming the relevant symbols>"}}
"""

# a question that points at an artifact the reader cannot see is unanswerable
_DANGLING = re.compile(
    r"\bthis (method|function|class|commit|code|file|change|module|object|"
    r"option|configuration|snippet|test|document|section|error)\b"
    r"|\bthe (above|following|given|shown) \b|\bhere\b\s*\?",
    re.I,
)

_STYLES = {
    "code_lookup": "You may name the symbol; ask where/how it is implemented or what it does.",
    "configuration": "Ask what the configuration option controls or how it is applied.",
    "tests": "Ask which tests exercise this behaviour, or how it is tested.",
    "change_impact": "Ask what would be affected if this symbol changed.",
    "historical": "Ask why the change was made or what problem it addressed.",
    "documentation": "Ask about the concept or workflow the documentation describes.",
    "debugging": "Ask what could cause the described failure mode.",
    "architecture": "Ask how the components involved fit together.",
}


@dataclass
class Candidate:
    """A repository fact selected to become a benchmark question.

    Gold labels are read off the chunk the candidate was built from, which
    is what makes them correct by construction rather than by review.

    Attributes:
      kind: Question category to generate.
      chunk: Chunk the question is built from.
      gold_files: Gold file paths.
      gold_symbols: Gold qualified symbol names.
      gold_chunks: Gold chunk ids.
      gold_commits: Gold commit SHAs.
      difficulty: "single_hop" or "multi_hop".
      extra_context: Additional text shown to the question-writing model.
      template: Question used when no LLM is available.
      template_answer: Reference answer for the templated question.
    """

    kind: str
    chunk: Chunk
    gold_files: list[str]
    gold_symbols: list[str]
    gold_chunks: list[str]
    gold_commits: list[str]
    difficulty: str = "single_hop"
    extra_context: str = ""
    template: str = ""
    template_answer: str = ""


class DatasetBuilder:
    """Generates benchmark questions from the indexed repository.

    Questions are built *from* the repository, so gold labels are correct by
    construction. The cost is that a generated question inherits vocabulary
    from its source chunk, which flatters lexical retrieval -- which is why
    the hand-written curated set is reported separately.

    Attributes:
      config: Configuration supplying the LLM and repository name.
      kb: Knowledge base to draw candidates from.
      rng: Seeded RNG, so a rebuild reproduces the same selection.
      seed: The RNG seed.
      use_llm: Phrase questions with an LLM rather than from templates.
    """

    def __init__(
        self,
        config: Config,
        kb: KnowledgeBase,
        *,
        use_llm: bool = True,
        llm: LLMClient | None = None,
        seed: int = 0,
    ) -> None:
        """Configure the builder.

        Args:
          config: Configuration supplying the LLM and repository name.
          kb: Knowledge base to draw candidates from.
          use_llm: Phrase questions with an LLM; templates are used when off.
          llm: Client to use; built from config on first use when omitted.
          seed: RNG seed, so a rebuild reproduces the same selection.
        """
        self.config = config
        self.kb = kb
        self.rng = random.Random(seed)
        self.seed = seed
        self._llm = llm
        self.use_llm = use_llm

    @property
    def llm(self) -> LLMClient | None:
        """LLMClient | None: Question-writing client, or `None` when disabled."""
        if not self.use_llm:
            return None
        if self._llm is None:
            self._llm = build_llm(self.config.llm)
        return self._llm

    # ------------------------------------------------------------------
    def build(
        self,
        *,
        n_auto: int = 120,
        include_curated: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> list[BenchmarkQuestion]:
        """Build a benchmark dataset.

        Candidates are drawn per category against fixed quotas, so the
        benchmark covers every question type rather than whichever type the
        repository happens to offer most of.

        Args:
          n_auto: How many generated questions to aim for.
          include_curated: Append the hand-written questions.
          progress: Called with a short status line as generation proceeds.

        Returns:
          The questions, with dev/test splits already assigned.
        """
        quotas = {
            "code_lookup": 0.24,
            "configuration": 0.12,
            "tests": 0.12,
            "change_impact": 0.14,
            "historical": 0.16,
            "documentation": 0.14,
            "debugging": 0.08,
        }
        generators: dict[str, Callable[[int], list[Candidate]]] = {
            "code_lookup": self._code_candidates,
            "configuration": self._config_candidates,
            "tests": self._test_candidates,
            "change_impact": self._impact_candidates,
            "historical": self._commit_candidates,
            "documentation": self._doc_candidates,
            "debugging": self._debug_candidates,
        }

        records: list[BenchmarkQuestion] = []
        for category, share in quotas.items():
            wanted = max(int(round(n_auto * share)), 1)
            candidates = generators[category](wanted * 2)
            self.rng.shuffle(candidates)
            made = 0
            for candidate in candidates:
                if made >= wanted:
                    break
                record = self._to_question(candidate, index=len(records))
                if record is None:
                    continue
                records.append(record)
                made += 1
            if progress:
                progress(f"{category}: {made} questions")

        if include_curated:
            curated = [c for c in curated_questions() if self._gold_exists(c)]
            records.extend(curated)
            if progress:
                progress(f"curated: {len(curated)} questions")
        return assign_splits(records)

    # ------------------------------------------------------------------
    def _to_question(self, candidate: Candidate, index: int) -> BenchmarkQuestion | None:
        question_text = candidate.template
        expected = candidate.template_answer
        source = "template"

        llm = self.llm
        if llm is not None:
            payload = self._ask_llm(candidate, llm)
            if payload and not _DANGLING.search(payload[0]):
                question_text, expected = payload
                source = "llm_phrased"
            elif payload:
                # keep the model's answer sketch, but use the self-contained
                # template phrasing instead of a dangling reference
                expected = payload[1] or expected
                source = "template_repaired"

        if not question_text or len(question_text) < 15:
            return None
        # a question that leaks the answer's path is not a retrieval test
        for path in candidate.gold_files:
            if path in question_text or Path(path).name in question_text:
                return None
        qid = f"{candidate.kind}-{index:03d}"
        return BenchmarkQuestion(
            id=qid,
            question=question_text.strip(),
            category=candidate.kind,
            difficulty=candidate.difficulty,
            expected_answer=expected.strip(),
            relevant_files=sorted(set(candidate.gold_files)),
            relevant_symbols=sorted(set(candidate.gold_symbols)),
            relevant_chunks=sorted(set(candidate.gold_chunks)),
            relevant_commits=sorted(set(candidate.gold_commits)),
            source=f"auto:{source}",
            provenance={
                "generated_from": candidate.chunk.location,
                "symbol": candidate.chunk.qualified_name,
                "seed": self.seed,
            },
        )

    def _ask_llm(self, candidate: Candidate, llm: LLMClient) -> tuple[str, str] | None:
        prompt = QUESTION_PROMPT.format(
            repository=self.config.repository,
            kind=candidate.chunk.symbol_type or candidate.kind,
            path=candidate.chunk.path,
            content=candidate.chunk.content[:3000],
            extra=candidate.extra_context,
            style=_STYLES.get(candidate.kind, ""),
        )
        try:
            response = llm.complete(prompt, max_tokens=400)
        except Exception:
            return None
        data = extract_json(response.text)
        if not data:
            return None
        question = str(data.get("question", "")).strip()
        expected = str(data.get("expected_answer", "")).strip()
        if not question.endswith("?") or len(question) > 400:
            return None
        return question, expected

    def _gold_exists(self, record: BenchmarkQuestion) -> bool:
        known = set(self.kb.store.paths)
        missing = [p for p in record.relevant_files if p not in known]
        return not missing

    # ------------------------------------------------------------------
    # candidate generators — each returns gold labels by construction
    # ------------------------------------------------------------------
    def _source_chunks(self, predicate: Callable[[Chunk], bool]) -> list[Chunk]:
        return [c for c in self.kb.store.chunks if predicate(c)]

    def _code_candidates(self, n: int) -> list[Candidate]:
        pool = self._source_chunks(
            lambda c: c.artifact_type == "source"
            and c.symbol_type in ("class", "function", "method")
            and c.docstring
            and len(c.docstring) > 120
            and c.path.startswith("vllm/")
            and c.end_line - c.start_line >= 8
        )
        self.rng.shuffle(pool)
        out: list[Candidate] = []
        for chunk in pool[: n * 2]:
            summary = " ".join((chunk.docstring or "").split())[:200]
            out.append(
                Candidate(
                    kind="code_lookup",
                    chunk=chunk,
                    gold_files=[chunk.path],
                    gold_symbols=[chunk.qualified_name or ""],
                    gold_chunks=[chunk.chunk_id],
                    gold_commits=[],
                    template=f"Where is `{chunk.symbol}` implemented and what does it do?",
                    template_answer=f"{chunk.qualified_name} in {chunk.path}: {summary}",
                )
            )
            if len(out) >= n:
                break
        return out

    def _config_candidates(self, n: int) -> list[Candidate]:
        pool = self._source_chunks(
            lambda c: c.artifact_type == "source"
            and (c.path.startswith("vllm/config/") or "Config" in (c.symbol or ""))
            and c.symbol_type in ("class", "method", "function")
            and len(c.content) > 300
        )
        self.rng.shuffle(pool)
        out: list[Candidate] = []
        for chunk in pool[: n * 2]:
            fields = re.findall(r"^\s{4}([a-z_][a-z0-9_]*)\s*:", chunk.content, re.M)[:8]
            if not fields and chunk.symbol_type == "class":
                continue
            out.append(
                Candidate(
                    kind="configuration",
                    chunk=chunk,
                    gold_files=[chunk.path],
                    gold_symbols=[chunk.qualified_name or ""],
                    gold_chunks=[chunk.chunk_id],
                    gold_commits=[],
                    template=(
                        f"Which configuration object defines the "
                        f"{', '.join(fields[:3]) or chunk.symbol} settings and how are they used?"
                    ),
                    template_answer=f"{chunk.qualified_name} in {chunk.path}.",
                )
            )
            if len(out) >= n:
                break
        return out

    def _test_candidates(self, n: int) -> list[Candidate]:
        graph = self.kb.graph
        if graph is None:
            return []
        symbols = [
            s for s, tests in graph.tests.items()
            if len(tests) >= 2 and s in graph.definitions and len(s) > 5
        ]
        self.rng.shuffle(symbols)
        out: list[Candidate] = []
        for symbol in symbols:
            definition = self.kb.store.get(graph.definitions[symbol][0])
            if definition is None or not definition.path.startswith("vllm/"):
                continue
            test_chunks = [
                self.kb.store.get(cid) for cid in graph.tests[symbol][:4]
            ]
            test_files = sorted({c.path for c in test_chunks if c})
            if not test_files:
                continue
            out.append(
                Candidate(
                    kind="tests",
                    chunk=definition,
                    gold_files=[definition.path] + test_files[:3],
                    gold_symbols=[definition.qualified_name or ""],
                    gold_chunks=[definition.chunk_id],
                    gold_commits=[],
                    difficulty="multi_hop",
                    extra_context=(
                        f"; it is exercised by the tests in {', '.join(test_files[:3])}"
                    ),
                    template=f"Which tests cover the behaviour of `{symbol}`?",
                    template_answer=(
                        f"{definition.qualified_name} is tested in {', '.join(test_files[:3])}."
                    ),
                )
            )
            if len(out) >= n:
                break
        return out

    def _impact_candidates(self, n: int) -> list[Candidate]:
        graph = self.kb.graph
        if graph is None:
            return []
        symbols = [s for s, callers in graph.callers.items() if len(callers) >= 4]
        self.rng.shuffle(symbols)
        out: list[Candidate] = []
        for symbol in symbols:
            if symbol not in graph.definitions:
                continue
            definition = self.kb.store.get(graph.definitions[symbol][0])
            if definition is None or not definition.path.startswith("vllm/"):
                continue
            caller_chunks = [self.kb.store.get(cid) for cid in graph.callers[symbol][:8]]
            caller_files = sorted(
                {c.path for c in caller_chunks if c and c.path != definition.path}
            )[:3]
            if len(caller_files) < 2:
                continue
            out.append(
                Candidate(
                    kind="change_impact",
                    chunk=definition,
                    gold_files=[definition.path] + caller_files,
                    gold_symbols=[definition.qualified_name or ""],
                    gold_chunks=[definition.chunk_id],
                    gold_commits=[],
                    difficulty="multi_hop",
                    extra_context=(
                        f"; it is called from {', '.join(caller_files)}"
                    ),
                    template=(
                        f"If the behaviour of `{symbol}` changed, which components "
                        f"would be affected?"
                    ),
                    template_answer=(
                        f"{definition.qualified_name} is used by {', '.join(caller_files)}."
                    ),
                )
            )
            if len(out) >= n:
                break
        return out

    def _commit_candidates(self, n: int) -> list[Candidate]:
        pool = [
            c for c in self.kb.store.chunks
            if c.artifact_type == "commit"
            and c.files_changed
            and len(c.content) > 500
            and any(f.endswith(".py") for f in c.files_changed)
            and len(c.files_changed) <= 6
            and not (c.extra.get("subject", "").lower().startswith(("bump", "chore")))
        ]
        self.rng.shuffle(pool)
        out: list[Candidate] = []
        for chunk in pool[: n * 2]:
            python_files = [f for f in chunk.files_changed if f.endswith(".py")][:3]
            subject = chunk.extra.get("subject", "")
            if len(subject) < 20:
                continue
            out.append(
                Candidate(
                    kind="historical",
                    chunk=chunk,
                    gold_files=python_files,
                    gold_symbols=[],
                    gold_chunks=[chunk.chunk_id],
                    gold_commits=[chunk.commit_sha or ""],
                    difficulty="multi_hop" if len(python_files) > 1 else "single_hop",
                    extra_context=" (a commit); the question must not quote the commit subject verbatim",
                    template=f"Why was the change described as \"{subject}\" made?",
                    template_answer=(
                        f"Commit {(chunk.commit_sha or '')[:10]} — {subject} — "
                        f"touching {', '.join(python_files)}."
                    ),
                )
            )
            if len(out) >= n:
                break
        return out

    def _doc_candidates(self, n: int) -> list[Candidate]:
        pool = self._source_chunks(
            lambda c: c.artifact_type == "doc"
            and len(c.content) > 600
            and c.heading_path
            and not c.path.startswith("docs/api")
        )
        self.rng.shuffle(pool)
        out: list[Candidate] = []
        for chunk in pool[: n * 2]:
            heading = " > ".join(chunk.heading_path)
            out.append(
                Candidate(
                    kind="documentation",
                    chunk=chunk,
                    gold_files=[chunk.path],
                    gold_symbols=[],
                    gold_chunks=[chunk.chunk_id],
                    gold_commits=[],
                    template=f"What does the documentation say about {heading}?",
                    template_answer=" ".join(chunk.content.split())[:300],
                )
            )
            if len(out) >= n:
                break
        return out

    def _debug_candidates(self, n: int) -> list[Candidate]:
        """Questions anchored on code that raises a specific error."""
        error_pattern = re.compile(
            r"raise\s+(\w*Error|\w*Exception)\(\s*\n?\s*[\"f]{1,2}([^\"]{25,200})", re.M
        )
        pool = self._source_chunks(
            lambda c: c.artifact_type == "source"
            and c.path.startswith("vllm/")
            and ("raise " in c.content or "logger.warning" in c.content)
            and len(c.content) > 400
        )
        self.rng.shuffle(pool)
        out: list[Candidate] = []
        for chunk in pool[: n * 4]:
            match = error_pattern.search(chunk.content)
            if not match:
                continue
            message = " ".join(match.group(2).split())[:160]
            out.append(
                Candidate(
                    kind="debugging",
                    chunk=chunk,
                    gold_files=[chunk.path],
                    gold_symbols=[chunk.qualified_name or ""],
                    gold_chunks=[chunk.chunk_id],
                    gold_commits=[],
                    extra_context=(
                        f"; the artifact raises {match.group(1)} with the message "
                        f"\"{message}\". Write the question as an engineer who hit "
                        f"that error and wants to know what causes it"
                    ),
                    template=(
                        f"An engineer sees the error \"{message}\" — what in the "
                        f"codebase raises it and why?"
                    ),
                    template_answer=(
                        f"{chunk.qualified_name} in {chunk.path} raises {match.group(1)}."
                    ),
                )
            )
            if len(out) >= n:
                break
        return out

    # ------------------------------------------------------------------
    def save(self, records: Iterable[BenchmarkQuestion], path: Path) -> Path:
        """Write questions to `path` and return it.

        Args:
          records: Questions to write.
          path: Destination JSON Lines file.

        Returns:
          The path written.
        """
        save_dataset(records, path)
        return path

    def summary(self, records: list[BenchmarkQuestion]) -> dict[str, Any]:
        """Summarise a built dataset's composition.

        Args:
          records: The questions.

        Returns:
          The same summary `dataset_summary` produces.
        """
        return dataset_summary(records)
