"""Query planning (PRD §18).

Classifies the question and proposes an initial retrieval plan.  A rule-based
classifier always runs; the LLM planner refines it when a model is configured.
Keeping the heuristic path means the agent degrades gracefully (and stays
testable) without an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..generation.llm import LLMClient, extract_json
from ..retrieval.symbol import extract_symbols

CATEGORIES = (
    "code_lookup",
    "architecture",
    "debugging",
    "change_impact",
    "historical",
    "documentation",
    "configuration",
    "tests",
)

# tool preferences per category (PRD §18)
CATEGORY_TOOLS: dict[str, list[str]] = {
    "code_lookup": ["symbol_search", "search"],
    "architecture": ["search", "symbol_search"],
    "debugging": ["search", "symbol_search"],
    "change_impact": ["impact", "symbol_search", "search"],
    "historical": ["history", "search"],
    "documentation": ["search"],
    "configuration": ["search", "symbol_search"],
    "tests": ["impact", "search"],
}

_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("historical", re.compile(
        r"\b(why (was|were|did)|when was|history|introduced|changed|regression|"
        r"commit|pull request|\bpr\b|deprecat)", re.I)),
    ("change_impact", re.compile(
        r"\b(if i (change|modify|remove|rename)|impact|affect|break|depends? on|"
        r"callers?|who calls|downstream)", re.I)),
    ("debugging", re.compile(
        r"\b(error|exception|traceback|fails?|failing|crash|out of memory|oom|"
        r"hang|deadlock|leak|debug|why (is|does).*(slow|fail))", re.I)),
    ("tests", re.compile(r"\b(tests?|pytest|test coverage|which tests)\b", re.I)),
    ("architecture", re.compile(
        r"\b(architecture|end.to.end|how does .* (flow|travel|work)|pipeline|"
        r"lifecycle|trace a request|interact|overall|components?)\b", re.I)),
    ("configuration", re.compile(
        r"\b(config|configuration|flag|environment variable|env var|parameter|"
        r"default value|setting)\b", re.I)),
    ("documentation", re.compile(r"\b(docs?|documentation|readme|guide|tutorial)\b", re.I)),
    ("code_lookup", re.compile(
        r"\b(where|which file|implemented|defined|located|function|class|method)\b", re.I)),
]


@dataclass
class QueryPlan:
    """How the agent intends to search, before it searches.

    Attributes:
      question: The question as asked.
      category: One of `CATEGORIES`.
      subqueries: Queries to issue, the original question always first.
      symbols: Identifiers worth looking up exactly.
      tools: Tool names in preference order for this category.
      rationale: One-sentence justification, when the LLM supplied one.
      source: "rules" or "llm" -- which path produced this plan.
      usage: Token usage of the planning call, empty for a rule-only plan.
    """

    question: str
    category: str = "code_lookup"
    subqueries: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    rationale: str = ""
    source: str = "rules"
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as a JSON-serialisable dict for traces and the API."""
        return {
            "category": self.category,
            "subqueries": self.subqueries,
            "symbols": self.symbols,
            "tools": self.tools,
            "rationale": self.rationale,
            "source": self.source,
        }


def classify(question: str) -> str:
    """Classify a question by matching ordered regex rules.

    The rules are ordered most-specific first, and the first match wins, so
    "why was this changed" is historical rather than merely a code lookup.

    Args:
      question: Question text.

    Returns:
      A member of `CATEGORIES`, defaulting to "code_lookup".
    """
    for category, pattern in _RULES:
        if pattern.search(question):
            return category
    return "code_lookup"


PLANNER_PROMPT = """\
You are the retrieval planner of a code-search system for the `{repository}` \
repository. Decide how to search; do not answer the question.

Question: {question}

Heuristic classification: {category}
Identifiers detected in the question: {symbols}

Respond with a JSON object only:
{{
  "category": one of {categories},
  "subqueries": [up to {max_subqueries} short search queries, each targeting a
                 different artifact: implementation, configuration, tests, docs,
                 or history. Use repository vocabulary, not full sentences],
  "symbols": [code identifiers worth looking up exactly],
  "rationale": "one sentence"
}}
"""


class QueryPlanner:
    """Classifies a question and proposes an initial retrieval plan.

    The rule-based classifier always runs; the LLM refines its output when a
    model is configured. Every LLM failure -- unreachable server, unparseable
    JSON, unknown category -- falls back to the rule plan, so the agent
    degrades rather than breaks, and stays testable with no model at all.

    Attributes:
      llm: Planning client, or `None` for rules only.
      repository: Repository name, given to the model as context.
      max_subqueries: Cap on sub-queries the model may add.
    """

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        repository: str = "repository",
        max_subqueries: int = 3,
    ) -> None:
        """Configure the planner.

        Args:
          llm: Planning client; omit for the rule-based path only.
          repository: Repository name, given to the model as context.
          max_subqueries: Cap on sub-queries the model may add.
        """
        self.llm = llm
        self.repository = repository
        self.max_subqueries = max_subqueries

    # ------------------------------------------------------------------
    def plan(self, question: str) -> QueryPlan:
        """Produce a retrieval plan for a question.

        Args:
          question: Question text.

        Returns:
          The plan. `source` is "llm" when the model was reached and its
          response parsed, and "rules" otherwise -- including every failure
          case, so a degraded plan is visible in the trace rather than
          indistinguishable from a healthy one.
        """
        category = classify(question)
        symbols = extract_symbols(question)
        plan = QueryPlan(
            question=question,
            category=category,
            subqueries=[question],
            symbols=symbols,
            tools=CATEGORY_TOOLS.get(category, ["search"]),
            source="rules",
        )
        if self.llm is None:
            return plan

        prompt = PLANNER_PROMPT.format(
            repository=self.repository,
            question=question,
            category=category,
            symbols=symbols or "none",
            categories=list(CATEGORIES),
            max_subqueries=self.max_subqueries,
        )
        try:
            response = self.llm.complete(prompt, max_tokens=400)
        except Exception:
            return plan
        data = extract_json(response.text)
        if not data:
            return plan

        llm_category = str(data.get("category", category))
        if llm_category in CATEGORIES:
            plan.category = llm_category
        subqueries = [
            str(q).strip() for q in (data.get("subqueries") or []) if str(q).strip()
        ]
        plan.subqueries = ([question] + subqueries)[: self.max_subqueries + 1]
        extra_symbols = [str(s).strip() for s in (data.get("symbols") or []) if str(s).strip()]
        plan.symbols = list(dict.fromkeys(symbols + extra_symbols))[:8]
        plan.tools = CATEGORY_TOOLS.get(plan.category, ["search"])
        plan.rationale = str(data.get("rationale", ""))[:400]
        plan.source = "llm"
        plan.usage = {  # type: ignore[attr-defined]
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "cost_usd": response.cost_usd,
        }
        return plan
