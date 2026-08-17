"""Typed, YAML-backed configuration.

Every knob that affects an index or a benchmark number lives here so that a run
can be reproduced from ``configs/*.yaml`` alone (PRD §29 reproducibility).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class IngestionConfig:
    """What to read out of a checkout and how to cut it into chunks.

    Attributes:
      include_globs: Patterns of files to consider, relative to the checkout.
      exclude_dirs: Directory names pruned before matching, for speed as much
        as for relevance.
      max_file_bytes: Files larger than this are skipped as generated data.
      max_chunk_lines: Symbols longer than this are split into several chunks.
      chunk_overlap_lines: Lines repeated across such a split, so a construct
        straddling the cut survives in at least one chunk.
      min_chunk_chars: Chunks shorter than this are dropped as noise.
      index_module_chunks: Emit one module-level summary chunk per file, which
        is what lets a question about a *file* retrieve anything at all.
      doc_max_chars: Document sections longer than this are split.
      doc_min_chars: Document sections shorter than this are dropped.
      git_max_commits: How much history to ingest, newest first.
      git_max_diff_chars: Per-commit diff truncation point.
      git_include_merges: Whether merge commits are ingested; they are usually
        pure noise, hence the default.
    """

    include_globs: list[str] = field(
        default_factory=lambda: ["**/*.py", "**/*.md", "**/*.rst", "**/*.yaml", "**/*.yml"]
    )
    exclude_dirs: list[str] = field(
        default_factory=lambda: [
            ".git", ".github/workflows", "node_modules", "__pycache__", ".venv",
            "build", "dist", "third_party", ".buildkite",
        ]
    )
    max_file_bytes: int = 400_000
    # code chunking
    max_chunk_lines: int = 220          # oversized symbols are split
    chunk_overlap_lines: int = 20
    min_chunk_chars: int = 60
    index_module_chunks: bool = True    # module-level "file summary" chunks
    # docs
    doc_max_chars: int = 4_000
    doc_min_chars: int = 120
    # git
    git_max_commits: int = 3_000
    git_max_diff_chars: int = 6_000
    git_include_merges: bool = False


@dataclass
class EmbeddingConfig:
    """Dense embedding model and how it is run.

    Attributes:
      model: Sentence-transformers model id.
      batch_size: Chunks encoded per forward pass.
      max_seq_length: Token cap per chunk; code runs ~2.35 chars/token, so 512
        tokens keeps roughly 1,200 characters.
      device: "auto", "cuda" or "cpu".
      normalize: L2-normalise embeddings so inner product is cosine.
      query_prefix: Instruction prefix some models require on queries.
      document_prefix: Instruction prefix some models require on documents.
      trust_remote_code: Allow the model repo to execute its own modelling
        code, which some non-MiniLM embedders need.
    """

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256
    max_seq_length: int = 512
    device: str = "auto"          # auto|cuda|cpu
    normalize: bool = True
    query_prefix: str = ""
    document_prefix: str = ""
    trust_remote_code: bool = False


@dataclass
class RetrievalConfig:
    """Per-retriever depth, fusion and the two cut-offs after it.

    Attributes:
      dense_k: Candidates taken from the vector index.
      bm25_k: Candidates taken from the lexical index.
      symbol_k: Candidates taken from exact/fuzzy symbol lookup.
      git_k: Candidates taken from commit history.
      fusion: Fusion method name, a key of `retrieval.fusion.FUSION_METHODS`.
      rrf_k: RRF rank offset. Small values weight the top ranks steeply; the
        tuned default of 10 is far below the customary 60.
      fusion_weights: Per-source weight applied to each source's contribution.
        Tuned on the dev split by `scripts/tune_fusion.py` and specific to the
        indexed repository -- a commit-heavy repo needs a different balance.
      candidate_k: Fused candidates handed to the reranker.
      final_k: Evidence chunks handed to the LLM.
    """

    dense_k: int = 40
    bm25_k: int = 40
    symbol_k: int = 10
    git_k: int = 10
    fusion: str = "rrf"           # rrf|score_sum
    rrf_k: int = 60
    fusion_weights: dict[str, float] = field(
        default_factory=lambda: {"dense": 1.0, "bm25": 1.0, "symbol": 1.0, "git": 0.6}
    )
    candidate_k: int = 50         # candidates handed to the reranker
    final_k: int = 10             # evidence chunks handed to the LLM


@dataclass
class RerankerConfig:
    """Cross-encoder reranking of the fused candidate list.

    Attributes:
      enabled: Whether to rerank at all. Measured as slightly negative for
        single-pass hybrid retrieval and clearly positive inside the agent,
        which is why it stays on by default.
      model: Cross-encoder model id.
      batch_size: Query/document pairs scored per forward pass.
      max_length: Token cap for the concatenated pair.
      max_doc_chars: Document truncation applied before tokenising.
      device: "auto", "cuda" or "cpu".
    """

    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 64
    max_length: int = 512
    max_doc_chars: int = 3_000
    device: str = "auto"


@dataclass
class LLMConfig:
    """Generation backend used for planning, inspection and answering.

    Attributes:
      provider: "openai" (any OpenAI-compatible server, including local
        vLLM), "anthropic", or "echo" for a deterministic test stub.
      model: Model id as the provider names it.
      base_url: API base; the default points at a local vLLM server.
      api_key_env: Environment variable holding the key, if the provider
        needs one. A local server needs none.
      temperature: Sampling temperature; 0.0 for reproducible benchmarks.
      max_tokens: Cap on generated tokens per call.
      timeout: Per-request timeout in seconds.
      max_retries: Retries on transient failures.
      price_in_per_mtok: Input price per million tokens, for trace cost
        estimates only.
      price_out_per_mtok: Output price per million tokens, likewise.
      extra_body: Provider-specific request extras, e.g. Qwen3's
        `{"chat_template_kwargs": {"enable_thinking": false}}`, which
        suppresses the `<think>` preamble.
    """

    provider: str = "openai"      # openai|anthropic|echo
    model: str = "Qwen/Qwen3-4B"
    base_url: str = "http://127.0.0.1:8099/v1"
    api_key_env: str = "REPO_RAG_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout: float = 180.0
    max_retries: int = 3
    # rough $/1M tokens, used only for cost estimates in traces
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    # provider-specific request extras, e.g. Qwen3's {"chat_template_kwargs":
    # {"enable_thinking": false}} which disables the <think> preamble
    extra_body: dict = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """How retrieved evidence is packed into the context window.

    Attributes:
      context_token_budget: Total token budget for evidence.
      max_chunk_tokens: Cap per chunk; longer chunks are head/tail truncated.
      per_file_chunk_cap: Chunks admitted from any one file, so a single large
        module cannot crowd out the rest of the answer.
      artifact_quota: Fraction of the budget per artifact type. Enforced
        softly -- a quota only blocks a chunk once the context is above 80%
        full, so an under-filled context is never left half empty on
        principle.
      include_related_symbols: Append one graph-related symbol per evidence
        chunk when budget remains.
    """

    context_token_budget: int = 6_000
    max_chunk_tokens: int = 900       # a single chunk may not exceed this
    per_file_chunk_cap: int = 3       # avoid one file eating the whole context
    artifact_quota: dict[str, float] = field(
        default_factory=lambda: {"source": 0.6, "test": 0.15, "doc": 0.15, "commit": 0.1}
    )
    include_related_symbols: bool = True


@dataclass
class AgentConfig:
    """Iterative agentic retrieval.

    Attributes:
      enabled: Off by default; single-pass retrieval is the shipped path and
        the agent costs roughly 18x the retrieval latency.
      max_iterations: Hard cap on retrieve/inspect rounds. Iteration 1 is
        deterministic plan execution and makes no LLM call.
      per_step_k: Results requested per tool call.
      max_evidence: Evidence chunks carried into generation.
      planner_max_subqueries: Sub-queries the planner may emit.
    """

    enabled: bool = False
    max_iterations: int = 4
    per_step_k: int = 8
    max_evidence: int = 16
    planner_max_subqueries: int = 3


@dataclass
class Config:
    """The whole system's configuration, loaded from one YAML file.

    A run is reproducible from this object alone, which is why every knob
    that can move a benchmark number lives here rather than in a call site.

    Attributes:
      repository: Name recorded on every chunk and in the index metadata.
      repo_path: Checkout to ingest.
      index_dir: Where the built indexes are written.
      ingestion: Scanning and chunking settings.
      embedding: Dense embedding model settings.
      retrieval: Retriever depths, fusion and cut-offs.
      reranker: Cross-encoder settings.
      llm: Generation backend.
      judge_llm: Separate backend for answer grading; falls back to `llm`.
      generation: Context assembly settings.
      agent: Agentic retrieval settings.
      seed: Random seed recorded in every run fingerprint.
      trace_dir: Where per-query traces are written.
      root: Base for every relative path above. Defaults to the project
        checkout; tests point it at a tmp dir so a fixture run cannot
        overwrite real data.
    """

    repository: str = "vllm"
    repo_path: str = "data/vllm"
    index_dir: str = "indexes/vllm"
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    judge_llm: LLMConfig | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    seed: int = 0
    trace_dir: str = "traces"
    # base for every relative path below. Defaults to the project checkout;
    # tests point it at a tmp dir so a fixture run cannot touch real data.
    root: str = ""

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, **overrides: Any) -> "Config":
        """Load a config from YAML, then apply dotted overrides.

        Args:
          path: YAML file to read. `None` yields the dataclass defaults.
          **overrides: Dotted keys such as `retrieval.final_k=8`, typically
            forwarded from CLI flags. Values of `None` are ignored, so an
            unset flag does not clobber the file.

        Returns:
          The populated config.

        Raises:
          ValueError: An override names a field that does not exist, or the
            YAML contains an unknown key.
        """
        data: dict[str, Any] = {}
        if path is not None:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        cfg = _from_dict(cls, data)
        for key, value in overrides.items():
            if value is not None:
                _apply_override(cfg, key, value)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a nested, JSON-serialisable dict."""
        return asdict(self)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the config to `path` as YAML, creating parent directories."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def resolve(self, value: str) -> Path:
        """Resolve a configured path against `root`.

        Args:
          value: Absolute or relative path from the config.

        Returns:
          `value` unchanged if absolute, otherwise joined onto `root` (or the
          project checkout when `root` is unset).
        """
        p = Path(value)
        if p.is_absolute():
            return p
        return (Path(self.root) if self.root else REPO_ROOT) / p

    @property
    def index_path(self) -> Path:
        """Path: Resolved `index_dir`."""
        return self.resolve(self.index_dir)

    @property
    def repo_dir(self) -> Path:
        """Path: Resolved `repo_path`."""
        return self.resolve(self.repo_path)

    def fingerprint(self) -> dict[str, Any]:
        """Summarise everything a benchmark run must record to be reproducible.

        Returns:
          Models, chunking, retrieval, generation and agent settings plus the
          seed, in the shape stored under `fingerprint` in `results/*.json`.
        """
        return {
            "repository": self.repository,
            "embedding_model": self.embedding.model,
            "reranker": self.reranker.model if self.reranker.enabled else None,
            "llm": f"{self.llm.provider}:{self.llm.model}",
            "judge_llm": (
                f"{self.judge_llm.provider}:{self.judge_llm.model}" if self.judge_llm else None
            ),
            "chunking": asdict(self.ingestion),
            "retrieval": asdict(self.retrieval),
            "generation": asdict(self.generation),
            "agent": asdict(self.agent),
            "seed": self.seed,
        }


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a plain mapping.

    Nested config sections are matched by their annotation name, because
    `from __future__ import annotations` leaves field types as strings.

    Args:
      cls: Dataclass type to construct.
      data: Mapping of field name to value.

    Returns:
      An instance of `cls`.

    Raises:
      ValueError: `data` contains a key that `cls` has no field for. Failing
        loudly here is what stops a typo in a YAML file from silently
        reverting a tuned setting to its default.
    """
    kwargs: dict[str, Any] = {}
    type_hints = {f.name: f for f in fields(cls)}
    for name, value in (data or {}).items():
        if name not in type_hints:
            raise ValueError(f"unknown config key {name!r} for {cls.__name__}")
        f = type_hints[name]
        target = f.type
        if isinstance(target, str):  # `from __future__ import annotations`
            target = {
                "IngestionConfig": IngestionConfig,
                "EmbeddingConfig": EmbeddingConfig,
                "RetrievalConfig": RetrievalConfig,
                "RerankerConfig": RerankerConfig,
                "LLMConfig": LLMConfig,
                "LLMConfig | None": LLMConfig,
                "GenerationConfig": GenerationConfig,
                "AgentConfig": AgentConfig,
            }.get(target)
        if target is not None and is_dataclass(target) and isinstance(value, dict):
            kwargs[name] = _from_dict(target, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _apply_override(cfg: Any, dotted: str, value: Any) -> None:
    """Set a nested config field addressed by a dotted path.

    Example:
      `_apply_override(cfg, "retrieval.final_k", 8)`.

    Args:
      cfg: Config object to mutate in place.
      dotted: Dotted path to the field.
      value: New value, assigned without coercion.

    Raises:
      ValueError: The final path component is not a field of its parent.
    """
    parts = dotted.split(".")
    target = cfg
    for part in parts[:-1]:
        target = getattr(target, part)
    if not hasattr(target, parts[-1]):
        raise ValueError(f"unknown config override {dotted!r}")
    setattr(target, parts[-1], value)


def default_config() -> Config:
    """Load `configs/default.yaml`, falling back to dataclass defaults.

    Returns:
      The default config, so that a fresh checkout works with no arguments.
    """
    path = REPO_ROOT / "configs" / "default.yaml"
    return Config.load(path) if path.exists() else Config()
