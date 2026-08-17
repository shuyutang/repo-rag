# PRD coverage

Every requirement in `prd.md`, and where it is implemented or measured.

| PRD | requirement | status | where |
| --- | --- | --- | --- |
| §7 FR1 | ingest a git repository: source, tests, docs, metadata, commits, provenance per artifact | done | `eka/ingestion/scanner.py`, `eka/schema.py` (`Chunk`) |
| §7 | optional issues/PR ingestion | not done | explicitly optional; commit history covers the "why" questions in the benchmark |
| §8 | structure-aware parsing: classes, functions, methods, modules, tests; symbol name, FQN, file, line range, type, source, parent, imports, references | done | `eka/ingestion/code_parser.py` |
| §8 | Python in V1 | done | AST-based |
| §8 | C++ stretch goal | not done | unparsed languages fall back to windowed chunks; a tree-sitter parser would slot in behind the same `parse()` interface |
| §9 | document chunking with document/section/heading hierarchy/line range/version | done | `eka/ingestion/document_parser.py` (markdown + rst) |
| §9 | commits with SHA, author, timestamp, message, modified files, diff | done | `eka/ingestion/git_parser.py` |
| §10 | embedding index with top-k, metadata/artifact/repository filtering | done | `eka/indexing/vector_index.py`, `eka/retrieval/base.py::mask_for` |
| §10 | simplicity and reproducibility over distributed scale | done | exact flat index; FAISS used only if installed |
| §11 | BM25 or equivalent lexical retrieval | done | `eka/indexing/bm25_index.py` + code-aware tokenizer |
| §12 | hybrid retrieval with RRF, scores and provenance exposed | done | `eka/retrieval/fusion.py`, `component_scores` on every result |
| §13 | exact and fuzzy symbol lookup, independently callable | done | `eka/indexing/symbol_index.py`, `eka/retrieval/symbol.py`, agent tool `symbol_search` |
| §14 | rerank top candidates, configurable k | done | `eka/retrieval/reranker.py`, `retrieval.candidate_k` / `final_k` |
| §15 | context builder: dedup, provenance, ranking, related symbols, per-type allocation, no single-file domination | done | `eka/generation/context_builder.py` |
| §16 | grounded generation with citations; fact / inference / unknown separation | done | `eka/generation/answer_generator.py` (system prompt + citation validation) |
| §17 | agentic iterative retrieval with a configurable iteration cap | done | `eka/agent/retrieval_agent.py`, `agent.max_iterations` |
| §18 | query planning/classification into categories, tool choice per type | done | `eka/agent/planner.py` (rules + LLM), `CATEGORY_TOOLS` |
| §19 | `Retriever` protocol with Dense/BM25/Symbol/Hybrid/Git implementations | done | `eka/retrieval/` |
| §20 | 100–200 question benchmark with the specified record schema; ≥25% multi-hop | done | 158 questions, 48.7% multi-hop; `eka/evaluation/dataset.py`, `dataset_builder.py`, `curated.py` |
| §21 | Recall@5, Recall@10, MRR, nDCG@10, globally and by category | done | `eka/evaluation/retrieval_metrics.py`; per-category and per-difficulty in every report |
| §22 | correctness, faithfulness, citation correctness/completeness, unsupported-claim rate; judge validated on a manual subset | done (audit sheet generated, not human-labelled) | `eka/evaluation/answer_metrics.py`, `rag audit` |
| §23 | ablation over dense / BM25 / hybrid / hybrid+rerank / agentic, no fabricated numbers | done | [results_table.md](results_table.md), `results/*.json` |
| §24 | per-query trace with latency, results, scores, reranker scores, tokens, LLM calls, cost, citations; inspectable in the UI | done | `eka/observability/tracing.py`, UI pipeline view |
| §25 | UI emphasising the pipeline: answer, trace, expandable evidence | done | `ui/index.html` — Ask / Search / Symbols / Benchmark / Traces, deep-linkable, plus live per-question evaluation against gold labels |
| §26 | CLI: ingest, index, ask, benchmark variants | done | `eka/cli.py` (plus `search`, `diagnose`, `impact`, `dataset`, `report`, `audit`, `serve`) |
| §27–28 | architecture and module layout | done | [architecture.md](architecture.md), module map in the README |
| §29 | reproducibility (config in version control, run fingerprint), testability (unit + integration + fixture repo), modularity, configuration | done | `eka/config.py`, `configs/*.yaml`, `BenchmarkReport.fingerprint`, `tests/` (60 tests on a fixture repo) |
| §30 | retrieval < 1 s, reranking < 2 s, total < 10 s, per-stage measurement | met | BM25 0.4 ms, hybrid 52 ms, +rerank 122 ms, answer ~5 s; per-stage latencies in every trace and report |
| §31 M1–M10 | milestones | all delivered | see the README quick start and [evaluation.md](evaluation.md) |
| §32 | five-minute demo | done | [demo.md](demo.md) |
| §33 | success criteria | all met | large repo indexed, structural chunking, dense/BM25/symbol, hybrid + rerank, inspectable citations, multi-hop agent, 158 questions, reproducible metrics, grounding evaluated, ablations, traces, end-to-end demo |
| §34 | each technique must earn its place | answered, including negative results | [failure_analysis.md](failure_analysis.md) §4 |

## Deliberate deviations

* **Issues/PR ingestion** (optional in §7) is not implemented — it needs a GitHub
  token and network access at index time, and commit history already supports
  the historical-reasoning questions in the benchmark.
* **C++ parsing** (stretch goal in §8) is not implemented.
* **The judge audit sheet is generated but not human-labelled**, so judge/human
  agreement is not reported. Reporting an unvalidated agreement number would be
  worse than reporting none.
* **The default configuration keeps reranking enabled** even though it is
  slightly negative for single-pass hybrid retrieval, because it is clearly
  positive for the agentic configuration; both are measured and either can be
  selected with one flag.
