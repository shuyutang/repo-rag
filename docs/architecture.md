# Architecture

This document explains *why* the system is put together the way it is. The
short version: a repository is not a pile of prose, so neither the chunking, the
retrieval, nor the context construction can be the generic "chat with your docs"
kind.

---

## 1. Ingestion — structure before size

`repo_rag/ingestion/code_parser.py`

Fixed-size chunking destroys the thing that makes code searchable: a symbol has
a name, a parent, a signature, a docstring, and a line range you can cite. The
Python parser walks the AST and emits:

* one **module** chunk per file — docstring, imports, and the public API
  (classes with their method names, top-level functions, constants). This is the
  chunk that answers "what is in this file".
* one **class overview** chunk — the class header, docstring, class-level
  attribute assignments and method *signatures*. Method bodies are not repeated
  here, so a class and its methods do not duplicate each other in the index.
* one chunk per **function/method**, with the full body, the decorators, the
  called names (`references`) and the file's imports.

Symbols longer than `max_chunk_lines` are split into overlapping line windows so
line numbers stay exact — a citation must point at real lines.

Files that fail to parse fall back to fixed windows rather than disappearing.

Documentation (`document_parser.py`) is split on the heading hierarchy, keeping
the full heading path (`Design > KV cache > Block allocation`), then on
paragraph boundaries if a section is too large.

Git history (`git_parser.py`) streams `git log --raw --patch` **once** for the
whole history and turns each commit into a chunk holding subject, body, touched
files and a truncated diff. One subprocess for 3,000 commits, ~2 s.

Every chunk keeps its provenance: repository, commit, path, symbol, qualified
name, symbol type, parent, line range.

---

## 2. Indexing — four views of the same chunks

`repo_rag/indexing/`

| index | answers | implementation |
| --- | --- | --- |
| vector | "something about paged memory reuse" | MiniLM-L6 embeddings, exact inner-product search over a normalised matrix (FAISS used automatically if installed) |
| BM25 | `allocate_gpu_cache`, `--gpu-memory-utilization`, an exact error string | scipy CSC term-document matrix, Okapi BM25, code-aware tokenizer |
| symbol | `CacheEngine.allocate_gpu_cache` | exact + variant keys (`camelCase`/`snake_case`), difflib fuzzy fallback |
| graph | "what calls this, what tests it" | name-resolution over the AST facts already extracted |

**The tokenizer is the interesting part.** `CacheEngine.allocate_gpu_cache` is
emitted whole *and* as `cache`, `engine`, `allocate`, `gpu`, so that a lexical
index can match both an exact identifier and an English paraphrase of it.

The dense index is exact (flat), not approximate: at 60k chunks × 384 dims a
matmul takes ~5 ms, and an exact index has no ANN hyper-parameters to drift
between runs — reproducibility beats scalability at this size (PRD §10).

The **symbol graph** is explicitly not a call graph (PRD §4 non-goal). It is a
recall-oriented heuristic: a call to `allocate` links to every `allocate`
definition. Precision is recovered downstream by reranking and by the answer
model, which sees the actual source. Noise names (`get`, `append`, …) are
excluded.

---

## 3. Retrieval — strategies that fail differently

`repo_rag/retrieval/`

Every retriever implements the same protocol (`retrieve(query, k) ->
list[RetrievedChunk]`), which is what makes the ablation possible: the benchmark
swaps the source list and changes nothing else.

They fail in different directions, which is the whole argument for fusion:

* dense retrieval finds paraphrases but is weak on identifiers and confidently
  returns plausible-but-wrong neighbours;
* BM25 nails identifiers and error strings but cannot bridge vocabulary gaps;
* symbol lookup is precise when the question names a symbol and silent
  otherwise;
* the git retriever is the only one that can answer "why".

Fusion is **Reciprocal Rank Fusion** — rank-based, so a cosine similarity and a
BM25 score never have to be normalised onto the same scale. Weights and the rank
constant are configuration, and were tuned on the dev split
(`scripts/tune_fusion.py`); see [evaluation.md](evaluation.md).

Reranking is a cross-encoder over the top `candidate_k` candidates. Retrieval
optimises recall over a large candidate set; the reranker optimises precision
over the few chunks that actually reach the LLM.

---

## 4. Agentic retrieval — bounded, inspectable

`repo_rag/agent/`

```
plan  ─> retrieve (multiple sub-queries + symbol/impact/history tools)
      ─> inspect evidence
      ─> "is a specific artifact still missing?" ── yes ─> one more tool call
                                                └─ no ─> rerank everything gathered
```

Three properties matter more than cleverness:

1. **Bounded.** `agent.max_iterations` caps the loop; a repeated query, an empty
   round, or an evidence cap also stop it.
2. **Degradable.** Without an LLM the planner falls back to a rule-based
   classifier, and the loop runs exactly one round. Nothing crashes, and the
   tests run offline.
3. **Traceable.** Every tool call, its arguments, its result count and the
   controller's stated reason go into the trace.

The planner classifies the question (code lookup / architecture / debugging /
change impact / historical / documentation / configuration / tests) and picks
tools accordingly: history questions get the git retriever, change-impact
questions get the symbol graph.

---

## 5. Context construction — the budget is the design

`repo_rag/generation/context_builder.py`

Retrieved evidence is not a context window. The builder:

* drops chunks whose line ranges **overlap** an already-kept chunk;
* caps chunks **per file** so one large file cannot eat the window;
* allocates the budget across **artifact types** (source/test/doc/commit) so the
  "why" evidence is not crowded out by the "what";
* truncates a single oversized chunk head+tail with an explicit elision marker;
* optionally pulls in the **definition of one symbol** the top evidence calls —
  the cheapest possible use of the graph;
* records everything it dropped, and why, into the answer's usage payload.

Each block is rendered with its own citation instruction
(`cite as [vllm/config/cache.py:41-88]`), which is what makes citation
validation possible afterwards.

---

## 6. Grounded generation

`repo_rag/generation/answer_generator.py`

The system prompt requires three epistemic levels to be kept apart: facts
supported by evidence, explicit `Inference:` for reasoning beyond it, and a
final "Not established from retrieved evidence" line for what is missing.

Answers are parsed for `[path:start-end]` citations and each one is checked
against the context blocks — exact location, or a line range inside a retrieved
block. Unsupported citations are stripped from `answer.citations`, kept in
`usage.unsupported_citations`, shown in red in the UI, and counted as
*citation validity* in the benchmark. A model that invents a path is caught by
construction, not by a judge's opinion.

---

## 7. Observability

`repo_rag/observability/tracing.py`

Every query produces a trace: one step per stage (each retriever, fusion,
rerank, each agent iteration, generation) with duration, the top results with
scores, token counts and estimated cost. Traces are JSON on disk; the CLI
renders them as a tree (`rag ask --trace`), the UI renders them as a pipeline
with expandable evidence, and the benchmark reads stage durations from the same
object it uses for metrics.

---

## 8. Configuration and reproducibility

`repo_rag/config.py`, `configs/default.yaml`

One typed config tree covers chunking, embeddings, retrieval, reranking, the
agent, generation and the LLM. `Config.fingerprint()` is embedded in every
benchmark result together with the repository commit, chunk count and a hash of
the dataset file, so a number in the ablation table can always be traced to the
configuration that produced it.

---

## 9. Module map

PRD §28 asks for a specific layout; this is where each part lives.

| PRD | here |
| --- | --- |
| `ingestion/` | [repo_rag/ingestion](../repo_rag/ingestion) — `code_parser.py`, `document_parser.py`, `git_parser.py`, `scanner.py` |
| `indexing/` | [repo_rag/indexing](../repo_rag/indexing) — `embeddings.py`, `vector_index.py`, `bm25_index.py`, `symbol_index.py`, `graph_index.py`, `knowledge_base.py` |
| `retrieval/` | [repo_rag/retrieval](../repo_rag/retrieval) — `base.py`, `dense.py`, `sparse.py`, `symbol.py`, `git.py`, `fusion.py`, `reranker.py`, `hybrid.py` |
| `agent/` | [repo_rag/agent](../repo_rag/agent) — `planner.py`, `tools.py`, `retrieval_agent.py` |
| `generation/` | [repo_rag/generation](../repo_rag/generation) — `context_builder.py`, `answer_generator.py`, `llm.py` |
| `evaluation/` | [repo_rag/evaluation](../repo_rag/evaluation) — `dataset.py`, `dataset_builder.py`, `curated.py`, `retrieval_metrics.py`, `answer_metrics.py`, `benchmark.py` |
| `observability/` | [repo_rag/observability/tracing.py](../repo_rag/observability/tracing.py) |
| `api/`, `ui/` | [repo_rag/api/server.py](../repo_rag/api/server.py), [ui/index.html](../ui/index.html) |

`repo_rag/pipeline.py` ties them together, and is the single object the CLI, the API
and the benchmark all share — so a number reported by the benchmark comes from
exactly the code path the demo runs.
