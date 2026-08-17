# Five-minute demo script

Prerequisites: `rag ingest && rag index` have been run, the local LLM server is
up (`./scripts/serve_llm.sh`), and `./scripts/start_ui.sh` is serving
<http://127.0.0.1:8100>.

Each demo below works in the UI and on the CLI; the CLI form is given so the
demo is reproducible in a terminal recording. Every UI view has a deep link, so
each step can be a bookmark rather than a sequence of clicks.

---

## 0. Frame it (20 s)

> "60,272 chunks from vLLM — source, tests, docs and 3,000 commits — indexed four
> ways. Every answer is grounded in retrieved evidence with line-level
> citations, and every component in the pipeline had to earn its place on a
> benchmark."

Show the header line: repository, commit, chunk count, models.

---

## 1. Direct implementation question (60 s)

```bash
rag ask "Where is KV cache memory allocated and how is its capacity determined?" --no-reranker --trace
```

UI: `/?q=Where+is+KV+cache+memory+allocated+and+how+is+its+capacity+determined%3F&reranker=0`

(`--no-reranker` because the benchmark says so — see demo 5. Worth calling out
live: the flag exists because it was measured, not because it was assumed.)

What to point at:

* the answer names the concrete symbols (`determine_available_memory`,
  `CacheConfig.gpu_memory_utilization`, the KV cache spec/blocks path) and cites
  `path:start-end` for each claim;
* the evidence table: source files, a design doc and a test, each with the
  retriever that surfaced it;
* click a citation in the UI — it expands the exact retrieved source.

---

## 2. Multi-hop architecture (75 s)

```bash
rag ask "Trace a request from the API server until model execution on the GPU." --agentic --reranker --trace
```

What to point at:

* the **planner** step: category `architecture`, decomposed sub-queries;
* **agent iteration #2**: the controller's stated reason for one more retrieval
  and which tool it chose;
* the final evidence spans `entrypoints/openai`, `v1/engine`, `v1/executor`,
  `v1/worker` — several files, not one semantically similar file;
* the trace's stage latencies and token counts.

---

## 3. Debugging (45 s)

```bash
rag ask "A server dies with CUDA out of memory while allocating the KV cache. Which components decide that allocation and what knobs control it?"
```

Point at the mix of evidence types: the profiling code in the worker, the
configuration object, and the block-count computation — plus the explicit
"Not established from retrieved evidence" line when the answer runs out of
support.

---

## 4. Change impact (45 s)

```bash
rag impact allocate_slots
rag ask "If the KV cache block allocation logic changed, which components and tests would need to be re-checked?"
```

Point at the symbol graph: definitions → callers → tests, then the answer that
combines those structural relations with retrieved source.

---

## 5. Evaluation — the actual point (75 s)

Switch to the **Benchmark** tab (or `rag report`).

```bash
rag report
```

Then click one question in the dataset table — it runs the live pipeline and
scores it against its gold labels (`/?qid=curated-arch-001` links straight to
one). Pick a curated multi-hop question and show the misses: the aggregate is
0.73 recall@10, this question is 0.0, and the reason is visible in the
retrieved list rather than hidden in a mean.

Talk track:

* 158 questions, 8 categories, 49% multi-hop, gold labels correct by
  construction, 28 hand-written architecture/debugging questions;
* fusion weights tuned on a 45-question dev split, everything reported on the
  disjoint 113-question test split;
* **BM25 alone beats dense retrieval by a wide margin on this corpus** —
  identifiers dominate the questions engineers actually ask;
* hybrid and agentic retrieval add on top of that; the generic cross-encoder
  reranker did **not** pay for itself, and that is reported rather than hidden;
* every row is reproducible: commit, models, config and dataset hash are stored
  with the run.

Close on the project's question: *does each additional RAG technique measurably
improve retrieval?* — and note that the honest answer here is "three of five
did".
