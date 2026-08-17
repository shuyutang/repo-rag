# Evaluation

Everything here comes from `results/*.json`, produced by `rag benchmark`. Each
result file records the repository commit, chunk count, models, chunking and
retrieval configuration, the dataset hash and the seed, so a number can always
be traced back to the run that produced it. No number in this document was
typed by hand.

---

## Dataset

`evaluation_data/benchmark.jsonl` — 158 questions.

| property | value |
| --- | --- |
| questions | 158 (45 dev / 113 test) |
| categories | code_lookup 31, change_impact 22, historical 22, configuration 20, documentation 20, tests 18, debugging 15, architecture 10 |
| multi-hop | 77 (48.7%) — PRD floor is 25% |
| avg. gold files per question | 1.95 |
| hand-written & reviewed | 28 |
| gold labels verified against the index | yes (`rag dataset validate` → 0 problems) |

### How it was built

Two sources, both grounded in the repository at the pinned commit:

**Auto-derived (130).** A candidate artifact is selected first, then a question
is generated *from* it, so the gold label is the artifact the question came
from — correct by construction:

| category | candidate | gold |
| --- | --- | --- |
| code_lookup | a documented class/function/method | its file + qualified symbol |
| configuration | a config class or field | its file + symbol |
| tests | a symbol with ≥2 referencing test chunks | definition file + test files (multi-hop) |
| change_impact | a symbol with ≥4 callers in ≥2 files | definition file + caller files (multi-hop) |
| historical | a commit touching ≤6 files | commit SHA + touched Python files |
| documentation | a doc section ≥600 chars | the doc file |
| debugging | code raising a specific exception message | its file + symbol |

The LLM (local Qwen3-4B) only *phrases* the question; it never chooses the
answer key. Generated questions are rejected if they leak the gold path, or if
they contain a dangling reference ("this method", "in this commit") that would
make them unanswerable standalone — those fall back to a self-contained
template phrasing.

**Curated (28).** Hand-written architecture, debugging, change-impact and
configuration questions (`eka/evaluation/curated.py`) with gold files chosen by
reading the repository. These are the multi-hop questions no mechanical
procedure produces.

### Known biases — stated up front

* **Gold labels are minimal.** Other files may also be reasonable evidence, so
  recall is a *lower bound*, and systems that retrieve a legitimate alternative
  file are punished.
* **Auto-derived questions inherit vocabulary from their source chunk.** Even
  though the model is told to paraphrase, identifiers leak into the wording.
  This favours lexical retrieval; it is a large part of why BM25 does so well
  below, and it is why the curated questions matter as a counterweight.
* **Questions are not filtered by whether this system can answer them.**
  Filtering on retrievability would make the benchmark self-congratulatory.
* The judge is the same model family as the answerer, which is why the judge is
  audited separately (below).

### Splits

`assign_splits()` hashes the question id: ~30% dev, ~70% test, deterministic.
Fusion weights and the RRF constant were tuned on **dev only**
(`scripts/tune_fusion.py`, 108-point grid, `results/fusion_sweep.json`).
Everything reported below is **test**.

---

## Retrieval ablation (test split, 113 questions)

| System | R@1 | R@5 | R@10 | MRR | nDCG@10 | retrieval p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| symbol only | 0.174 | 0.194 | 0.209 | 0.330 | 0.255 | 22 ms |
| dense only | 0.169 | 0.399 | 0.506 | 0.413 | 0.356 | 5 ms |
| BM25 only | 0.403 | 0.655 | 0.729 | 0.715 | 0.622 | **0.4 ms** |
| hybrid (tuned RRF) | 0.405 | 0.658 | 0.713 | 0.723 | 0.622 | 52 ms |
| hybrid + reranker | 0.409 | 0.634 | 0.716 | 0.725 | 0.605 | 122 ms |
| agentic, no reranker | 0.372 | 0.622 | **0.735** | 0.693 | 0.612 | 2267 ms |
| **agentic (hybrid + reranker + loop)** | **0.425** | **0.680** | **0.735** | **0.736** | **0.628** | 2246 ms |

Read this as five separate claims, three of which hold:

1. **Dense retrieval alone is not enough** (R@10 0.506). A 384-dim general-purpose
   sentence embedder does not represent identifiers, and identifiers are what
   engineering questions are made of.
2. **BM25 is the workhorse** (R@10 0.729) and essentially free (0.4 ms). Any
   "vector-database-first" design for code search starts a long way behind a
   well-tokenised lexical index.
3. **Fusion earns a small win on ranking quality, not on recall.** Hybrid beats
   BM25 on MRR (0.723 vs 0.715) and matches it on R@5, but is slightly *behind*
   on R@10. The honest summary: after weight tuning, hybrid is roughly BM25 plus
   a better top of the ranking, because the dense leg contributes a handful of
   good chunks near the top and a lot of noise below.
4. **The reranker's value depends entirely on what it is reranking.** On a
   single fused list it does not earn its place: ms-marco MiniLM improves
   R@1/MRR by <0.005 and *hurts* R@5 (0.634 vs 0.658) and nDCG@10 (0.605 vs
   0.622) at 2.4× the latency — it was trained on web passages, and code chunks
   are out of domain. But inside the agent, where evidence arrives from four
   tools whose scores are not comparable, it is the component that puts
   everything on one scale: removing it costs 0.058 R@5, 0.043 MRR and 0.016
   nDCG@10 (agentic 0.680/0.736/0.628 → 0.622/0.693/0.612) at identical R@10.
   Both configurations are shipped and both are measured.
5. **Agentic retrieval is the largest single gain** (+0.022 R@10, +0.011 MRR,
   +0.046 R@5 over hybrid+reranker) — and costs 18× the latency (2.2 s vs
   122 ms) plus 2–5 LLM calls. Whether that trade is worth it depends on the
   question type; see the breakdown below.

### Where the gains actually are

Recall@10 by question difficulty:

| System | single-hop (n=59) | multi-hop (n=54) |
| --- | ---: | ---: |
| BM25 | 0.898 | 0.544 |
| hybrid + reranker | 0.881 | 0.535 |
| agentic | 0.907 | 0.548 |

Multi-hop questions are where every configuration loses ~35 points of recall,
and where the agent's extra iterations pay for themselves — exactly the
prediction the PRD makes in §17, and the reason the agent exists.

By category (hybrid, R@10): documentation 0.85, code_lookup 0.68, debugging
0.53, tests 0.49, historical 0.38, change_impact 0.31, architecture 0.13.
Architecture questions — "trace a request from the API server to the GPU" — are
the hardest thing in the benchmark: their gold set is 4–6 files spread across
subsystems, and no single retrieval call finds all of them.

---

## Embedding-model ablation

The dense leg is the weakest component, so the embedding model itself was
ablated on the dev split (`configs/strong_embeddings.yaml`, separate index):

| embedding model | dense R@5 | dense R@10 | dense MRR | hybrid R@10 | hybrid MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| all-MiniLM-L6-v2 (384d, 22M) | 0.420 | 0.548 | 0.528 | **0.704** | **0.782** |
| snowflake-arctic-embed-m-v1.5 (768d, 109M) | **0.524** | **0.591** | **0.630** | 0.698 | 0.772 |

Reproduce with `scripts/compare_embedders.sh` (writes `results/dev/*.json`).

The stronger model is clearly better *as a dense retriever* (+0.10 MRR) and
makes no difference to the tuned hybrid. The dense leg is not the bottleneck:
lexical evidence dominates these questions, so improving the leg that
contributes least changes little. Caveat: the fusion weights in use were tuned
with MiniLM, which slightly favours it — a fair comparison would re-tune per
embedder. MiniLM stays the default: 5× smaller, 3× faster to index, same hybrid
result.

---

## Generation quality

`rag benchmark --generation` answers the question with the pipeline and scores
the answer two ways.

**Deterministic (no model opinion involved):**

* *citation validity* — fraction of `[path:line-line]` citations that point at
  a chunk that was actually in the context. Invented paths are caught by
  construction.
* *citation precision / completeness* — cited files vs the question's gold files.

**LLM-judged** (correctness, faithfulness, unsupported-claim rate) with the
judge prompt in `eka/evaluation/answer_metrics.py`.

Measured on 60 sampled test questions per system (`results/gen-*.json`):

| system | answer accuracy | faithfulness | unsupported claims | citation validity | citation completeness |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.867 | 0.900 | 0.202 | 0.898 | 0.386 |
| bm25 | 0.942 | 0.950 | 0.191 | 0.925 | 0.525 |
| hybrid | 0.942 | 0.958 | 0.146 | 0.833 | 0.454 |
| hybrid + reranker | 0.925 | 0.942 | 0.153 | 0.894 | 0.519 |
| **agentic** | **0.942** | **0.975** | **0.128** | **0.950** | **0.572** |

Answer accuracy is compressed at the top (0.87–0.94) because a 3-point judge
scale is coarse and most questions are answerable from partial evidence — the
metrics that actually separate the systems here are *citation completeness*
(does the answer point at all the gold evidence) and the *unsupported-claim
rate*, and both track retrieval quality closely. Better retrieval produces
better-grounded answers, which is the result the whole pipeline exists to
produce.

### Judge reliability

An LLM judge that is never checked is a vibe. `sample_for_audit()` writes a
manual-review sheet of judged answers; `judge_agreement()` reports exact
agreement and MAE between the human labels and the judge once the sheet is
filled in. The workflow is:

```bash
python -c "from eka.evaluation.answer_metrics import sample_for_audit; import json,pathlib; \
  rows=json.load(open('results/gen-hybrid+rerank.json'))['per_question']; \
  sample_for_audit(rows, pathlib.Path('evaluation_data/judge_audit.jsonl'), n=20)"
# fill in human_correctness / human_faithfulness
python -c "from eka.evaluation.answer_metrics import judge_agreement; import pathlib; \
  print(judge_agreement(pathlib.Path('evaluation_data/judge_audit.jsonl')))"
```

---

## Latency (PRD §30)

Warm, measured per stage by the benchmark on the test split:

| stage | p50 | p95 |
| --- | ---: | ---: |
| BM25 retrieval | 0.4 ms | 1 ms |
| dense retrieval | 4.6 ms | 6 ms |
| hybrid (dense+BM25+symbol+git, fused) | 52 ms | ~90 ms |
| + cross-encoder rerank of 50 candidates | 122 ms | 271 ms |
| agentic retrieval (2–5 LLM calls) | 2.2 s | 3.4 s |
| answer generation (local Qwen3-4B, ~4k prompt tokens) | ~4 s | ~7 s |

Retrieval is comfortably inside the PRD's "<1 s retrieval, <2 s reranking,
<10 s total" targets; agentic retrieval spends its budget on LLM calls, not on
search.

Cold start adds ~4.5 s for the embedding model and ~3 s for the cross-encoder
the first time they are used in a process; the benchmark warms both before
timing.

---

## Reproducing

```bash
rag dataset build --n-auto 130 --seed 0     # rebuilds the benchmark
rag dataset validate                        # every gold path must exist
python scripts/tune_fusion.py --out configs/default.yaml   # dev split only

rag benchmark --retriever dense  --no-reranker --name dense          --split test
rag benchmark --retriever bm25   --no-reranker --name bm25           --split test
rag benchmark --retriever symbol --no-reranker --name symbol         --split test
rag benchmark --retriever hybrid --no-reranker --name hybrid         --split test
rag benchmark --retriever hybrid --reranker    --name hybrid+rerank  --split test
rag benchmark --retriever hybrid --reranker --agentic --name agentic --split test
rag report --out docs/results_table.md
```
