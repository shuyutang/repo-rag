# Failure analysis

Numbers come from `results/*.json` on the 113-question test split (retrieval)
and a 60-question sample of it (generation). The point of this document is the
part the demo does not show: where the system loses, and which of the PRD's five
techniques actually paid for themselves.

---

## 1. The headline: the benchmark contains two very different populations

| system | auto-derived questions (n=94) | hand-written curated questions (n=19) |
| --- | ---: | ---: |
| dense | 0.555 | 0.263 |
| BM25 | 0.815 | 0.307 |
| hybrid | 0.795 | 0.307 |
| hybrid + reranker | 0.792 | **0.342** |
| agentic | **0.823** | 0.303 |

*(Recall@10.)*

Auto-derived questions are generated *from* an artifact, so even after
paraphrasing they inherit its vocabulary — which is exactly what BM25 is good
at. The 19 hand-written architecture/debugging questions are the honest measure
of the hard case, and **every configuration loses roughly half its recall on
them**.

Two consequences:

* A headline "recall@10 = 0.73" for this system is only meaningful next to the
  0.30 on hand-written multi-hop questions. Both are reported.
* Any future work on this project should be measured on the curated subset, not
  the aggregate, or it will optimise for a lexical artifact of the generator.

---

## 2. Complete misses (recall@10 = 0), hybrid + reranker

14 of 113 questions (12.4%) retrieved no gold evidence at all:

| category | misses |
| --- | ---: |
| debugging | 3 |
| architecture | 3 |
| configuration | 2 |
| historical | 2 |
| change_impact | 2 |
| code_lookup | 1 |
| documentation | 1 |

### Failure mode A — under-specified question, many valid targets (3 of 14)

> *"Where is `forward` implemented and what does it do?"* → gold
> `vllm/model_executor/models/phi4mm_utils.py`

vLLM defines `forward` hundreds of times. The retriever returned three perfectly
reasonable `forward` implementations from other model files. This is a **dataset
defect**, not a retrieval defect: the generator produced a question whose gold
key is arbitrary. Filtering candidate symbols by name frequency would remove
these; it is not done today because the filter would also silently remove
legitimately hard questions, and the effect is measurable and small.

### Failure mode B — the sibling-file trap (the largest group)

> *"Why does the code raise a ValueError indicating that the audio is too short
> …"* → gold `qwen2_5_omni_thinker.py`, retrieved `qwen2_audio.py`,
> `moss_audio.py`

In a repository with 4,000+ Python files, families of near-identical model
implementations exist. Both dense and lexical retrieval put the wrong family
member first, and the cross-encoder — which sees only the chunk text — has no
signal to separate them either. Fixing this needs a discriminating feature the
current system does not have (exact error-string matching, or file-level
disambiguation by the model in a second pass).

### Failure mode C — tests crowd out the implementation (3 of 14)

> *"What could cause the error message 'Invalid attention backend for XPU, with
> use_mla: True'?"* → gold `vllm/platforms/xpu.py`; the top 10 were dominated by
> `tests/v1/attention/test_*.py`

Test files quote error strings verbatim and repeat them across parametrisations,
so BM25 scores them above the single source line that raises the error. Tests
are 32% of the corpus and 26% of retrieved top-10 chunks overall.

**This one is actionable**: a per-category artifact prior (down-weight `test`
chunks for `debugging` and `code_lookup` questions, up-weight them for `tests`
and `change_impact`) is a small change to `RetrievalRequest`. It is listed as
future work rather than shipped because it must be validated on the dev split
first, and adopting it would invalidate every number in this report.

---

## 3. Partial recall: multi-hop is the real gap

35 of 113 questions retrieved *some* gold evidence but not all — dominated by
`historical` (12) and `change_impact` (11), the two categories whose gold set is
"the commit **and** the files it touched" or "the definition **and** its
callers".

Recall@10 by difficulty (hybrid + reranker): single-hop **0.881**, multi-hop
**0.535**. A single retrieval call ranked by similarity to one query cannot be
expected to cover 4–6 files across subsystems, which is precisely the argument
for the agent.

Agentic retrieval, question by question, fixed **4** complete misses and
introduced **3** new ones (evidence gathered over several steps can push a gold
chunk out of the final top-k). Net +0.022 recall@10 — real, but small enough
that it should be reported as "modest gain at 18× latency", not as a headline.

---

## 4. Did each technique earn its place?

| technique | verdict | evidence |
| --- | --- | --- |
| structure-aware chunking | **yes** (by construction) | citations resolve to real symbols and line ranges; symbol/graph retrieval and change-impact are impossible without it |
| BM25 with a code-aware tokenizer | **yes, decisively** | 0.729 R@10 at 0.4 ms — the strongest single component |
| dense retrieval | **only as a component** | 0.506 R@10 alone; contributes ranking quality to the fusion, not recall |
| symbol retrieval | **yes, but narrowly** | 0.209 R@10 alone (it is silent when a question names no identifier); it is what fixes identifier questions inside the fusion, and what the agent calls directly |
| RRF fusion | **marginal** | +0.008 MRR over BM25, −0.016 R@10; it needed weight tuning on dev just to reach parity |
| cross-encoder reranking, single fused list | **no** (as configured) | −0.024 R@5, −0.017 nDCG@10, +0.010 MRR, 2.4× latency; ms-marco MiniLM is out of domain for code |
| cross-encoder reranking, agent evidence | **yes** | the agent's four tools produce scores on incomparable scales; dropping the reranker costs 0.058 R@5 / 0.043 MRR (0.680→0.622, 0.736→0.693) even after the agent's own rank fusion |
| agentic retrieval | **yes, for multi-hop** | best on every retrieval metric (+0.022 R@10, +0.046 R@5 over hybrid+reranker) at 2.2 s/query |
| a stronger embedding model | **no** (on this benchmark) | arctic-embed-m-v1.5 improves the *dense leg* on dev (R@10 0.548 → 0.591, MRR 0.528 → 0.630) but the tuned hybrid does not move (0.704 → 0.698); the bottleneck is not embedding quality, it is that lexical evidence dominates these questions. Caveat: fusion weights were tuned for MiniLM, so this comparison slightly favours it. |

Three of the five systems in the PRD's ablation list paid for themselves, and
the fourth (reranking) turned out to be conditional rather than simply "no" —
which only became visible because the agent was ablated with *and* without it.
The project's guiding question (§34) is answered honestly, including where the
answer is "no".

One measurement bug worth recording: the first agentic-without-reranker run
scored 0.625 R@10, far below everything else. The cause was not the retrieval —
it was that the agent's fallback ranking sorted evidence by raw score across
tools whose scores live on different scales (RRF ≈0.03, symbol ≈1.0, graph
≈0.7). The agent now rank-fuses its own tool outputs, which moved that
configuration to 0.735. A number that looks anomalous is usually a bug in the
harness before it is a fact about the world.

---

## 5. Generation

Measured on 60 sampled test questions per system, local Qwen3-4B as both
answerer and judge.

| system | answer accuracy | faithfulness | unsupported claims | citation validity | citation completeness |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.867 | 0.900 | 0.202 | 0.898 | 0.386 |
| bm25 | 0.942 | 0.950 | 0.191 | 0.925 | 0.525 |
| hybrid | 0.942 | 0.958 | 0.146 | 0.833 | 0.454 |
| hybrid + reranker | 0.925 | 0.942 | 0.153 | 0.894 | 0.519 |
| agentic | 0.942 | 0.975 | 0.128 | 0.950 | 0.572 |

Answer accuracy saturates: a 3-point judge scale and questions that are often
answerable from partial evidence leave little room between systems. The
discriminating metrics are citation completeness (0.386 → 0.572 from worst to
best retrieval) and the unsupported-claim rate (0.202 → 0.128) — both track
retrieval quality, which is the causal chain the system is built on.

The recurring failure modes in the answers are:

* **Missing citations rather than wrong ones.** Citation *validity* is high
  (~0.9 — the model rarely invents a path, and when it does the validator
  strips it), but citation *completeness* against gold files is 0.39–0.57
  depending on the retriever: a 4B model often explains the mechanism correctly
  and cites one location instead of the three that support it. Answers with zero citations occur and are visible
  in the per-question records.
* **Unsupported-claim rate ~0.20** as judged: the model fills gaps from general
  knowledge about inference servers. The system prompt's `Inference:` /
  "Not established from retrieved evidence" convention is followed
  inconsistently at this model size — a larger generation model is the obvious
  lever, and the provider abstraction makes swapping one in a config change.
* **The judge is the same model family as the answerer**, which is a known
  reliability risk; `rag audit` exists to sample answers for human review and
  report judge/human agreement. The audit sheet is generated but not
  human-labelled in this run — the agreement numbers are therefore *not*
  reported, rather than reported unvalidated.

---

## 6. What I would do next, in priority order

1. **Category-conditioned artifact priors** (failure mode C) — cheapest
   expected win, validate on dev.
2. **Query decomposition for architecture questions** — the agent currently
   issues 2–4 sub-queries from one plan; multi-hop recall suggests it should
   iterate over *components* named in its own intermediate evidence.
3. **A code-domain reranker** — the reranking result is a statement about
   ms-marco MiniLM, not about reranking. A code-trained cross-encoder is the
   obvious retry.
4. **Human-labelled judge audit** (20–30 answers) to put an error bar on the
   generation numbers.
5. **Grow the curated set** from 28 to ~60 questions: it is the only part of the
   benchmark that measures the hard case, and 19 test-split questions is a thin
   basis for the conclusions in §1.
