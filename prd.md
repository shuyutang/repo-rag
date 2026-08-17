
# Product Requirements Document

## Engineering Knowledge Agent

**Version:** 1.0
**Status:** Proposed
**Primary demo repository:** vLLM
**Product type:** Retrieval-Augmented Generation / Code Intelligence System

---

## 1. Product Summary

Engineering Knowledge Agent is a production-oriented RAG system for answering complex engineering questions about large software repositories.

Unlike conventional "chat with your docs" systems, the agent builds a unified knowledge base across:

* Source code
* Tests
* Documentation
* Architecture/design documents
* Git history
* Issues and pull requests
* Benchmark and performance reports

The system uses structure-aware code indexing, hybrid retrieval, reranking, iterative/agentic retrieval, and evidence-grounded generation to answer questions with precise file-, symbol-, and line-level citations.

The initial demonstration will use the vLLM repository.

The project is intended both as a useful engineering tool and as a demonstration of production-quality RAG architecture, retrieval evaluation, agentic reasoning, and ML systems engineering.

---

# 2. Problem

Engineers working with large repositories frequently need to answer questions such as:

> Where is KV-cache memory allocated?

> How does a request move from the API server to GPU execution?

> Why was this implementation changed?

> Which components depend on this class?

> What tests could be affected if I modify this function?

> Could a recent commit explain this memory regression?

The information necessary to answer these questions is fragmented across source code, documentation, tests, commits, issues, and design discussions.

Traditional repository search works well when the engineer already knows what symbol or string to search for.

Generic vector RAG has the opposite problem: it can retrieve semantically similar text but frequently fails to understand code structure, exact identifiers, dependencies, or relationships between multiple files.

The proposed system combines lexical, semantic, structural, and iterative retrieval.

---

# 3. Goals

The system should demonstrate five core capabilities.

### G1 — Repository understanding

Answer natural-language questions about a large unfamiliar repository.

### G2 — High-quality retrieval

Retrieve the correct source files, symbols, documentation, tests, and historical evidence using multiple retrieval strategies.

### G3 — Multi-hop engineering reasoning

Answer questions requiring evidence distributed across several repository artifacts.

### G4 — Grounded answers

Every substantive technical claim should be traceable to retrieved evidence.

### G5 — Measurable RAG quality

Retrieval and answer quality must be evaluated quantitatively rather than demonstrated only through cherry-picked examples.

---

# 4. Non-Goals

Version 1 will not attempt to:

* Replace a full IDE or language server.
* Automatically modify production code.
* Execute arbitrary generated code.
* Build a complete compiler-level call graph.
* Provide autonomous pull-request generation.
* Index arbitrary private enterprise systems.
* Train a new foundation model.
* Fine-tune the generation LLM.

The primary objective is retrieval, reasoning, grounding, and evaluation.

---

# 5. Target Users

### Primary persona — Software engineer

An engineer joining or debugging a large codebase who needs to quickly understand implementation details.

### Secondary persona — ML infrastructure engineer

An engineer investigating model-serving architecture, GPU behavior, performance regressions, or inference pipelines.

### Secondary persona — Technical lead

A lead assessing architectural dependencies and the impact of proposed changes.

---

# 6. Primary User Stories

## US1 — Code Understanding

As an engineer, I want to ask:

> Where is KV-cache memory allocated and how is its size determined?

The system should identify relevant symbols, configuration objects, and allocation code and synthesize an explanation.

---

## US2 — Architecture Understanding

As an engineer, I want to ask:

> Explain how a request travels from the API server to GPU model execution.

The system should discover and connect multiple components rather than returning one semantically similar file.

---

## US3 — Debugging

Given an error such as:

```text
CUDA out of memory during decode
```

I want to ask:

> What components in this repository could cause this?

The system should retrieve implementation, configuration, and relevant diagnostic information.

---

## US4 — Change Impact

As an engineer, I want to ask:

> If I change PagedAttention, what components and tests are potentially affected?

The system should use symbol relationships in addition to semantic similarity.

---

## US5 — Historical Reasoning

As an engineer, I want to ask:

> Why was this batching behavior introduced?

The system should retrieve relevant code, commits, issues, PRs, and documentation and reconstruct the rationale when sufficient evidence exists.

---

# 7. Functional Requirements

## FR1 — Repository Ingestion

The system must ingest a Git repository.

It should extract:

* Source files
* Tests
* Markdown/documentation
* Repository metadata
* Git commits

Optional integrations may additionally ingest:

* Issues
* Pull requests
* PR discussions

Each indexed artifact must retain provenance.

Example:

```json
{
  "repository": "vllm",
  "commit": "abc123",
  "path": "vllm/worker/model_runner.py",
  "language": "python",
  "artifact_type": "source",
  "symbol": "ModelRunner.execute_model",
  "start_line": 812,
  "end_line": 936
}
```

---

# 8. Structure-Aware Code Parsing

Source code must not be indexed solely through fixed token-length chunking.

The ingestion pipeline should recognize structural units including:

* Classes
* Functions
* Methods
* Modules
* Tests

For each symbol, store:

```text
symbol name
fully qualified name
file
line range
symbol type
source text
parent symbol
imports
referenced symbols when available
```

Python should be supported in V1.

C++ support is a stretch goal.

---

# 9. Document Chunking

Documentation may use semantic or section-based chunking.

Chunks should preserve:

```text
document
section
heading hierarchy
line/paragraph range
repository version
```

Git commits should preserve:

```text
commit SHA
author
timestamp
message
modified files
diff
```

---

# 10. Embedding Index

Every retrievable chunk should have a semantic embedding.

The vector index must support:

```text
top-k search
metadata filtering
artifact-type filtering
repository filtering
```

Potential implementations include FAISS, Qdrant, or PostgreSQL/pgvector.

V1 should favor simplicity and reproducibility over distributed scalability.

---

# 11. Sparse Retrieval

The system must provide lexical retrieval using BM25 or an equivalent algorithm.

This is particularly important for:

* Function names
* Class names
* Configuration parameters
* Error messages
* Exact identifiers

---

# 12. Hybrid Retrieval

Dense and sparse retrieval results should be combined.

Initial implementation:

```text
Dense retrieval ──┐
                  ├── Reciprocal Rank Fusion
BM25 retrieval ───┘
                         ↓
                  candidate set
```

Retrieval components must expose scores and provenance for debugging and evaluation.

---

# 13. Symbol Retrieval

The system should support exact and fuzzy symbol lookup.

Example:

```text
PagedAttention
CacheEngine
execute_model
```

Symbol retrieval should be independently callable by the retrieval agent.

---

# 14. Reranking

The top candidates from hybrid retrieval should be reranked using a stronger relevance model.

Example:

```text
Hybrid retrieval
      ↓
Top 50
      ↓
Reranker
      ↓
Top 10
```

The exact values must be configurable.

---

# 15. Context Construction

Retrieved evidence must be transformed into an LLM context while respecting the context-token budget.

The context builder should:

* Deduplicate overlapping chunks.
* Preserve provenance.
* Prefer higher-ranked evidence.
* Preserve related symbols when useful.
* Allocate context across different evidence types.
* Avoid filling the entire context with one large source file.

---

# 16. Grounded Answer Generation

The generation model receives:

```text
question
retrieved evidence
citation metadata
answer instructions
```

Answers must include citations to evidence.

Code citations should preferably contain:

```text
filename
symbol
line range
```

Example:

```text
KV-cache allocation occurs inside CacheEngine.allocate_gpu_cache().

[vllm/worker/cache_engine.py:95–142]
```

The model must be instructed to distinguish between:

1. Facts directly supported by evidence.
2. Reasonable inference from evidence.
3. Information that cannot be established from retrieved evidence.

---

# 17. Agentic Retrieval

V2 should support iterative retrieval.

Instead of:

```text
Question
   ↓
Retrieve once
   ↓
Answer
```

the system should support:

```text
Question
   ↓
Planner
   ↓
Retrieval
   ↓
Evidence inspection
   ↓
Need more information?
   ├── YES → formulate next retrieval
   │             ↓
   │          Retrieve
   │             ↓
   │          Inspect
   │
   └── NO → Answer
```

Example:

```text
Question:
"How is KV-cache size determined?"

Step 1
Find CacheEngine.

Step 2
CacheEngine references CacheConfig.

Step 3
Retrieve CacheConfig.

Step 4
Configuration references block-size calculation.

Step 5
Retrieve calculation.

Step 6
Synthesize answer.
```

The number of iterations must have a configurable maximum to prevent uncontrolled loops.

---

# 18. Query Planning

The system should classify or decompose queries into categories such as:

```text
code lookup
architecture
debugging
change impact
historical reasoning
documentation
```

The planner may choose different retrieval tools depending on query type.

Example:

```text
"Where is X implemented?"

→ symbol search + BM25

"Why was X introduced?"

→ code search + Git/PR search

"What depends on X?"

→ symbol/dependency search
```

---

# 19. Retrieval API

Retrieval components should implement clean interfaces.

Conceptually:

```python
class Retriever(Protocol):

    def retrieve(
        self,
        query: str,
        k: int
    ) -> list[RetrievedChunk]:
        ...
```

Implementations:

```text
DenseRetriever
BM25Retriever
SymbolRetriever
HybridRetriever
GitRetriever
```

This allows retrieval strategies to be independently tested and ablated.

---

# 20. Evaluation Dataset

The project must contain a curated benchmark.

Target V1:

**100–200 questions.**

Each record should contain:

```json
{
  "question": "...",
  "category": "architecture",
  "difficulty": "multi_hop",
  "expected_answer": "...",
  "relevant_files": [],
  "relevant_symbols": [],
  "relevant_chunks": []
}
```

Questions should cover:

```text
code lookup
architecture
configuration
debugging
tests
change impact
historical reasoning
multi-hop reasoning
```

At least 25% of the benchmark should require evidence from multiple files or artifacts.

---

# 21. Retrieval Evaluation

Retrieval should be evaluated independently of generation.

Required metrics:

```text
Recall@5
Recall@10
MRR
nDCG@10
```

Metrics should be available globally and by question category.

---

# 22. Generation Evaluation

Answer evaluation should measure:

```text
correctness
faithfulness
citation correctness
citation completeness
unsupported-claim rate
```

Automated LLM judging may be used, but a manually reviewed subset should validate judge reliability.

---

# 23. Required Ablation Study

The final project should compare at minimum:

```text
Dense only
BM25 only
Hybrid
Hybrid + reranking
Agentic retrieval
```

Example final report:

| System            | Recall@10 | MRR | Answer Accuracy | Citation Accuracy |
| ----------------- | --------: | --: | --------------: | ----------------: |
| Dense             |       TBD | TBD |             TBD |               TBD |
| BM25              |       TBD | TBD |             TBD |               TBD |
| Hybrid            |       TBD | TBD |             TBD |               TBD |
| Hybrid + reranker |       TBD | TBD |             TBD |               TBD |
| Agentic RAG       |       TBD | TBD |             TBD |               TBD |

No target numbers should be fabricated in the demo; all reported metrics must come from reproducible benchmark runs.

---

# 24. Observability

Every query should produce a trace.

Example:

```text
Query
 │
 ├─ Query classification
 │
 ├─ Dense search
 │    ├─ candidates
 │    └─ scores
 │
 ├─ BM25 search
 │    ├─ candidates
 │    └─ scores
 │
 ├─ Fusion
 │
 ├─ Reranking
 │
 ├─ Agent retrieval iteration #2
 │
 ├─ Final context
 │
 └─ Generation
```

Store:

```text
latency
retrieval results
retrieval scores
reranker scores
token counts
LLM calls
estimated cost
final citations
```

This trace should be inspectable from the demo UI.

---

# 25. User Interface

The UI should emphasize the RAG pipeline rather than resemble a generic chatbot.

Main screen:

```text
┌────────────────────────────────────────────┐
│ Ask about the vLLM repository             │
│                                            │
│ Why is KV cache memory allocated this way?│
└────────────────────────────────────────────┘

              ANSWER

KV cache capacity is determined by ...

[cache_engine.py:95–142]
[config.py:718–760]


              RETRIEVAL TRACE

Planner
  ↓
"Find cache allocation implementation"

Dense    20 candidates
BM25     20 candidates
Symbol    5 candidates
  ↓
Fusion
  ↓
Reranker
  ↓
10 evidence chunks


              EVIDENCE

1. cache_engine.py
   CacheEngine.allocate_gpu_cache()
   score: 0.94

2. config.py
   CacheConfig
   score: 0.89

3. worker.py
   determine_num_available_blocks()
   score: 0.84
```

Users should be able to expand evidence and inspect the actual retrieved source.

---

# 26. CLI

The project should also work without the UI.

Example:

```bash
rag ingest ./vllm

rag index

rag ask \
  "Where is KV cache allocated?"

rag benchmark \
  --retriever hybrid

rag benchmark \
  --retriever hybrid \
  --reranker

rag benchmark \
  --agentic
```

---

# 27. System Architecture

```text
                 Repository
                     │
             ┌───────┴────────┐
             │                │
         Code Parser      Doc/Git Parser
             │                │
             └───────┬────────┘
                     │
                  Chunks
                     │
        ┌────────────┼────────────┐
        │            │            │
     Vector       BM25         Symbol
      Index        Index         Index
        │            │            │
        └────────────┼────────────┘
                     │
              Retrieval Layer
                     │
                   Fusion
                     │
                 Reranker
                     │
              Retrieval Agent
                     │
              Context Builder
                     │
                    LLM
                     │
             Grounded Answer
                     │
                API / UI
```

---

# 28. Suggested Repository Structure

```text
engineering-rag/
│
├── ingestion/
│   ├── code_parser.py
│   ├── document_parser.py
│   ├── git_parser.py
│   └── chunker.py
│
├── indexing/
│   ├── embeddings.py
│   ├── vector_index.py
│   ├── bm25_index.py
│   └── symbol_index.py
│
├── retrieval/
│   ├── base.py
│   ├── dense.py
│   ├── sparse.py
│   ├── symbol.py
│   ├── fusion.py
│   └── reranker.py
│
├── agent/
│   ├── planner.py
│   ├── retrieval_agent.py
│   └── tools.py
│
├── generation/
│   ├── context_builder.py
│   └── answer_generator.py
│
├── evaluation/
│   ├── dataset.py
│   ├── retrieval_metrics.py
│   ├── answer_metrics.py
│   └── benchmark.py
│
├── observability/
│   └── tracing.py
│
├── api/
├── ui/
├── tests/
├── configs/
└── docs/
```

---

# 29. Non-Functional Requirements

## Reproducibility

All indexing and evaluation configurations must be version controlled.

A benchmark run should record:

```text
repository commit
embedding model
reranker
LLM
chunking configuration
retrieval configuration
evaluation dataset version
random seed where applicable
```

## Testability

Core components should have unit tests.

Integration tests should cover:

```text
repository → ingestion → indexing → retrieval → answer
```

A small fixture repository should allow CI testing without indexing all of vLLM.

## Modularity

Models and databases must be replaceable through typed interfaces rather than tightly coupled implementation code.

## Configuration

Important parameters should be configurable:

```text
embedding model
chunk size
retrieval k
fusion method
reranker k
final context k
agent iteration limit
generation model
```

---

# 30. Performance Requirements

For a repository approximately the size of vLLM, after indexing:

**Simple query**

Target:

```text
retrieval < 1 second
reranking < 2 seconds
total answer < 10 seconds
```

These are demo-oriented targets rather than strict production SLAs.

Performance measurements should report each stage independently.

---

# 31. Development Milestones

## M1 — Repository Ingestion

Deliver:

* Repository scanner
* Python AST parser
* Structure-aware chunks
* Documentation ingestion
* Metadata schema
* Tests

Demo:

```text
rag ingest ./vllm
```

and inspect parsed symbols.

---

## M2 — Baseline RAG

Deliver:

* Embedding generation
* Vector index
* Dense retrieval
* Basic answer generation
* Source citations

Establish the first end-to-end baseline.

---

## M3 — Hybrid Retrieval

Deliver:

* BM25
* Symbol search
* Reciprocal Rank Fusion
* Retrieval diagnostics

Compare:

```text
dense
BM25
hybrid
```

---

## M4 — Reranking

Deliver:

* Reranker interface
* Candidate reranking
* Context selection

Measure the retrieval improvement.

---

## M5 — Evaluation Harness

Deliver:

* Benchmark schema
* Initial 100+ questions
* Recall@K
* MRR
* nDCG
* Answer evaluation
* Reproducible benchmark CLI

This milestone converts the project from a RAG demo into an ML/IR engineering project.

---

## M6 — Agentic Retrieval

Deliver:

* Query planner
* Retrieval tools
* Multi-step retrieval
* Stopping criteria
* Agent trace

Evaluate whether additional retrieval steps actually improve multi-hop questions.

---

## M7 — Git/History Retrieval

Deliver:

* Commit ingestion
* Diff indexing
* Historical retrieval
* Optional issue/PR integration

Enable:

> Why was this behavior introduced?

---

## M8 — Change-Impact Reasoning

Deliver lightweight relationships such as:

```text
symbol → callers
symbol → imports
symbol → tests
module → dependencies
```

Combine graph relationships with RAG.

---

## M9 — Observability + UI

Deliver interactive UI exposing:

```text
answer
citations
retrieval results
scores
reranking
agent steps
latencies
token usage
```

---

## M10 — Final Benchmark and Demo

Run all configurations:

```text
Dense
BM25
Hybrid
Hybrid + reranker
Agentic RAG
```

Produce final quantitative results, failure analysis, architecture documentation, and demo scenarios.

---

# 32. Final Demo

The primary demo should take approximately five minutes.

### Demo 1 — Direct implementation question

Ask:

> Where is KV-cache memory allocated and how is its capacity determined?

Show retrieved symbols and citations.

### Demo 2 — Multi-hop architecture

Ask:

> Trace a request from the API server until model execution on the GPU.

Show multiple retrieval iterations.

### Demo 3 — Debugging

Provide a GPU-memory error and ask:

> Which components could explain this behavior?

Show evidence retrieval.

### Demo 4 — Change impact

Ask:

> What could break if I modify this cache-management function?

Show source and test relationships.

### Demo 5 — Evaluation

Finish with the benchmark dashboard comparing retrieval strategies.

The final screen should make the central result immediately visible:

```text
                    Recall@10   Answer Accuracy
Dense                   XX%          XX%
BM25                    XX%          XX%
Hybrid                  XX%          XX%
Hybrid + Reranker       XX%          XX%
Agentic RAG             XX%          XX%
```

---

# 33. Success Criteria

The project is considered complete when:

* A large real-world repository can be indexed automatically.
* Code is parsed using structural rather than purely fixed-size chunking.
* Dense, BM25, and symbol retrieval are implemented.
* Hybrid retrieval and reranking are implemented.
* Answers contain inspectable source citations.
* Multi-hop agentic retrieval is supported.
* At least 100 evaluation questions exist.
* Retrieval metrics are reproducible.
* Generation grounding is evaluated.
* Ablations quantify the value of each RAG component.
* Retrieval traces can be inspected.
* The system supports a polished end-to-end demo on vLLM.

---

# 34. Key Project Principle

The project should answer one question throughout development:

> **Does each additional RAG technique measurably improve an engineer's ability to retrieve the correct evidence and answer questions about an unfamiliar codebase?**

Architecture complexity should not be added merely because it is fashionable.

Dense retrieval, hybrid search, reranking, graph relationships, and agentic retrieval should each earn their place through benchmark results.

That evaluation-driven approach is the primary distinction between this project and a conventional RAG chatbot demo.
