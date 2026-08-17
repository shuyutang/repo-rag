| System | Recall@5 | Recall@10 | MRR | nDCG@10 | Answer accuracy | Faithfulness | Unsupported claims | Citation validity | Citation completeness | Retrieval p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.399 | 0.506 | 0.413 | 0.356 | 0.867 | 0.900 | 0.202 | 0.898 | 0.386 | 5 |
| bm25 | 0.655 | 0.729 | 0.715 | 0.622 | 0.942 | 0.950 | 0.191 | 0.925 | 0.525 | 0 |
| symbol | 0.194 | 0.209 | 0.330 | 0.255 | – | – | – | – | – | 22 |
| hybrid | 0.658 | 0.713 | 0.723 | 0.622 | 0.942 | 0.958 | 0.146 | 0.833 | 0.454 | 52 |
| hybrid+rerank | 0.634 | 0.716 | 0.725 | 0.605 | 0.925 | 0.942 | 0.153 | 0.894 | 0.519 | 122 |
| agentic | 0.680 | 0.735 | 0.736 | 0.628 | 0.942 | 0.975 | 0.128 | 0.950 | 0.572 | 2246 |
| agentic-norerank | 0.622 | 0.735 | 0.693 | 0.612 | – | – | – | – | – | 2267 |

Retrieval metrics: 113 questions (test split). Generation metrics: 60 sampled questions from the same split.

repository commit `63344913a6` · 60272 chunks · embedding `sentence-transformers/all-MiniLM-L6-v2` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` · LLM `openai:Qwen/Qwen3-4B` · dataset `benchmark.jsonl@f7cc8afc2d89` (158 questions)