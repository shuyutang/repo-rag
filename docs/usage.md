# Usage

Everything you can do with the system once it is installed. For *why* it is
built this way see [architecture.md](architecture.md); for what the numbers mean
see [evaluation.md](evaluation.md).

---

## Indexing a repository

One script handles both first-time setup and refresh:

```bash
./scripts/index_repo.sh https://github.com/huggingface/trl     # clone + index
./scripts/index_repo.sh https://github.com/huggingface/trl     # again: pull, rebuild only if moved
./scripts/index_repo.sh ~/src/my-project --serve               # local checkout, then open the UI
```

It clones (or pulls), writes `configs/<name>.yaml`, and rebuilds the index
**only when the checkout has moved** — so it is safe to run from cron. The
rebuild happens in `indexes/<name>.build` and is swapped in atomically, so an
interrupted run leaves the previous index intact.

| flag | effect |
| --- | --- |
| `--serve` | start (or restart) the web UI when indexing finishes |
| `--port N` | UI port, default 8100 |
| `--force` | rebuild even when the commit already matches the index |
| `--branch NAME` | check out this branch or tag |
| `--depth N` | shallow clone; a later refresh keeps the depth it finds |
| `--no-git` | skip commit ingestion |
| `--max-commits N` | override how much history to ingest |
| `--keep-config` | never touch an existing `configs/<name>.yaml` |

Generated configs record absolute paths for the checkout and index, so they are
gitignored — only `configs/default.yaml` and `configs/strong_embeddings.yaml`,
the two the published benchmark numbers were produced with, are tracked.

Every subsequent command takes `-c configs/<name>.yaml`, and commands run
without `-c` use `configs/default.yaml`, which points at vLLM. Running the
script on vLLM therefore produces a `configs/vllm.yaml` equivalent to the
default one and sharing its `indexes/vllm` directory — tune whichever you
actually pass to `rag`.

Two things the script deliberately does not do:

* **It does not re-tune fusion weights.** The generated config carries the
  weights tuned on vLLM, which are not optimal elsewhere — on a commit-heavy
  repository they let history crowd out source. To fix that for your repository:

  ```bash
  rag dataset build -c configs/<name>.yaml            # needs the LLM up
  python scripts/tune_fusion.py --config configs/<name>.yaml --out configs/<name>.yaml
  ```

* **It does not update incrementally.** Every rebuild is a full one (~13 s for a
  small repository, ~70 s for vLLM).

The manual equivalent is `rag ingest && rag index`.

### Scope

V1 parses **Python** with the `ast` module; markdown and rst as documents; git
history as commit chunks. Other languages are skipped rather than chunked, so
a Go or C++ repository indexes only its Python fraction and loses symbol
lookup, the change-impact graph and structure-aware citations along with it.

---

## Web UI

```bash
./scripts/start_ui.sh                # http://127.0.0.1:8100
./scripts/start_ui.sh --host 0.0.0.0 # reachable from another machine on the LAN
./scripts/start_ui.sh --port 9000
```

The launcher checks that an index exists and warns (rather than fails) if the
generation backend is down — only the **Ask** tab needs it. Two dots in the
header report the live state of the index and the LLM.

| tab | what it is for |
| --- | --- |
| **Ask** | full pipeline: grounded answer, clickable citations, per-stage trace, expandable evidence. Switch retriever, reranker, agentic mode and `k` to feel the difference. |
| **Search** | retrieval only, no LLM — milliseconds per query, with the per-retriever component scores. The fastest way to see *why* something was ranked where it was. |
| **Symbols** | change impact for a symbol: definitions → callers → tests → importers. |
| **Benchmark** | the recorded ablation table, plus all 158 benchmark questions. **Click any question to run it live**: gold labels are scored against what retrieval actually returned, hits marked green and misses red. |
| **Traces** | every question asked is traced to disk; browse and inspect the raw JSON. |

Any view can be linked to directly, so a specific test case is shareable:

```
/?q=Where+is+KV+cache+memory+allocated%3F&reranker=0   # run a question on load
/?tab=search&q=allocate_slots&retriever=bm25           # retrieval only
/?sym=allocate_slots                                   # impact of a symbol
/?qid=curated-arch-001                                 # score one benchmark question
```

The live-evaluation view is the honest one: open `?qid=curated-arch-001` and
watch a hand-written multi-hop question retrieve documentation pages instead of
the six implementation files it needs. That gap is the subject of
[failure_analysis.md](failure_analysis.md).

Note that the server loads the index once, on first request, and caches it — so
after re-indexing, restart the process (`index_repo.sh … --serve` does this for
you).

---

## API

Everything the UI does is a documented endpoint; `/docs` serves the generated
OpenAPI page.

| endpoint | purpose |
| --- | --- |
| `GET /api/health` | index + generation backend reachability |
| `GET /api/meta` | repository, commit, chunk counts, models |
| `POST /api/ask` | `{question, retriever, reranker, agentic, k}` → answer + citations + trace + evidence |
| `POST /api/search` | retrieval only |
| `POST /api/evaluate` | `{question_id, …}` → live retrieval metrics against gold labels |
| `GET /api/impact/{symbol}` | callers, tests, importers |
| `GET /api/dataset` | benchmark questions, filterable by category/split/source |
| `GET /api/benchmark` | recorded ablation runs |
| `GET /api/traces`, `/api/traces/{id}` | recent queries and full traces |

---

## CLI

| command | what it does |
| --- | --- |
| `rag ingest [path]` | parse a checkout into structure-aware chunks |
| `rag index` | build vector, BM25, symbol and graph indexes |
| `rag ask "…"` | grounded answer with citations (`--trace`, `--agentic`, `--retriever`, `--no-reranker`) |
| `rag search "…"` | retrieval only, no LLM (`--show` prints the source) |
| `rag diagnose "…"` | per-retriever candidates, scores and overlap |
| `rag impact SYMBOL` | callers, tests and importers of a symbol |
| `rag dataset build/validate/stats` | build and check the evaluation benchmark |
| `rag benchmark` | reproducible run (`--retriever`, `--reranker`, `--agentic`, `--generation`, `--split`) |
| `rag report` | aggregate runs into the ablation table |
| `rag serve` | API + demo UI |
| `rag stats` | index fingerprint |

All of them accept `-c/--config`.

`rag dataset validate` re-checks every gold path against the current index —
worth running after a refresh, since benchmark gold labels are paths plus line
ranges and a repository update quietly invalidates some of them.

---

## Generation backend

A local vLLM server, so no API key is needed anywhere:

```bash
./scripts/serve_llm.sh &     # Qwen3-4B on :8099
```

To use a hosted model instead, point `llm.base_url` / `llm.provider` in the
config at OpenAI or Anthropic and export the key named by `llm.api_key_env`.

---

## Tests

```bash
.venv/bin/python -m pytest       # 60 tests, under a second, no network, no GPU
```

A CI-sized fixture repository (`tests/conftest.py`) exercises the whole path:
ingest → index → retrieve → fuse → build context → answer → score.

---

## Code style

Python follows the [Google Python style
guide](https://google.github.io/styleguide/pyguide.html): every module, class,
function and method carries a docstring with `Args:` / `Returns:` / `Raises:`
sections where they apply. That is enforced, not merely intended:

```bash
.venv/bin/ruff check eka scripts tests    # pydocstyle, convention = "google"
```

The one deliberate deviation is line length. Google specifies 80 columns; the
codebase was written to 88, and reflowing it is a separate concern from
documentation style, so `E501` is not selected. The rule lives in
`[tool.ruff.lint]` in `pyproject.toml`.

Shell scripts follow the [Google shell style
guide](https://google.github.io/styleguide/shellguide.html): a file header, a
`main` function called with `"$@"`, per-function header comments listing
Globals / Arguments / Outputs / Returns, `local` for function variables and
`readonly` constants. They are shellcheck-clean:

```bash
shellcheck scripts/*.sh
```

Both checks, plus the test suite on Python 3.10 and 3.12, run in CI
(`.github/workflows/ci.yml`) on every push and pull request.

---

## Reproducibility

Every benchmark run records the repository commit, chunk count, embedding model,
reranker, LLM, chunking/retrieval/generation configuration, dataset hash and
random seed in its result file (`results/*.json`). The config file is the single
source of truth for all of it, and is version-controlled.

Result files also mean the recorded ablation is tied to a specific corpus: after
re-indexing at a new commit, earlier runs in `results/` describe a repository
state that no longer exists. `rag report` will still aggregate them — check the
fingerprints before comparing across a refresh.
