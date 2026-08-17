#!/bin/bash
#
# Embedding-model ablation on the dev split (see docs/evaluation.md).
#
# Builds a second index with a code-trained embedding model and compares the
# dense leg -- and the hybrid that depends on it -- against the default
# general-purpose model.  Dev split only: this is a tuning decision, and
# consulting the test split here would make the reported numbers a fit.
#
# The second index reuses the first one's chunks.jsonl, so only the embeddings
# are recomputed and the two runs are compared over an identical corpus.

set -euo pipefail

readonly BASE_INDEX="indexes/vllm"
readonly ALT_INDEX="indexes/vllm-arctic"
readonly ALT_CONFIG="configs/strong_embeddings.yaml"
readonly BASE_CONFIG="configs/default.yaml"
readonly RESULTS_DIR="results/dev"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT
readonly RAG="${ROOT}/.venv/bin/rag"

#######################################
# Embed the shared chunk file with the alternative model, if not already done.
# Globals:
#   BASE_INDEX, ALT_INDEX, ALT_CONFIG, RESULTS_DIR, RAG (read)
#######################################
build_alt_index() {
  mkdir -p "${ALT_INDEX}" "${RESULTS_DIR}"
  cp -n "${BASE_INDEX}/chunks.jsonl" "${ALT_INDEX}/" || true
  if [[ ! -f "${ALT_INDEX}/embeddings.npy" ]]; then
    "${RAG}" index --config "${ALT_CONFIG}"
  fi
}

#######################################
# Benchmark the dense and hybrid retrievers under one config.
# Globals:
#   RESULTS_DIR, RAG (read)
# Arguments:
#   Config file to benchmark.
#   Short label for the result filenames.
#######################################
benchmark_config() {
  local config="$1"
  local label="$2"
  "${RAG}" benchmark -c "${config}" --retriever dense --no-reranker \
    --split dev --out "${RESULTS_DIR}" --name "dense-${label}"
  "${RAG}" benchmark -c "${config}" --retriever hybrid --no-reranker \
    --split dev --out "${RESULTS_DIR}" --name "hybrid-${label}"
}

#######################################
# Print the four runs side by side.
# Globals:
#   RESULTS_DIR (read)
# Outputs:
#   One line per run: recall@5, recall@10, MRR and nDCG@10.
#######################################
print_comparison() {
  python3 - "${RESULTS_DIR}" <<'PY'
"""Print retrieval metrics for each embedder/retriever combination."""
import json
import pathlib
import sys

results_dir = pathlib.Path(sys.argv[1])
for name in ["dense-minilm", "dense-arctic", "hybrid-minilm", "hybrid-arctic"]:
    path = results_dir / f"{name}.json"
    if not path.exists():
        continue
    r = json.loads(path.read_text())["retrieval"]
    print(f"{name:<20} R@5={r['recall@5']:.4f} R@10={r['recall@10']:.4f} "
          f"MRR={r['mrr']:.4f} nDCG@10={r['ndcg@10']:.4f}")
PY
}

#######################################
# Build the alternative index, benchmark both, and print the comparison.
#######################################
main() {
  cd "${ROOT}"
  build_alt_index
  benchmark_config "${BASE_CONFIG}" "minilm"
  benchmark_config "${ALT_CONFIG}" "arctic"
  print_comparison
}

main "$@"
