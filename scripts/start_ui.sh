#!/bin/bash
#
# Bring up the web UI, after checking that what it needs is actually there.
#
# The LLM is only needed by the Ask tab; Search, Symbols, Benchmark and Traces
# all work without it, so a missing generation backend is a warning rather than
# an error.  A missing index is an error: there would be nothing to serve.
#
# Globals set by parse_args and read throughout: HOST, PORT.

set -euo pipefail

readonly DEFAULT_HOST="127.0.0.1"
readonly DEFAULT_PORT=8100
readonly DEFAULT_LLM_PORT=8099
readonly DEFAULT_INDEX_DIR="indexes/vllm"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT

HOST="${DEFAULT_HOST}"
PORT="${DEFAULT_PORT}"
RAG=""

#######################################
# Print usage to stdout.
# Outputs:
#   Usage text.
#######################################
usage() {
  cat <<'EOF'
Bring up the web UI, after checking that what it needs is actually there.

Usage:
  ./scripts/start_ui.sh                 # http://127.0.0.1:8100
  ./scripts/start_ui.sh --port 9000
  ./scripts/start_ui.sh --host 0.0.0.0  # reachable from other machines

Options:
  --host ADDR   bind address, default 127.0.0.1
  --port N      listen port, default 8100
  -h, --help    show this message
EOF
}

#######################################
# Print an error to stderr.
# Arguments:
#   Message to print.
# Outputs:
#   The message on stderr.
#######################################
err() {
  printf 'error: %s\n' "$*" >&2
}

#######################################
# Parse the command line into HOST and PORT.
# Globals:
#   HOST, PORT (set)
# Arguments:
#   The script's own arguments.
# Returns:
#   0 on success; exits 2 on an unusable command line.
#######################################
parse_args() {
  while (( $# > 0 )); do
    case "$1" in
      --host) HOST="$2"; shift 2 ;;
      --port) PORT="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) err "unknown argument: $1"; usage >&2; exit 2 ;;
    esac
  done
}

#######################################
# Locate the project's own rag CLI.
# Globals:
#   ROOT (read), RAG (set)
# Returns:
#   0 on success; exits 1 when the project is not installed.
#######################################
resolve_rag() {
  RAG="rag"
  if [[ -x "${ROOT}/.venv/bin/rag" ]]; then
    RAG=".venv/bin/rag"
  fi
  if ! command -v "${RAG}" >/dev/null 2>&1 && [[ ! -x "${RAG}" ]]; then
    err "'rag' not found -- install with: pip install -e ."
    exit 1
  fi
}

#######################################
# Fail early when the configured index has not been built.
# Globals:
#   RAG (read)
# Returns:
#   0 on success; exits 1 when there is no index to serve.
#######################################
check_index() {
  local index_dir
  index_dir="$(python3 - <<'PY' 2>/dev/null || echo "indexes/vllm"
"""Print the index directory the default config points at."""
import yaml

print((yaml.safe_load(open("configs/default.yaml")) or {}).get(
    "index_dir", "indexes/vllm"))
PY
)"
  : "${index_dir:=${DEFAULT_INDEX_DIR}}"

  if [[ ! -d "${index_dir}" ]]; then
    err "no index at ${index_dir} -- build one first:"
    printf '  %s ingest && %s index\n' "${RAG}" "${RAG}" >&2
    exit 1
  fi
}

#######################################
# Report whether the generation backend is reachable.
# Globals:
#   EKA_LLM_PORT (read)
# Outputs:
#   A status line on stdout, or an explanation on stderr when it is down.
#######################################
check_llm() {
  local llm_port="${EKA_LLM_PORT:-${DEFAULT_LLM_PORT}}"
  if curl -sf -m 3 "http://127.0.0.1:${llm_port}/v1/models" >/dev/null 2>&1; then
    echo "generation backend: up on :${llm_port}"
    return
  fi
  {
    printf 'warning: no generation backend on :%s -- the Ask tab will fail.\n' \
      "${llm_port}"
    printf '         start one with ./scripts/serve_llm.sh (or set llm.provider in\n'
    printf '         configs/default.yaml to a remote model).\n'
  } >&2
}

#######################################
# Check the prerequisites, then replace this process with the server.
# Arguments:
#   The script's own arguments.
#######################################
main() {
  cd "${ROOT}"
  parse_args "$@"
  resolve_rag
  check_index
  check_llm

  local shown_host="${HOST}"
  if [[ "${HOST}" == "0.0.0.0" ]]; then
    shown_host="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi
  echo "UI: http://${shown_host}:${PORT}"
  exec "${RAG}" serve --host "${HOST}" --port "${PORT}"
}

main "$@"
