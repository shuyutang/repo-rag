#!/bin/bash
#
# Index any Python repository -- first run or refresh -- from one command.
#
# The same command does setup and update: it clones the repository if it is
# missing, pulls if it is already there, and rebuilds the index only when the
# checkout moved (or --force).  The rebuild happens in a scratch directory and
# is swapped in atomically, so an interrupted run never leaves a half-written
# index behind, which makes the script safe to put on a cron schedule.
#
# Two things it deliberately does not do: it does not re-tune fusion weights
# for the new repository (the generated config carries the weights tuned on
# vLLM), and it does not update incrementally -- every rebuild is a full one.
#
# Globals set by parse_args and read throughout: TARGET, PORT, SERVE, FORCE,
# BRANCH, DEPTH, KEEP_CONFIG, INGEST_FLAGS.

set -euo pipefail

readonly DEFAULT_PORT=8100
readonly DEFAULT_LLM_PORT=8099

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT

# Parsed from the command line.
TARGET=""
PORT="${DEFAULT_PORT}"
SERVE=0
FORCE=0
BRANCH=""
DEPTH=""
KEEP_CONFIG=0
INGEST_FLAGS=()

# Resolved by resolve_toolchain and resolve_checkout.
RAG=""
PYTHON=""
NAME=""
REPO_DIR=""
CONFIG=""
INDEX_DIR=""
HEAD_SHA=""

# Set by build_index, read by its EXIT trap.
BUILD_DIR=""
BUILD_CFG=""

#######################################
# Print usage to stdout.
# Outputs:
#   Usage text.
#######################################
usage() {
  cat <<'EOF'
Index any Python repository -- first run or refresh -- from one command.

Usage:
  ./scripts/index_repo.sh <repository> [options]

  ./scripts/index_repo.sh https://github.com/huggingface/trl
  ./scripts/index_repo.sh git@github.com:psf/requests.git --serve
  ./scripts/index_repo.sh ~/src/my-project --force

The repository may be a clone URL or the path of an existing local checkout.

Options:
  --serve            start (or restart) the web UI when indexing finishes
  --port N           UI port, default 8100
  --force            rebuild even when the commit already matches the index
  --branch NAME      check out this branch/tag instead of the default
  --depth N          shallow clone to N commits (faster; limits history depth)
  --no-git           skip commit ingestion entirely
  --max-commits N    override how much history to ingest (default: config)
  --keep-config      never touch an existing configs/<name>.yaml
  -h, --help         show this message

Notes:
  * V1 parses Python with the AST; other languages are skipped, not chunked.
  * Fusion weights in the generated config are the ones tuned on vLLM.  See
    the hint printed at the end for how to re-tune them for your repository.
EOF
}

#######################################
# Print a progress line to stdout.
# Arguments:
#   Message to print.
# Outputs:
#   The message, prefixed and emphasised.
#######################################
say() {
  printf '\033[1m==>\033[0m %s\n' "$*"
}

#######################################
# Print a warning to stderr, without stopping the run.
# Arguments:
#   Message to print.
# Outputs:
#   The message on stderr.
#######################################
warn() {
  printf '\033[33mwarning:\033[0m %s\n' "$*" >&2
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
# Parse the command line into the globals declared above.
# Globals:
#   TARGET, PORT, SERVE, FORCE, BRANCH, DEPTH, KEEP_CONFIG, INGEST_FLAGS
# Arguments:
#   The script's own arguments.
# Returns:
#   0 on success; exits 2 on an unusable command line.
#######################################
parse_args() {
  while (( $# > 0 )); do
    case "$1" in
      --serve) SERVE=1; shift ;;
      --port) PORT="$2"; shift 2 ;;
      --force) FORCE=1; shift ;;
      --branch) BRANCH="$2"; shift 2 ;;
      --depth) DEPTH="$2"; shift 2 ;;
      --keep-config) KEEP_CONFIG=1; shift ;;
      --no-git) INGEST_FLAGS+=(--no-git); shift ;;
      --max-commits) INGEST_FLAGS+=(--max-commits "$2"); shift 2 ;;
      -h|--help) usage; exit 0 ;;
      -*) err "unknown option: $1"; usage >&2; exit 2 ;;
      *)
        if [[ -n "${TARGET}" ]]; then
          err "expected one repository, got '${TARGET}' and '$1'"
          usage >&2
          exit 2
        fi
        TARGET="$1"
        shift
        ;;
    esac
  done
  if [[ -z "${TARGET}" ]]; then
    usage >&2
    exit 2
  fi
}

#######################################
# Locate the project's own rag CLI and Python interpreter.
# Globals:
#   ROOT (read), RAG (set), PYTHON (set)
# Returns:
#   0 on success; exits 1 when the project is not installed.
#######################################
resolve_toolchain() {
  RAG="rag"
  if [[ -x "${ROOT}/.venv/bin/rag" ]]; then
    RAG="${ROOT}/.venv/bin/rag"
  fi
  PYTHON="python3"
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON="${ROOT}/.venv/bin/python"
  fi

  if ! command -v "${RAG}" >/dev/null 2>&1 && [[ ! -x "${RAG}" ]]; then
    err "'rag' not found -- install with: uv pip install -e ."
    exit 1
  fi
  if ! "${PYTHON}" -c "import yaml" 2>/dev/null; then
    err "PyYAML missing from ${PYTHON} -- install the project first"
    exit 1
  fi
}

#######################################
# Use a local checkout where it is, or clone the target into data/<name>.
# Globals:
#   TARGET, BRANCH, DEPTH, ROOT (read); NAME, REPO_DIR (set)
#######################################
resolve_checkout() {
  if [[ -d "${TARGET}/.git" ]]; then
    REPO_DIR="$(cd "${TARGET}" && pwd)"
    NAME="$(basename "${REPO_DIR}")"
    say "using existing checkout ${REPO_DIR}"
    return
  fi

  NAME="$(basename "${TARGET%.git}")"
  REPO_DIR="${ROOT}/data/${NAME}"
  if [[ ! -d "${REPO_DIR}/.git" ]]; then
    say "cloning ${TARGET} -> data/${NAME}"
    mkdir -p "${ROOT}/data"
    local clone_cmd=(git clone --quiet)
    if [[ -n "${DEPTH}" ]]; then
      clone_cmd+=(--depth "${DEPTH}")
    fi
    if [[ -n "${BRANCH}" ]]; then
      clone_cmd+=(--branch "${BRANCH}")
    fi
    "${clone_cmd[@]}" "${TARGET}" "${REPO_DIR}"
  fi
}

#######################################
# Check out the requested branch and fast-forward to the remote.
#
# A shallow checkout keeps its own depth unless --depth says otherwise;
# defaulting to 1 here would truncate history and gut commit ingestion.
#
# Globals:
#   REPO_DIR, BRANCH, DEPTH (read); HEAD_SHA (set)
# Returns:
#   0 on success; exits 1 when the checkout has no commits at all.
#######################################
fetch_latest() {
  if [[ -n "${BRANCH}" ]]; then
    git -C "${REPO_DIR}" checkout --quiet "${BRANCH}"
  fi

  if git -C "${REPO_DIR}" remote get-url origin >/dev/null 2>&1; then
    say "fetching latest commits"
    local git_dir
    git_dir="$(git -C "${REPO_DIR}" rev-parse --git-dir)"
    if [[ -f "${git_dir}/shallow" ]]; then
      local have
      have="$(git -C "${REPO_DIR}" rev-list --count HEAD)"
      git -C "${REPO_DIR}" pull --quiet --depth="${DEPTH:-${have}}" --ff-only ||
        warn "pull failed -- indexing the checkout as it stands"
    else
      git -C "${REPO_DIR}" pull --quiet --ff-only ||
        warn "pull failed (local changes or diverged branch) -- indexing as-is"
    fi
  else
    warn "no 'origin' remote -- nothing to pull"
  fi

  HEAD_SHA="$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ "${HEAD_SHA}" == "unknown" ]]; then
    err "${REPO_DIR} has no commits; the ingester needs real git history"
    exit 1
  fi
}

#######################################
# Warn when the checkout is mostly not Python, which V1 cannot parse.
# Globals:
#   REPO_DIR (read)
# Returns:
#   0 on success; exits 1 when the checkout has no Python at all.
#######################################
check_python_files() {
  local n_py n_all
  n_py="$(git -C "${REPO_DIR}" ls-files '*.py' | wc -l)"
  n_all="$(git -C "${REPO_DIR}" ls-files | wc -l)"

  if (( n_py == 0 )); then
    err "no .py files tracked in ${REPO_DIR} -- this version parses Python only"
    exit 1
  fi
  if (( n_all > 0 )) && (( n_py * 4 < n_all )); then
    warn "only ${n_py} of ${n_all} tracked files are Python; the rest are not parsed"
  fi
}

#######################################
# Write configs/<name>.yaml, then read the index directory back out of it.
#
# Only the four location keys are rewritten, so any tuning already applied to
# this repository's config -- fusion weights, model choices, budgets --
# survives a refresh.
#
# Globals:
#   ROOT, NAME, REPO_DIR, KEEP_CONFIG, PYTHON (read); CONFIG, INDEX_DIR (set)
#######################################
write_config() {
  CONFIG="${ROOT}/configs/${NAME}.yaml"

  if [[ -f "${CONFIG}" && "${KEEP_CONFIG}" -eq 1 ]]; then
    say "reusing configs/${NAME}.yaml unchanged"
  else
    local base="${CONFIG}"
    if [[ ! -f "${CONFIG}" ]]; then
      base="${ROOT}/configs/default.yaml"
    fi
    "${PYTHON}" - "${base}" "${CONFIG}" "${NAME}" "${REPO_DIR}" "${ROOT}" <<'PY'
"""Rewrite only the location keys of a config, preserving any tuning."""
import sys

import yaml

base, out, name, repo_dir, root = sys.argv[1:6]
cfg = yaml.safe_load(open(base)) or {}
cfg["repository"] = name
cfg["repo_path"] = repo_dir
cfg["index_dir"] = f"{root}/indexes/{name}"
cfg["trace_dir"] = f"{root}/traces/{name}"
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
PY
    say "wrote configs/${NAME}.yaml"
  fi

  INDEX_DIR="$("${PYTHON}" - "${CONFIG}" <<'PY'
"""Print the index directory a config points at."""
import sys

import yaml

print(yaml.safe_load(open(sys.argv[1]))["index_dir"])
PY
)"
}

#######################################
# Print the commit an existing index was built from.
# Globals:
#   INDEX_DIR, PYTHON (read)
# Outputs:
#   The commit SHA, or an empty line when there is no index yet.
#######################################
indexed_commit() {
  "${PYTHON}" - "${INDEX_DIR}" <<'PY'
"""Print the commit recorded in an index's metadata, or nothing."""
import json
import pathlib
import sys

meta = pathlib.Path(sys.argv[1]) / "meta.json"
print(json.loads(meta.read_text()).get("commit", "") if meta.exists() else "")
PY
}

#######################################
# Remove a partially built index and its temporary config.
# Globals:
#   BUILD_DIR, BUILD_CFG (read)
#######################################
cleanup_build() {
  rm -rf "${BUILD_DIR}" "${BUILD_CFG}"
}

#######################################
# Ingest and index into a scratch directory, then swap it in atomically.
# Globals:
#   INDEX_DIR, CONFIG, NAME, HEAD_SHA, RAG, PYTHON, INGEST_FLAGS (read);
#   BUILD_DIR, BUILD_CFG (set, for cleanup_build)
#######################################
build_index() {
  BUILD_DIR="${INDEX_DIR}.build"
  BUILD_CFG="$(mktemp -t eka-build-XXXXXX.yaml)"
  trap cleanup_build EXIT
  rm -rf "${BUILD_DIR}"

  "${PYTHON}" - "${CONFIG}" "${BUILD_CFG}" "${BUILD_DIR}" <<'PY'
"""Copy a config with its index directory pointed at the scratch build."""
import sys

import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
cfg["index_dir"] = sys.argv[3]
yaml.safe_dump(cfg, open(sys.argv[2], "w"), sort_keys=False)
PY

  say "ingesting ${NAME} @ ${HEAD_SHA:0:10}"
  "${RAG}" ingest -c "${BUILD_CFG}" "${INGEST_FLAGS[@]+"${INGEST_FLAGS[@]}"}"
  say "building indexes (vector, BM25, symbol, graph)"
  "${RAG}" index -c "${BUILD_CFG}"

  say "swapping in the new index"
  rm -rf "${INDEX_DIR}.old"
  if [[ -d "${INDEX_DIR}" ]]; then
    mv "${INDEX_DIR}" "${INDEX_DIR}.old"
  fi
  mv "${BUILD_DIR}" "${INDEX_DIR}"
  rm -rf "${INDEX_DIR}.old"

  trap - EXIT
  rm -f "${BUILD_CFG}"
}

#######################################
# Stop any server already on the UI port, then serve this config.
#
# The old server is killed by the PID holding the listening socket.  Never
# `pkill -f "rag serve"`: that pattern also matches the shell running this
# script.
#
# Globals:
#   PORT, CONFIG, RAG (read)
#######################################
serve_ui() {
  local old_pid
  old_pid="$(ss -ltnp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print $0}' |
             grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)"
  if [[ -n "${old_pid}" ]]; then
    say "stopping the server already on :${PORT} (pid ${old_pid})"
    kill "${old_pid}" 2>/dev/null || true
    local _
    for _ in $(seq 20); do
      kill -0 "${old_pid}" 2>/dev/null || break
      sleep 0.25
    done
  fi

  local llm_port="${EKA_LLM_PORT:-${DEFAULT_LLM_PORT}}"
  if ! curl -sf -m 3 "http://127.0.0.1:${llm_port}/v1/models" >/dev/null 2>&1; then
    warn "no generation backend on :${llm_port} -- the Ask tab will fail (./scripts/serve_llm.sh)"
  fi

  say "UI: http://127.0.0.1:${PORT}"
  exec "${RAG}" serve --host 127.0.0.1 --port "${PORT}" -c "${CONFIG}"
}

#######################################
# Print what was indexed and how to query it.
# Globals:
#   INDEX_DIR, NAME, TARGET, RAG, PYTHON (read)
# Outputs:
#   Chunk counts and a few example commands.
#######################################
print_summary() {
  "${PYTHON}" - "${INDEX_DIR}" <<'PY'
"""Print an index's repository, commit and per-artifact chunk counts."""
import json
import pathlib
import sys

meta = json.loads((pathlib.Path(sys.argv[1]) / "meta.json").read_text())
by = meta.get("by_artifact", {})
print(f"\n  {meta['repository']} @ {meta['commit'][:10]}  ·  {meta['n_chunks']:,} chunks")
print("  " + "  ".join(f"{k} {v:,}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])))
PY

  cat <<EOF

  ask     ${RAG} ask "how does X work?" -c configs/${NAME}.yaml --trace
  search  ${RAG} search "identifier" -c configs/${NAME}.yaml
  impact  ${RAG} impact SYMBOL -c configs/${NAME}.yaml
  ui      $0 ${TARGET} --serve

  Fusion weights in configs/${NAME}.yaml were tuned on vLLM.  To tune them here:
    ${RAG} dataset build -c configs/${NAME}.yaml
    ${PYTHON} scripts/tune_fusion.py --config configs/${NAME}.yaml --out configs/${NAME}.yaml
EOF
}

#######################################
# Clone or refresh the repository, rebuild the index if it moved, then serve
# or summarise.
# Arguments:
#   The script's own arguments.
#######################################
main() {
  cd "${ROOT}"
  parse_args "$@"
  resolve_toolchain
  resolve_checkout
  fetch_latest
  check_python_files
  write_config

  local indexed_sha
  indexed_sha="$(indexed_commit)"
  if [[ "${indexed_sha}" == "${HEAD_SHA}" && "${FORCE}" -eq 0 ]]; then
    say "index already at ${HEAD_SHA:0:10} -- nothing to rebuild (use --force to redo)"
    if (( SERVE == 0 )); then
      echo "  ask: ${RAG} ask \"...\" -c configs/${NAME}.yaml"
      exit 0
    fi
  else
    if [[ -n "${indexed_sha}" ]]; then
      say "index at ${indexed_sha:0:10}, checkout at ${HEAD_SHA:0:10} -- rebuilding"
    fi
    build_index
  fi

  if (( SERVE == 1 )); then
    serve_ui
  fi
  print_summary
}

main "$@"
