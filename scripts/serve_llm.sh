#!/bin/bash
#
# Local generation backend: vLLM's OpenAI-compatible server.
#
# The whole system runs against this by default, so no API key is required.
# Point `llm.base_url` in configs/default.yaml elsewhere (OpenAI, Anthropic, a
# hosted endpoint) if you would rather use a remote model.
#
# Environment:
#   EKA_LLM_MODEL     model to serve, default Qwen/Qwen3-4B
#   EKA_LLM_PORT      listen port, default 8099
#   EKA_VLLM_PYTHON   interpreter with vllm installed
#   EKA_LLM_MAX_LEN   context length, default 16384
#   EKA_LLM_GPU_UTIL  fraction of device memory to reserve, default 0.55

set -euo pipefail

readonly DEFAULT_MODEL="Qwen/Qwen3-4B"
readonly DEFAULT_PORT=8099
readonly DEFAULT_MAX_LEN=16384
readonly DEFAULT_GPU_UTIL=0.55

#######################################
# Replace this process with a vLLM OpenAI-compatible server.
# Globals:
#   EKA_LLM_MODEL, EKA_LLM_PORT, EKA_VLLM_PYTHON, EKA_LLM_MAX_LEN,
#   EKA_LLM_GPU_UTIL (read)
# Outputs:
#   vLLM's own server log on stdout and stderr.
#######################################
main() {
  local model="${EKA_LLM_MODEL:-${DEFAULT_MODEL}}"
  local port="${EKA_LLM_PORT:-${DEFAULT_PORT}}"
  local python="${EKA_VLLM_PYTHON:-${HOME}/miniconda3/envs/vllm/bin/python}"

  # vllm 0.24 crashes during engine init with the FlashInfer sampler enabled.
  export VLLM_USE_FLASHINFER_SAMPLER=0

  exec "${python}" -m vllm.entrypoints.openai.api_server \
    --model "${model}" \
    --served-model-name "${model}" \
    --port "${port}" \
    --max-model-len "${EKA_LLM_MAX_LEN:-${DEFAULT_MAX_LEN}}" \
    --gpu-memory-utilization "${EKA_LLM_GPU_UTIL:-${DEFAULT_GPU_UTIL}}" \
    --no-enable-log-requests
}

main "$@"
