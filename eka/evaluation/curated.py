"""Hand-written multi-hop benchmark questions for the vLLM demo repository.

These cover the question types that cannot be derived mechanically from a
single artifact — architecture walk-throughs, debugging, and change impact —
and they are what pushes the benchmark past the PRD's 25% multi-hop floor.

Gold files were chosen by reading the repository at the pinned commit; every
path is checked against the index by ``rag dataset validate`` (and by
``tests/test_evaluation.py``), so a repository upgrade that moves a file fails
loudly instead of silently degrading the metric.

The expected answers are deliberately short: they state the mechanism and the
symbols a correct answer must name, not a full essay.
"""

from __future__ import annotations

from .dataset import BenchmarkQuestion

_CURATED: list[dict] = [
    # ---------------------------------------------------------------- architecture
    {
        "id": "curated-arch-001",
        "question": "How does a completion request travel from the OpenAI-compatible HTTP server to model execution on the GPU?",
        "category": "architecture",
        "difficulty": "multi_hop",
        "expected_answer": (
            "The API server hands the request to AsyncLLM, which processes the input and "
            "submits it to the EngineCore (usually in a separate process via the core client). "
            "EngineCore schedules the request and calls the executor, which dispatches "
            "execute_model to the workers; the GPU worker runs the model runner, which builds "
            "the input batch, runs the forward pass and samples tokens. Outputs travel back "
            "through the output processor/detokenizer to the HTTP response."
        ),
        "relevant_files": [
            "vllm/entrypoints/openai/api_server.py",
            "vllm/v1/engine/async_llm.py",
            "vllm/v1/engine/core.py",
            "vllm/v1/executor/multiproc_executor.py",
            "vllm/v1/worker/gpu_worker.py",
            "vllm/v1/worker/gpu_model_runner.py",
        ],
        "relevant_symbols": [],
    },
    {
        "id": "curated-arch-002",
        "question": "How is the amount of GPU memory available for the KV cache determined at startup, and how is it turned into a number of cache blocks?",
        "category": "architecture",
        "difficulty": "multi_hop",
        "expected_answer": (
            "The worker profiles a forward pass and computes the memory left for the KV cache "
            "(determine_available_memory in the GPU worker), bounded by "
            "gpu_memory_utilization from CacheConfig. The KV cache specs of each attention "
            "layer are then combined with that budget in kv_cache_utils to produce the KV "
            "cache configuration and the number of blocks of block_size tokens."
        ),
        "relevant_files": [
            "vllm/v1/worker/gpu_worker.py",
            "vllm/v1/core/kv_cache_utils.py",
            "vllm/v1/kv_cache_interface.py",
            "vllm/config/cache.py",
        ],
        "relevant_symbols": ["determine_available_memory", "CacheConfig"],
    },
    {
        "id": "curated-arch-003",
        "question": "How does the scheduler decide which requests are executed in the next engine step, and what limits how many tokens it can batch?",
        "category": "architecture",
        "difficulty": "multi_hop",
        "expected_answer": (
            "Scheduler.schedule walks the waiting and running queues, allocating KV cache "
            "blocks through the KV cache manager and filling a token budget "
            "(max_num_batched_tokens) and a request budget (max_num_seqs) from "
            "SchedulerConfig; the result is a SchedulerOutput consumed by the model runner."
        ),
        "relevant_files": [
            "vllm/v1/core/sched/scheduler.py",
            "vllm/v1/core/sched/output.py",
            "vllm/config/scheduler.py",
        ],
        "relevant_symbols": ["Scheduler", "SchedulerConfig"],
    },
    {
        "id": "curated-arch-004",
        "question": "How does automatic prefix caching decide that a new request can reuse blocks from an earlier one?",
        "category": "architecture",
        "difficulty": "multi_hop",
        "expected_answer": (
            "Full blocks of tokens are hashed (block hashing in kv_cache_utils); the block "
            "pool keeps a cached-block map keyed by that hash, and the KV cache manager looks "
            "up the longest cached prefix for a new request before allocating new blocks."
        ),
        "relevant_files": [
            "vllm/v1/core/block_pool.py",
            "vllm/v1/core/kv_cache_utils.py",
            "vllm/v1/core/kv_cache_manager.py",
        ],
        "relevant_symbols": ["BlockPool", "KVCacheManager"],
    },
    {
        "id": "curated-arch-005",
        "question": "Where does the engine run relative to the API server process, and how do the two communicate?",
        "category": "architecture",
        "difficulty": "multi_hop",
        "expected_answer": (
            "The engine core runs in its own process; the front-end talks to it through the "
            "core client over ZeroMQ sockets, and the executor spawns worker processes for "
            "tensor/pipeline parallelism."
        ),
        "relevant_files": [
            "vllm/v1/engine/core_client.py",
            "vllm/v1/engine/core.py",
            "vllm/v1/executor/multiproc_executor.py",
            "docs/design/multiprocessing.md",
        ],
        "relevant_symbols": [],
    },
    {
        "id": "curated-arch-006",
        "question": "How are output tokens turned back into text and streamed to the client?",
        "category": "architecture",
        "difficulty": "multi_hop",
        "expected_answer": (
            "EngineCoreOutputs are handled by the output processor, which uses the "
            "incremental detokenizer per request and pushes RequestOutputs to the async "
            "generator that the serving layer streams from."
        ),
        "relevant_files": [
            "vllm/v1/engine/output_processor.py",
            "vllm/v1/engine/detokenizer.py",
            "vllm/v1/engine/async_llm.py",
        ],
        "relevant_symbols": [],
    },
    {
        "id": "curated-arch-007",
        "question": "How does speculative decoding propose draft tokens and decide which of them to keep?",
        "category": "architecture",
        "difficulty": "multi_hop",
        "expected_answer": (
            "A proposer (for example the n-gram proposer or an EAGLE draft model) produces "
            "draft tokens according to SpeculativeConfig; the rejection sampler then accepts "
            "or rejects them against the target model's distribution inside the sampling step."
        ),
        "relevant_files": [
            "vllm/v1/spec_decode/ngram_proposer.py",
            "vllm/v1/sample/rejection_sampler.py",
            "vllm/config/speculative.py",
        ],
        "relevant_symbols": ["RejectionSampler", "SpeculativeConfig"],
    },
    {
        "id": "curated-arch-008",
        "question": "How is a sampling request with temperature, top-p and penalties actually applied to the logits?",
        "category": "architecture",
        "difficulty": "multi_hop",
        "expected_answer": (
            "SamplingParams from the request become sampling metadata for the batch; the "
            "Sampler applies logits processors, penalties and temperature/top-p before "
            "sampling the next token."
        ),
        "relevant_files": [
            "vllm/sampling_params.py",
            "vllm/v1/sample/sampler.py",
            "vllm/v1/sample/metadata.py",
        ],
        "relevant_symbols": ["Sampler", "SamplingParams"],
    },
    {
        "id": "curated-arch-009",
        "question": "How does the model runner build the flattened batch of token positions and slot mappings that the attention kernel consumes?",
        "category": "architecture",
        "difficulty": "multi_hop",
        "expected_answer": (
            "The input batch keeps per-request token state; the model runner turns the "
            "scheduler output into flattened positions and uses the block table to map tokens "
            "to KV cache slots for the attention backend's metadata."
        ),
        "relevant_files": [
            "vllm/v1/worker/gpu_input_batch.py",
            "vllm/v1/worker/block_table.py",
            "vllm/v1/worker/gpu_model_runner.py",
        ],
        "relevant_symbols": [],
    },
    {
        "id": "curated-arch-010",
        "question": "How are structured-output constraints such as a JSON schema enforced during generation?",
        "category": "architecture",
        "difficulty": "multi_hop",
        "expected_answer": (
            "The structured output manager compiles the grammar for a request and produces a "
            "bitmask of allowed tokens each step, which is applied to the logits before "
            "sampling."
        ),
        "relevant_files": [
            "vllm/v1/structured_output/__init__.py",
            "vllm/v1/core/sched/scheduler.py",
            "vllm/v1/sample/sampler.py",
        ],
        "relevant_symbols": ["StructuredOutputManager"],
    },
    # ---------------------------------------------------------------- debugging
    {
        "id": "curated-debug-001",
        "question": "A server dies with 'CUDA out of memory' shortly after startup while allocating the KV cache. Which components decide that allocation and what knobs control it?",
        "category": "debugging",
        "difficulty": "multi_hop",
        "expected_answer": (
            "Memory profiling in the GPU worker computes the KV cache budget from "
            "gpu_memory_utilization; kv_cache_utils turns it into blocks. Reducing "
            "gpu_memory_utilization, max_model_len, max_num_batched_tokens or the number of "
            "parallel sequences reduces the requirement."
        ),
        "relevant_files": [
            "vllm/v1/worker/gpu_worker.py",
            "vllm/config/cache.py",
            "vllm/v1/core/kv_cache_utils.py",
        ],
        "relevant_symbols": ["determine_available_memory", "CacheConfig"],
    },
    {
        "id": "curated-debug-002",
        "question": "Throughput collapses under load and the logs mention preemption. What causes a running request to be preempted and what happens to its KV blocks?",
        "category": "debugging",
        "difficulty": "multi_hop",
        "expected_answer": (
            "When the KV cache manager cannot allocate new blocks for a running request, the "
            "scheduler preempts the lowest-priority running request, frees its blocks and "
            "moves it back to the waiting queue to be recomputed later."
        ),
        "relevant_files": [
            "vllm/v1/core/sched/scheduler.py",
            "vllm/v1/core/kv_cache_manager.py",
        ],
        "relevant_symbols": ["Scheduler", "KVCacheManager"],
    },
    {
        "id": "curated-debug-003",
        "question": "Requests fail with an error saying the prompt is longer than the maximum model length. Where is that check made and which settings govern it?",
        "category": "debugging",
        "difficulty": "single_hop",
        "expected_answer": (
            "The input processor validates prompt length against max_model_len from the model "
            "configuration before the request is queued."
        ),
        "relevant_files": [
            "vllm/v1/engine/input_processor.py",
            "vllm/config/model.py",
        ],
        "relevant_symbols": [],
    },
    {
        "id": "curated-debug-004",
        "question": "Where does the engine report throughput, running/waiting request counts and KV cache usage, so a performance regression can be observed?",
        "category": "debugging",
        "difficulty": "multi_hop",
        "expected_answer": (
            "Scheduler statistics are collected into the metrics stats objects and rendered by "
            "the metrics loggers, including Prometheus gauges for KV cache usage and queue "
            "sizes."
        ),
        "relevant_files": [
            "vllm/v1/metrics/stats.py",
            "vllm/v1/metrics/loggers.py",
        ],
        "relevant_symbols": [],
    },
    {
        "id": "curated-debug-005",
        "question": "A multi-GPU run hangs at startup before any request is served. Which parts of the codebase set up the distributed process group and worker processes?",
        "category": "debugging",
        "difficulty": "multi_hop",
        "expected_answer": (
            "Parallel state initialises the distributed groups, the multiproc executor spawns "
            "and handshakes with worker processes, and ParallelConfig determines the world "
            "size."
        ),
        "relevant_files": [
            "vllm/distributed/parallel_state.py",
            "vllm/v1/executor/multiproc_executor.py",
            "vllm/config/parallel.py",
        ],
        "relevant_symbols": [],
    },
    # ---------------------------------------------------------------- change impact
    {
        "id": "curated-impact-001",
        "question": "If the KV cache block allocation logic changed, which components and tests would need to be re-checked?",
        "category": "change_impact",
        "difficulty": "multi_hop",
        "expected_answer": (
            "The block pool and KV cache manager are used by the scheduler and, through the "
            "KV cache configuration, by the workers; the v1 core tests for the KV cache "
            "manager, block pool and scheduler cover the behaviour."
        ),
        "relevant_files": [
            "vllm/v1/core/block_pool.py",
            "vllm/v1/core/kv_cache_manager.py",
            "vllm/v1/core/sched/scheduler.py",
        ],
        "relevant_symbols": ["BlockPool", "KVCacheManager"],
    },
    {
        "id": "curated-impact-002",
        "question": "What depends on the scheduler output structure, so that adding a field to it would require changes elsewhere?",
        "category": "change_impact",
        "difficulty": "multi_hop",
        "expected_answer": (
            "SchedulerOutput is produced by the scheduler and consumed by the engine core and "
            "the model runners, so all of them (and the serialization used to send it to "
            "worker processes) are affected."
        ),
        "relevant_files": [
            "vllm/v1/core/sched/output.py",
            "vllm/v1/core/sched/scheduler.py",
            "vllm/v1/worker/gpu_model_runner.py",
            "vllm/v1/engine/core.py",
        ],
        "relevant_symbols": ["SchedulerOutput"],
    },
    {
        "id": "curated-impact-003",
        "question": "Which parts of the system would be affected by changing the signature of the worker's execute_model entry point?",
        "category": "change_impact",
        "difficulty": "multi_hop",
        "expected_answer": (
            "The executor calls into the worker base class, which delegates to the platform "
            "model runners (GPU, CPU); all executors and worker implementations would need to "
            "match the new signature."
        ),
        "relevant_files": [
            "vllm/v1/worker/worker_base.py",
            "vllm/v1/executor/abstract.py",
            "vllm/v1/worker/gpu_worker.py",
            "vllm/v1/worker/gpu_model_runner.py",
        ],
        "relevant_symbols": ["execute_model"],
    },
    {
        "id": "curated-impact-004",
        "question": "If LoRA adapter loading changed, which components in the serving path would be affected?",
        "category": "change_impact",
        "difficulty": "multi_hop",
        "expected_answer": (
            "The LoRA model manager and the worker-side LoRA manager are used by the model "
            "runner's LoRA mixin; LoRARequest carries adapters from the API layer."
        ),
        "relevant_files": [
            "vllm/lora/model_manager.py",
            "vllm/lora/worker_manager.py",
            "vllm/v1/worker/lora_model_runner_mixin.py",
        ],
        "relevant_symbols": ["LoRAModelManager"],
    },
    # ---------------------------------------------------------------- configuration
    {
        "id": "curated-config-001",
        "question": "Which setting controls the fraction of GPU memory vLLM is allowed to use, and where is it consumed?",
        "category": "configuration",
        "difficulty": "multi_hop",
        "expected_answer": (
            "gpu_memory_utilization on CacheConfig; the GPU worker uses it while profiling to "
            "compute the memory available for the KV cache."
        ),
        "relevant_files": ["vllm/config/cache.py", "vllm/v1/worker/gpu_worker.py"],
        "relevant_symbols": ["CacheConfig"],
    },
    {
        "id": "curated-config-002",
        "question": "Which configuration fields bound the size of a scheduling step, and what are their defaults?",
        "category": "configuration",
        "difficulty": "single_hop",
        "expected_answer": (
            "SchedulerConfig.max_num_batched_tokens and max_num_seqs (plus long-prefill "
            "related settings) bound tokens and requests per step."
        ),
        "relevant_files": ["vllm/config/scheduler.py"],
        "relevant_symbols": ["SchedulerConfig"],
    },
    {
        "id": "curated-config-003",
        "question": "How is the KV cache block size configured and what constrains the values it may take?",
        "category": "configuration",
        "difficulty": "multi_hop",
        "expected_answer": (
            "CacheConfig.block_size, validated against the attention backend's supported "
            "block sizes; it determines the token granularity of KV cache blocks used by the "
            "KV cache specs."
        ),
        "relevant_files": ["vllm/config/cache.py", "vllm/v1/kv_cache_interface.py"],
        "relevant_symbols": ["CacheConfig"],
    },
    {
        "id": "curated-config-004",
        "question": "Which environment variables change vLLM's runtime behaviour, and where are they read?",
        "category": "configuration",
        "difficulty": "single_hop",
        "expected_answer": (
            "vllm/envs.py declares the VLLM_* environment variables and their defaults; other "
            "modules import envs and read attributes from it."
        ),
        "relevant_files": ["vllm/envs.py"],
        "relevant_symbols": [],
    },
    # ---------------------------------------------------------------- tests
    {
        "id": "curated-tests-001",
        "question": "Which tests would catch a regression in prefix-cache block reuse?",
        "category": "tests",
        "difficulty": "multi_hop",
        "expected_answer": (
            "The v1 core tests for the KV cache manager, the block pool and prefix caching "
            "exercise cache hits and block reuse."
        ),
        "relevant_files": [
            "tests/v1/core/test_prefix_caching.py",
            "vllm/v1/core/kv_cache_manager.py",
        ],
        "relevant_symbols": [],
    },
    {
        "id": "curated-tests-002",
        "question": "How is scheduler behaviour tested without running a real model?",
        "category": "tests",
        "difficulty": "multi_hop",
        "expected_answer": (
            "The v1 core scheduler tests construct a scheduler with mock configurations and "
            "requests and assert on the SchedulerOutput."
        ),
        "relevant_files": [
            "tests/v1/core/test_scheduler.py",
            "vllm/v1/core/sched/scheduler.py",
        ],
        "relevant_symbols": [],
    },
    # ---------------------------------------------------------------- documentation
    {
        "id": "curated-doc-001",
        "question": "What does the design documentation say about how the hybrid KV cache manager handles models that mix attention types?",
        "category": "documentation",
        "difficulty": "single_hop",
        "expected_answer": (
            "The hybrid KV cache manager design doc explains grouping layers with different "
            "KV cache specs (for example full attention and sliding window) into KV cache "
            "groups sharing one memory pool."
        ),
        "relevant_files": ["docs/design/hybrid_kv_cache_manager.md"],
        "relevant_symbols": [],
    },
    {
        "id": "curated-doc-002",
        "question": "Where is the high-level architecture of the system described, including the entrypoints and the engine?",
        "category": "documentation",
        "difficulty": "single_hop",
        "expected_answer": (
            "docs/design/arch_overview.md walks through entrypoints (LLM class and OpenAI "
            "server), the engine, the worker and the model runner."
        ),
        "relevant_files": ["docs/design/arch_overview.md"],
        "relevant_symbols": [],
    },
    # ---------------------------------------------------------------- historical
    {
        "id": "curated-hist-001",
        "question": "Why does the engine run its core loop in a separate process instead of the API server process?",
        "category": "historical",
        "difficulty": "multi_hop",
        "expected_answer": (
            "To keep the Python GIL of the HTTP front-end from stalling the engine loop; the "
            "multiprocessing design document and the engine core client describe the split "
            "and the IPC that replaced the in-process loop."
        ),
        "relevant_files": [
            "docs/design/multiprocessing.md",
            "vllm/v1/engine/core_client.py",
        ],
        "relevant_symbols": [],
    },
]


def curated_questions() -> list[BenchmarkQuestion]:
    """Return the hand-written benchmark questions.

    These are written against the pinned commit and marked reviewed. They are
    the honest half of the benchmark: recall on them is roughly 0.30 for
    every configuration, against roughly 0.73 on the generated questions, so
    any change should be measured here rather than on the aggregate.

    Returns:
      The curated questions, each with `source` set to "curated".
    """
    return [
        BenchmarkQuestion(
            id=item["id"],
            question=item["question"],
            category=item["category"],
            difficulty=item["difficulty"],
            expected_answer=item["expected_answer"],
            relevant_files=item["relevant_files"],
            relevant_symbols=item.get("relevant_symbols", []),
            source="curated",
            reviewed=True,
            provenance={"author": "hand-written against the pinned commit"},
        )
        for item in _CURATED
    ]
