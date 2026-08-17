"""Test fixtures.

A miniature repository stands in for vLLM so the whole pipeline —
ingest → index → retrieve → answer → evaluate — runs in CI in a couple of
seconds with no model downloads and no network (PRD §29 testability).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repo_rag.config import (
    AgentConfig,
    Config,
    EmbeddingConfig,
    GenerationConfig,
    IngestionConfig,
    LLMConfig,
    RerankerConfig,
    RetrievalConfig,
)
from repo_rag.indexing.knowledge_base import KnowledgeBase
from repo_rag.ingestion.scanner import RepositoryScanner

FIXTURE_FILES: dict[str, str] = {
    "minirepo/cache_engine.py": '''\
"""Cache engine: owns the KV cache tensors for one worker."""

from minirepo.config import CacheConfig


class CacheEngine:
    """Allocates and frees the paged KV cache on the device."""

    def __init__(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.gpu_cache = None

    def allocate_gpu_cache(self, num_blocks: int) -> list:
        """Allocate ``num_blocks`` KV cache blocks on the GPU.

        The size of one block is block_size * num_heads * head_size elements,
        so the total allocation is num_blocks times that value.
        """
        block_elems = self.cache_config.block_size * self.cache_config.num_heads
        self.gpu_cache = [[0] * block_elems for _ in range(num_blocks)]
        return self.gpu_cache

    def free(self) -> None:
        """Release the KV cache."""
        self.gpu_cache = None
''',
    "minirepo/config.py": '''\
"""Configuration objects."""


class CacheConfig:
    """How much memory the KV cache may use.

    gpu_memory_utilization is the fraction of device memory vLLM may occupy.
    """

    def __init__(self, block_size: int = 16, num_heads: int = 8,
                 gpu_memory_utilization: float = 0.9) -> None:
        self.block_size = block_size
        self.num_heads = num_heads
        self.gpu_memory_utilization = gpu_memory_utilization

    def num_blocks_for(self, available_bytes: int) -> int:
        """Number of blocks that fit into ``available_bytes``."""
        per_block = self.block_size * self.num_heads * 2
        return int(available_bytes * self.gpu_memory_utilization) // per_block
''',
    "minirepo/worker.py": '''\
"""Worker: drives the cache engine and the model runner."""

from minirepo.cache_engine import CacheEngine
from minirepo.config import CacheConfig


class Worker:
    """One device worker."""

    def __init__(self) -> None:
        self.cache_config = CacheConfig()
        self.cache_engine = CacheEngine(self.cache_config)

    def determine_num_available_blocks(self, free_bytes: int) -> int:
        """Profile memory and decide how many KV cache blocks fit."""
        return self.cache_config.num_blocks_for(free_bytes)

    def initialize_cache(self, free_bytes: int) -> None:
        """Allocate the KV cache for this worker."""
        num_blocks = self.determine_num_available_blocks(free_bytes)
        self.cache_engine.allocate_gpu_cache(num_blocks)

    def execute_model(self, tokens: list) -> list:
        """Run one decoding step."""
        if self.cache_engine.gpu_cache is None:
            raise RuntimeError("KV cache has not been initialised")
        return [t + 1 for t in tokens]
''',
    "tests/test_cache_engine.py": '''\
"""Tests for the cache engine."""

from minirepo.cache_engine import CacheEngine
from minirepo.config import CacheConfig


def test_allocate_gpu_cache():
    engine = CacheEngine(CacheConfig())
    blocks = engine.allocate_gpu_cache(4)
    assert len(blocks) == 4


def test_num_blocks_for():
    config = CacheConfig()
    assert config.num_blocks_for(1024) > 0
''',
    "docs/design.md": """\
# Mini repository design

## KV cache

The cache engine allocates paged KV cache blocks. The number of blocks is
derived from the free device memory and `gpu_memory_utilization`.

## Execution

The worker initialises the cache and then calls `execute_model` once per step.
""",
}


@pytest.fixture(scope="session")
def fixture_repo(tmp_path_factory) -> Path:
    """Write the fixture files to a tmp dir and commit them, so git history exists."""
    root = tmp_path_factory.mktemp("minirepo")
    for rel, content in FIXTURE_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e",
           "PATH": "/usr/bin:/bin"}
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-q", "-m", "Add paged KV cache allocation to the cache engine"],
        cwd=root, check=True, env=env,
    )
    return root


@pytest.fixture(scope="session")
def config(fixture_repo, tmp_path_factory) -> Config:
    """Configure the whole system for the fixture repo: hashing embedder, echo LLM, no reranker."""
    index_dir = tmp_path_factory.mktemp("index")
    return Config(
        repository="minirepo",
        repo_path=str(fixture_repo),
        index_dir=str(index_dir),
        ingestion=IngestionConfig(git_max_commits=10, min_chunk_chars=30),
        embedding=EmbeddingConfig(model="hashing"),
        retrieval=RetrievalConfig(dense_k=10, bm25_k=10, symbol_k=5, candidate_k=10, final_k=5),
        reranker=RerankerConfig(enabled=False, model="identity"),
        llm=LLMConfig(provider="echo", model="echo"),
        generation=GenerationConfig(context_token_budget=2000, max_chunk_tokens=400),
        agent=AgentConfig(enabled=False, max_iterations=2, per_step_k=4, max_evidence=6),
        trace_dir=str(tmp_path_factory.mktemp("traces")),
        # every remaining relative path (evaluation_data/, results/) lands here
        # rather than in the checkout, so a test can never clobber real data
        root=str(tmp_path_factory.mktemp("root")),
    )


@pytest.fixture(scope="session")
def chunks(config):
    """Ingest the fixture repository once per session, git history included."""
    scanner = RepositoryScanner(config)
    parsed, _stats = scanner.scan(config.repo_dir, include_git=True)
    return parsed


@pytest.fixture(scope="session")
def kb(config, chunks) -> KnowledgeBase:
    """Build every index over the fixture chunks once per session."""
    return KnowledgeBase.build(config, chunks)
