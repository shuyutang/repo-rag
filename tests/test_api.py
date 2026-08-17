"""API surface used by the demo UI."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eka.api.server import create_app
from eka.generation.llm import EchoClient
from eka.indexing.knowledge_base import KnowledgeBase
from eka.pipeline import Pipeline


@pytest.fixture(scope="module")
def client(config, kb, tmp_path_factory):
    """A TestClient over the app, wired to the fixture config and index."""
    kb.save(config.index_path)
    app = create_app(config)
    with TestClient(app) as test_client:
        # inject an offline pipeline so no model is contacted
        loaded = KnowledgeBase.load(config)
        for route in app.routes:  # trigger lazy state creation via a cheap call
            pass
        test_client.app.dependency_overrides = {}
        yield test_client


def test_meta_endpoint(client, monkeypatch):
    """Meta reports the repository, commit and chunk counts."""
    monkeypatch.setattr(Pipeline, "llm", property(lambda self: EchoClient()))
    response = client.get("/api/meta")
    assert response.status_code == 200
    payload = response.json()
    assert payload["n_chunks"] > 0
    assert payload["repository"] == "minirepo"


def test_search_endpoint(client):
    """Search returns ranked results with their component scores."""
    response = client.post("/api/search", json={"query": "allocate_gpu_cache", "k": 3})
    assert response.status_code == 200
    results = response.json()["results"]
    assert results and results[0]["location"]
    assert "content" in results[0]


def test_ask_endpoint_returns_trace_and_evidence(client, monkeypatch):
    """Ask returns an answer alongside its trace and evidence blocks."""
    monkeypatch.setattr(Pipeline, "llm", property(lambda self: EchoClient()))
    response = client.post(
        "/api/ask",
        json={"question": "Where is the KV cache allocated?", "retriever": "bm25",
              "reranker": False},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence"]
    assert payload["trace"]["steps"]
    assert "usage" in payload

    trace_id = payload["trace"]["trace_id"]
    stored = client.get(f"/api/traces/{trace_id}")
    assert stored.status_code == 200
    assert stored.json()["question"]


def test_chunk_and_impact_endpoints(client):
    """Chunk lookup and symbol impact both resolve against the index."""
    results = client.post("/api/search", json={"query": "allocate_gpu_cache", "k": 1}).json()
    chunk_id = results["results"][0]["chunk_id"]
    assert client.get(f"/api/chunk/{chunk_id}").json()["path"]
    assert client.get("/api/chunk/deadbeef").status_code == 404

    impact = client.get("/api/impact/allocate_gpu_cache").json()
    assert impact["definitions"]
    assert impact["callers"]


def test_ask_rejects_empty_question(client):
    """An empty question is rejected rather than sent to the model."""
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400


def test_benchmark_and_dataset_endpoints(client):
    """The recorded ablation and the dataset browser both serve."""
    assert client.get("/api/benchmark").status_code == 200
    assert client.get("/api/dataset").status_code == 200


def test_health_reports_index_and_llm(client, monkeypatch):
    """Health reports index and generation-backend reachability separately."""
    monkeypatch.setattr(Pipeline, "llm", property(lambda self: EchoClient()))
    payload = client.get("/api/health").json()
    assert payload["index"] == "ready"
    assert payload["llm"] == "ready"


def _write_benchmark(config, chunk_path: str) -> str:
    """One question whose gold file is a path we know is in the fixture index."""
    from eka.evaluation.dataset import BenchmarkQuestion, save_dataset

    question = BenchmarkQuestion(
        id="ui-001",
        question="Where is the GPU cache allocated?",
        category="code_lookup",
        relevant_files=[chunk_path],
        source="curated",
    )
    save_dataset([question], config.resolve("evaluation_data/benchmark.jsonl"))
    return question.id


def test_dataset_browser_filters(client, config):
    """The dataset browser filters by category, split and source."""
    top = client.post("/api/search", json={"query": "allocate_gpu_cache", "k": 1}).json()
    _write_benchmark(config, top["results"][0]["path"])

    payload = client.get("/api/dataset").json()
    assert payload["n_questions"] == 1
    assert payload["questions"][0]["id"] == "ui-001"
    assert "code_lookup" in payload["categories"]

    # `source` is matched as a family prefix, and filters actually filter
    assert client.get("/api/dataset?source=curated").json()["n_matching"] == 1
    assert client.get("/api/dataset?source=auto").json()["n_matching"] == 0
    assert client.get("/api/dataset?q=nothing matches").json()["n_matching"] == 0


def test_evaluate_endpoint_scores_a_question(client, config, monkeypatch):
    """Live evaluation scores one question against its gold labels."""
    monkeypatch.setattr(Pipeline, "llm", property(lambda self: EchoClient()))
    top = client.post("/api/search", json={"query": "allocate_gpu_cache", "k": 1}).json()
    gold = top["results"][0]["path"]
    _write_benchmark(config, gold)

    response = client.post(
        "/api/evaluate",
        json={"question_id": "ui-001", "retriever": "bm25", "reranker": False},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval"]["recall_at"]["10"] == 1.0
    assert payload["gold_found"] == [gold]
    assert not payload["gold_missed"]
    assert any(row["is_gold"] for row in payload["evidence"])
    assert "answer" in payload and "citation_metrics" in payload

    # retrieval-only mode skips the LLM but still scores
    retrieval_only = client.post(
        "/api/evaluate",
        json={"question_id": "ui-001", "retriever": "bm25", "generate": False},
    ).json()
    assert retrieval_only["retrieval"]["n_gold"] >= 1
    assert "answer" not in retrieval_only

    assert client.post("/api/evaluate", json={"question_id": "nope"}).status_code == 404
