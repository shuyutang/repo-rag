"""Fusion-weight sweep on the dev split (PRD §34).

Hybrid retrieval only earns its place if it beats its own components.  On the
first run it did not — BM25 alone outscored the RRF fusion, because the dense
leg contributes many confidently-wrong candidates near the top of its ranking.

This script sweeps the RRF weights and the rank constant on the *dev* split and
writes the winning configuration to configs/default.yaml.  Test-split numbers
are never consulted here.

Example:
  python scripts/tune_fusion.py --out configs/default.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import yaml

from eka.config import REPO_ROOT, Config
from eka.evaluation.dataset import load_dataset
from eka.evaluation.retrieval_metrics import aggregate, evaluate_question
from eka.indexing.knowledge_base import KnowledgeBase
from eka.retrieval.dense import DenseRetriever
from eka.retrieval.fusion import fuse
from eka.retrieval.git import GitRetriever
from eka.retrieval.sparse import BM25Retriever
from eka.retrieval.symbol import SymbolRetriever


def main() -> None:
    """Sweep fusion weights on the dev split and report the winner.

    Retrieval runs once per question and is then re-fused offline for every
    grid point, so the sweep costs one retrieval pass rather than 108 of
    them. The objective is `recall@10 + mrr`.

    Writes the full sweep to `--results`, and the winning weights back into
    the config named by `--out` when one is given.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--dataset", default=str(REPO_ROOT / "evaluation_data" / "benchmark.jsonl"))
    parser.add_argument("--out", default="")
    parser.add_argument("--results", default=str(REPO_ROOT / "results" / "fusion_sweep.json"))
    args = parser.parse_args()

    config = Config.load(args.config)
    kb = KnowledgeBase.load(config)
    questions = [q for q in load_dataset(Path(args.dataset)) if q.split == "dev"]
    print(f"{len(questions)} dev questions")

    retrievers = {
        "dense": DenseRetriever(kb),
        "bm25": BM25Retriever(kb),
        "symbol": SymbolRetriever(kb),
        "git": GitRetriever(kb),
    }
    depth = {"dense": config.retrieval.dense_k, "bm25": config.retrieval.bm25_k,
             "symbol": config.retrieval.symbol_k, "git": config.retrieval.git_k}

    # retrieve once per question per source, then re-fuse offline
    cached: list[tuple] = []
    for question in questions:
        per_source = {
            name: retriever.retrieve(question.question, depth[name])
            for name, retriever in retrievers.items()
        }
        cached.append((question, per_source))

    grid = {
        "dense": [0.3, 0.5, 0.7, 1.0],
        "bm25": [1.0],                      # reference leg, held at 1.0
        "symbol": [0.3, 0.6, 1.0],
        "git": [0.2, 0.4, 0.6],
        "rrf_k": [10, 30, 60],
    }
    rows = []
    for dense_w, symbol_w, git_w, rrf_k in itertools.product(
        grid["dense"], grid["symbol"], grid["git"], grid["rrf_k"]
    ):
        weights = {"dense": dense_w, "bm25": 1.0, "symbol": symbol_w, "git": git_w}
        metrics = []
        for question, per_source in cached:
            fused = fuse(per_source, method="rrf", k=20, rrf_k=rrf_k, weights=weights)
            metrics.append(evaluate_question(question, fused))
        summary = aggregate(metrics)
        rows.append(
            {
                "weights": weights,
                "rrf_k": rrf_k,
                "recall@10": summary["recall@10"],
                "mrr": summary["mrr"],
                "ndcg@10": summary["ndcg@10"],
                "objective": round(summary["recall@10"] + summary["mrr"], 4),
            }
        )
        print(
            f"dense={dense_w} symbol={symbol_w} git={git_w} rrf_k={rrf_k} "
            f"-> recall@10={summary['recall@10']:.4f} mrr={summary['mrr']:.4f}"
        )

    rows.sort(key=lambda r: -r["objective"])
    best = rows[0]
    print("\nbest on dev:", json.dumps(best, indent=2))

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(
        json.dumps({"n_dev": len(questions), "grid": grid, "rows": rows}, indent=2)
    )

    if args.out:
        raw = yaml.safe_load(Path(args.out).read_text())
        raw["retrieval"]["fusion_weights"] = best["weights"]
        raw["retrieval"]["rrf_k"] = best["rrf_k"]
        Path(args.out).write_text(yaml.safe_dump(raw, sort_keys=False))
        print(f"wrote winning fusion configuration to {args.out}")


if __name__ == "__main__":
    main()
