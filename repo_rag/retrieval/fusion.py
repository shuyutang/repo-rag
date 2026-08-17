"""Result fusion (PRD §12): Reciprocal Rank Fusion and weighted score sum."""

from __future__ import annotations

from collections import defaultdict

from ..schema import RetrievedChunk


def reciprocal_rank_fusion(
    result_sets: dict[str, list[RetrievedChunk]],
    *,
    k: int = 10,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists by `weight / (rrf_k + rank)`, summed over sources.

    RRF is rank-based, so it needs no score normalisation between a cosine
    similarity and a BM25 score -- the reason it is the default here.

    Args:
      result_sets: Source name to its ranked results.
      k: Maximum fused results.
      rrf_k: Rank offset. Small values weight the top ranks steeply; the
        tuned default for this corpus is 10, far below the customary 60.
      weights: Per-source weight; sources default to 1.0.

    Returns:
      Fused results ranked best first. Each carries every contributing
      source's raw score *and* rank in `component_scores`, so a ranking can
      be explained after the fact.
    """
    weights = weights or {}
    fused: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievedChunk] = {}
    components: dict[str, dict[str, float]] = defaultdict(dict)

    for source, results in result_sets.items():
        weight = weights.get(source, 1.0)
        for rank, item in enumerate(results, start=1):
            cid = item.chunk_id
            fused[cid] += weight / (rrf_k + rank)
            components[cid][source] = round(item.score, 6)
            components[cid][f"{source}_rank"] = rank
            if cid not in best or item.score > best[cid].score:
                best[cid] = item

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    out: list[RetrievedChunk] = []
    for rank, (cid, score) in enumerate(ordered, start=1):
        source = best[cid]
        out.append(
            RetrievedChunk(
                chunk=source.chunk,
                score=float(score),
                retriever="hybrid",
                rank=rank,
                component_scores={**components[cid], "rrf": round(float(score), 6)},
                query=source.query,
            )
        )
    return out


def _min_max(values: list[float]) -> list[float]:
    """Scale values to [0, 1].

    Args:
      values: Scores from one source.

    Returns:
      The scaled values; all 1.0 when the range is degenerate, since with no
      spread there is nothing to distinguish.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def weighted_score_fusion(
    result_sets: dict[str, list[RetrievedChunk]],
    *,
    k: int = 10,
    weights: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    """Min-max normalise each source's scores, then take a weighted sum.

    The score-based alternative to RRF. It keeps score *margins* that RRF
    discards, at the cost of depending on per-query score ranges, which makes
    the weights harder to transfer between repositories.

    Args:
      result_sets: Source name to its ranked results.
      k: Maximum fused results.
      weights: Per-source weight; sources default to 1.0.

    Returns:
      Fused results ranked best first.
    """
    weights = weights or {}
    fused: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievedChunk] = {}
    components: dict[str, dict[str, float]] = defaultdict(dict)

    for source, results in result_sets.items():
        weight = weights.get(source, 1.0)
        normalised = _min_max([r.score for r in results])
        for item, value in zip(results, normalised):
            cid = item.chunk_id
            fused[cid] += weight * value
            components[cid][source] = round(item.score, 6)
            if cid not in best or item.score > best[cid].score:
                best[cid] = item

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    return [
        RetrievedChunk(
            chunk=best[cid].chunk,
            score=float(score),
            retriever="hybrid",
            rank=rank,
            component_scores={**components[cid], "fused": round(float(score), 6)},
            query=best[cid].query,
        )
        for rank, (cid, score) in enumerate(ordered, start=1)
    ]


FUSION_METHODS = {
    "rrf": reciprocal_rank_fusion,
    "score_sum": weighted_score_fusion,
}


def fuse(
    result_sets: dict[str, list[RetrievedChunk]],
    *,
    method: str = "rrf",
    k: int = 10,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse result sets with the named method.

    Args:
      result_sets: Source name to its ranked results.
      method: A key of `FUSION_METHODS`.
      k: Maximum fused results.
      rrf_k: Rank offset, used by RRF only.
      weights: Per-source weight.

    Returns:
      Fused results ranked best first.

    Raises:
      ValueError: `method` is not a known fusion method.
    """
    if method == "rrf":
        return reciprocal_rank_fusion(result_sets, k=k, rrf_k=rrf_k, weights=weights)
    if method == "score_sum":
        return weighted_score_fusion(result_sets, k=k, weights=weights)
    raise ValueError(f"unknown fusion method {method!r}")
