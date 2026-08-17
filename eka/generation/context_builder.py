"""Context construction under a token budget (PRD §15).

Rules implemented here:

* deduplicate overlapping chunks (same file, overlapping line ranges)
* prefer higher-ranked evidence
* keep provenance attached to every block
* allocate the budget across artifact types so one big source file cannot
  crowd out the docs/tests/commits that answer the "why"
* optionally pull in structurally related symbols for the top hit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..config import GenerationConfig
from ..indexing.graph_index import SymbolGraph
from ..indexing.store import ChunkStore
from ..schema import Chunk, RetrievedChunk


def estimate_tokens(text: str) -> int:
    """Estimate a token count without tokenising.

    Deliberately model-independent: the budget only has to be approximately
    right, and a real tokenizer would tie the context builder to whichever
    model happens to be configured.

    Args:
      text: Text to measure.

    Returns:
      An estimate at roughly 4 characters per token, never below 1.
    """
    return max(1, len(text) // 4)


@dataclass
class ContextBlock:
    """One piece of evidence as it will appear in the prompt.

    Attributes:
      item: The retrieval result this block came from.
      text: Block text, already truncated to fit.
      tokens: Estimated token count of `text`.
      reason: Why the block is here: "retrieved", or an explanation for a
        block pulled in from the symbol graph.
    """

    item: RetrievedChunk
    text: str
    tokens: int
    reason: str = "retrieved"

    @property
    def chunk(self) -> Chunk:
        """Chunk: The underlying chunk."""
        return self.item.chunk


@dataclass
class BuiltContext:
    """An assembled context window, plus what did not fit.

    Attributes:
      blocks: Evidence blocks in prompt order.
      dropped: Locations that were excluded, each with its reason -- kept so
        a thin answer can be traced to what was left out.
      total_tokens: Estimated total across all blocks.
    """

    blocks: list[ContextBlock] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    total_tokens: int = 0

    def render(self) -> str:
        """Render the blocks as the evidence section of the prompt.

        Returns:
          Delimited blocks, each header stating the exact bracket form to
          cite it by. Giving the model the citation string verbatim is what
          makes citation validation a check rather than a guess.
        """
        parts: list[str] = []
        for i, block in enumerate(self.blocks, start=1):
            chunk = block.chunk
            meta = [f"source {i}", chunk.path]
            if chunk.qualified_name:
                meta.append(chunk.qualified_name)
            if chunk.artifact_type == "commit":
                meta.append(f"commit {chunk.commit_sha[:10] if chunk.commit_sha else ''}")
                if chunk.timestamp:
                    meta.append(chunk.timestamp[:10])
            else:
                meta.append(f"lines {chunk.start_line}-{chunk.end_line}")
            parts.append(
                f"<<<EVIDENCE {i} | {' | '.join(meta)} | cite as [{chunk.location}]>>>\n"
                f"{block.text}\n<<<END EVIDENCE {i}>>>"
            )
        return "\n\n".join(parts)

    def citation_map(self) -> dict[str, RetrievedChunk]:
        """Map each block's citation location to the result behind it.

        Returns:
          A mapping used by citation validation to decide whether a citation
          the model emitted is backed by evidence.
        """
        return {b.chunk.location: b.item for b in self.blocks}


def _overlaps(a: Chunk, b: Chunk) -> bool:
    """Report whether two chunks cover overlapping lines of one file.

    Args:
      a: First chunk.
      b: Second chunk.

    Returns:
      True when both come from the same file and their line ranges
      intersect. Commit chunks never overlap, having no line range.
    """
    if a.path != b.path or a.artifact_type == "commit":
        return False
    return not (a.end_line < b.start_line or b.end_line < a.start_line)


class ContextBuilder:
    """Packs retrieved evidence into the context window.

    Applies, in order: overlap deduplication, a per-file cap, soft
    per-artifact-type quotas, and the overall token budget. The quotas are
    what stop one large source file from crowding out the doc or commit that
    answers the "why" half of a question.

    Attributes:
      config: Generation configuration: budget, caps and quotas.
      store: Chunk store, needed only to attach related symbols.
      graph: Symbol graph, needed only to attach related symbols.
    """

    def __init__(
        self,
        config: GenerationConfig,
        *,
        store: ChunkStore | None = None,
        graph: SymbolGraph | None = None,
    ) -> None:
        """Configure the builder.

        Args:
          config: Generation configuration.
          store: Chunk store; without it, related symbols are skipped.
          graph: Symbol graph; without it, related symbols are skipped.
        """
        self.config = config
        self.store = store
        self.graph = graph

    # ------------------------------------------------------------------
    def build(
        self,
        results: Sequence[RetrievedChunk],
        *,
        budget: int | None = None,
    ) -> BuiltContext:
        """Assemble a context window from ranked results.

        Quotas are enforced *softly*: a chunk over its type's quota is only
        dropped once the context is already 80% full. An under-filled context
        would otherwise be left half empty on principle, which helps nobody.

        Args:
          results: Retrieval results, best first. Order is respected, so
            reranking decides what survives the budget.
          budget: Token budget; defaults to `context_token_budget`.

        Returns:
          The assembled context, with every exclusion recorded in `dropped`.
        """
        budget = budget or self.config.context_token_budget
        context = BuiltContext()
        used_by_type: dict[str, int] = {}
        per_file: dict[str, int] = {}
        kept: list[Chunk] = []

        quotas = {
            k: int(budget * v) for k, v in self.config.artifact_quota.items()
        }

        for item in results:
            chunk = item.chunk
            if any(_overlaps(chunk, other) for other in kept):
                context.dropped.append(f"{chunk.location} (overlaps kept evidence)")
                continue
            if per_file.get(chunk.path, 0) >= self.config.per_file_chunk_cap:
                context.dropped.append(f"{chunk.location} (per-file cap)")
                continue

            text = self._truncate(chunk)
            tokens = estimate_tokens(text)
            atype = chunk.artifact_type
            quota = quotas.get(atype)
            if quota is not None and used_by_type.get(atype, 0) + tokens > quota:
                # quotas are soft: only enforced while other types still fit
                if context.total_tokens + tokens > budget * 0.8:
                    context.dropped.append(f"{chunk.location} ({atype} quota)")
                    continue
            if context.total_tokens + tokens > budget:
                context.dropped.append(f"{chunk.location} (budget)")
                continue

            context.blocks.append(ContextBlock(item=item, text=text, tokens=tokens))
            context.total_tokens += tokens
            used_by_type[atype] = used_by_type.get(atype, 0) + tokens
            per_file[chunk.path] = per_file.get(chunk.path, 0) + 1
            kept.append(chunk)

        if self.config.include_related_symbols and self.store and self.graph:
            self._add_related(context, budget)
        return context

    # ------------------------------------------------------------------
    def _truncate(self, chunk: Chunk) -> str:
        """Truncate an oversized chunk, keeping its head and tail.

        Head and tail rather than a prefix, because a function's signature
        and its return statement are usually both load-bearing.

        Args:
          chunk: Chunk to render.

        Returns:
          The content unchanged when it fits, otherwise the first 70% and
          last 25% of the allowance with a marker naming the elided length.
        """
        max_chars = self.config.max_chunk_tokens * 4
        text = chunk.content
        if len(text) <= max_chars:
            return text
        head = text[: int(max_chars * 0.7)]
        tail = text[-int(max_chars * 0.25) :]
        return f"{head}\n... [{len(text) - len(head) - len(tail)} chars elided] ...\n{tail}"

    def _add_related(self, context: BuiltContext, budget: int) -> None:
        """Attach the definition of a symbol the top evidence calls.

        At most one such block per query, and only from a different file --
        this is a small precision aid, not a second retrieval pass.

        Args:
          context: Context to extend in place.
          budget: Token budget the addition must fit inside.
        """
        if not context.blocks or self.graph is None or self.store is None:
            return
        top = context.blocks[0].chunk
        present = {b.chunk.chunk_id for b in context.blocks}
        for ref in top.references[:20]:
            for chunk_id in self.graph.definitions.get(ref, [])[:1]:
                if chunk_id in present:
                    continue
                chunk = self.store.get(chunk_id)
                if chunk is None or chunk.path == top.path:
                    continue
                text = self._truncate(chunk)
                tokens = estimate_tokens(text)
                if context.total_tokens + tokens > budget:
                    return
                item = RetrievedChunk(
                    chunk=chunk, score=0.0, retriever="graph:related", rank=0
                )
                context.blocks.append(
                    ContextBlock(
                        item=item, text=text, tokens=tokens,
                        reason=f"definition of `{ref}` referenced by top evidence",
                    )
                )
                context.total_tokens += tokens
                present.add(chunk_id)
                return  # at most one related symbol per query
