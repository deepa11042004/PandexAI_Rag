"""Hybrid retrieval: combine ChromaDB vector similarity with a lightweight lexical overlap score,
then rerank. This is a heuristic reranker (no cross-encoder model) chosen to keep the install
footprint small - see retrieved_chunk.hybrid_score for the combination formula.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document

from backend.config import get_settings
from backend.vector_store import VectorStoreManager

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


@dataclass
class RetrievedChunk:
    document: Document
    vector_distance: float  # chroma cosine distance; lower = more similar
    lexical_overlap: float  # fraction of query terms present in the chunk; higher = more similar
    hybrid_score: float  # lower = better (same direction as vector_distance)

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")


def _hybrid_score(vector_distance: float, lexical_overlap: float) -> float:
    """Blend vector distance (0=identical) with lexical overlap (1=all query terms present).

    `1 - lexical_overlap` puts both terms on a "lower is better" scale so they combine linearly.
    """
    return 0.75 * vector_distance + 0.25 * (1 - lexical_overlap)


def vector_search(manager: VectorStoreManager, query: str, top_k: int | None = None) -> list[tuple[Document, float]]:
    """Step 1 of retrieval: pure semantic search against ChromaDB (LangGraph 'retrieve' node)."""
    top_k = top_k or get_settings().retrieval_top_k
    return manager.similarity_search_with_score(query, k=top_k)


def vector_search_per_source(
    manager: VectorStoreManager, query: str, source_ids: list[str], per_source_k: int = 4, top_k: int | None = None
) -> list[tuple[Document, float]]:
    """Like `vector_search`, but queries each source's chunks separately and combines the results.

    A single flat `similarity_search_with_score(query, k=top_k)` across the whole session ranks
    purely by vector distance with no notion of "one search per document" - a document with 100+
    chunks (e.g. a long report) has vastly more chances to land in the top-K than a 5-chunk resume,
    so it can fill the entire candidate pool before reranking even sees the other documents' chunks.
    Downstream diversification (`hybrid_rerank`) only helps if every source's chunks actually made
    it into the pool in the first place - this guarantees that regardless of chunk-count imbalance.

    Only kicks in for multi-document sessions; a single source has nothing to be starved by, so it
    falls back to one plain query (also avoids one needless Chroma round-trip).
    """
    if len(source_ids) <= 1:
        return vector_search(manager, query, top_k)

    seen: set[tuple[str, str]] = set()
    combined: list[tuple[Document, float]] = []
    for source_id in source_ids:
        for doc, distance in manager.similarity_search_with_score(query, k=per_source_k, source_id=source_id):
            key = (doc.metadata.get("source_id", ""), doc.page_content)
            if key not in seen:
                seen.add(key)
                combined.append((doc, distance))
    return combined


def hybrid_rerank(
    pairs: list[tuple[Document, float]], query: str, top_k: int | None = None, max_distance: float | None = None
) -> list[RetrievedChunk]:
    """Step 2 of retrieval: blend vector distance with lexical overlap and keep the best few
    (LangGraph 'rerank' node)."""
    settings = get_settings()
    top_k = top_k or settings.rerank_top_k
    max_distance = max_distance if max_distance is not None else settings.max_distance

    query_terms = _tokenize(query)
    candidates: list[RetrievedChunk] = []
    for doc, distance in pairs:
        chunk_terms = _tokenize(doc.page_content)
        overlap = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0.0
        candidates.append(
            RetrievedChunk(
                document=doc,
                vector_distance=distance,
                lexical_overlap=overlap,
                hybrid_score=_hybrid_score(distance, overlap),
            )
        )

    candidates.sort(key=lambda c: c.hybrid_score)
    filtered = [c for c in candidates if c.vector_distance <= max_distance]
    return _diversify_by_source(filtered, top_k)


def _diversify_by_source(ranked: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Guarantee every unique source gets a shot at a context slot before any source gets a second one.

    Plain top-K reranking has no notion of "one per document" - if several of a session's best-
    scoring chunks all happen to come from the same file, that file can silently crowd out every
    other uploaded document from the LLM's context. That's invisible for a question about one
    specific document, but breaks broad questions like "summarize each document", where the answer
    should draw from every source, not whichever one happened to score highest.
    """
    if top_k <= 0 or not ranked:
        return ranked[:top_k]

    picked: list[RetrievedChunk] = []
    seen_sources: set[str] = set()
    leftover: list[RetrievedChunk] = []
    for chunk in ranked:
        if chunk.source not in seen_sources:
            picked.append(chunk)
            seen_sources.add(chunk.source)
        else:
            leftover.append(chunk)

    return (picked + leftover)[:top_k]


def retrieve(manager: VectorStoreManager, query: str, top_k_initial: int | None = None) -> list[RetrievedChunk]:
    """Convenience wrapper: vector search then hybrid-rerank in one call."""
    pairs = vector_search(manager, query, top_k_initial)
    if not pairs:
        return []
    return hybrid_rerank(pairs, query)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Combine retrieved chunks into a single labeled context block for the LLM prompt."""
    blocks = [f"[Source {i}: {c.source}]\n{c.document.page_content}" for i, c in enumerate(chunks, start=1)]
    return "\n\n---\n\n".join(blocks)


def unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Ordered, de-duplicated list of source names used in a retrieval result."""
    seen: list[str] = []
    for chunk in chunks:
        if chunk.source not in seen:
            seen.append(chunk.source)
    return seen
