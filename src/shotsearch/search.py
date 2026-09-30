"""Hybrid search: full-text, OCR-text vectors and image vectors, fused with RRF."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from shotsearch.db import Database

if TYPE_CHECKING:
    import numpy as np

RRF_K = 60

# Signal names, in display order.
FTS = "fts"
TEXT_VEC = "text"
IMAGE_VEC = "image"
SIGNALS = (FTS, TEXT_VEC, IMAGE_VEC)

# Weighted RRF. The vector signals return every image, relevant or not, and e5-small
# separates relevant from irrelevant OCR text by only ~0.01-0.04 cosine, so its
# ranking counts half. On a labelled set of 18 queries over real screenshots this
# moved top-1 from 12 to 14 without losing any query.
SIGNAL_WEIGHTS: Mapping[str, float] = {FTS: 1.0, TEXT_VEC: 0.5, IMAGE_VEC: 1.0}

# SigLIP's own calibrated match probability. Below this, an image is treated as
# unrelated to the query instead of taking a rank. Relevant matches seen in testing
# scored 0.045 and up; unrelated screenshots mostly scored below 0.001.
IMAGE_MIN_PROBABILITY = 0.01

_SNIPPET_CHARS = 180


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> np.ndarray: ...
    def embed_image_query(self, query: str) -> np.ndarray: ...
    def image_match_probability(self, cosine: float) -> float: ...


@dataclass
class Hit:
    """Where one image landed in one signal."""

    rank: int
    score: float  # BM25, e5 cosine, or SigLIP match probability


@dataclass
class SearchResult:
    image_id: int
    path: str
    score: float  # fused RRF score
    snippet: str
    hits: dict[str, Hit] = field(default_factory=dict)


def rrf(
    rankings: Mapping[str, Sequence[int]],
    k: int = RRF_K,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: sum of weight / (k + rank) over every list an item is in.

    Ties are broken by id so results are deterministic.
    """
    scores: dict[int, float] = defaultdict(float)
    for name, ranking in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        for rank, item in enumerate(ranking, start=1):
            scores[item] += weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def best_possible_score(signals: Sequence[str], k: int = RRF_K) -> float:
    """Fused score of an image ranked first by every given signal."""
    return sum(SIGNAL_WEIGHTS[name] for name in signals) / (k + 1)


def search(
    db: Database,
    query: str,
    limit: int = 5,
    embedder: QueryEmbedder | None = None,
    *,
    k: int = RRF_K,
    depth: int | None = None,
    image_min_probability: float = IMAGE_MIN_PROBABILITY,
) -> list[SearchResult]:
    """Rank images for a query. Without an embedder, only full-text search is used."""
    depth = depth or max(limit * 4, 30)
    hits: dict[str, dict[int, Hit]] = {name: {} for name in SIGNALS}
    snippets: dict[int, str] = {}
    chunks: dict[int, str] = {}

    for rank, (image_id, score, snippet) in enumerate(db.search_text(query, depth), 1):
        hits[FTS][image_id] = Hit(rank, score)
        snippets[image_id] = snippet

    if embedder is not None and db.vector_count() > 0:
        text_hits = db.search_text_vectors(embedder.embed_query(query), depth)
        for rank, (image_id, score, chunk) in enumerate(text_hits, 1):
            hits[TEXT_VEC][image_id] = Hit(rank, score)
            chunks[image_id] = chunk

        image_hits = db.search_image_vectors(embedder.embed_image_query(query), depth)
        probabilities = [
            (image_id, embedder.image_match_probability(cosine)) for image_id, cosine in image_hits
        ]
        kept = [(i, p) for i, p in probabilities if p >= image_min_probability]
        for rank, (image_id, probability) in enumerate(kept, 1):
            hits[IMAGE_VEC][image_id] = Hit(rank, probability)

    rankings = {
        name: sorted(by_id, key=lambda i, by_id=by_id: by_id[i].rank)
        for name, by_id in hits.items()
    }
    fused = rrf(rankings, k, SIGNAL_WEIGHTS)[:limit]
    rows = db.get_many([image_id for image_id, _ in fused])

    results = []
    for image_id, score in fused:
        row = rows.get(image_id)
        if row is None:
            continue
        snippet = snippets.get(image_id) or _shorten(chunks.get(image_id) or row["ocr_text"])
        results.append(
            SearchResult(
                image_id=image_id,
                path=row["path"],
                score=score,
                snippet=snippet,
                hits={name: hits[name][image_id] for name in SIGNALS if image_id in hits[name]},
            )
        )
    return results


def _shorten(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _SNIPPET_CHARS else text[:_SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
