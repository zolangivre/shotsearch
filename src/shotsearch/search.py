"""Query the index."""

from __future__ import annotations

from dataclasses import dataclass

from shotsearch.db import Database


@dataclass
class SearchResult:
    image_id: int
    path: str
    score: float
    snippet: str


def search(db: Database, query: str, limit: int = 5) -> list[SearchResult]:
    hits = db.search_text(query, limit)
    rows = db.get_many([image_id for image_id, _, _ in hits])
    return [
        SearchResult(image_id=image_id, path=rows[image_id]["path"], score=score, snippet=snippet)
        for image_id, score, snippet in hits
        if image_id in rows
    ]
