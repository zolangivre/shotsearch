from __future__ import annotations

import numpy as np
import pytest

from shotsearch.db import Database, ImageRecord
from shotsearch.search import FTS, IMAGE_VEC, TEXT_VEC, rrf, search

# -- rrf() -------------------------------------------------------------------


def test_single_ranking_keeps_its_order() -> None:
    assert [i for i, _ in rrf({"a": [3, 1, 2]})] == [3, 1, 2]


def test_score_is_sum_of_reciprocal_ranks() -> None:
    fused = dict(rrf({"a": [1, 2], "b": [2]}, k=60))
    assert fused[1] == pytest.approx(1 / 61)
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)


def test_agreement_beats_a_single_first_place() -> None:
    # 7 is first in one list only; 8 is second in all three.
    fused = rrf({"a": [7, 8], "b": [9, 8], "c": [10, 8]})
    assert fused[0][0] == 8


def test_empty_and_missing_signals() -> None:
    assert rrf({}) == []
    assert rrf({"a": [], "b": []}) == []
    assert [i for i, _ in rrf({"a": [], "b": [5, 6]})] == [5, 6]


def test_ties_are_deterministic() -> None:
    # Same rank in two different lists: identical scores, ordered by id.
    assert [i for i, _ in rrf({"a": [9], "b": [4]})] == [4, 9]


def test_weights_scale_each_signal() -> None:
    fused = dict(rrf({"a": [1], "b": [2]}, k=60, weights={"a": 1.0, "b": 0.5}))
    assert fused[1] == pytest.approx(1 / 61)
    assert fused[2] == pytest.approx(0.5 / 61)
    # Signals missing from the weights count fully.
    assert dict(rrf({"c": [3]}, weights={"a": 0.1}))[3] == pytest.approx(1 / 61)


def test_small_k_rewards_top_ranks_more() -> None:
    rankings = {"a": [1, 2, 3], "b": [3, 2, 1], "c": [1, 3, 2]}
    # With a tiny k, 1 (two first places) clearly wins; with a huge k, ranks barely matter.
    small = dict(rrf(rankings, k=1))
    large = dict(rrf(rankings, k=10_000))
    assert small[1] > small[2]
    assert large[1] - large[2] < small[1] - small[2]


# -- search() ----------------------------------------------------------------


def unit(values: list[float]) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeQueryEmbedder:
    """Queries are mapped to fixed directions so each signal is predictable."""

    def __init__(self, text: list[float], image: list[float]) -> None:
        self.text, self.image = unit(text), unit(image)

    def embed_query(self, query: str) -> np.ndarray:
        return self.text

    def embed_image_query(self, query: str) -> np.ndarray:
        return self.image

    def image_match_probability(self, cosine: float) -> float:
        return max(cosine, 0.0)


@pytest.fixture
def corpus() -> tuple[Database, dict[str, int]]:
    db = Database(":memory:", text_dim=2, image_dim=2)
    db.reset_vectors("t", 2, "i", 2)
    ids = {}
    for name, text, text_vec, image_vec in [
        ("build", "Build failed with 1 error", [1, 0], [1, 0]),
        ("invoice", "Facture montant total", [0, 1], [0, 1]),
        ("sunset", "", None, [0.2, 1]),
    ]:
        ids[name] = db.upsert(ImageRecord(f"/{name}.png", 1, 1, "s", 1, 1, text))
        chunks = [text] if text else []
        vectors = [unit(text_vec)] if text_vec else []
        db.store_vectors(ids[name], chunks, vectors, unit(image_vec))
    return db, ids


def test_fts_only_without_embedder(corpus) -> None:
    db, ids = corpus
    results = search(db, "build", 5)
    assert [r.image_id for r in results] == [ids["build"]]
    assert set(results[0].hits) == {FTS}


def test_semantic_match_without_shared_words(corpus) -> None:
    db, ids = corpus
    # "erreur" matches no OCR word, but both vector signals point at the build screenshot.
    results = search(db, "erreur", 5, FakeQueryEmbedder(text=[1, 0.1], image=[1, 0.2]))
    assert results[0].image_id == ids["build"]
    assert FTS not in results[0].hits
    assert results[0].hits[TEXT_VEC].rank == 1
    assert results[0].hits[IMAGE_VEC].rank == 1
    assert results[0].snippet == "Build failed with 1 error"


def test_image_only_match_still_ranks(corpus) -> None:
    db, ids = corpus
    results = search(db, "coucher de soleil", 5, FakeQueryEmbedder(text=[0, 1], image=[0.2, 1]))
    sunset = next(r for r in results if r.image_id == ids["sunset"])
    assert sunset.hits[IMAGE_VEC].rank == 1
    assert TEXT_VEC not in sunset.hits  # no OCR text, so no text vector
    assert sunset.snippet == ""


def test_signals_agreeing_beat_one_strong_signal(corpus) -> None:
    db, ids = corpus
    # BM25 prefers "build", both vector signals prefer "invoice". depth=1 keeps only
    # each signal's top hit, as a large library would push weak matches out.
    results = search(db, "build", 5, FakeQueryEmbedder(text=[0, 1], image=[0, 1]), depth=1)
    assert results[0].image_id == ids["invoice"]
    assert results[1].image_id == ids["build"]


def test_unlikely_image_matches_are_dropped(corpus) -> None:
    db, ids = corpus
    embedder = FakeQueryEmbedder(text=[0, 1], image=[0.2, 1])
    # sunset scores 1.0, invoice ~0.98, build ~0.2: a high bar keeps only the first two.
    results = search(db, "coucher de soleil", 5, embedder, image_min_probability=0.9)
    with_image = {r.image_id for r in results if IMAGE_VEC in r.hits}
    assert with_image == {ids["sunset"], ids["invoice"]}


def test_confident_image_beats_weak_text_signal(corpus) -> None:
    db, ids = corpus
    # Text vectors rank invoice first; only the sunset clears the image threshold.
    embedder = FakeQueryEmbedder(text=[0, 1], image=[0.2, 1])
    results = search(db, "coucher de soleil", 5, embedder, image_min_probability=0.999)
    assert results[0].image_id == ids["sunset"]


def test_limit_and_highlighted_snippet(corpus) -> None:
    db, _ = corpus
    results = search(db, "build", 1, FakeQueryEmbedder(text=[1, 0], image=[1, 0]))
    assert len(results) == 1
    assert "\x02" in results[0].snippet


def test_query_of_only_stopwords_uses_vectors(corpus) -> None:
    db, ids = corpus
    results = search(db, "de la", 5, FakeQueryEmbedder(text=[0, 1], image=[0, 1]))
    assert results[0].image_id == ids["invoice"]
