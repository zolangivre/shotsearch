"""Checks against the real models. Run with `uv run pytest -m slow`."""

from __future__ import annotations

import numpy as np
import pytest

from shotsearch import ocr
from shotsearch.embeddings import Embedder

from .conftest import FIXTURES

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def embedder(real_models_dir) -> Embedder:
    return Embedder()


def test_e5_matches_across_languages(embedder: Embedder) -> None:
    names = ["build_failed.png", "facture.png", "meeting_notes.jpg"]
    texts = [ocr.extract_text(ocr.load_image(FIXTURES / n)) for n in names]
    passages = embedder.embed_passages(texts)
    assert passages.shape == (3, 384)
    assert np.allclose(np.linalg.norm(passages, axis=1), 1, atol=1e-4)

    # Cross-lingual margins are thin with e5-small (~0.01-0.04): hybrid ranking helps.
    for query, expected in [
        ("erreur de build", 0),
        ("invoice due date", 1),
        ("bank transfer payment", 1),
        ("réunion jeudi", 2),
    ]:
        scores = passages @ embedder.embed_query(query)
        assert int(scores.argmax()) == expected, (query, scores)


def test_siglip_finds_image_without_text(embedder: Embedder) -> None:
    names = ["build_failed.png", "facture.png", "sunset.png", "meeting_notes.jpg"]
    images = embedder.embed_images([ocr.load_image(FIXTURES / n) for n in names])
    assert images.shape == (4, 768)
    for query in ["coucher de soleil sur la mer", "sunset over the ocean"]:
        scores = images @ embedder.embed_image_query(query)
        assert names[int(scores.argmax())] == "sunset.png", (query, scores)
