from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from shotsearch import indexer as indexer_mod
from shotsearch.db import Database, ImageRecord
from shotsearch.embeddings import ModelSpec, chunk_text
from shotsearch.indexer import Indexer, ModelChangedError

from .conftest import FIXTURES


def unit(values: list[float]) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def text_vector(text: str, dim: int = 8) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
    return unit(np.random.default_rng(seed).normal(size=dim).tolist())


class FakeEmbedder:
    text_spec = ModelSpec("fake/text", 8)
    image_spec = ModelSpec("fake/image", 3)

    def __init__(self) -> None:
        self.images_seen = 0

    def embed_passages(self, texts):
        return np.stack([text_vector(t) for t in texts])

    def embed_images(self, images):
        self.images_seen += len(images)
        # Mean colour: a sunset is orange-ish, a terminal is dark grey.
        return np.stack([unit(np.asarray(im).reshape(-1, 3).mean(0) + 1e-3) for im in images])


def add(db: Database, path: str, text: str = "") -> int:
    return db.upsert(ImageRecord(path, 1, 1, "sha", 10, 10, text))


@pytest.fixture
def vdb() -> Database:
    database = Database(":memory:", text_dim=8, image_dim=3)
    database.reset_vectors("fake/text", 8, "fake/image", 3)
    yield database
    database.close()


def test_image_vector_search_orders_by_similarity(vdb: Database) -> None:
    red = add(vdb, "/red.png")
    blue = add(vdb, "/blue.png")
    vdb.store_vectors(red, [], [], unit([1, 0, 0]))
    vdb.store_vectors(blue, [], [], unit([0, 0, 1]))
    hits = vdb.search_image_vectors(unit([0.9, 0.1, 0]), 5)
    assert [h[0] for h in hits] == [red, blue]
    assert hits[0][1] == pytest.approx(0.994, abs=1e-3)


def test_text_vector_search_keeps_best_chunk_per_image(vdb: Database) -> None:
    a = add(vdb, "/a.png", "x")
    b = add(vdb, "/b.png", "y")
    vdb.store_vectors(
        a, ["alpha", "beta"], [text_vector("alpha"), text_vector("beta")], unit([1, 0, 0])
    )
    vdb.store_vectors(b, ["gamma"], [text_vector("gamma")], unit([0, 1, 0]))
    hits = vdb.search_text_vectors(text_vector("beta"), 5)
    assert hits[0][0] == a
    assert hits[0][2] == "beta"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len({h[0] for h in hits}) == len(hits)


def test_store_vectors_replaces_previous(vdb: Database) -> None:
    a = add(vdb, "/a.png")
    vdb.store_vectors(a, ["one", "two"], [text_vector("one"), text_vector("two")], unit([1, 0, 0]))
    vdb.store_vectors(a, ["three"], [text_vector("three")], unit([0, 1, 0]))
    assert vdb.conn.execute("SELECT COUNT(*) FROM text_chunks").fetchone()[0] == 1
    assert vdb.conn.execute("SELECT COUNT(*) FROM vec_text").fetchone()[0] == 1
    assert vdb.vector_count() == 1


def test_deleting_an_image_removes_its_vectors(vdb: Database) -> None:
    a = add(vdb, "/a.png")
    vdb.store_vectors(a, ["one"], [text_vector("one")], unit([1, 0, 0]))
    vdb.delete_paths(["/a.png"])
    for table in ("text_chunks", "vec_text", "vec_image"):
        assert vdb.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_ids_without_vectors(vdb: Database) -> None:
    a = add(vdb, "/a.png")
    b = add(vdb, "/b.png")
    vdb.store_vectors(a, [], [], unit([1, 0, 0]))
    assert vdb.ids_without_vectors() == [b]


def test_chunk_text() -> None:
    assert chunk_text("") == []
    assert chunk_text("a b c") == ["a b c"]
    words = [f"w{i}" for i in range(250)]
    chunks = chunk_text(" ".join(words), max_words=100, overlap=20)
    assert chunks[0].split()[0] == "w0"
    assert chunks[1].split()[0] == "w80"
    assert chunks[-1].split()[-1] == "w249"


# -- indexer with embeddings -------------------------------------------------


@pytest.fixture
def folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shots = tmp_path / "shots"
    shots.mkdir()
    for name in ("build_failed.png", "sunset.png"):
        shutil.copy(FIXTURES / name, shots / name)
    monkeypatch.setattr(indexer_mod.ocr, "extract_text", lambda image: "some words here")
    return shots


def test_indexing_stores_vectors(folder: Path) -> None:
    db = Database(":memory:", text_dim=8, image_dim=3)
    report = Indexer(db, FakeEmbedder()).index_folder(folder)
    assert report.added == 2
    assert db.vector_count() == 2
    assert db.vector_models() == ("fake/text", "fake/image")


def test_images_indexed_without_models_get_embedded_later(folder: Path) -> None:
    db = Database(":memory:", text_dim=8, image_dim=3)
    Indexer(db).index_folder(folder)
    assert db.vector_count() == 0

    embedder = FakeEmbedder()
    report = Indexer(db, embedder).index_folder(folder)
    assert (report.added, report.updated, report.embedded) == (0, 0, 2)
    assert db.vector_count() == 2

    report = Indexer(db, embedder).index_folder(folder)
    assert report.unchanged == 2
    assert embedder.images_seen == 2


def test_model_change_requires_reindex(folder: Path) -> None:
    db = Database(":memory:", text_dim=8, image_dim=3)
    Indexer(db, FakeEmbedder()).index_folder(folder)

    class Other(FakeEmbedder):
        image_spec = ModelSpec("fake/other-image", 4)

        def embed_images(self, images):
            return np.stack([unit([1, 2, 3, 4]) for _ in images])

    with pytest.raises(ModelChangedError):
        Indexer(db, Other()).index_folder(folder)

    report = Indexer(db, Other()).index_folder(folder, reindex=True)
    assert report.updated == 2
    assert db.vector_models() == ("fake/text", "fake/other-image")
    assert db.vector_dims() == (8, 4)
    assert db.vector_count() == 2
