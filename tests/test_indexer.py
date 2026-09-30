from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from shotsearch import indexer as indexer_mod
from shotsearch import ocr
from shotsearch.db import Database
from shotsearch.indexer import Indexer

from .conftest import FIXTURES


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    shots = tmp_path / "shots"
    shots.mkdir()
    shutil.copy(FIXTURES / "build_failed.png", shots / "a.png")
    shutil.copy(FIXTURES / "facture.png", shots / "b.png")
    (shots / "notes.txt").write_text("not an image")
    (shots / ".hidden.png").write_bytes(b"temp file being written")
    return shots


@pytest.fixture
def fake_ocr(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Skip Vision; record which files were actually processed."""
    calls: list[Path] = []
    real_load = ocr.load_image

    def load(path: Path):
        calls.append(path)
        return real_load(path)

    monkeypatch.setattr(indexer_mod.ocr, "load_image", load)
    monkeypatch.setattr(indexer_mod.ocr, "extract_text", lambda image: "fake text")
    return calls


def test_only_images_are_listed(folder: Path) -> None:
    assert [p.name for p in indexer_mod.list_images(folder)] == ["a.png", "b.png"]


def test_second_run_skips_unchanged(db: Database, folder: Path, fake_ocr: list[Path]) -> None:
    first = Indexer(db).index_folder(folder)
    assert (first.added, first.unchanged) == (2, 0)

    fake_ocr.clear()
    second = Indexer(db).index_folder(folder)
    assert (second.added, second.updated, second.unchanged) == (0, 0, 2)
    assert fake_ocr == []


def test_touched_file_is_not_reprocessed(db: Database, folder: Path, fake_ocr: list[Path]) -> None:
    Indexer(db).index_folder(folder)
    fake_ocr.clear()
    os.utime(folder / "a.png", (1_000_000, 1_000_000))
    report = Indexer(db).index_folder(folder)
    assert report.unchanged == 2
    assert fake_ocr == []
    # The new mtime is recorded so the hash isn't recomputed next time.
    row = db.get_by_path(str((folder / "a.png").resolve()))
    assert row["mtime_ns"] == (folder / "a.png").stat().st_mtime_ns


def test_modified_file_is_reprocessed(db: Database, folder: Path, fake_ocr: list[Path]) -> None:
    Indexer(db).index_folder(folder)
    shutil.copy(FIXTURES / "sunset.png", folder / "a.png")
    report = Indexer(db).index_folder(folder)
    assert (report.updated, report.unchanged) == (1, 1)


def test_reindex_processes_everything(db: Database, folder: Path, fake_ocr: list[Path]) -> None:
    Indexer(db).index_folder(folder)
    report = Indexer(db).index_folder(folder, reindex=True)
    assert report.updated == 2


def test_deleted_file_is_removed(db: Database, folder: Path, fake_ocr: list[Path]) -> None:
    Indexer(db).index_folder(folder)
    (folder / "b.png").unlink()
    report = Indexer(db).index_folder(folder)
    assert report.removed == 1
    assert db.count() == 1


def test_corrupt_image_is_reported_not_fatal(
    db: Database, folder: Path, fake_ocr: list[Path]
) -> None:
    (folder / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n garbage")
    report = Indexer(db).index_folder(folder)
    assert report.added == 2
    assert [p.name for p, _ in report.failed] == ["broken.png"]


def test_real_ocr_finds_build_failed(db: Database, folder: Path) -> None:
    Indexer(db).index_folder(folder)
    row = db.get_by_path(str((folder / "a.png").resolve()))
    assert "Build failed" in row["ocr_text"]


@pytest.mark.parametrize("name", ["meeting_notes.jpg", "recette.heic"])
def test_non_png_formats(name: str) -> None:
    text = ocr.extract_text(ocr.load_image(FIXTURES / name))
    assert text.strip()
