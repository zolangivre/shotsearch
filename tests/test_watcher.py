from __future__ import annotations

import plistlib
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from shotsearch import indexer as indexer_mod
from shotsearch.db import Database
from shotsearch.indexer import Indexer, is_screenshot
from shotsearch.watcher import PendingFiles, is_candidate, watch

from .conftest import FIXTURES


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_candidates() -> None:
    assert is_candidate(Path("/s/Capture d\u2019écran 2026-09-30 à 10.00.00.png"))
    assert is_candidate(Path("/s/photo.HEIC"))
    assert not is_candidate(Path("/s/.Capture d\u2019écran 2026-09-30 à 10.00.00.png"))
    assert not is_candidate(Path("/s/Enregistrement.mov"))


def test_file_is_ready_once_size_is_stable(tmp_path: Path) -> None:
    clock = FakeClock()
    pending = PendingFiles(settle=1.0, clock=clock)
    f = tmp_path / "a.png"
    f.write_bytes(b"x" * 10)
    pending.touch(f)

    assert pending.ready() == []  # first look records the size
    clock.now = 0.5
    f.write_bytes(b"x" * 20)  # still being written
    assert pending.ready() == []
    clock.now = 1.2
    assert pending.ready() == []  # stable, but only for 0.7 s
    clock.now = 1.6
    assert pending.ready() == [f]
    assert len(pending) == 0


def test_empty_or_vanished_files_are_never_ready(tmp_path: Path) -> None:
    clock = FakeClock()
    pending = PendingFiles(settle=0.1, clock=clock)
    empty, gone = tmp_path / "empty.png", tmp_path / "gone.png"
    empty.touch()
    gone.write_bytes(b"x")
    pending.touch(empty)
    pending.touch(gone)
    gone.unlink()
    for clock.now in (0.0, 1.0, 2.0):
        assert pending.ready() == []
    assert len(pending) == 1  # the empty file is still waiting for content


def test_screenshot_flag_is_read_from_xattr(tmp_path: Path) -> None:
    f = tmp_path / "shot.png"
    shutil.copy(FIXTURES / "build_failed.png", f)
    assert not is_screenshot(f)
    value = plistlib.dumps(True, fmt=plistlib.FMT_BINARY).hex()
    subprocess.run(
        ["xattr", "-wx", "com.apple.metadata:kMDItemIsScreenCapture", value, str(f)], check=True
    )
    assert is_screenshot(f)


@pytest.fixture
def fake_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(indexer_mod.ocr, "extract_text", lambda image: "Build failed")


def _run_watch(folder: Path, actions, until) -> tuple[Database, list[str]]:
    """Watch `folder` while `actions` run in another thread; stop when `until(log)`."""
    db = Database(":memory:")
    log: list[str] = []
    stop = threading.Event()
    started = threading.Event()

    def record(kind: str):
        def callback(paths, *_):
            log.extend(f"{kind} {p.name}" for p in paths)
            if until(log):
                stop.set()

        return callback

    def drive() -> None:
        started.wait(5)
        time.sleep(0.3)  # let FSEvents settle
        actions()

    threading.Thread(target=drive, daemon=True).start()
    threading.Timer(15, stop.set).start()  # safety net
    watch(
        folder,
        Indexer(db),
        settle=0.3,
        interval=0.1,
        stop=stop,
        on_ready=started.set,
        on_indexed=record("+"),
        on_removed=record("-"),
    )
    return db, log


def test_watch_indexes_renamed_screenshot_and_removes_deleted(tmp_path: Path, fake_ocr) -> None:
    folder = (tmp_path / "shots").resolve()
    folder.mkdir()
    final = folder / "Capture d\u2019écran 2026-09-30 à 10.00.00.png"

    def actions() -> None:
        # What screencapture does: write a hidden temp file, then rename it.
        hidden = folder / ".Capture d\u2019écran 2026-09-30 à 10.00.00.png"
        shutil.copy(FIXTURES / "build_failed.png", hidden)
        hidden.rename(final)
        (folder / "notes.txt").write_text("ignored")
        time.sleep(1.5)
        final.unlink()

    db, log = _run_watch(folder, actions, until=lambda log: any(e.startswith("-") for e in log))
    assert log == [f"+ {final.name}", f"- {final.name}"]
    assert db.count() == 0
