"""Watch a folder and index screenshots as they appear.

macOS first writes a screenshot to a hidden temporary file (".Capture d'écran …"),
then renames it. So hidden files are ignored, renames count as new files, and a
file is only indexed once its size has stopped changing for a moment.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from shotsearch.config import IMAGE_EXTENSIONS
from shotsearch.indexer import Indexer, IndexReport, is_screenshot

log = logging.getLogger(__name__)


def is_candidate(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")


class PendingFiles:
    """Files seen recently. One is ready once its size is stable for `settle` seconds."""

    def __init__(self, settle: float = 1.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.settle = settle
        self._clock = clock
        self._lock = threading.Lock()
        self._files: dict[Path, tuple[int, float]] = {}  # path -> (last size, since)

    def touch(self, path: Path) -> None:
        with self._lock:
            self._files[path] = (-1, self._clock())

    def discard(self, path: Path) -> None:
        with self._lock:
            self._files.pop(path, None)

    def __len__(self) -> int:
        return len(self._files)

    def ready(self) -> list[Path]:
        now = self._clock()
        done = []
        with self._lock:
            for path, (last_size, since) in list(self._files.items()):
                try:
                    size = path.stat().st_size
                except FileNotFoundError:
                    del self._files[path]
                    continue
                if size != last_size or size == 0:
                    self._files[path] = (size, now)
                elif now - since >= self.settle:
                    del self._files[path]
                    done.append(path)
        return sorted(done)


class _Handler(FileSystemEventHandler):
    def __init__(self, pending: PendingFiles) -> None:
        self.pending = pending
        self._lock = threading.Lock()
        self._deleted: set[Path] = set()

    def take_deleted(self) -> list[Path]:
        with self._lock:
            deleted, self._deleted = sorted(self._deleted), set()
        return deleted

    def _gone(self, path: Path) -> None:
        self.pending.discard(path)
        if is_candidate(path):
            with self._lock:
                self._deleted.add(path)

    def _seen(self, path: Path) -> None:
        if is_candidate(path):
            with self._lock:
                self._deleted.discard(path)
            self.pending.touch(path)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._seen(Path(str(event.src_path)))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._seen(Path(str(event.src_path)))

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._gone(Path(str(event.src_path)))
            self._seen(Path(str(event.dest_path)))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._gone(Path(str(event.src_path)))


def watch(
    folder: Path,
    indexer: Indexer,
    *,
    only_screenshots: bool = False,
    settle: float = 1.0,
    interval: float = 0.25,
    stop: threading.Event | None = None,
    on_ready: Callable[[], None] | None = None,
    on_indexed: Callable[[list[Path], IndexReport], None] | None = None,
    on_removed: Callable[[list[Path]], None] | None = None,
) -> None:
    """Block until `stop` is set (or KeyboardInterrupt), indexing changes as they happen.

    The database is only touched from the calling thread.
    """
    folder = folder.expanduser().resolve()
    stop = stop or threading.Event()
    pending = PendingFiles(settle)
    handler = _Handler(pending)
    observer = Observer()
    observer.schedule(handler, str(folder), recursive=False)
    observer.start()
    if on_ready:
        on_ready()
    try:
        while not stop.wait(interval):
            gone = [p.resolve() for p in handler.take_deleted() if not p.exists()]
            if gone and indexer.db.delete_paths([str(p) for p in gone]) and on_removed:
                on_removed(gone)

            ready = [p.resolve() for p in pending.ready()]
            if only_screenshots:
                ready = [p for p in ready if is_screenshot(p)]
            if ready:
                report = indexer.index_paths(ready)
                if on_indexed:
                    on_indexed(ready, report)
    finally:
        observer.stop()
        observer.join()
