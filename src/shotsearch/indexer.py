"""Incremental indexing of a screenshot folder."""

from __future__ import annotations

import hashlib
import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from shotsearch import ocr
from shotsearch.config import IMAGE_EXTENSIONS
from shotsearch.db import Database, ImageRecord

log = logging.getLogger(__name__)


@dataclass
class IndexReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: list[tuple[Path, str]] = field(default_factory=list)


def is_image_file(path: Path) -> bool:
    return (
        path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".") and path.is_file()
    )


def is_screenshot(path: Path) -> bool:
    """Ask Spotlight whether macOS flagged this file as a screen capture."""
    try:
        out = subprocess.run(
            ["mdls", "-raw", "-name", "kMDItemIsScreenCapture", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.stdout.strip() == "1"


def list_images(folder: Path, only_screenshots: bool = False) -> list[Path]:
    images = sorted(p for p in folder.iterdir() if is_image_file(p))
    if only_screenshots:
        images = [p for p in images if is_screenshot(p)]
    return images


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Indexer:
    def __init__(self, db: Database) -> None:
        self.db = db

    def needs_indexing(self, path: Path, reindex: bool = False) -> bool:
        """Cheap check. Size and mtime first, then the hash to catch mere touches."""
        row = self.db.get_by_path(str(path))
        if row is None or reindex:
            return True
        stat = path.stat()
        if row["size"] == stat.st_size and row["mtime_ns"] == stat.st_mtime_ns:
            return False
        if row["sha256"] == sha256_of(path):
            self.db.touch(row["id"], stat.st_size, stat.st_mtime_ns)
            return False
        return True

    def index_file(self, path: Path) -> bool:
        """OCR and store one image. Returns True if it was new, False if updated."""
        path = path.resolve()
        existed = self.db.get_by_path(str(path)) is not None
        stat = path.stat()
        image = ocr.load_image(path)
        text = ocr.extract_text(image)
        self.db.upsert(
            ImageRecord(
                path=str(path),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sha256=sha256_of(path),
                width=image.width,
                height=image.height,
                ocr_text=text,
            )
        )
        return not existed

    def remove_missing(self, folder: Path) -> int:
        gone = [p for p in self.db.paths_under(folder) if not Path(p).exists()]
        return self.db.delete_paths(gone) if gone else 0

    def index_folder(
        self,
        folder: Path,
        *,
        reindex: bool = False,
        only_screenshots: bool = False,
        on_start: Callable[[int], None] | None = None,
        on_file: Callable[[Path], None] | None = None,
    ) -> IndexReport:
        folder = folder.expanduser().resolve()
        report = IndexReport()
        report.removed = self.remove_missing(folder)

        todo: list[Path] = []
        for path in list_images(folder, only_screenshots):
            try:
                if self.needs_indexing(path, reindex):
                    todo.append(path)
                else:
                    report.unchanged += 1
            except OSError as exc:
                report.failed.append((path, str(exc)))

        if on_start:
            on_start(len(todo))
        for path in todo:
            try:
                if self.index_file(path):
                    report.added += 1
                else:
                    report.updated += 1
            except (ocr.UnreadableImageError, OSError) as exc:
                log.warning("skipping %s: %s", path, exc)
                report.failed.append((path, str(exc)))
            if on_file:
                on_file(path)
        return report
