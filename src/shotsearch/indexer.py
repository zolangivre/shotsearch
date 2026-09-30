"""Incremental indexing of a screenshot folder."""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import plistlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from shotsearch import ocr
from shotsearch.config import IMAGE_EXTENSIONS
from shotsearch.db import Database, ImageRecord
from shotsearch.embeddings import ModelSpec, chunk_text

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

log = logging.getLogger(__name__)


class Embedder(Protocol):
    text_spec: ModelSpec
    image_spec: ModelSpec

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...


class ModelChangedError(RuntimeError):
    def __init__(self, indexed: tuple[str, str], current: tuple[str, str]) -> None:
        super().__init__(
            f"The index was built with {indexed[0]} + {indexed[1]}, but shotsearch now uses "
            f"{current[0]} + {current[1]}. Run `shotsearch index --reindex` to rebuild it."
        )


@dataclass
class IndexReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    embedded: int = 0  # unchanged files that only needed embeddings
    failed: list[tuple[Path, str]] = field(default_factory=list)


@dataclass
class _Job:
    path: Path
    ocr: bool  # False: content unchanged, only embeddings are missing


def is_image_file(path: Path) -> bool:
    return (
        path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".") and path.is_file()
    )


_SCREENSHOT_XATTR = b"com.apple.metadata:kMDItemIsScreenCapture"
_libc = ctypes.CDLL(None, use_errno=True)
_libc.getxattr.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_uint32,
    ctypes.c_int,
]
_libc.getxattr.restype = ctypes.c_ssize_t


def is_screenshot(path: Path) -> bool:
    """Whether macOS flagged this file as a screen capture.

    screencapture stores the flag as an extended attribute, which is what Spotlight
    reads too. Reading it directly avoids waiting for Spotlight on brand-new files.
    Python has no getxattr on macOS, hence ctypes.
    """
    name = os.fsencode(path)
    size = _libc.getxattr(name, _SCREENSHOT_XATTR, None, 0, 0, 0)
    if size <= 0:
        return False
    buf = ctypes.create_string_buffer(size)
    if _libc.getxattr(name, _SCREENSHOT_XATTR, buf, size, 0, 0) != size:
        return False
    try:
        return bool(plistlib.loads(buf.raw))
    except plistlib.InvalidFileException:
        return False


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
    def __init__(self, db: Database, embedder: Embedder | None = None, batch_size: int = 16):
        """Without an embedder, only OCR and full-text indexing are done."""
        self.db = db
        self.embedder = embedder
        self.batch_size = batch_size

    def prepare_vectors(self, reindex: bool = False) -> None:
        """Make sure the vector tables match the embedder's models."""
        if self.embedder is None:
            return
        text, image = self.embedder.text_spec, self.embedder.image_spec
        indexed = self.db.vector_models()
        if indexed is None or (reindex and indexed != (text.repo, image.repo)):
            self.db.reset_vectors(text.repo, text.dim, image.repo, image.dim)
        elif indexed != (text.repo, image.repo):
            raise ModelChangedError(indexed, (text.repo, image.repo))

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

    def remove_missing(self, folder: Path) -> int:
        gone = [p for p in self.db.paths_under(folder) if not Path(p).exists()]
        return self.db.delete_paths(gone) if gone else 0

    def index_paths(
        self,
        paths: Sequence[Path],
        *,
        reindex: bool = False,
        on_start: Callable[[int], None] | None = None,
        on_file: Callable[[Path], None] | None = None,
    ) -> IndexReport:
        self.prepare_vectors(reindex)
        report = IndexReport()
        missing_vectors = set(self.db.ids_without_vectors()) if self.embedder else set()

        jobs: list[_Job] = []
        for path in paths:
            path = path.resolve()
            try:
                if self.needs_indexing(path, reindex):
                    jobs.append(_Job(path, ocr=True))
                    continue
                row = self.db.get_by_path(str(path))
                if row["id"] in missing_vectors:
                    jobs.append(_Job(path, ocr=False))
                else:
                    report.unchanged += 1
            except OSError as exc:
                report.failed.append((path, str(exc)))

        if on_start:
            on_start(len(jobs))
        for batch in batched(jobs, self.batch_size):
            self._process_batch(batch, report, on_file)
        return report

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
        removed = self.remove_missing(folder)
        report = self.index_paths(
            list_images(folder, only_screenshots),
            reindex=reindex,
            on_start=on_start,
            on_file=on_file,
        )
        report.removed = removed
        return report

    def _process_batch(
        self,
        batch: Sequence[_Job],
        report: IndexReport,
        on_file: Callable[[Path], None] | None,
    ) -> None:
        done: list[tuple[_Job, int, Image.Image, str, bool]] = []
        for job in batch:
            try:
                done.append((job, *self._read(job)))
            except (ocr.UnreadableImageError, OSError) as exc:
                log.warning("skipping %s: %s", job.path, exc)
                report.failed.append((job.path, str(exc)))
                if on_file:
                    on_file(job.path)

        if self.embedder and done:
            self._embed([(image_id, image, text) for _, image_id, image, text, _ in done])

        for job, _, _, _, is_new in done:
            if not job.ocr:
                report.embedded += 1
            elif is_new:
                report.added += 1
            else:
                report.updated += 1
            if on_file:
                on_file(job.path)

    def _read(self, job: _Job) -> tuple[int, Image.Image, str, bool]:
        """Load the image and get its text, running OCR only when needed."""
        image = ocr.load_image(job.path)
        row = self.db.get_by_path(str(job.path))
        if not job.ocr and row is not None:
            return row["id"], image, row["ocr_text"], False

        stat = job.path.stat()
        text = ocr.extract_text(image)
        image_id = self.db.upsert(
            ImageRecord(
                path=str(job.path),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sha256=sha256_of(job.path),
                width=image.width,
                height=image.height,
                ocr_text=text,
            )
        )
        return image_id, image, text, row is None

    def _embed(self, items: list[tuple[int, Image.Image, str]]) -> None:
        assert self.embedder is not None
        image_vectors = self.embedder.embed_images([image for _, image, _ in items])
        chunks = [chunk_text(text) for _, _, text in items]
        flat = [chunk for per_image in chunks for chunk in per_image]
        chunk_vectors = self.embedder.embed_passages(flat) if flat else []

        offset = 0
        for (image_id, _, _), image_vector, image_chunks in zip(
            items, image_vectors, chunks, strict=True
        ):
            vectors = chunk_vectors[offset : offset + len(image_chunks)]
            offset += len(image_chunks)
            self.db.store_vectors(image_id, image_chunks, vectors, image_vector)
