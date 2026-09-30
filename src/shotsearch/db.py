"""SQLite storage: image metadata, OCR text (FTS5) and embeddings (sqlite-vec)."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlite_vec

SCHEMA_VERSION = "2"

# Matches are wrapped in these control characters; the CLI turns them into styling.
HIGHLIGHT_START = "\x02"
HIGHLIGHT_END = "\x03"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id         INTEGER PRIMARY KEY,
    path       TEXT NOT NULL UNIQUE,
    size       INTEGER NOT NULL,
    mtime_ns   INTEGER NOT NULL,
    sha256     TEXT NOT NULL,
    width      INTEGER,
    height     INTEGER,
    ocr_text   TEXT NOT NULL DEFAULT '',
    indexed_at REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
    ocr_text,
    content='images',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
    INSERT INTO images_fts(rowid, ocr_text) VALUES (new.id, new.ocr_text);
END;

CREATE TRIGGER IF NOT EXISTS images_au AFTER UPDATE OF ocr_text ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, ocr_text) VALUES ('delete', old.id, old.ocr_text);
    INSERT INTO images_fts(rowid, ocr_text) VALUES (new.id, new.ocr_text);
END;

CREATE TABLE IF NOT EXISTS text_chunks (
    id       INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id),
    chunk    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS text_chunks_image ON text_chunks(image_id);
"""

# Vector tables depend on the model dimensions, so they're created separately.
_VECTOR_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS vec_text USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding float[{text_dim}]
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_image USING vec0(
    image_id INTEGER PRIMARY KEY,
    embedding float[{image_dim}]
);

CREATE TRIGGER IF NOT EXISTS images_bd BEFORE DELETE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, ocr_text) VALUES ('delete', old.id, old.ocr_text);
    DELETE FROM vec_image WHERE image_id = old.id;
    DELETE FROM vec_text WHERE chunk_id IN (SELECT id FROM text_chunks WHERE image_id = old.id);
    DELETE FROM text_chunks WHERE image_id = old.id;
END;
"""

# Very common French/English words that only add noise to an OR query.
_STOPWORDS = frozenset(
    (
        "le la les un une des du de d l et ou en au aux a à pour par sur dans avec ce ces"
        " est sont the of and or to in on at for with is are an"
    ).split()
)


@dataclass(frozen=True)
class ImageRecord:
    path: str
    size: int
    mtime_ns: int
    sha256: str
    width: int | None
    height: int | None
    ocr_text: str


def fts_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 query: quoted prefix terms joined by OR."""
    words = [w for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS]
    if not words:
        return None
    return " OR ".join(f'"{w}"*' for w in dict.fromkeys(words))


def _blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _similarity(l2_distance: float) -> float:
    """Cosine similarity from the L2 distance between two unit vectors."""
    return 1.0 - (l2_distance**2) / 2.0


class Database:
    def __init__(self, path: Path | str, text_dim: int = 384, image_dim: int = 768) -> None:
        self.path = Path(path) if path != ":memory:" else None
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path is not None:
            self.conn.execute("PRAGMA journal_mode = WAL")
        # v1's AFTER DELETE trigger only cleaned FTS; images_bd replaces it.
        self.conn.execute("DROP TRIGGER IF EXISTS images_ad")
        self.conn.executescript(_SCHEMA)
        dims = self.vector_dims()
        self.conn.executescript(
            _VECTOR_SCHEMA.format(
                text_dim=dims[0] if dims else text_dim, image_dim=dims[1] if dims else image_dim
            )
        )
        self.set_meta("schema_version", SCHEMA_VERSION)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- meta ---------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def get_meta_safe(self, key: str) -> str | None:
        """get_meta that also works before the schema exists."""
        try:
            return self.get_meta(key)
        except sqlite3.OperationalError:
            return None

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # -- images -------------------------------------------------------------

    def get_by_path(self, path: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM images WHERE path = ?", (path,)).fetchone()

    def get_many(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(f"SELECT * FROM images WHERE id IN ({marks})", ids)
        return {row["id"]: row for row in rows}

    def upsert(self, record: ImageRecord) -> int:
        with self.conn:
            row = self.conn.execute(
                """
                INSERT INTO images (path, size, mtime_ns, sha256, width, height, ocr_text,
                                    indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    size = excluded.size, mtime_ns = excluded.mtime_ns,
                    sha256 = excluded.sha256, width = excluded.width,
                    height = excluded.height, ocr_text = excluded.ocr_text,
                    indexed_at = excluded.indexed_at
                RETURNING id
                """,
                (
                    record.path,
                    record.size,
                    record.mtime_ns,
                    record.sha256,
                    record.width,
                    record.height,
                    record.ocr_text,
                    time.time(),
                ),
            ).fetchone()
        return row["id"]

    def touch(self, image_id: int, size: int, mtime_ns: int) -> None:
        """Record a new size/mtime for a file whose content did not change."""
        with self.conn:
            self.conn.execute(
                "UPDATE images SET size = ?, mtime_ns = ? WHERE id = ?",
                (size, mtime_ns, image_id),
            )

    def delete_paths(self, paths: list[str]) -> int:
        with self.conn:
            cur = self.conn.executemany("DELETE FROM images WHERE path = ?", [(p,) for p in paths])
        return cur.rowcount

    def paths_under(self, folder: Path) -> list[str]:
        prefix = str(folder).rstrip("/") + "/"
        rows = self.conn.execute(
            "SELECT path FROM images WHERE substr(path, 1, ?) = ?", (len(prefix), prefix)
        )
        return [row["path"] for row in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def size_on_disk(self) -> int:
        if self.path is None:
            return 0
        files = [self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")]
        return sum(f.stat().st_size for f in files if f.exists())

    # -- vectors ------------------------------------------------------------

    def vector_dims(self) -> tuple[int, int] | None:
        text_dim, image_dim = self.get_meta_safe("text_dim"), self.get_meta_safe("image_dim")
        return (int(text_dim), int(image_dim)) if text_dim and image_dim else None

    def vector_models(self) -> tuple[str, str] | None:
        text, image = self.get_meta("text_model"), self.get_meta("image_model")
        return (text, image) if text and image else None

    def reset_vectors(self, text_model: str, text_dim: int, image_model: str, image_dim: int):
        """Drop every embedding and prepare tables for (possibly new) models."""
        with self.conn:
            self.conn.execute("DROP TRIGGER IF EXISTS images_bd")
            self.conn.execute("DROP TABLE IF EXISTS vec_text")
            self.conn.execute("DROP TABLE IF EXISTS vec_image")
            self.conn.execute("DELETE FROM text_chunks")
        self.conn.executescript(_VECTOR_SCHEMA.format(text_dim=text_dim, image_dim=image_dim))
        for key, value in {
            "text_model": text_model,
            "text_dim": str(text_dim),
            "image_model": image_model,
            "image_dim": str(image_dim),
        }.items():
            self.set_meta(key, value)

    def store_vectors(
        self,
        image_id: int,
        chunks: list[str],
        chunk_vectors: np.ndarray,
        image_vector: np.ndarray,
    ) -> None:
        with self.conn:
            old = [
                r[0]
                for r in self.conn.execute(
                    "SELECT id FROM text_chunks WHERE image_id = ?", (image_id,)
                )
            ]
            self.conn.executemany("DELETE FROM vec_text WHERE chunk_id = ?", [(i,) for i in old])
            self.conn.execute("DELETE FROM text_chunks WHERE image_id = ?", (image_id,))
            self.conn.execute("DELETE FROM vec_image WHERE image_id = ?", (image_id,))

            self.conn.execute(
                "INSERT INTO vec_image(image_id, embedding) VALUES (?, ?)",
                (image_id, _blob(image_vector)),
            )
            for chunk, vector in zip(chunks, chunk_vectors, strict=True):
                chunk_id = self.conn.execute(
                    "INSERT INTO text_chunks(image_id, chunk) VALUES (?, ?) RETURNING id",
                    (image_id, chunk),
                ).fetchone()[0]
                self.conn.execute(
                    "INSERT INTO vec_text(chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, _blob(vector)),
                )

    def ids_without_vectors(self) -> list[int]:
        rows = self.conn.execute(
            "SELECT id FROM images WHERE id NOT IN (SELECT image_id FROM vec_image)"
        )
        return [row[0] for row in rows]

    def vector_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM vec_image").fetchone()[0]

    # -- search -------------------------------------------------------------

    def search_text(self, query: str, limit: int) -> list[tuple[int, float, str]]:
        """BM25 matches as (image id, score, snippet); higher score is better."""
        match = fts_query(query)
        if match is None:
            return []
        rows = self.conn.execute(
            """
            SELECT rowid, -bm25(images_fts) AS score,
                   snippet(images_fts, 0, ?, ?, '…', 16) AS snippet
            FROM images_fts
            WHERE images_fts MATCH ?
            ORDER BY bm25(images_fts)
            LIMIT ?
            """,
            (HIGHLIGHT_START, HIGHLIGHT_END, match, limit),
        )
        return [(row["rowid"], row["score"], row["snippet"]) for row in rows]

    def search_text_vectors(self, query: np.ndarray, limit: int) -> list[tuple[int, float, str]]:
        """Nearest OCR chunks, best chunk per image: (image id, cosine, chunk)."""
        rows = self.conn.execute(
            """
            SELECT c.image_id, v.distance, c.chunk
            FROM vec_text v JOIN text_chunks c ON c.id = v.chunk_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (_blob(query), limit * 4),
        )
        best: dict[int, tuple[int, float, str]] = {}
        for image_id, distance, chunk in rows:
            if image_id not in best:
                best[image_id] = (image_id, _similarity(distance), chunk)
        return list(best.values())[:limit]

    def search_image_vectors(self, query: np.ndarray, limit: int) -> list[tuple[int, float]]:
        """Nearest images to a SigLIP text embedding: (image id, cosine)."""
        rows = self.conn.execute(
            """
            SELECT image_id, distance FROM vec_image
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (_blob(query), limit),
        )
        return [(image_id, _similarity(distance)) for image_id, distance in rows]
