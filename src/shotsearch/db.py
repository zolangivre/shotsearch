"""SQLite storage: image metadata, OCR text and its FTS5 index."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "1"

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

CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, ocr_text) VALUES ('delete', old.id, old.ocr_text);
END;

CREATE TRIGGER IF NOT EXISTS images_au AFTER UPDATE OF ocr_text ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, ocr_text) VALUES ('delete', old.id, old.ocr_text);
    INSERT INTO images_fts(rowid, ocr_text) VALUES (new.id, new.ocr_text);
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


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path) if path != ":memory:" else None
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path is not None:
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(_SCHEMA)
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
