from __future__ import annotations

from pathlib import Path

from shotsearch.db import HIGHLIGHT_START, Database, ImageRecord, fts_query


def record(path: str, text: str, sha: str = "abc") -> ImageRecord:
    return ImageRecord(
        path=path, size=10, mtime_ns=1, sha256=sha, width=100, height=50, ocr_text=text
    )


def test_schema_and_meta(db: Database) -> None:
    assert db.get_meta("schema_version") == "1"
    db.set_meta("k", "v1")
    db.set_meta("k", "v2")
    assert db.get_meta("k") == "v2"
    assert db.get_meta("missing") is None


def test_upsert_is_idempotent_on_path(db: Database) -> None:
    first = db.upsert(record("/a.png", "hello"))
    second = db.upsert(record("/a.png", "hello world", sha="def"))
    assert first == second
    assert db.count() == 1
    assert db.get_by_path("/a.png")["sha256"] == "def"


def test_fts_follows_updates_and_deletes(db: Database) -> None:
    db.upsert(record("/a.png", "Build failed"))
    assert [hit[0] for hit in db.search_text("build", 5)]

    db.upsert(record("/a.png", "all green"))
    assert db.search_text("build", 5) == []
    assert db.search_text("green", 5)

    db.delete_paths(["/a.png"])
    assert db.search_text("green", 5) == []
    assert db.count() == 0


def test_fts_ignores_accents_and_matches_prefixes(db: Database) -> None:
    db.upsert(record("/f.png", "Échéance : 15 octobre"))
    assert db.search_text("echeance", 5)
    assert db.search_text("ÉCHÉANCE", 5)
    db.upsert(record("/b.png", "Build failed"))
    assert db.search_text("fail", 5)


def test_fts_ranks_better_match_first(db: Database) -> None:
    db.upsert(record("/weak.png", "the build is green, nothing failed today maybe"))
    db.upsert(record("/strong.png", "Build failed"))
    db.upsert(record("/other.png", "invoice"))
    hits = db.search_text("build failed", 5)
    rows = db.get_many([h[0] for h in hits])
    assert rows[hits[0][0]]["path"] == "/strong.png"
    assert len(hits) == 2
    assert HIGHLIGHT_START in hits[0][2]


def test_fts_query_is_safe() -> None:
    assert fts_query("") is None
    assert fts_query("de la") is None
    assert fts_query('build "failed" OR NOT (x*') == '"build"* OR "failed"* OR "not"* OR "x"*'


def test_hostile_query_does_not_crash(db: Database) -> None:
    db.upsert(record("/a.png", "hello"))
    for query in ['"', "*", "NEAR(", "AND OR", "a:b", "'; DROP TABLE images; --"]:
        db.search_text(query, 5)
    assert db.count() == 1


def test_paths_under(db: Database) -> None:
    db.upsert(record("/shots/a.png", ""))
    db.upsert(record("/shots/sub/b.png", ""))
    db.upsert(record("/shotsother/c.png", ""))
    assert sorted(db.paths_under(Path("/shots"))) == ["/shots/a.png", "/shots/sub/b.png"]


def test_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "t.db"
    with Database(path) as db:
        db.upsert(record("/a.png", "persisted"))
    with Database(path) as db:
        assert db.count() == 1
        assert db.size_on_disk() > 0
