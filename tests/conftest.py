from __future__ import annotations

from pathlib import Path

import pytest

from shotsearch.db import Database

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never touch the real ~/Library/Application Support/shotsearch in tests."""
    home = tmp_path / "home"
    monkeypatch.setenv("SHOTSEARCH_HOME", str(home))
    return home


@pytest.fixture
def db() -> Database:
    database = Database(":memory:")
    yield database
    database.close()
