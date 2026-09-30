from __future__ import annotations

from pathlib import Path

import pytest

from shotsearch.config import default_models_dir
from shotsearch.db import Database

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never touch the real ~/Library/Application Support/shotsearch in tests."""
    home = tmp_path / "home"
    monkeypatch.setenv("SHOTSEARCH_HOME", str(home))
    return home


@pytest.fixture(scope="session")
def real_models_dir() -> Path:
    """The models downloaded by `shotsearch setup`; the database stays isolated."""
    models = default_models_dir()
    if not (models / "hub").is_dir():
        pytest.skip("models not downloaded: run `shotsearch setup`")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SHOTSEARCH_MODELS", str(models))
        yield models


@pytest.fixture
def db() -> Database:
    database = Database(":memory:")
    yield database
    database.close()
