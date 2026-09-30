# shotsearch

Local semantic search over Mac screenshots. CLI in Python 3.12, managed with uv. macOS + Apple Silicon only (Vision OCR, `mps`).

## Commands

- `uv run shotsearch --help`
- `uv run pytest` (fast) / `uv run pytest -m slow` (needs models)
- `uv run ruff format . && uv run ruff check .`
- `uv run python scripts/make_fixtures.py` regenerates `tests/fixtures/`

## Rules

- Nothing may leave the machine: no network calls except the one-time model download.
- Tests must never touch the real data dir: `SHOTSEARCH_HOME` is redirected in `tests/conftest.py`.
- For manual CLI runs against fixtures, set `SHOTSEARCH_HOME` to a scratch dir and
  `SHOTSEARCH_MODELS` to `~/Library/Application Support/shotsearch/models` to reuse the models.
- huggingface_hub reads its env vars at import time: see `embeddings._configure_hub`.
- Modules: `config` (paths), `ocr`, `embeddings`, `db` (SQLite + FTS5 + sqlite-vec), `indexer`, `search`, `watcher`, `cli`.
