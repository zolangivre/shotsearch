# shotsearch

Find any screenshot on your Mac by describing it — in English or French — without sending a single pixel anywhere.

```console
$ shotsearch search "build error"
 #  Score  File                                   Text
 1  2.140  Screenshot 2026-09-12 at 10.41.03.png  … Build failed with 1 error.
```

> **Status: early.** Full-text search over OCR works today. Semantic search (embeddings,
> hybrid ranking) and folder watching are in progress — see the [roadmap](#roadmap).

## How it works

1. **OCR** — each image goes through Apple's Vision framework (via [`ocrmac`](https://github.com/straussmaximilian/ocrmac)), in French and English.
2. **Storage** — metadata and text go into a single SQLite file, indexed with FTS5
   (accent-insensitive, so `echeance` finds `Échéance`).
3. **Search** — your query is matched against the OCR text with BM25 ranking.

Indexing is incremental: unchanged files (same size and mtime, or same SHA-256) are
skipped, and files deleted from disk are dropped from the index.

## Install

Requires macOS on Apple Silicon and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> shotsearch && cd shotsearch
uv sync
uv run shotsearch --help
```

To have `shotsearch` on your PATH: `uv tool install .`

## Usage

```bash
shotsearch index                      # your macOS screenshot folder
shotsearch index --path ~/Downloads   # any folder
shotsearch index --only-screenshots   # skip images Spotlight doesn't flag as screenshots
shotsearch index --reindex            # re-process everything

shotsearch search "facture octobre" -n 10
shotsearch search "build failed" --open   # open the best match in Preview

shotsearch stats
```

PNG, JPEG and HEIC are supported. Unreadable files are reported and skipped.

## Privacy

- Everything runs locally. No API calls, no analytics, no telemetry.
- The index lives in `~/Library/Application Support/shotsearch/` (override with
  `SHOTSEARCH_HOME`). Delete that folder and it's gone.
- The index contains the text found in your screenshots, which may include sensitive
  information. It is stored unencrypted, like the screenshots themselves.

## Development

```bash
uv run python scripts/make_fixtures.py   # regenerate tests/fixtures
uv run pytest                            # fast tests
uv run pytest -m slow                    # tests that need the ML models
uv run ruff check . && uv run ruff format .
```

## Roadmap

- [x] OCR + full-text search, incremental indexing
- [ ] Multilingual embeddings: image (SigLIP) and OCR text (multilingual-e5)
- [ ] Hybrid search with reciprocal rank fusion
- [ ] `watch` mode: index new screenshots as they're taken
- [ ] Local FastAPI server
- [ ] SwiftUI menu bar app

## License

MIT
