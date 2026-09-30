# shotsearch

Find any screenshot on your Mac by describing it — in English or French — without sending a single pixel anywhere.

```console
$ shotsearch search "erreur de build"
 #  Score  File                                   Text
 1   1.00  Screenshot 2026-09-30 at 16.30.00.png  $ npm run build > vite build … Build failed with 1 error.
```

> **Status: early.** The command-line tool works end to end: indexing, hybrid search and
> folder watching. A local server and a menu bar app come next — see the [roadmap](#roadmap).

## How it works

```
screenshot ──► Vision OCR (fr, en) ──► text ──┬─► FTS5 (BM25)            ─┐
     │                                        └─► multilingual-e5 vectors ├─► weighted RRF ─► results
     └──────────────────────────────────────────► SigLIP 2 image vector  ─┘
```

1. **OCR** — each image goes through Apple's Vision framework (via
   [`ocrmac`](https://github.com/straussmaximilian/ocrmac)), in French and English.
2. **Embeddings** — the OCR text is embedded with multilingual-e5 (so `erreur de build`
   can match `Build failed`), and the image itself with SigLIP 2 (so `coucher de soleil`
   can find a photo with no text at all). Both run on the Apple GPU (`mps`).
3. **Storage** — metadata, text and vectors live in a single SQLite file: FTS5 for text
   (accent-insensitive, so `echeance` finds `Échéance`), [sqlite-vec](https://github.com/asg017/sqlite-vec) for vectors.
4. **Search** — three rankings are fused with reciprocal rank fusion:
   - **BM25** over the OCR text: exact words.
   - **OCR meaning**: e5 similarity between your query and the text. Weighted 0.5,
     because e5-small only barely separates relevant from unrelated text.
   - **Image**: SigLIP's own calibrated match probability. Images below 1 % don't take
     part, so unrelated screenshots can't crowd out a photo that clearly matches.

   A score of 1.00 means all three signals ranked the screenshot first. Use `--explain`
   to see each signal's rank and score.

Indexing is incremental: unchanged files (same size and mtime, or same SHA-256) are
skipped, and files deleted from disk are dropped from the index.

## Install

Requires macOS on Apple Silicon and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> shotsearch && cd shotsearch
uv sync
uv run shotsearch setup   # one-time download of the models (~2 GB)
```

`setup` is the only command that touches the network. It fetches two models from
Hugging Face into the local cache; after that, everything runs offline.

| Role | Model | Size |
|---|---|---|
| OCR text → vector | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) | 470 MB |
| Image ↔ text | [`google/siglip2-base-patch16-256`](https://huggingface.co/google/siglip2-base-patch16-256) | 1.5 GB |

SigLIP 2 was picked over `siglip-base-patch16-256-multilingual` after a side-by-side
test: it got 8/8 French and English test queries right, versus 6/8.

To have `shotsearch` on your PATH: `uv tool install .`

## Usage

```bash
shotsearch index                      # your macOS screenshot folder
shotsearch index --path ~/Downloads   # any folder
shotsearch index --only-screenshots   # skip images macOS didn't flag as screenshots
shotsearch index --reindex            # re-process everything

shotsearch watch                      # catch up, then index each new screenshot (Ctrl+C to stop)

shotsearch search "facture octobre" -n 10
shotsearch search "build failed" --open   # open the best match in Preview
shotsearch search "coucher de soleil" --explain   # show why each result ranks where it does

shotsearch stats
```

PNG, JPEG and HEIC are supported. Unreadable files are reported and skipped.

`watch` handles the way macOS saves screenshots (a hidden temporary file that is then
renamed) and waits until a file stops growing before indexing it. A new screenshot is
searchable about a second after it's taken. `--only-screenshots` reads the flag macOS
stores on each screenshot, so it works on brand-new files too.

## Privacy

- Everything runs locally. No API calls, no analytics, no telemetry.
- The index and models live in `~/Library/Application Support/shotsearch/` (override
  with `SHOTSEARCH_HOME`, and the models alone with `SHOTSEARCH_MODELS`). Delete that
  folder and it's gone.
- Hugging Face runs in offline mode with telemetry disabled; models are always loaded
  from the local cache.
- The index contains the text found in your screenshots, which may include sensitive
  information. It is stored unencrypted, like the screenshots themselves.

## Limitations

- **Cross-language matches are weak when no word is shared.** `meeting moved to thursday`
  finds an English chat screenshot, but `réunion jeudi` doesn't: multilingual-e5-small
  ranks it 5th among 25 screenshots and BM25 has nothing to match. A larger text model
  would help, at the cost of size.
- **The first search in a shell takes a few seconds** while the models load. The planned
  local server will keep them in memory.
- `watch` runs in the foreground; it isn't installed as a login item yet.
- macOS and Apple Silicon only: OCR relies on the Vision framework.

## Development

```bash
uv run python scripts/make_fixtures.py   # regenerate tests/fixtures
uv run pytest                            # fast tests
uv run pytest -m slow                    # tests that need the ML models
uv run ruff check . && uv run ruff format .
```

## Roadmap

- [x] OCR + full-text search, incremental indexing
- [x] Multilingual embeddings: image (SigLIP 2) and OCR text (multilingual-e5), stored with sqlite-vec
- [x] Hybrid search with reciprocal rank fusion
- [x] `watch` mode: index new screenshots as they're taken
- [ ] Local FastAPI server
- [ ] SwiftUI menu bar app

## License

MIT
