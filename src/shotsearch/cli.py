"""Command-line interface."""

from __future__ import annotations

import datetime as dt
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from shotsearch import config, embeddings
from shotsearch.db import HIGHLIGHT_END, HIGHLIGHT_START, Database
from shotsearch.embeddings import IMAGE_MODEL, TEXT_MODEL
from shotsearch.indexer import Indexer, ModelChangedError
from shotsearch.search import FTS, IMAGE_VEC, RRF_K, SIGNALS, TEXT_VEC, best_possible_score
from shotsearch.search import search as run_search

if TYPE_CHECKING:
    from shotsearch.embeddings import Embedder

app = typer.Typer(
    name="shotsearch",
    help="Local semantic search for your Mac screenshots. Nothing leaves your machine.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_SIGNAL_LABELS = {FTS: "BM25", TEXT_VEC: "OCR meaning", IMAGE_VEC: "Image"}


def _open_db() -> Database:
    return Database(config.db_path(), TEXT_MODEL.dim, IMAGE_MODEL.dim)


def _load_embedder() -> Embedder:
    try:
        with console.status("Loading models…"):
            return embeddings.Embedder()
    except embeddings.ModelsNotInstalledError as exc:
        console.print(f"[yellow]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc


def _human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _pretty_path(path: Path) -> str:
    home = str(Path.home())
    text = str(path)
    return escape("~" + text[len(home) :] if text.startswith(home + "/") else text)


def _render_snippet(snippet: str) -> str:
    text = escape(" ".join(snippet.split()))
    return text.replace(HIGHLIGHT_START, "[bold yellow]").replace(HIGHLIGHT_END, "[/]")


@app.command()
def setup() -> None:
    """Download the models (once) and check that they run."""
    missing = embeddings.missing_models()
    if missing:
        console.print(
            "Downloading models. [dim]This is the only time shotsearch uses the network.[/]"
        )
        for repo in missing:
            console.print(f"  • {repo}")
        embeddings.download_models(missing)
    else:
        console.print("[green]Models already downloaded.[/]")

    embedder = _load_embedder()
    from PIL import Image

    with console.status("Checking models…"):
        image_dim = embedder.embed_images([Image.new("RGB", (64, 64), "white")]).shape[1]
        text_dim = embedder.embed_passages(["hello"]).shape[1]
        query_dim = embedder.embed_image_query("bonjour").shape[0]
    assert image_dim == query_dim == IMAGE_MODEL.dim and text_dim == TEXT_MODEL.dim
    console.print(f"[green]✓[/] {TEXT_MODEL.repo} [dim]({text_dim} dims)[/]")
    console.print(f"[green]✓[/] {IMAGE_MODEL.repo} [dim]({image_dim} dims)[/]")
    console.print(
        f"Running on [bold]{embedder.device}[/]. "
        f"Models stored in [dim]{_pretty_path(config.models_dir())}[/]"
    )


@app.command()
def index(
    path: Annotated[
        Path | None,
        typer.Option("--path", help="Folder to index. Default: macOS screenshot folder."),
    ] = None,
    reindex: Annotated[bool, typer.Option("--reindex", help="Re-process every image.")] = False,
    only_screenshots: Annotated[
        bool,
        typer.Option(
            "--only-screenshots", help="Skip images Spotlight doesn't flag as screenshots."
        ),
    ] = False,
) -> None:
    """Index new and modified images in a folder."""
    folder = (path or config.default_screenshot_dir()).expanduser()
    if not folder.is_dir():
        console.print(f"[red]Not a folder:[/] {folder}")
        raise typer.Exit(1)

    embedder = _load_embedder()
    console.print(f"Indexing [bold]{_pretty_path(folder)}[/]")
    with (
        _open_db() as db,
        Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress,
    ):
        task = progress.add_task("Scanning…", total=None)

        def on_start(total: int) -> None:
            progress.update(task, description="Indexing", total=total)

        def on_file(p: Path) -> None:
            progress.update(task, advance=1, description=escape(p.name[:40]))

        try:
            report = Indexer(db, embedder).index_folder(
                folder,
                reindex=reindex,
                only_screenshots=only_screenshots,
                on_start=on_start,
                on_file=on_file,
            )
        except ModelChangedError as exc:
            progress.stop()
            console.print(f"[yellow]{escape(str(exc))}[/]")
            raise typer.Exit(1) from exc
        db.set_meta("last_indexed_at", str(dt.datetime.now().timestamp()))

    console.print(
        f"[green]{report.added} added[/], {report.updated} updated, "
        f"{report.unchanged} unchanged, {report.removed} removed"
        + (f", {report.embedded} embedded" if report.embedded else "")
        + (f", [red]{len(report.failed)} failed[/]" if report.failed else "")
    )
    for failed_path, reason in report.failed:
        console.print(f"  [red]✗[/] {escape(failed_path.name)} [dim]{escape(reason)}[/]")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What you're looking for, in French or English.")],
    limit: Annotated[int, typer.Option("-n", "--limit", min=1, help="Number of results.")] = 5,
    open_first: Annotated[
        bool, typer.Option("--open", help="Open the best match in Preview.")
    ] = False,
    explain: Annotated[
        bool, typer.Option("--explain", help="Show each result's rank in every signal.")
    ] = False,
    rrf_k: Annotated[
        int, typer.Option("--rrf-k", min=1, help="RRF constant: higher flattens rank differences.")
    ] = RRF_K,
) -> None:
    """Search indexed screenshots.

    Score is 1.00 when all signals (full text, OCR meaning, image content) rank a
    screenshot first.
    """
    with _open_db() as db:
        if db.count() == 0:
            console.print("[yellow]The index is empty.[/] Run [bold]shotsearch index[/] first.")
            raise typer.Exit(1)
        embedder = None
        if embeddings.missing_models():
            console.print("[yellow]Models not installed: full-text search only.[/]")
        elif db.vector_count() == 0:
            console.print("[yellow]No embeddings yet: full-text search only.[/]")
        else:
            embedder = _load_embedder()
        results = run_search(db, query, limit, embedder, k=rrf_k)

    if not results:
        console.print("No match.")
        raise typer.Exit(0)

    signals = SIGNALS if embedder else (FTS,)
    best_possible = best_possible_score(signals, rrf_k)

    table = Table(show_lines=True, expand=True)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Score", justify="right", width=6)
    table.add_column("File", ratio=2, overflow="fold")
    table.add_column("Text", ratio=3)
    if explain:
        for name in signals:
            table.add_column(_SIGNAL_LABELS[name], justify="right", width=12)
    for rank, result in enumerate(results, 1):
        path = Path(result.path)
        link = f"[link=file://{escape(str(path))}]{escape(path.name)}[/link]"
        row = [
            str(rank),
            f"{result.score / best_possible:.2f}",
            f"{link}\n[dim]{_pretty_path(path.parent)}[/]",
            _render_snippet(result.snippet) or "[dim]no text[/]",
        ]
        if explain:
            for name in signals:
                hit = result.hits.get(name)
                row.append(f"#{hit.rank} [dim]{hit.score:.3f}[/]" if hit else "[dim]·[/]")
        table.add_row(*row)
    console.print(table)
    if explain:
        console.print(
            "[dim]BM25: full-text on OCR · OCR meaning: e5 cosine · "
            "Image: SigLIP match probability. Weighted RRF, k="
            f"{rrf_k}.[/]"
        )

    if open_first:
        subprocess.run(["open", "-a", "Preview", results[0].path], check=False)


@app.command()
def stats() -> None:
    """Show what's in the index."""
    with _open_db() as db:
        count = db.count()
        embedded = db.vector_count()
        size = db.size_on_disk()
        last = db.get_meta("last_indexed_at")
        models = db.vector_models()

    table = Table(show_header=False, box=None)
    table.add_row("Images indexed", f"[bold]{count}[/]")
    if embedded < count:
        table.add_row("With embeddings", f"[yellow]{embedded}[/]  [dim]run shotsearch index[/]")
    table.add_row("Database", f"{_human_size(size)}  [dim]{_pretty_path(config.db_path())}[/]")
    if models:
        table.add_row("Models", f"{models[0]}\n{models[1]}")
    if last:
        when = dt.datetime.fromtimestamp(float(last)).strftime("%Y-%m-%d %H:%M")
        table.add_row("Last indexed", when)
    console.print(table)


def main() -> None:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    app()
