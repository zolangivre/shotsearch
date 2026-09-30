"""Command-line interface."""

from __future__ import annotations

import datetime as dt
import logging
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from shotsearch import config
from shotsearch.db import HIGHLIGHT_END, HIGHLIGHT_START, Database
from shotsearch.indexer import Indexer
from shotsearch.search import search as run_search

app = typer.Typer(
    name="shotsearch",
    help="Local semantic search for your Mac screenshots. Nothing leaves your machine.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _open_db() -> Database:
    return Database(config.db_path())


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
            progress.update(task, description="OCR", total=total)

        def on_file(p: Path) -> None:
            progress.update(task, advance=1, description=f"OCR {escape(p.name[:40])}")

        report = Indexer(db).index_folder(
            folder,
            reindex=reindex,
            only_screenshots=only_screenshots,
            on_start=on_start,
            on_file=on_file,
        )
        db.set_meta("last_indexed_at", str(dt.datetime.now().timestamp()))

    console.print(
        f"[green]{report.added} added[/], {report.updated} updated, "
        f"{report.unchanged} unchanged, {report.removed} removed"
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
) -> None:
    """Search indexed screenshots."""
    with _open_db() as db:
        if db.count() == 0:
            console.print("[yellow]The index is empty.[/] Run [bold]shotsearch index[/] first.")
            raise typer.Exit(1)
        results = run_search(db, query, limit)

    if not results:
        console.print("No match.")
        raise typer.Exit(0)

    table = Table(show_lines=True, expand=True)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Score", justify="right", width=7)
    table.add_column("File", ratio=2, overflow="fold")
    table.add_column("Text", ratio=3)
    for rank, result in enumerate(results, 1):
        path = Path(result.path)
        link = f"[link=file://{escape(str(path))}]{escape(path.name)}[/link]"
        table.add_row(
            str(rank),
            f"{result.score:.3f}",
            f"{link}\n[dim]{_pretty_path(path.parent)}[/]",
            _render_snippet(result.snippet) or "[dim]no text[/]",
        )
    console.print(table)

    if open_first:
        subprocess.run(["open", "-a", "Preview", results[0].path], check=False)


@app.command()
def stats() -> None:
    """Show what's in the index."""
    with _open_db() as db:
        count = db.count()
        size = db.size_on_disk()
        last = db.get_meta("last_indexed_at")

    table = Table(show_header=False, box=None)
    table.add_row("Images indexed", f"[bold]{count}[/]")
    table.add_row("Database", f"{_human_size(size)}  [dim]{_pretty_path(config.db_path())}[/]")
    if last:
        when = dt.datetime.fromtimestamp(float(last)).strftime("%Y-%m-%d %H:%M")
        table.add_row("Last indexed", when)
    console.print(table)


def main() -> None:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    app()
