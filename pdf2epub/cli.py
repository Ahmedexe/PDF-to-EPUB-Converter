"""Command line: ``pdf2epub convert | review | redo | check | build | status | eval | sample | validate``.

    pdf2epub convert sharh.pdf --layout two-column --notes popup --commentary mixed
    pdf2epub review sharh             # opens the review screen
    pdf2epub redo sharh --pages 37,112
    pdf2epub build sharh --notes inline   # rebuilds the EPUB from the page JSON
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import click

from . import __version__, pipeline

WORK_DEFAULT = os.environ.get("PDF2EPUB_WORK", "work")


def _pages(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def book_options(f):
    """Options that set per-book settings; unset options never change saved settings."""
    opts = [
        click.option("--title", help="Book title (EPUB metadata)."),
        click.option("--author", help="Author (EPUB metadata)."),
        click.option("--layout", "poem_layout", type=click.Choice(["two-column", "stacked"]),
                     help="Poem layout: two half-lines side by side, or stacked for small screens."),
        click.option("--notes", type=click.Choice(["popup", "inline", "chapter-end"]), help="How حاشية notes are shown."),
        click.option("--quran-refs", type=click.Choice(["label", "footnote", "none"]),
                     help="Show sura:aya references for verses."),
        click.option("--editor-notes", type=click.Choice(["show", "hide"]), help="Show the editor's notes."),
        click.option("--author-notes", type=click.Choice(["show", "hide"]), help="Show the author's notes."),
        click.option("--note-markers", type=click.Choice(["renumber", "printed"]), help="Note markers in the EPUB."),
        click.option("--font", "body_font", type=click.Choice(["amiri", "noto-naskh"]), help="Embedded body font."),
        click.option("--frames", type=click.Choice(["section", "separate-epub"]),
                     help="A second work printed in a frame: extra section or its own EPUB."),
        click.option("--commentary", type=click.Choice(["none", "mixed"]),
                     help="mixed: a commentary weaving the main text in (شرح ممزوج)."),
        click.option("--hashiya", help="Note forms in this book, comma-separated: foot,lemma,margin,frame,layered,editor."),
        click.option("--quran-style", type=click.Choice(["auto", "uthmani", "uthmani-plain", "uthmani-min", "simple",
                                                         "simple-plain", "simple-clean"]),
                     help="Tanzil text used to replace verses (auto: as the book prints them)."),
        click.option("--poem-columns", type=click.Choice(["auto", "two", "single"])),
        click.option("--editor-markers", type=click.Choice(["none", "letters", "numbers", "symbols"]),
                     help="Marker style that identifies the editor's notes."),
        click.option("--regions", type=click.Choice(["auto", "always", "never"]),
                     help="Layout step for margin/framed notes (auto: only where needed)."),
        click.option("--backend", type=click.Choice(["openai", "fixture", "tesseract"]), help="OCR backend."),
        click.option("--endpoint", help="OpenAI-compatible URL of the model server, ending in /v1."),
        click.option("--model", help="Model name served by the endpoint."),
        click.option("--fixtures", type=click.Path(file_okay=False), help="Fixture backend: folder of NNNN.tags."),
        click.option("--concurrency", type=int, help="Pages read at the same time."),
        click.option("--dpi", type=int, help="Render resolution for pages that are not plain scans."),
        click.option("--tesseract/--no-tesseract", default=None, help="Tesseract word-count cross-check."),
        click.option("--epubcheck/--no-epubcheck", default=None, help="Validate the EPUB with EPUBCheck."),
    ]
    for opt in reversed(opts):
        f = opt(f)
    return f


def overrides_from(kw: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "title": "title", "author": "author", "poem_layout": "build.poem_layout", "notes": "build.notes",
        "quran_refs": "build.quran_refs", "editor_notes": "build.editor_notes", "author_notes": "build.author_notes",
        "note_markers": "build.note_markers", "body_font": "build.body_font", "frames": "build.frames",
        "commentary": "text.commentary", "quran_style": "text.quran_style", "poem_columns": "text.poem_columns",
        "editor_markers": "text.editor_markers", "regions": "layout.regions", "backend": "ocr.backend",
        "endpoint": "ocr.endpoint", "model": "ocr.model", "fixtures": "ocr.fixtures",
        "concurrency": "ocr.concurrency", "dpi": "ingest.dpi", "tesseract": "checks.tesseract",
        "epubcheck": "build.epubcheck",
    }
    out = {mapping[k]: v for k, v in kw.items() if k in mapping and v is not None}
    if kw.get("fixtures"):
        out["ocr.fixtures"] = str(Path(kw["fixtures"]).resolve())
        out.setdefault("ocr.backend", "fixture")
    if kw.get("hashiya"):
        out["text.hashiya_forms"] = [x.strip() for x in kw["hashiya"].split(",") if x.strip()]
    return out


@click.group()
@click.version_option(__version__)
@click.option("--work", default=WORK_DEFAULT, show_default=True, type=click.Path(file_okay=False),
              help="Folder holding one work folder per book (env PDF2EPUB_WORK).")
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def main(ctx: click.Context, work: str, verbose: bool) -> None:
    """Convert scanned Arabic PDF books into right-to-left EPUB3 for Kindle and Boox."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ctx.obj = {"work": Path(work)}


@main.command()
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--slug", help="Name of the book's work folder (default: from the file name).")
@click.option("--pdf-pages", help="Only these PDF pages, e.g. 1-20 (a trial run).")
@click.option("--force", is_flag=True, help="Re-ingest and re-read everything.")
@book_options
@click.pass_obj
def convert(obj: dict[str, Any], pdf: Path, slug: str | None, pdf_pages: str | None, force: bool, **kw: Any) -> None:
    """Convert a PDF: ingest, OCR, checks, and the EPUB when no page needs review."""
    report = pipeline.convert(pdf, obj["work"], overrides_from(kw), slug=slug,
                              pdf_pages=set(_pages(pdf_pages)) if pdf_pages else None, force=force, echo=click.echo)
    if report.blocked and not report.epubs:
        sys.exit(2)
    if not report.ok:
        sys.exit(1)


@main.command()
@click.argument("slug")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True, type=int)
@click.option("--no-browser", is_flag=True)
@click.pass_obj
def review(obj: dict[str, Any], slug: str, host: str, port: int, no_browser: bool) -> None:
    """Open the review screen for a book's flagged pages."""
    from .review.app import serve

    wd, _cfg = pipeline.open_book(obj["work"], slug)
    serve(wd, host=host, port=port, open_browser=not no_browser)


@main.command()
@click.argument("slug")
@click.option("--pages", required=True, help="Book page numbers (indices), e.g. 37,112 or 10-14.")
@click.option("--discard-review", is_flag=True, help="Also re-read pages that were reviewed (keeps a .bak).")
@book_options
@click.pass_obj
def redo(obj: dict[str, Any], slug: str, pages: str, discard_review: bool, **kw: Any) -> None:
    """Re-read some pages with the model and re-check the book."""
    wd, cfg = pipeline.open_book(obj["work"], slug)
    cfg = _update(wd, cfg, kw)
    done = pipeline.redo(wd, cfg, _pages(pages), discard_review=discard_review, echo=click.echo)
    click.echo(f"Re-read {len(done)} page(s)")
    pipeline.report_flags([p for p in (wd.best_page(i) for i in wd.page_indices()) if p is not None], click.echo)


@main.command()
@click.argument("slug")
@click.option("--reparse", is_flag=True, help="Re-run the parser on the cached model output first.")
@book_options
@click.pass_obj
def check(obj: dict[str, Any], slug: str, reparse: bool, **kw: Any) -> None:
    """Re-run the checks (and optionally the parser) without calling the model."""
    wd, cfg = pipeline.open_book(obj["work"], slug)
    cfg = _update(wd, cfg, kw)
    if reparse:
        pipeline.reparse(wd, cfg)
    pages = pipeline.check_book(wd, cfg)
    pipeline.report_flags(pages, click.echo)


@main.command()
@click.argument("slug")
@click.option("--draft", is_flag=True, help="Build even if pages still need review (marks unresolved text).")
@book_options
@click.pass_obj
def build(obj: dict[str, Any], slug: str, draft: bool, **kw: Any) -> None:
    """Rebuild the EPUB from the page JSON (display options can change freely)."""
    wd, cfg = pipeline.open_book(obj["work"], slug)
    cfg = _update(wd, cfg, kw)
    report = pipeline.build(wd, cfg, draft=draft or None, echo=click.echo)
    if report.blocked and not report.epubs:
        click.echo(f"{len(report.blocked)} page(s) need review: {report.blocked[:20]} "
                   f"(pdf2epub review {slug}, or --draft)")
        sys.exit(2)
    for path in report.epubs:
        click.echo(f"Wrote {path}")
    if not report.ok:
        sys.exit(1)


@main.command()
@click.argument("slug")
@click.pass_obj
def status(obj: dict[str, Any], slug: str) -> None:
    """Page counts by status, pages not yet read, built EPUBs."""
    wd, _cfg = pipeline.open_book(obj["work"], slug)
    click.echo(json.dumps(pipeline.status(wd), ensure_ascii=False, indent=1))


@main.command(name="eval")
@click.argument("testset", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Where predictions, scores.json and report.md go (default: <testset>/results/<model>).")
@click.option("--compare", "compare", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="Other scores.json files to put in the comparison table.")
@book_options
def eval_cmd(testset: Path, out_dir: Path | None, compare: tuple[Path, ...], **kw: Any) -> None:
    """Score a model on a hand-verified test set (spec §9)."""
    from .config import BookConfig, apply_overrides
    from .eval.run import compare_report, evaluate, load_scores

    cfg = apply_overrides(BookConfig(slug="eval"), overrides_from(kw))
    name = (cfg.ocr.model if cfg.ocr.backend == "openai" else cfg.ocr.backend).replace("/", "_")
    out_dir = out_dir or testset / "results" / name
    scores = evaluate(testset, cfg, out_dir=out_dir)
    results = {name: scores}
    for path in compare:
        results[path.parent.name] = load_scores(path)
    click.echo(compare_report(results))


@main.command()
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--no-noise", is_flag=True, help="Clean renders instead of simulated scans.")
def sample(out_dir: Path, no_noise: bool) -> None:
    """Write a small synthetic scanned book with ground truth, to try the pipeline offline."""
    from .devtools.sample import make_sample_book

    book = make_sample_book(out_dir, scan_noise=not no_noise)
    click.echo(f"Wrote {book.pdf} ({book.pages} pages) and model outputs in {book.fixtures}")
    click.echo(f"Try: pdf2epub convert {book.pdf} --fixtures {book.fixtures}")


@main.command()
@click.argument("epub", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(epub: Path) -> None:
    """Run EPUBCheck on an EPUB."""
    from .epub import run_epubcheck

    report = run_epubcheck(epub)
    click.echo(report.summary())
    for line in report.fatal + report.errors + report.warnings:
        click.echo("  " + line)
    if report.available and not report.ok:
        sys.exit(1)


@main.group()
def bench() -> None:
    """Phase-1 model comparison on Modal: deploy, run on the test PDF, report."""


@bench.command(name="list")
def bench_list() -> None:
    """Candidate models and their recorded endpoints."""
    from .bench import endpoints
    from .ocr.models import CANDIDATES

    eps = endpoints()
    for key, c in CANDIDATES.items():
        click.echo(f"{key:14} {c.hf_id:45} {c.params_b:4.1f}B  {c.gpu:5} {c.preset:6} {eps.get(key, '-')}")
        click.echo(f"{'':14} {c.note}")


@bench.command(name="deploy")
@click.argument("models", nargs=-1, required=True)
def bench_deploy(models: tuple[str, ...]) -> None:
    """Deploy one or more candidates on Modal (needs `modal setup` once)."""
    from .bench import BenchError, deploy

    try:
        for key in models:
            deploy(key, echo=click.echo)
    except (BenchError, KeyError) as exc:
        raise click.ClickException(str(exc))


@bench.command(name="run")
@click.argument("models", nargs=-1, required=True)
@click.option("--test", "test_pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Test PDF (default: sample/test/part.pdf).")
@click.option("--endpoint", help="Use this endpoint instead of the recorded one.")
@click.option("--concurrency", default=8, show_default=True)
@click.option("--force", is_flag=True, help="Re-read pages already read.")
@click.option("--max-side", type=int, help="Longest image side sent to the model (default 2400).")
@click.option("--tag", default="", help="Suffix for the run's name, to keep variants apart (e.g. -hires).")
@click.option("--bands", type=int, help="Read each page in this many main-text bands plus the notes.")
@click.pass_obj
def bench_run(obj: dict[str, Any], models: tuple[str, ...], test_pdf: Path | None, endpoint: str | None,
              concurrency: int, force: bool, max_side: int | None, tag: str, bands: int | None) -> None:
    """Read the test PDF with each model (OCR, parse, checks)."""
    from .bench import DEFAULT_TEST, BenchError, run

    try:
        for key in models:
            run(key, test_pdf=test_pdf or DEFAULT_TEST, work_root=obj["work"], endpoint=endpoint,
                concurrency=concurrency, force=force, max_side=max_side, tag=tag, bands=bands, echo=click.echo)
    except (BenchError, KeyError) as exc:
        raise click.ClickException(str(exc))


@bench.command(name="report")
@click.option("--truth", type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Folder of reviewed pages (NNNN.json) to score CER against.")
@click.option("--reference", type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Folder of reference texts (NNNN.txt) from a digital edition, for letter/word error rates.")
@click.pass_obj
def bench_report(obj: dict[str, Any], truth: Path | None, reference: Path | None) -> None:
    """Compare every bench run (writes bench-report.md and bench-compare.html in the work folder)."""
    from .bench import BenchError, report

    try:
        click.echo(report(obj["work"], truth=truth, reference=reference))
    except BenchError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Side by side: {Path(obj['work']) / 'bench-compare.html'}")


@bench.command(name="rebuild")
@click.argument("models", nargs=-1, required=True)
@click.pass_obj
def bench_rebuild(obj: dict[str, Any], models: tuple[str, ...]) -> None:
    """Rebuild runs from the model replies already saved (no GPU calls)."""
    from .bench import rebuild

    for key in models:
        click.echo(f"{key}: rebuilt {rebuild(key, obj['work'], echo=click.echo)} pages")


@bench.command(name="stop")
@click.argument("models", nargs=-1, required=True)
def bench_stop(models: tuple[str, ...]) -> None:
    """Stop a model's Modal app now (it also stops by itself after 5 idle minutes)."""
    from .bench import BenchError, stop

    try:
        for key in models:
            stop(key, echo=click.echo)
    except (BenchError, KeyError) as exc:
        raise click.ClickException(str(exc))


@bench.command(name="export-truth")
@click.argument("slug")
@click.argument("out", type=click.Path(file_okay=False, path_type=Path))
@click.pass_obj
def bench_export_truth(obj: dict[str, Any], slug: str, out: Path) -> None:
    """Copy the pages you approved in the review screen into a truth folder."""
    from .bench import export_truth

    click.echo(f"Exported {export_truth(slug, obj['work'], out)} reviewed pages to {out}")


def _update(wd, cfg, kw):
    overrides = overrides_from(kw)
    if overrides:
        from .config import apply_overrides

        cfg = apply_overrides(cfg, overrides)
        wd.save_config(cfg)
    return cfg


if __name__ == "__main__":  # pragma: no cover
    main()
