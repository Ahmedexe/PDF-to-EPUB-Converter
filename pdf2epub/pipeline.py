"""The conversion pipeline: ingest -> OCR -> parse -> checks -> (review) -> assemble -> EPUB (spec §2).

Every stage caches its output in the book's work folder, so a crash or a re-run
picks up where it stopped and one page can be redone alone. A page reaches the
EPUB only when it is ``clean`` or ``reviewed``, unless the build is a draft.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from .assemble import assemble_book
from .checks import run_checks
from .config import BookConfig, apply_overrides, slugify
from .epub import build_epub, run_epubcheck
from .ingest.render import ingest_pdf
from .model import Figure, OcrInfo, Page, PageEntry, PageLayout, iter_blocks, load_page, save_model
from .ocr.client import OcrClient
from .ocr.run import PageOcr, ocr_book
from .parse import parse_tags
from .workdir import WorkDir, book_workdir

log = logging.getLogger(__name__)
Echo = Callable[[str], None]


def _quiet(_msg: str) -> None:
    return None


@dataclass
class BuildReport:
    epubs: list[Path] = field(default_factory=list)
    blocked: list[int] = field(default_factory=list)  # pages still needing review
    epubcheck: list[dict[str, Any]] = field(default_factory=list)
    draft: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.epubs) and all(r.get("ok", True) or not r.get("available", True) for r in self.epubcheck)


# --------------------------------------------------------------------------- book setup


def open_book(work_root: Path | str, slug: str) -> tuple[WorkDir, BookConfig]:
    wd = book_workdir(work_root, slug)
    if not wd.config_path.exists():
        raise FileNotFoundError(f"no book '{slug}' in {work_root} (run `pdf2epub convert` first)")
    return wd, wd.load_config()


def setup_book(pdf: Path, work_root: Path | str, overrides: dict[str, Any], slug: str | None = None) -> tuple[WorkDir, BookConfig]:
    slug = slug or slugify(pdf.name)
    wd = book_workdir(work_root, slug).ensure()
    if wd.config_path.exists():
        cfg = wd.load_config()
    else:
        cfg = BookConfig(slug=slug, title=pdf.stem, source_pdf=str(pdf.resolve()))
    cfg = apply_overrides(cfg, overrides)
    wd.save_config(cfg)
    return wd, cfg


# --------------------------------------------------------------------------- parse


def crop_figures(page: Page, wd: WorkDir) -> None:
    """Crop each figure's box from the colour source (or the page image) into ``figures/``."""
    src = wd.page_source_image(page.index)
    path = src if src.exists() else wd.page_image(page.index)
    img = None
    for block, _owner in iter_blocks(page):
        if not isinstance(block, Figure) or not block.bbox:
            continue
        if img is None:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return
        h, w = img.shape[:2]
        x0, y0, x1, y1 = (int(round(v)) for v in block.bbox)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        out = wd.figure_image(block.id)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img[y0:y1, x0:x1])
        block.image = wd.rel(out)


def base_page(entry: PageEntry, info: OcrInfo | None, layout: PageLayout | None) -> Page:
    page = Page(index=entry.index, pdf_page=entry.pdf_page, side=entry.side, image=entry.image,
                width=entry.width, height=entry.height, ocr=info, blank=entry.blank)
    if layout is not None:
        main = next((r for r in layout.regions if r.kind == "main"), None)
        page.main_bbox = list(main.bbox) if main else None
    return page


def parse_page(entry: PageEntry, ocr: PageOcr, wd: WorkDir, cfg: BookConfig) -> Page:
    """Model output -> ``pages/NNNN.parsed.json`` (with figures cropped)."""
    page = parse_tags(ocr.tags, base_page(entry, ocr.info, ocr.layout), commentary=cfg.text.commentary).page
    crop_figures(page, wd)
    wd.save_page(page, "parsed")
    return page


def _load_ocr(wd: WorkDir, entry: PageEntry) -> PageOcr | None:
    meta = wd.page_ocr_info(entry.index)
    if not meta.exists():
        return None
    info = OcrInfo.model_validate(json.loads(meta.read_text(encoding="utf-8"))["info"])
    tags = wd.page_tags(entry.index).read_text(encoding="utf-8") if wd.page_tags(entry.index).exists() else ""
    layout = None
    if wd.page_regions(entry.index).exists():
        layout = PageLayout.model_validate_json(wd.page_regions(entry.index).read_text(encoding="utf-8"))
    return PageOcr(tags, info, layout)


def reparse(wd: WorkDir, cfg: BookConfig, indices: list[int] | None = None) -> None:
    """Re-run the parser from the cached model output (cheap; no GPU)."""
    manifest = wd.load_manifest()
    for entry in manifest.pages:
        if indices is not None and entry.index not in indices:
            continue
        ocr = _load_ocr(wd, entry)
        if ocr is not None:
            parse_page(entry, ocr, wd, cfg)


# --------------------------------------------------------------------------- checks


def check_inputs(wd: WorkDir) -> list[tuple[Page, str]]:
    """Check input per page: reviewed.json if present, else parsed.json (§5 rule 1)."""
    out = []
    for entry in wd.load_manifest().pages:
        if wd.page_reviewed(entry.index).exists():
            out.append((load_page(wd.page_reviewed(entry.index)), "reviewed"))
        elif wd.page_parsed(entry.index).exists():
            out.append((load_page(wd.page_parsed(entry.index)), "checked"))
    return out


def check_book(wd: WorkDir, cfg: BookConfig, *, only: set[int] | None = None) -> list[Page]:
    """Run the book-level checks and save every page back to the file it came from."""
    inputs = check_inputs(wd)
    pages = [p for p, _kind in inputs]
    run_checks(pages, wd, cfg, only=only)
    for page, kind in inputs:
        wd.save_page(page, kind)
    return pages


def needs_review(pages: list[Page]) -> list[int]:
    return [p.index for p in pages if p.status not in ("clean", "reviewed")]


# --------------------------------------------------------------------------- build


def build(wd: WorkDir, cfg: BookConfig, *, draft: bool | None = None, modified: str | None = None,
          echo: Echo = _quiet) -> BuildReport:
    """Assemble the best version of every page and write the EPUB(s) into ``out/``."""
    draft = cfg.build.allow_flagged if draft is None else draft
    pages = [p for p in (wd.best_page(i) for i in wd.page_indices()) if p is not None]
    report = BuildReport(draft=draft)
    report.blocked = needs_review(pages)
    if report.blocked and not draft:
        return report
    if not pages:
        raise RuntimeError("no pages to build (run convert first)")
    title = cfg.title or cfg.slug
    book = assemble_book(pages, cfg, slug=cfg.slug, title=title, author=cfg.author)
    save_model(book, wd.book_model_path)
    cover = None
    if cfg.build.cover == "first-page" and pages:
        first = pages[0].index
        cover = wd.page_source_image(first) if wd.page_source_image(first).exists() else wd.page_image(first)
    suffix = "-draft" if draft and report.blocked else ""
    report.epubs = build_epub(book, cfg, wd, wd.epub_path(cfg.slug, suffix), modified=modified, draft=draft,
                              cover=cover)
    if cfg.build.epubcheck:
        for path in report.epubs:
            r = run_epubcheck(path)
            report.epubcheck.append({"path": str(path), "ok": r.ok, "available": r.available, "errors": r.errors,
                                     "fatal": r.fatal, "warnings": r.warnings, "summary": r.summary()})
            echo(r.summary())
        wd.write_json(wd.epubcheck_report, report.epubcheck)
    return report


# --------------------------------------------------------------------------- whole commands


def ocr_pages(wd: WorkDir, cfg: BookConfig, entries: list[PageEntry], *, force: bool = False,
              echo: Echo = _quiet, client: OcrClient | None = None) -> dict[int, PageOcr]:
    def progress(done: int, total: int, res: PageOcr | None) -> None:
        if res is not None and res.failed:
            echo(f"  page {done}/{total}: OCR failed (will be retried on the next run)")
        elif done == total or done % 10 == 0:
            echo(f"  read {done}/{total} pages")

    return asyncio.run(ocr_book(entries, wd, cfg, force=force, client=client, progress=progress))


def convert(pdf: Path, work_root: Path | str, overrides: dict[str, Any], *, slug: str | None = None,
            pdf_pages: set[int] | None = None, force: bool = False, echo: Echo = _quiet,
            modified: str | None = None, client: OcrClient | None = None) -> BuildReport:
    """Run the whole pipeline for one PDF; builds the EPUB when no page needs review."""
    wd, cfg = setup_book(Path(pdf), work_root, overrides, slug)
    echo(f"Book '{cfg.slug}' in {wd.root}")
    manifest = ingest_pdf(Path(pdf), wd, cfg, pdf_pages=pdf_pages, force=force,
                          progress=lambda n, t: echo(f"  ingested {n}/{t} PDF pages") if n == t else None)
    engine = f"{cfg.ocr.backend} ({cfg.ocr.model})" if cfg.ocr.backend == "openai" else cfg.ocr.backend
    echo(f"Reading {len(manifest.pages)} pages with {engine}")
    results = ocr_pages(wd, cfg, manifest.pages, force=force, echo=echo, client=client)
    for entry in manifest.pages:
        if entry.index in results:
            parse_page(entry, results[entry.index], wd, cfg)
    pages = check_book(wd, cfg)
    if cfg.checks.tesseract and rescue_short_pages(wd, cfg, echo=echo, client=client):
        pages = check_book(wd, cfg)
    report_flags(pages, echo)
    report = build(wd, cfg, modified=modified, echo=echo)
    if report.blocked and not report.epubs:
        echo(f"{len(report.blocked)} page(s) need review before the EPUB is built: "
             f"pdf2epub review {cfg.slug}   (or build a draft: pdf2epub build {cfg.slug} --draft)")
    for path in report.epubs:
        echo(f"Wrote {path}")
    return report


def redo(wd: WorkDir, cfg: BookConfig, indices: list[int], *, discard_review: bool = False,
         echo: Echo = _quiet, client: OcrClient | None = None) -> list[int]:
    """Re-read the given pages with the model, re-parse them and re-check the book."""
    manifest = wd.load_manifest()
    by_index = {e.index: e for e in manifest.pages}
    todo: list[PageEntry] = []
    for i in indices:
        if i not in by_index:
            echo(f"  page {i} does not exist")
            continue
        reviewed = wd.page_reviewed(i)
        if reviewed.exists():
            if not discard_review:
                echo(f"  page {i} was reviewed; pass --discard-review to re-read it anyway")
                continue
            reviewed.replace(reviewed.with_suffix(".json.bak"))
        todo.append(by_index[i])
    if todo:
        results = ocr_pages(wd, cfg, todo, force=True, echo=echo, client=client)
        for entry in todo:
            parse_page(entry, results[entry.index], wd, cfg)
        check_book(wd, cfg)
    return [e.index for e in todo]


def rescue_short_pages(wd: WorkDir, cfg: BookConfig, *, ratio: float = 0.6, echo: Echo = _quiet,
                       client: OcrClient | None = None) -> list[int]:
    """Re-read pages where the model stopped early, in horizontal bands, and keep the new reading
    when it holds more text.

    A model can end a page after a few lines (a premature stop, not a parse failure), which a
    plain re-read repeats exactly. A page whose reading has fewer than ``ratio`` of Tesseract's
    words is read again as bands (``layout.regions = bands``); the reading with more words wins,
    unless it loops.
    Reviewed pages are left alone. Returns the pages whose reading was replaced.
    """
    from .checks.flags import find_repetition, tesseract_text
    from .eval.metrics import page_text

    def words(i: int) -> int:
        page = wd.best_page(i)
        return len(page_text(page, notes=True).split()) if page else 0

    manifest = wd.load_manifest()
    short = []
    for entry in manifest.pages:
        if entry.blank or wd.page_reviewed(entry.index).exists() or not wd.page_tags(entry.index).exists():
            continue
        tess = len((tesseract_text(wd, entry, cfg) or "").split())
        if tess >= 40 and words(entry.index) < ratio * tess:
            short.append(entry)
    if not short or cfg.layout.regions == "bands":
        return []
    echo(f"Re-reading {len(short)} page(s) that look cut short, in bands")
    before = {e.index: words(e.index) for e in short}
    saved = {e.index: {f: f.read_bytes() for f in (wd.page_tags(e.index), wd.page_ocr_info(e.index),
                                                  wd.page_parsed(e.index), wd.page_regions(e.index),
                                                  wd.page_checked(e.index)) if f.exists()} for e in short}
    banded = apply_overrides(cfg, {"layout.regions": "bands"})
    results = ocr_pages(wd, banded, short, force=True, echo=echo, client=client)
    replaced = []
    for entry in short:
        if entry.index in results and not results[entry.index].failed:
            parse_page(entry, results[entry.index], wd, cfg)
            wd.page_checked(entry.index).unlink(missing_ok=True)
        looped = wd.page_tags(entry.index).exists() and find_repetition(
            wd.page_tags(entry.index).read_text(encoding="utf-8"), 3)
        # More words only counts when they are new text: a band read can loop (on a decorative
        # frame, say, which Tesseract also misreads as many words).
        if words(entry.index) > before[entry.index] and not looped:
            replaced.append(entry.index)
            continue
        for f in (wd.page_tags(entry.index), wd.page_ocr_info(entry.index), wd.page_parsed(entry.index),
                  wd.page_regions(entry.index), wd.page_checked(entry.index)):
            if f in saved[entry.index]:
                f.write_bytes(saved[entry.index][f])
            else:
                f.unlink(missing_ok=True)
    echo(f"  kept the banded reading for {len(replaced)} of them")
    return replaced


def report_flags(pages: list[Page], echo: Echo) -> None:
    flagged = [p for p in pages if p.status == "flagged"]
    echo(f"Checked {len(pages)} pages: {len(pages) - len(flagged)} ready, {len(flagged)} to review")
    for page in flagged[:30]:
        codes = sorted({f.code for f in page.flags if f.severity != "info"})
        echo(f"  page {page.index} ({page.printed_page or '-'}): {', '.join(codes)}")
    if len(flagged) > 30:
        echo(f"  ... and {len(flagged) - 30} more")


def status(wd: WorkDir) -> dict[str, Any]:
    pages = [p for p in (wd.best_page(i) for i in wd.page_indices()) if p is not None]
    counts: dict[str, int] = {}
    for p in pages:
        counts[p.status] = counts.get(p.status, 0) + 1
    missing = [i for i in wd.page_indices() if not wd.page_tags(i).exists()]
    return {"pages": len(wd.page_indices()), "status": counts, "not_read": missing,
            "epubs": sorted(str(p) for p in wd.out_dir.glob("*.epub"))}
