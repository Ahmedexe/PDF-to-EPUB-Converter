"""The conversion pipeline: ingest -> OCR -> parse -> checks -> (review) -> assemble -> EPUB (spec §2).

Every stage caches its output in the book's work folder, so a crash or a re-run
picks up where it stopped and one page can be redone alone. A page reaches the
EPUB only when it is ``clean`` or ``reviewed``, unless the build is a draft.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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
    blocked: list[int] = field(default_factory=list)  # pages the checks still flag
    quality: Path | None = None  # unattended builds: the report of what the checks still flag
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
    unattended = cfg.build.review == "auto" and not draft
    if report.blocked and not draft and not unattended:
        return report
    if unattended and pages:
        report.quality = write_quality_report(wd, cfg, pages)
    if not pages:
        raise RuntimeError("no pages to build (run convert first)")
    title = cfg.title or cfg.slug
    book = assemble_book(pages, cfg, slug=cfg.slug, title=title, author=cfg.author)
    save_model(book, wd.book_model_path)
    cover = None
    if cfg.build.cover == "first-page" and pages:
        first = pages[0].index
        cover = wd.page_source_image(first) if wd.page_source_image(first).exists() else wd.page_image(first)
    suffix = "-draft" if draft and report.blocked else ""  # unattended builds are final: no suffix
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
    if cfg.rescore.enabled and cfg.rescore.lines and cfg.ocr.backend == "openai":
        read_lines(wd, cfg, echo=echo, client=client)
    rescore_pages(wd, cfg, echo=echo)
    pages = check_book(wd, cfg)
    if cfg.checks.tesseract and rescue_short_pages(wd, cfg, echo=echo, client=client):
        pages = check_book(wd, cfg)
    if cfg.checks.completeness and cfg.ocr.backend == "openai" and complete_pages(wd, cfg, echo=echo, client=client):
        pages = check_book(wd, cfg)
    if cfg.build.review == "auto" and needs_review(pages):
        repaired = repair_flagged(wd, cfg, echo=echo, client=client)
        rescore_pages(wd, cfg, echo=echo, indices=sorted({i for v in repaired.values() for i in v}))
        pages = check_book(wd, cfg)
    report_flags(pages, echo)
    report = build(wd, cfg, modified=modified, echo=echo)
    if report.blocked and not report.epubs:
        echo(f"{len(report.blocked)} page(s) need review before the EPUB is built: "
             f"pdf2epub review {cfg.slug}   (or build a draft: pdf2epub build {cfg.slug} --draft)")
    for path in report.epubs:
        echo(f"Wrote {path}")
    if report.quality:
        echo(f"Quality report: {report.quality}")
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


def rescore_pages(wd: WorkDir, cfg: BookConfig, *, echo: Echo = _quiet, indices: list[int] | None = None,
                  scorer: Any = None) -> list[int]:
    """Language-model rescoring (pdf2epub/rescore.py) of every page whose reading came with its
    token alternatives. Pages are skipped once rescored for their current text, and reviewed
    pages are left alone. ``scorer`` (for tests) maps a list of windows to log10 scores; by
    default the deployed ``Scorer`` on Modal. Returns the pages whose text changed."""
    import hashlib

    from . import rescore as rs

    if not cfg.rescore.enabled:
        return []
    todo = []
    for entry in wd.load_manifest().pages:
        i = entry.index
        tok_file = wd.page_tags(i).with_name(f"{wd.stem(i)}.tokens.json")
        rec_file = wd.page_tags(i).with_name(f"{wd.stem(i)}.rescore.json")
        if entry.blank or not tok_file.exists() or not wd.page_tags(i).exists() or wd.page_reviewed(i).exists():
            continue
        if indices is not None and i not in indices:
            continue
        tags = wd.page_tags(i).read_text(encoding="utf-8")
        sha = hashlib.sha256(tags.encode()).hexdigest()[:16]
        if rec_file.exists() and json.loads(rec_file.read_text(encoding="utf-8")).get("tags_sha") == sha:
            continue
        tokens = json.loads(tok_file.read_text(encoding="utf-8"))
        if "".join(t["t"] for t in tokens) != tags:
            continue  # the text was changed since it was read (an inserted line, a review)
        words = rs.words_of(tokens)
        lines_file = wd.page_tags(i).with_name(f"{wd.stem(i)}.lines.json")
        if cfg.rescore.lines and lines_file.exists():
            rs.add_line_candidates(words, [x["text"] for x in json.loads(lines_file.read_text(encoding="utf-8"))["lines"]],
                                   cfg.rescore.line_score)
        todo.append((entry, tags, words, rs.windows(words, cfg.rescore.doubt)))
    if not todo:
        return []
    if scorer is None:
        try:
            import modal

            remote = modal.Cls.from_name(cfg.rescore.app, "Scorer")()
            scorer = remote.score.remote
        except Exception as exc:  # no Modal, or the app is not deployed: keep the readings as they are
            echo(f"Language-model rescoring skipped ({type(exc).__name__}: {exc})")
            return []
    flat = [w for _e, _t, _w, jobs in todo for _i, _c, w in jobs]
    echo(f"Rescoring doubtful words with the language model ({len(flat)} windows, {len(todo)} pages)")
    scores: list[float] = []
    for k in range(0, len(flat), 5000):
        scores.extend(scorer(flat[k:k + 5000]))
    changed, pos, n_words = [], 0, 0
    for entry, tags, words, jobs in todo:
        page_scores = {(i, c): scores[pos + k] for k, (i, c, _w) in enumerate(jobs)}
        pos += len(jobs)
        changes = rs.choose(words, page_scores, weight=cfg.rescore.weight, margin=cfg.rescore.margin,
                            doubt=cfg.rescore.doubt)
        new = rs.apply(tags, words, changes) if changes else tags
        if changes and not parse_tags(new, Page(index=entry.index)).ok:
            changes, new = [], tags
        i = entry.index
        record = {"tags_sha": hashlib.sha256(new.encode()).hexdigest()[:16],
                  "changes": [{"from": words[k].raw, "to": raw, "gain": gain} for k, raw, _n, gain in changes]}
        if changes:
            # the tokens no longer spell the text: keep them under the corrected text's name
            tok_file = wd.page_tags(i).with_name(f"{wd.stem(i)}.tokens.json")
            tok_file.replace(tok_file.with_name(f"{wd.stem(i)}.tokens.read.json"))
            wd.write_text(wd.page_tags(i), new)
            ocr = _load_ocr(wd, entry)
            if ocr is not None:
                parse_page(entry, ocr, wd, cfg)
                wd.page_checked(i).unlink(missing_ok=True)
            changed.append(i)
            n_words += len(changes)
        wd.write_json(wd.page_tags(i).with_name(f"{wd.stem(i)}.rescore.json"), record)
    echo(f"  corrected {n_words} word(s) on {len(changed)} page(s)")
    return changed


def read_lines(wd: WorkDir, cfg: BookConfig, *, echo: Echo = _quiet, client: OcrClient | None = None,
               indices: list[int] | None = None) -> int:
    """Read every printed line of the main text on its own (a second, independent view of the page,
    used by the rescoring and the completeness check), cached in ``pages/NNNN.lines.json``.
    Returns the number of pages read."""
    from .checks import completeness as cm
    from .ocr.client import encode_image

    entries = [e for e in wd.load_manifest().pages if not e.blank and wd.page_tags(e.index).exists()
               and not wd.page_reviewed(e.index).exists() and (indices is None or e.index in indices)
               and not wd.page_tags(e.index).with_name(f"{wd.stem(e.index)}.lines.json").exists()]
    if not entries:
        return 0
    own = client is None
    client = client or OcrClient(cfg, wd)
    sem = asyncio.Semaphore(max(1, cfg.ocr.concurrency) * 2)

    async def read(img, request_id: str) -> str:
        data, _ = encode_image(img, cfg.ocr.max_side)
        async with sem:
            c = await client.backend.complete(data, cm.LINE_SYSTEM, cm.LINE_USER, request_id=request_id,
                                              max_tokens=400, temperature=0.0, extra=dict(cfg.ocr.extra_body))
        return " ".join(c.text.split())

    async def one(entry: PageEntry) -> None:
        gray = cv2.imread(str(wd.page_image(entry.index)), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return
        lines = cm.page_lines(gray)
        texts = await asyncio.gather(*(read(gray[y0:y1, x0:x1], f"{wd.stem(entry.index)}.line{k}")
                                       for k, ln in enumerate(lines) for x0, y0, x1, y1 in [ln.box]))
        wd.write_json(wd.page_tags(entry.index).with_name(f"{wd.stem(entry.index)}.lines.json"),
                      {"lines": [{"box": list(ln.box), "halves": [list(h) for h in ln.halves] if ln.halves else None,
                                  "text": t} for ln, t in zip(lines, texts)]})

    async def run_all() -> None:
        try:
            await asyncio.gather(*(one(e) for e in entries))
        finally:
            if own:
                await client.aclose()

    echo(f"Reading each printed line on its own ({len(entries)} pages)")
    asyncio.run(run_all())
    return len(entries)


def complete_pages(wd: WorkDir, cfg: BookConfig, *, echo: Echo = _quiet, client: OcrClient | None = None,
                   indices: list[int] | None = None) -> list[int]:
    """Completeness check (checks/completeness.py): read every printed line of the main text on
    its own, insert the lines the page reading skipped, and record runs of text no printed line
    matches (flagged as extra text by the checks). Reviewed pages are left alone. Line readings
    and what was done are kept in ``pages/NNNN.lines.json``. Returns the pages that gained lines."""
    from .checks import completeness as cm
    from .ocr.client import encode_image

    entries = [e for e in wd.load_manifest().pages if not e.blank and wd.page_tags(e.index).exists()
               and not wd.page_reviewed(e.index).exists() and (indices is None or e.index in indices)]
    if not entries:
        return []
    own = client is None
    client = client or OcrClient(cfg, wd)
    sem = asyncio.Semaphore(max(1, cfg.ocr.concurrency) * 2)

    async def read(img, request_id: str) -> str:
        data, _ = encode_image(img, cfg.ocr.max_side)
        async with sem:
            c = await client.backend.complete(data, cm.LINE_SYSTEM, cm.LINE_USER, request_id=request_id,
                                              max_tokens=400, temperature=0.0, extra=dict(cfg.ocr.extra_body))
        return " ".join(c.text.split())

    async def one(entry: PageEntry) -> int | None:
        i = entry.index
        record_path = wd.page_tags(i).with_name(f"{wd.stem(i)}.lines.json")
        gray = cv2.imread(str(wd.page_image(i)), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("done"):
                return None
            lines = [cm.Line(tuple(x["box"]), tuple(map(tuple, x["halves"])) if x["halves"] else None, x["text"])
                     for x in record["lines"]]
        else:
            lines = cm.page_lines(gray)
            texts = await asyncio.gather(*(read(gray[y0:y1, x0:x1], f"{wd.stem(i)}.line{k}")
                                           for k, ln in enumerate(lines) for x0, y0, x1, y1 in [ln.box]))
            for ln, t in zip(lines, texts):
                ln.text = t
        tags = wd.page_tags(i).read_text(encoding="utf-8")
        typical = float(np.median([len(ln.text.split()) for ln in lines])) if lines else 10.0
        considered = [k for k, ln in enumerate(lines)
                      if cm.plausible(ln.text, typical) and not re.fullmatch(r"[\s٠-٩0-9\-–—()]*", ln.text)]
        words = cm.page_words(tags)                       # where lines are inserted
        everything = cm.page_words(tags, main_only=False)  # what a printed line may be matched against
        missing_all, _ = cm.match([lines[k].text.split() for k in considered], [w for w, _a, _b in everything])
        found_nowhere = {considered[li] for li, _after in missing_all}
        main_considered = considered
        missing, extra = cm.match([lines[k].text.split() for k in main_considered], [w for w, _a, _b in words])
        missing = [(main_considered[li], after) for li, after in missing if main_considered[li] in found_nowhere]
        inserts = []
        if missing and len(missing) <= max(2, len(considered) // 2):
            for li, after in missing:
                ln = lines[li]
                halves = None
                if ln.halves:
                    (r0, r1), (l0, l1) = ln.halves
                    y0, y1 = ln.box[1], ln.box[3]
                    halves = tuple(await asyncio.gather(read(gray[y0:y1, r0:r1], f"{wd.stem(i)}.line{li}r"),
                                                        read(gray[y0:y1, l0:l1], f"{wd.stem(i)}.line{li}l")))
                inserts.append((after, ln.text, halves))
        new = cm.insert_lines(tags, words, inserts) if inserts else tags
        if inserts and not parse_tags(new, Page(index=i)).ok:
            inserts, new = [], tags
        record = {"done": True, "lines": [{"box": list(ln.box), "halves": [list(h) for h in ln.halves] if ln.halves else None,
                                           "text": ln.text} for ln in lines],
                  "inserted": [{"line": li, "after": after, "text": lines[li].text} for li, after in missing] if inserts else [],
                  "skipped_missing": [] if inserts else [{"line": li, "text": lines[li].text} for li, _a in missing],
                  "extra": [{"words": " ".join(w for w, _a, _b in words[a:b])} for a, b in extra]}
        wd.write_json(record_path, record)
        if inserts:
            wd.write_text(wd.page_tags(i), new)
            ocr = _load_ocr(wd, entry)
            if ocr is not None:
                parse_page(entry, ocr, wd, cfg)
                wd.page_checked(i).unlink(missing_ok=True)
            return i
        return None

    async def run_all() -> list[int]:
        try:
            done = await asyncio.gather(*(one(e) for e in entries))
        finally:
            if own:
                await client.aclose()
        return [i for i in done if i is not None]

    echo(f"Checking that every printed line was read ({len(entries)} pages)")
    changed = asyncio.run(run_all())
    echo(f"  inserted skipped lines on {len(changed)} page(s)")
    return changed


REPAIR_METHODS: dict[str, dict[str, Any]] = {
    "hires": {"ocr.max_side": 3400},       # small marks (note markers, dots) read better larger
    "bands": {"layout.regions": "bands"},  # main text in strips and the notes on their own
}


def _flag_score(page: Page | None) -> tuple[int, int]:
    """Lower is better: (problems found by the checks, minus words read)."""
    from .eval.metrics import page_text

    if page is None:
        return (10_000, 0)
    problems = sum(1 for f in page.flags if f.severity != "info")
    return (problems, -len(page_text(page, notes=True).split()))


def repair_flagged(wd: WorkDir, cfg: BookConfig, *, methods: tuple[str, ...] = ("hires", "bands"),
                   echo: Echo = _quiet, client: OcrClient | None = None) -> dict[str, list[int]]:
    """Unattended review: re-read every flagged page with each method in turn and keep a new
    reading only when the checks find fewer problems in it (ties go to the reading with more
    text). Reviewed pages are left alone. Returns the pages each method improved."""
    from .checks.flags import find_repetition

    improved: dict[str, list[int]] = {}
    files = (wd.page_tags, wd.page_ocr_info, wd.page_parsed, wd.page_regions, wd.page_checked)
    for method in methods:
        by_index = {e.index: e for e in wd.load_manifest().pages}
        flagged = [by_index[i] for i in wd.page_indices()
                   if i in by_index and not wd.page_reviewed(i).exists()
                   and (p := wd.best_page(i)) is not None and p.status not in ("clean", "reviewed")]
        if not flagged:
            break
        echo(f"Re-reading {len(flagged)} flagged page(s) ({method})")
        before = {e.index: _flag_score(wd.best_page(e.index)) for e in flagged}
        saved = {e.index: {f(e.index): f(e.index).read_bytes() for f in files if f(e.index).exists()} for e in flagged}
        results = ocr_pages(wd, apply_overrides(cfg, REPAIR_METHODS[method]), flagged, force=True, echo=echo,
                            client=client)
        for e in flagged:
            if e.index in results and not results[e.index].failed:
                parse_page(e, results[e.index], wd, cfg)
                wd.page_checked(e.index).unlink(missing_ok=True)
        check_book(wd, cfg)
        kept = []
        for e in flagged:
            tags = wd.page_tags(e.index)
            looped = tags.exists() and find_repetition(tags.read_text(encoding="utf-8"), 3)
            if not looped and _flag_score(wd.best_page(e.index)) < before[e.index]:
                kept.append(e.index)
                continue
            for f in files:
                if f(e.index) in saved[e.index]:
                    f(e.index).write_bytes(saved[e.index][f(e.index)])
                else:
                    f(e.index).unlink(missing_ok=True)
        check_book(wd, cfg)
        improved[method] = kept
        echo(f"  kept the new reading for {len(kept)} of them")
    return improved


FLAG_HELP = {
    "note_without_ref": "a note whose marker was not found in the text (the note is still in the book, unlinked)",
    "ref_without_note": "a marker whose note was not found (shown as a plain number)",
    "poem_rhyme": "a verse whose rhyme differs from the poem's (a misread or misplaced half-line)",
    "missing_notes": "the page shows a notes rule but no notes were read",
    "untagged_text": "text the model left outside any paragraph or verse",
    "wordcount_mismatch": "much less (or more) text than Tesseract finds on the page",
}


def write_quality_report(wd: WorkDir, cfg: BookConfig, pages: list[Page]) -> Path:
    """``out/<slug>-quality.md``: what the checks still flag after an unattended conversion, by
    kind and by page, for spot checks (nothing waits on it)."""
    from collections import Counter

    flagged = [p for p in pages if p.status not in ("clean", "reviewed")]
    kinds: Counter = Counter(code for p in flagged for code in {f.code for f in p.flags if f.severity != "info"})
    lines = [f"# Quality report: {cfg.title or cfg.slug}", "",
             f"{len(pages) - len(flagged)} of {len(pages)} pages pass every check; {len(flagged)} still have "
             "something the checks flag. The EPUB was built without review: each item below is in the "
             "book as the model read it.", "", "## By kind", ""]
    for code, n in kinds.most_common():
        help_text = FLAG_HELP.get(code)
        lines.append(f"- **{code}** ({n} pages)" + (f": {help_text}" if help_text else ""))
    lines += ["", "## By page", "", "| Page | Printed | Problems |", "|---|---|---|"]
    for p in flagged:
        msgs = "; ".join(sorted({f.message for f in p.flags if f.severity != "info"}))[:300]
        lines.append(f"| {p.index} | {p.printed_page or ''} | {msgs.replace('|', '/')} |")
    path = wd.root / "out" / f"{cfg.slug}-quality.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
