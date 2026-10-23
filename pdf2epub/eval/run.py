"""Run a model over the test set and score it (spec §9; phase 1 model comparison).

Test-set folder::

    pages/<id>.png            page images (ids are any names; sorted order is page order)
    truth/<id>.json           a reviewed Page (best: holds verified links), or
    truth/<id>.tags           ground truth as tags (marker links are scored, lemma links not), or
    truth/<id>.txt | .md      plain text (text metrics only, e.g. Misraj-DocOCR)
    meta.json                 optional: {"books": {name: {"overrides": {"text.commentary": "mixed"}}},
                                         "pages": {id: {"book": name, "index": n}}}

Each book's pages are checked together (neighbours only when their indices
are adjacent) with that book's settings. Results go to ``out_dir``:
``work/`` (predictions), ``scores.json`` and ``report.md``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from ..checks import run_checks
from ..checks.hashiya import link_notes
from ..checks.quran import check_quran
from ..config import BookConfig, apply_overrides
from ..model import Page, PageEntry, Paragraph, load_page
from ..ocr.client import OcrClient
from ..ocr.run import ocr_book
from ..parse import parse_tags
from ..workdir import WorkDir
from .metrics import TARGETS, SetScores, score_page


@dataclass
class TestPage:
    id: str
    image: Path
    book: str
    index: int
    truth: Page | None  # structured truth
    truth_text: str | None  # text-only truth
    reliable_links: set[str] | None  # note ids whose truth anchors are trustworthy (None = all)


def load_testset(path: Path) -> list[TestPage]:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8")) if (path / "meta.json").exists() else {}
    page_meta = meta.get("pages", {})
    out: list[TestPage] = []
    counters: dict[str, int] = {}
    for image in sorted((path / "pages").glob("*.png")):
        pid = image.stem
        info = page_meta.get(pid, {})
        book = info.get("book", "default")
        counters[book] = counters.get(book, 0) + 1
        index = int(info.get("index", counters[book]))
        truth: Page | None = None
        text: str | None = None
        reliable: set[str] | None = None
        tdir = path / "truth"
        if (tdir / f"{pid}.json").exists():
            truth = load_page(tdir / f"{pid}.json")
            truth.index = index
        elif (tdir / f"{pid}.tags").exists():
            truth = parse_tags((tdir / f"{pid}.tags").read_text(encoding="utf-8"), Page(index=index)).page
        else:
            for suffix in (".txt", ".md"):
                if (tdir / f"{pid}{suffix}").exists():
                    text = (tdir / f"{pid}{suffix}").read_text(encoding="utf-8")
                    break
        out.append(TestPage(pid, image, book, index, truth, text, reliable))
    return out


def _prepare_truth(tests: list[TestPage], cfg: BookConfig) -> None:
    """Truth from tags has no links or verse refs yet: add them (marker links only count)."""
    pages = [t.truth for t in tests if t.truth is not None and not any(n.link.ref for n in t.truth.notes())]
    if not pages:
        return
    check_quran(pages, cfg)
    link_notes(pages, cfg)
    for t in tests:
        if t.truth in pages:
            t.reliable_links = {n.id for n in t.truth.notes() if n.link.method == "marker"}


def evaluate(testset: Path, cfg: BookConfig, *, out_dir: Path, client: OcrClient | None = None,
             checks: Any = None) -> SetScores:
    """OCR, parse and check every test page with ``cfg``'s backend; score against the truth."""
    from ..pipeline import parse_page

    testset, out_dir = Path(testset), Path(out_dir)
    meta = json.loads((testset / "meta.json").read_text(encoding="utf-8")) if (testset / "meta.json").exists() else {}
    tests = load_testset(testset)
    checks = checks or (lambda pages, wd, c: run_checks(pages, wd, c))
    scores = SetScores()
    by_book: dict[str, list[TestPage]] = {}
    for t in tests:
        by_book.setdefault(t.book, []).append(t)
    for book, items in by_book.items():
        bcfg = apply_overrides(cfg, (meta.get("books", {}).get(book, {}) or {}).get("overrides", {}))
        bcfg = bcfg.model_copy(update={"slug": book})
        _prepare_truth(items, bcfg)
        wd = WorkDir(out_dir / "work" / book).ensure()
        entries = []
        for t in items:
            dest = wd.page_image(t.index)
            shutil.copyfile(t.image, dest)
            img = cv2.imread(str(dest), cv2.IMREAD_GRAYSCALE)
            entries.append(PageEntry(index=t.index, pdf_page=t.index, image=wd.rel(dest), width=int(img.shape[1]),
                                     height=int(img.shape[0]), sha256=t.id))
        # Without a client, ocr_book opens and closes its own in the same event loop.
        results = asyncio.run(ocr_book(entries, wd, bcfg, client=client))
        preds = [parse_page(e, results[e.index], wd, bcfg) for e in entries]
        checks(preds, wd, bcfg)
        for page in preds:
            wd.save_page(page, "checked")
        for t, pred in zip(items, preds):
            if t.truth is not None:
                scores.pages.append(score_page(t.truth, pred, page_id=t.id, reliable_links=t.reliable_links))
            elif t.truth_text is not None:
                truth = Page(index=t.index, blocks=[Paragraph(c=[t.truth_text])])
                s = score_page(truth, pred, page_id=t.id)
                s.blocks_total = s.blocks_correct = 0
                scores.pages.append(s)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores.json").write_text(json.dumps(scores.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "report.md").write_text(compare_report({cfg.ocr.model if cfg.ocr.backend == "openai" else cfg.ocr.backend: scores}),
                                       encoding="utf-8")
    return scores


def load_scores(path: Path) -> SetScores:
    return SetScores.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _fmt(value: float | None, pct: bool = True) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%" if pct else f"{value:g}"


def compare_report(results: dict[str, SetScores]) -> str:
    """Markdown table, one column per model, best whole-page CER first."""
    order = sorted(results, key=lambda k: (results[k].summary()["cer_whole_page"] is None,
                                           results[k].summary()["cer_whole_page"] or 0))
    rows = [
        ("Whole-page CER (ranking)", "cer_whole_page"), ("CER, diacritics removed", "cer_no_diacritics"),
        ("CER, diacritics kept", "cer_with_diacritics"), ("WER", "wer"), ("Note text CER", "cer_notes"),
        ("Quran wrong-verse rate", "quran_wrong_verse_rate"), ("Quran auto-resolved", "quran_auto_resolved_rate"),
        ("Poem verse accuracy", "poem_verse_accuracy"), ("Block-type accuracy", "block_type_accuracy"),
        ("Note recall", "note_recall"), ("Note link accuracy", "note_link_accuracy"),
        ("Region detection", "region_detection"), ("Missed-error rate", "missed_error_rate"),
    ]
    head = "| Metric | Target | " + " | ".join(order) + " |\n|---|---|" + "---|" * len(order) + "\n"
    lines = []
    for label, key in rows:
        target = TARGETS.get(key)
        tgt = f"{target[0]} {target[1] * 100:g}%" if target else ""
        lines.append(f"| {label} | {tgt} | " + " | ".join(_fmt(results[m].summary()[key]) for m in order) + " |")
    pages = " | ".join(str(results[m].summary()["pages"]) for m in order)
    flagged = " | ".join(str(results[m].summary()["flagged_pages"]) for m in order)
    lines.append(f"| Pages scored | | {pages} |")
    lines.append(f"| Pages flagged | | {flagged} |")
    return head + "\n".join(lines) + "\n"
