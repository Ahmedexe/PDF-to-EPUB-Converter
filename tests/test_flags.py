"""Bad-page detection (spec §4) and page status."""

from __future__ import annotations

import cv2
import numpy as np
import pymupdf
import pytest

from conftest import fixture_page, requires_tesseract
from pdf2epub.checks import run_checks
from pdf2epub.checks.flags import check_health, find_repetition, finalize_status, model_word_count, tesseract_text
from pdf2epub.config import BookConfig
from pdf2epub.model import OcrInfo, Page
from pdf2epub.parse import parse_tags
from pdf2epub.workdir import WorkDir


def codes(page):
    return {f.code for f in page.flags}


def test_repetition():
    assert find_repetition("قال الشيخ رحمه الله تعالى " * 4) is not None
    looping = "\n".join(["هذا سطر يتكرر مرة بعد مرة"] * 3)
    assert find_repetition(looping) == "هذا سطر يتكرر مرة بعد مرة"
    assert find_repetition("كلام عادي لا تكرار فيه أبدا ولا يعود") is None


def test_quran_refrain_is_not_repetition():
    verse = "<quran>فَبِأَيِّ آلَاءِ رَبِّكُمَا تُكَذِّبَانِ</quran>"
    raw = "<p>" + " ثم ".join([verse] * 5) + "</p>"
    page = parse_tags(raw, Page(index=1)).page
    check_health(page, BookConfig(slug="x"))
    assert "repetition" not in codes(page)


def test_word_count_counts_everything():
    page = fixture_page("basic")
    # main text, notes, markers, header and page number all count
    assert model_word_count(page) > 40


def test_health_flags():
    cfg = BookConfig(slug="x")
    page = parse_tags("<p>نص <unclear>كلمه</unclear> آخر</p>", Page(index=1)).page
    page.ocr = OcrInfo(finish_reasons=["length"])
    check_health(page, cfg)
    assert {"unclear_text", "truncated"} <= codes(page)
    empty = Page(index=2, ocr=OcrInfo(finish_reasons=["stop"]))
    check_health(empty, cfg)
    assert "empty_output" in codes(empty)
    blank = Page(index=3, blank=True)
    check_health(blank, cfg)
    assert not blank.flags
    failed = Page(index=4, ocr=OcrInfo(finish_reasons=["error"]))
    check_health(failed, cfg)
    assert "ocr_failed" in codes(failed)


def test_wordcount_mismatch():
    cfg = BookConfig(slug="x")
    page = fixture_page("basic")
    check_health(page, cfg, tess_text="كلمة " * 200)
    assert "wordcount_mismatch" in codes(page)
    check_health(page, cfg, tess_text="كلمة " * model_word_count(page))
    assert "wordcount_mismatch" not in codes(page)


def test_finalize_status():
    cfg = BookConfig(slug="x")
    page = fixture_page("basic")
    finalize_status(page, cfg)
    assert page.status == "clean"
    page.add_flag("x", severity="warning")
    finalize_status(page, cfg)
    assert page.status == "flagged"
    page.status = "reviewed"
    finalize_status(page, cfg)
    assert page.status == "reviewed"


def test_run_checks_end_to_end_on_fixtures():
    cfg = BookConfig(slug="x")
    pages = [fixture_page(f"seq_{i:04d}", i) for i in (1, 2, 3)]
    run_checks(pages, None, cfg, tesseract=False)
    assert [p.status for p in pages] == ["clean", "clean", "clean"], [p.flags for p in pages]


@requires_tesseract
@pytest.mark.slow
def test_tesseract_cached(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=300)
    page.insert_htmlbox(pymupdf.Rect(20, 20, 400, 280), "<p>الحمد لله رب العالمين والصلاة والسلام على رسول الله</p>",
                        css="* {font-size: 20pt; direction: rtl;}")
    pix = page.get_pixmap(dpi=200)
    wd = WorkDir(tmp_path / "w").ensure()
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    cv2.imwrite(str(wd.page_image(1)), img)
    p = Page(index=1, image=wd.rel(wd.page_image(1)))
    text = tesseract_text(wd, p, BookConfig(slug="x"))
    assert text and "الله" in text
    assert wd.page_tesseract(1).exists()


def test_missing_notes_and_stray_markers(tmp_path):
    from pdf2epub.devtools.sample import make_sample_book
    from pdf2epub.ingest.render import ingest_pdf

    book = make_sample_book(tmp_path / "s")
    wd = WorkDir(tmp_path / "w")
    manifest = ingest_pdf(book.pdf, wd, BookConfig(slug="x"))
    entry = manifest.pages[1]  # page 2 has two foot notes under a rule
    # A model that read the text but merged the notes into it.
    page = parse_tags("<p>وقال النبي: «والصبر ضياء» (١)، أي نور يستضيء به صاحبه</p>"
                      "<p>(١) رواه مسلم في كتاب الطهارة.</p>",
                      Page(index=2, image=entry.image, width=entry.width, height=entry.height)).page
    check_health(page, BookConfig(slug="x"), wd=wd)
    assert {"missing_notes", "marker_in_text"} <= codes(page)
    # The same page read correctly raises neither.
    good = parse_tags((book.fixtures / "0002.tags").read_text(encoding="utf-8"),
                      Page(index=2, image=entry.image, width=entry.width, height=entry.height)).page
    check_health(good, BookConfig(slug="x"), wd=wd)
    assert not {"missing_notes", "marker_in_text"} & codes(good)
