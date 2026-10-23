"""Scoring (spec §9)."""

from __future__ import annotations

import json
import shutil

import pymupdf
import pytest

from conftest import fixture_page
from pdf2epub.checks import run_checks
from pdf2epub.config import BookConfig
from pdf2epub.devtools.sample import make_sample_book
from pdf2epub.eval import cer, compare_report, evaluate, load_scores, page_text, score_page, wer
from pdf2epub.eval.metrics import review_minutes_per_100_pages
from pdf2epub.model import Page
from pdf2epub.parse import parse_tags


def test_cer_wer():
    assert cer("abcd", "abcd") == 0.0
    assert cer("abcd", "abed") == 0.25
    assert cer("", "") == 0.0 and cer("", "x") == 1.0
    assert wer("a b c d", "a x c d") == 0.25


def checked(name, index=1):
    page = fixture_page(name, index)
    run_checks([page], None, BookConfig(slug="e"), tesseract=False)
    return page


def test_perfect_prediction_scores_perfectly():
    truth, pred = checked("basic"), checked("basic")
    s = score_page(truth, pred)
    assert s.edits_nodia == 0 and s.edits_dia == 0 and s.word_edits == 0
    assert s.quran_total == 1 and s.quran_auto == 1 and s.quran_wrong == 0
    assert s.poem_verses == 2 and s.poem_correct == 2
    assert s.notes_found == s.notes_truth == 2 and s.links_correct == s.links_total == 2
    assert s.blocks_correct == s.blocks_total and not s.has_error


def test_errors_are_measured():
    truth = checked("basic")
    damaged = fixture_page("basic").model_copy(deep=True)
    raw = (open("tests/fixtures/tags/basic.tags", encoding="utf-8").read()
           .replace("وتعظم في عين الصغير صغارها</s><a>وتصغر", "وتصغر في عين العظيم العظائم</s><a>وتعظم")
           .replace("الصَّابِرُونَ", "الصابرين")
           .replace('<note n="١">رواه مسلم في كتاب الطهارة.</note>', ""))
    damaged = parse_tags(raw, Page(index=1)).page
    run_checks([damaged], None, BookConfig(slug="e"), tesseract=False)
    s = score_page(truth, damaged)
    assert s.poem_correct == 1  # the swapped verse is wrong
    assert s.notes_found == 1 and s.has_error
    assert s.edits_dia > 0


def test_diacritics_scored_separately():
    t = parse_tags("<p>الصَّبْرُ ضِيَاءٌ</p>", Page(index=1)).page
    p = parse_tags("<p>الصبر ضياء</p>", Page(index=1)).page
    s = score_page(t, p)
    assert s.edits_nodia == 0 and s.edits_dia > 0


def test_page_text_order():
    page = fixture_page("basic")
    text = page_text(page, notes=True)
    assert text.index("باب الصبر") < text.index("رواه مسلم")
    assert "كتاب الصبر" not in text  # running header excluded


def test_review_time():
    assert review_minutes_per_100_pages([{"seconds": 60}, {"seconds": 120}], 10) == pytest.approx(30.0)


def test_evaluate_with_fixture_backend(tmp_path):
    book = make_sample_book(tmp_path / "sample")
    ts = tmp_path / "testset"
    (ts / "pages").mkdir(parents=True)
    (ts / "truth").mkdir()
    doc = pymupdf.open(str(book.pdf))
    for i, page in enumerate(doc, 1):
        page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY).save(str(ts / "pages" / f"{i:04d}.png"))
        shutil.copyfile(book.truth / f"{i:04d}.tags", ts / "truth" / f"{i:04d}.tags")
    # A model that misreads page 3's poem: swap the halves of the first verse.
    preds = tmp_path / "preds"
    shutil.copytree(book.fixtures, preds)
    p3 = (preds / "0003.tags").read_text(encoding="utf-8")
    p3 = p3.replace("<s>على قدر أهل العزم تأتي العزائم</s><a>وتأتي على قدر الكرام المكارم</a>",
                    "<s>وتأتي على قدر الكرام المكارم</s><a>على قدر أهل العزم تأتي العزائم</a>")
    (preds / "0003.tags").write_text(p3, encoding="utf-8")
    (ts / "meta.json").write_text(json.dumps({"books": {"sabr": {"overrides": {}}},
                                              "pages": {f"{i:04d}": {"book": "sabr", "index": i} for i in range(1, 6)}}))
    cfg = BookConfig(slug="e")
    cfg.ocr.backend, cfg.ocr.fixtures = "fixture", str(preds)
    cfg.checks.tesseract = False
    scores = evaluate(ts, cfg, out_dir=tmp_path / "out")
    summary = scores.summary()
    assert summary["pages"] == 5
    assert summary["poem_verse_accuracy"] == pytest.approx(3 / 4)
    assert summary["quran_wrong_verse_rate"] == 0.0 and summary["note_recall"] == 1.0
    assert (tmp_path / "out" / "report.md").exists()
    loaded = load_scores(tmp_path / "out" / "scores.json")
    report = compare_report({"model-a": loaded, "model-b": scores})
    assert "| Poem verse accuracy |" in report and "model-a" in report
