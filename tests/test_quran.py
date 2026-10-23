"""Quran matching against Tanzil (spec §5.1)."""

from __future__ import annotations

import json
import unicodedata

import pytest

from conftest import fixture_page
from pdf2epub.arabic import normalize_for_match
from pdf2epub.checks.quran import QuranIndex, check_quran, detect_quran_style, tag_untagged_verses
from pdf2epub.config import BookConfig
from pdf2epub.model import Page, Paragraph, Quran, QuranBlock, Ref, block_text
from pdf2epub.parse import parse_tags


@pytest.fixture(scope="module")
def idx() -> QuranIndex:
    return QuranIndex.load()


def nfc(text: str | None) -> str:
    return unicodedata.normalize("NFC", text or "")


def cfg(style: str = "auto") -> BookConfig:
    c = BookConfig(slug="q")
    c.text.quran_style = style  # type: ignore[assignment]
    return c


def test_tanzil_data_and_basmala(idx):
    assert idx.verse_text(1, 1, "simple-clean") == "بسم الله الرحمن الرحيم"
    # Verse 1 of other suras is stored without the prepended basmala.
    assert idx.verse_text(2, 1, "simple-clean") == "الم"
    assert idx.verse_text(9, 1, "simple-clean").startswith("براءة")
    assert idx.sura_name(2) == "البقرة"


@pytest.mark.parametrize("text,ref,status", [
    ("إِنَّمَا يُوَفَّى الصَّابِرُونَ أَجْرَهُم بِغَيْرِ حِسَابٍ", "39:10", "verified"),
    ("انما يوفى الصابرون اجرهم بغير حساب", "39:10", "verified"),
    ("انما يوفي الصابرين اجرهم بغبر حساب", "39:10", "corrected"),
    ("وَبَشِّرِ الصَّابِرِينَ الَّذِينَ إِذَا أَصَابَتْهُم مُّصِيبَةٌ قَالُوا إِنَّا لِلَّهِ وَإِنَّا إِلَيْهِ رَاجِعُونَ",
     "2:155-156", "verified"),
    ("ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ", "1:2", "verified"),
    ("يا أيها الذين آمنوا اصبروا وصابروا ورابطوا", "3:200", "verified"),
    ("بسم الله الرحمن الرحيم", "1:1", "verified"),
    ("الم ذلك الكتاب لا ريب فيه", "2:1-2", "verified"),
    ("قل أعوذ برب الفلق", "113:1", "verified"),
    ("والعصر إن الإنسان لفي خسر", "103:1-2", "verified"),
])
def test_match(idx, text, ref, status):
    m = idx.match(text, style="simple-plain")
    assert (m.ref, m.status) == (ref, status)
    assert m.text  # replaced


def test_not_quran_is_flagged(idx):
    m = idx.match("هذا كلام عربي ليس من القرآن الكريم في شيء", style="simple-plain")
    assert m.status == "flagged" and m.text is None


def test_replacement_uses_book_style_and_quoted_words_only(idx):
    m = idx.match("انما يوفى الصابرون اجرهم بغير حساب", style="simple-plain")
    assert nfc(m.text) == nfc("إِنَّمَا يُوَفَّى الصَّابِرُونَ أَجْرَهُم بِغَيْرِ حِسَابٍ")  # only the end of 39:10
    m = idx.match("انما يوفى الصابرون اجرهم بغير حساب", style="uthmani-plain")
    assert nfc(m.text) == nfc("إِنَّمَا يُوَفَّى ٱلصَّٰبِرُونَ أَجْرَهُم بِغَيْرِ حِسَابٍ")
    m = idx.match("انما يوفى الصابرون اجرهم بغير حساب", style="simple-clean")
    assert m.text == "إنما يوفى الصابرون أجرهم بغير حساب"


def test_cross_family_mapping(idx):
    m = idx.match("يا أيها الذين آمنوا اصبروا", style="uthmani-plain")
    assert nfc(m.text).startswith(nfc("يَٰٓأَيُّهَا")) and nfc(m.text).endswith(nfc("ٱصْبِرُوا۟"))
    m = idx.match("يَٰٓأَيُّهَا ٱلَّذِينَ ءَامَنُوا۟ ٱصْبِرُوا۟", style="simple-plain")
    assert nfc(m.text) == nfc("يَا أَيُّهَا الَّذِينَ آمَنُوا اصْبِرُوا")


def test_repeated_passages_are_identical_alternatives(idx):
    m = idx.match("فبأي آلاء ربكما تكذبان", style="simple-plain")
    assert m.status == "verified" and m.ref == "55:13" and "55:16" in m.alternatives


def test_printed_verse_numbers_are_kept(idx):
    m = idx.match("الحمد لله رب العالمين (٢) الرحمن الرحيم (٣)", style="simple-plain")
    assert m.ref == "1:2-3"
    assert nfc(m.text) == nfc("الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ (٢) الرَّحْمَٰنِ الرَّحِيمِ (٣)")


def test_short_passages(idx):
    m = idx.match("واصبر", style="simple-plain")
    assert m.status == "verified"  # every location has the same Tanzil text
    m = idx.match("واصبرو", style="simple-plain")
    assert m.status == "flagged"


def test_hint_from_printed_reference(idx):
    m = idx.match("إن الله مع الصابرين", style="simple-plain", hint=(8, 46))
    assert m.ref == "8:46"


def test_detect_style():
    assert detect_quran_style(["ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ"]) == "uthmani-plain"
    assert detect_quran_style(["ذَٰلِكَ الْكِتَابُ لَا رَيْبَ ۛ فِيهِ ۛ هُدًى"]) == "simple"
    assert detect_quran_style(["الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"]) == "simple-plain"
    assert detect_quran_style(["الحمد لله رب العالمين"]) == "simple-clean"


def test_check_quran_on_pages(tmp_path, idx):
    pages = [fixture_page("basic", 1), fixture_page("commentary", 2), fixture_page("seq_0003", 3)]
    log = tmp_path / "quran.jsonl"
    style = check_quran(pages, cfg(), log_path=log, index=idx)
    assert style == "simple-plain"
    spans = [n for p in pages for b in p.blocks for n in getattr(b, "c", []) if isinstance(n, Quran)]
    assert spans and all(s.status in ("verified", "corrected") for s in spans)
    refs = {s.ref for s in spans}
    assert {"39:10", "1:2", "2:155-156", "14:7"} <= refs
    # The untagged ﴿…﴾ in seq_0003 was wrapped and its brackets removed.
    assert "﴿" not in block_text(pages[2].blocks[2])
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == len(spans)
    # Idempotent: a second run gives the same result.
    before = [s.model_dump() for s in spans]
    check_quran(pages, cfg(), index=idx)
    spans2 = [n for p in pages for b in p.blocks for n in getattr(b, "c", []) if isinstance(n, Quran)]
    assert [s.model_dump() for s in spans2] == before


def test_flagged_span_keeps_ocr_text_and_flags_page(idx):
    page = parse_tags("<p>قال تعالى: <quran>هذا كلام عربي ليس من القرآن الكريم</quran></p>", Page(index=1)).page
    check_quran([page], cfg("simple-plain"), index=idx)
    span = page.blocks[0].c[1]
    assert span.status == "flagged" and span.text == "هذا كلام عربي ليس من القرآن الكريم"
    assert [f.code for f in page.flags] == ["quran_unmatched"]
    assert page.flags[0].severity == "warning"


def test_no_span_left_unchecked_and_short_flag_is_warning(idx):
    page = parse_tags("<p>قال: <quran>واصبرو</quran> و<quran>الم</quran></p>", Page(index=1)).page
    check_quran([page], cfg("simple-plain"), index=idx)
    spans = [n for n in page.blocks[0].c if isinstance(n, Quran)]
    assert all(s.status != "unchecked" for s in spans)
    short = [f for f in page.flags if f.code == "quran_short"]
    assert short and short[0].severity == "warning"


def test_reviewed_and_accepted_spans_untouched(idx):
    page = parse_tags("<quran>كلام ليس من القرآن الكريم أبدا</quran>", Page(index=1)).page
    page.blocks[0].c[0].status = "accepted"
    check_quran([page], cfg("simple-plain"), index=idx)
    assert page.blocks[0].c[0].status == "accepted" and not page.flags
    reviewed = parse_tags("<quran>كلام ليس من القرآن الكريم أبدا</quran>", Page(index=2)).page
    reviewed.status = "reviewed"
    check_quran([reviewed], cfg("simple-plain"), index=idx)
    assert reviewed.blocks[0].c[0].status == "unchecked" and not reviewed.flags


def test_tag_untagged_keeps_refs():
    page = Page(index=1, blocks=[Paragraph(id="p0001-b1", c=["قال تعالى: ﴿الحمد لله", Ref(id="r", n="1"),
                                                               " رب العالمين﴾ ثم"])])
    assert tag_untagged_verses(page, ["﴿﴾", "﴾﴿"]) == 1
    kinds = [type(n).__name__ for n in page.blocks[0].c]
    assert kinds == ["str", "Quran", "Ref", "str"]
    assert page.blocks[0].c[1].text == "الحمد لله رب العالمين"


def test_quran_in_notes_is_checked(idx):
    page = parse_tags('<p>نص<ref n="1"/></p><hashiya><note n="1">قال تعالى: <quran>فبأي آلاء ربكما تكذبان</quran></note></hashiya>',
                      Page(index=1)).page
    check_quran([page], cfg("simple-plain"), index=idx)
    span = [n for n in page.notes()[0].blocks[0].c if isinstance(n, Quran)][0]
    assert span.status == "verified" and span.ref == "55:13"
    assert normalize_for_match(span.text) == "فباي الاء ربكما تكذبان"


def test_quran_block_replacement(idx):
    page = fixture_page("basic", 7)
    check_quran([page], cfg("simple-plain"), index=idx)
    block = page.blocks[2]
    assert isinstance(block, QuranBlock) and block.c[0].ref == "39:10"
    assert nfc(block.c[0].ocr_text) == nfc("إِنَّمَا يُوَفَّى الصَّابِرُونَ أَجْرَهُم بِغَيْرِ حِسَابٍ")
