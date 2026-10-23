"""EPUB building (spec §6.6, §7)."""

from __future__ import annotations

import re
import zipfile

import pytest
from lxml import etree

from conftest import fixture_page, fixture_sequence, requires_epubcheck
from pdf2epub.arabic import ORNATE_CLOSE, ORNATE_OPEN
from pdf2epub.assemble import assemble_book
from pdf2epub.checks import run_checks
from pdf2epub.config import BookConfig
from pdf2epub.epub import build_epub, run_epubcheck
from pdf2epub.parse import parse_tags
from pdf2epub.model import Page

STAMP = "2026-10-19T00:00:00Z"


def make_book(cfg: BookConfig, pages=None):
    pages = pages or fixture_sequence() + [fixture_page("basic", 4), fixture_page("commentary", 5),
                                           fixture_page("margin", 6, width=1000, height=1400)]
    run_checks(pages, None, cfg, tesseract=False)
    return assemble_book(pages, cfg, slug="t", title="كتاب تجريبي", author="مؤلف")


def build(tmp_path, **settings):
    cfg = BookConfig(slug="t")
    for key, value in settings.items():
        group, name = key.split("__")
        setattr(getattr(cfg, group), name, value)
    book = make_book(cfg)
    out = build_epub(book, cfg, None, tmp_path / "t.epub", modified=STAMP)
    return out, cfg


def read(path, name):
    with zipfile.ZipFile(path) as zf:
        return zf.read(name).decode("utf-8")


def chapter(path):
    """All chapter files of the main text, concatenated (notes stay in their own chapter)."""
    with zipfile.ZipFile(path) as zf:
        names = sorted(n for n in zf.namelist() if n.startswith("OEBPS/text/ch"))
        return "\n".join(zf.read(n).decode("utf-8") for n in names)


def test_package_rules(tmp_path):
    out, _ = build(tmp_path)
    path = out[0]
    with zipfile.ZipFile(path) as zf:
        first = zf.infolist()[0]
        assert first.filename == "mimetype" and first.compress_type == zipfile.ZIP_STORED
        names = zf.namelist()
        for name in names:
            if name.endswith((".xhtml", ".opf", ".ncx", ".xml")):
                etree.fromstring(zf.read(name))  # well-formed XML
            if name.endswith(".xhtml"):
                html = zf.read(name).decode()
                assert 'xml:lang="ar"' in html and 'dir="rtl"' in html.split("<head>")[0]
    opf = read(path, "OEBPS/content.opf")
    assert 'page-progression-direction="rtl"' in opf
    assert "font/ttf" in opf and 'properties="nav"' in opf
    css = read(path, "OEBPS/css/book.css")
    assert not re.search(r"(?<![-\w])direction\s*:", css) and "unicode-bidi" not in css
    nav = read(path, "OEBPS/nav.xhtml")
    assert 'epub:type="page-list"' in nav and "١٢" in nav
    assert "tanzil.net" in read(path, "OEBPS/text/credits.xhtml")


def test_deterministic(tmp_path):
    a, _ = build(tmp_path / "a")
    b, _ = build(tmp_path / "b")
    assert a[0].read_bytes() == b[0].read_bytes()


def test_quran_brackets_logical_order_and_labels(tmp_path):
    out, _ = build(tmp_path, build__quran_refs="label")
    html = chapter(out[0])
    m = re.search(r'<span class="quran">(.)(.*?)(.)</span>', html)
    assert m and m.group(1) == ORNATE_OPEN and m.group(3) == ORNATE_CLOSE
    assert "[آل عمران: ٢٠٠]" in html
    out, _ = build(tmp_path / "n", build__quran_refs="none")
    assert 'class="qref"' not in chapter(out[0])


def test_poem_layouts(tmp_path):
    out, _ = build(tmp_path, build__poem_layout="two-column")
    html = chapter(out[0])
    row = re.search(r'<tr><td class="sadr">(.*?)</td><td class="ajuz">(.*?)</td></tr>', html)
    assert row and "دع الأيام" in row.group(1) and "القضاء" in row.group(2)
    out, _ = build(tmp_path / "s", build__poem_layout="stacked")
    assert 'class="poem stacked"' in chapter(out[0])


@pytest.mark.parametrize("mode,marker", [("popup", 'epub:type="footnote"'), ("inline", 'class="notes-inline"'),
                                         ("chapter-end", 'epub:type="endnotes"')])
def test_note_modes(tmp_path, mode, marker):
    out, _ = build(tmp_path, build__notes=mode)
    html = chapter(out[0])
    assert marker in html
    # Every noteref points at an element in the same file.
    with zipfile.ZipFile(out[0]) as zf:
        for name in [n for n in zf.namelist() if n.startswith("OEBPS/text/ch")]:
            one = zf.read(name).decode("utf-8")
            targets = set(re.findall(r'href="#(n-[^"]+)"', one))
            ids = set(re.findall(r'id="(n-[^"]+)"', one))
            assert targets <= ids, name  # every noteref points into the same file
    # Lemma notes open with the bold قوله template.
    assert '<b class="lemma">قوله: (والصبر ضياء)</b>' in html


def test_hidden_editor_notes_and_markers(tmp_path):
    out, _ = build(tmp_path, build__editor_notes="hide")
    html = chapter(out[0])
    assert "زيادة من نسخة أخرى" not in html and ">أ</a>" not in html
    out, _ = build(tmp_path / "p", build__note_markers="printed")
    assert ">*</a>" in chapter(out[0])


def test_unanchored_note_is_visible(tmp_path):
    cfg = BookConfig(slug="t")
    page = parse_tags('<p>نص</p><hashiya><note n="9">حاشية بلا علامة</note></hashiya>', Page(index=1)).page
    book = make_book(cfg, [page])
    out = build_epub(book, cfg, None, tmp_path / "u.epub", modified=STAMP)
    html = chapter(out[0])
    assert "حاشية بلا علامة" in html
    assert re.search(r'<div class="endnote" id="n-[^"]+">', html)  # no epub:type: never hidden


def test_separate_frame_epub(tmp_path):
    out, _ = build(tmp_path, build__frames="separate-epub")
    assert [p.name for p in out] == ["t.epub", "t-frame.epub"]
    with zipfile.ZipFile(out[1]) as zf:
        chapters = [n for n in zf.namelist() if n.startswith("OEBPS/text/ch")]
        assert any("متن الكتاب" in zf.read(n).decode() for n in chapters)
    with zipfile.ZipFile(out[0]) as zf:
        assert not any("متن الكتاب" in zf.read(n).decode() for n in zf.namelist() if n.endswith(".xhtml"))


@requires_epubcheck
@pytest.mark.slow
@pytest.mark.parametrize("mode", ["popup", "inline", "chapter-end"])
def test_epubcheck_passes(tmp_path, mode):
    out, _ = build(tmp_path, build__notes=mode, build__quran_refs="footnote" if mode == "inline" else "label",
                   build__frames="separate-epub" if mode == "chapter-end" else "section")
    for path in out:
        report = run_epubcheck(path)
        assert report.ok, report.errors + report.fatal
        assert not report.warnings, report.warnings
