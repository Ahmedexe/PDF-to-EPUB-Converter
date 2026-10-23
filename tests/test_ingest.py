"""Ingest, image clean-up and the sample book."""

from __future__ import annotations

import cv2
import numpy as np
import pymupdf
import pytest

from pdf2epub.config import BookConfig
from pdf2epub.devtools.sample import make_sample_book
from pdf2epub.ingest.cleanup import (
    content_box,
    crop_borders,
    deskew,
    estimate_skew,
    find_spread_split,
    is_blank,
    rotate,
    split_spread,
)
from pdf2epub.ingest.render import IngestError, ingest_pdf
from pdf2epub.parse import parse_tags
from pdf2epub.model import Page
from pdf2epub.workdir import WorkDir


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return make_sample_book(tmp_path_factory.mktemp("sample"))


def text_page(w=1200, h=1700, lines=25, x0=150, x1=1050, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 255, np.uint8)
    for k in range(lines):
        y = 150 + k * 55
        x = x0
        while x < x1:
            ww = int(rng.integers(30, 90))
            cv2.rectangle(img, (x, y), (min(x + ww, x1), y + 18), 0, -1)
            x += ww + int(rng.integers(12, 25))
    return img


def test_blank_detection():
    assert is_blank(np.full((800, 600), 250, np.uint8))
    assert not is_blank(text_page())
    dark_border = np.full((800, 600), 250, np.uint8)
    dark_border[:, :25] = 10
    assert is_blank(dark_border)


def test_skew_is_found_and_corrected():
    page = text_page()
    tilted = rotate(page, 2.0)
    assert abs(estimate_skew(tilted) - 2.0) < 0.3
    fixed, angle = deskew(tilted)
    assert abs(angle - 2.0) < 0.3 and abs(estimate_skew(fixed)) < 0.3
    assert deskew(page)[1] == 0.0


def test_crop_keeps_margin_text_and_removes_border():
    page = text_page()
    cv2.rectangle(page, (20, 300), (60, 700), 0, -1)  # a margin note
    page[:, -15:] = 0  # dark scanner edge
    cropped, (x0, y0, x1, y1) = crop_borders(page, pad=10)
    assert x0 <= 12 and x1 < page.shape[1] - 10
    assert content_box(page)[0] <= 22  # the margin note is kept


def test_spread_split_right_page_first():
    left, right = text_page(seed=1), text_page(seed=2)
    right[100:140, 600:700] = 0  # mark the right-hand page
    spread = np.hstack([left, np.full((1700, 80), 255, np.uint8), right])
    x = find_spread_split(spread)
    assert x is not None and 1150 < x < 1330
    first, second = split_spread(spread, x)
    assert first[100:140, :].mean() < second[100:140, :].mean()  # the marked page comes first
    assert find_spread_split(text_page()) is None


def make_pdf(path, pages):
    doc = pymupdf.open()
    for img in pages:
        ok, buf = cv2.imencode(".png", img)
        h, w = img.shape[:2]
        page = doc.new_page(width=595 * w / max(w, h), height=595 * h / max(w, h))
        page.insert_image(page.rect, stream=buf.tobytes())
    doc.save(str(path))


def test_ingest_sample_book_extracts_scans(sample, tmp_path):
    wd = WorkDir(tmp_path / "w")
    manifest = ingest_pdf(sample.pdf, wd, BookConfig(slug="s"))
    assert [p.index for p in manifest.pages] == [1, 2, 3, 4, 5]
    assert all(p.source == "extract" and p.side == "full" and not p.blank for p in manifest.pages)
    assert all(wd.page_image(p.index).exists() for p in manifest.pages)
    assert manifest.pages[0].dpi and 190 <= manifest.pages[0].dpi <= 210
    # A second run does no work and keeps everything.
    before = [p.sha256 for p in manifest.pages]
    again = ingest_pdf(sample.pdf, wd, BookConfig(slug="s"))
    assert [p.sha256 for p in again.pages] == before


def test_ingest_vector_page_renders(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello")
    pdf = tmp_path / "v.pdf"
    doc.save(str(pdf))
    m = ingest_pdf(pdf, WorkDir(tmp_path / "w"), BookConfig(slug="v"))
    assert m.pages[0].source == "render" and m.pages[0].dpi == 300


def test_ingest_spreads_and_blank(tmp_path):
    spread = np.hstack([text_page(seed=1), np.full((1700, 80), 255, np.uint8), text_page(seed=2)])
    pdf = tmp_path / "s.pdf"
    make_pdf(pdf, [spread, np.full((1700, 1200), 255, np.uint8)])
    m = ingest_pdf(pdf, WorkDir(tmp_path / "w"), BookConfig(slug="s"))
    assert [(p.index, p.pdf_page, p.side) for p in m.pages] == [(1, 1, "right"), (2, 1, "left"), (3, 2, "full")]
    assert m.pages[2].blank


def test_indices_are_stable(tmp_path, sample):
    wd = WorkDir(tmp_path / "w")
    m = ingest_pdf(sample.pdf, wd, BookConfig(slug="s"), pdf_pages={1, 2})
    assert [p.index for p in m.pages] == [1, 2]
    m = ingest_pdf(sample.pdf, wd, BookConfig(slug="s"))  # appending later pages is fine
    assert [p.pdf_page for p in m.pages] == [1, 2, 3, 4, 5]
    wd2 = WorkDir(tmp_path / "w2")
    ingest_pdf(sample.pdf, wd2, BookConfig(slug="s"), pdf_pages={4, 5})
    with pytest.raises(IngestError):
        ingest_pdf(sample.pdf, wd2, BookConfig(slug="s"))  # pages 1-3 would shift indices
    m = ingest_pdf(sample.pdf, wd2, BookConfig(slug="s"), force=True)
    assert [p.pdf_page for p in m.pages] == [1, 2, 3, 4, 5]


def test_sample_ground_truth_parses_cleanly(sample):
    for path in sorted(sample.truth.glob("*.tags")):
        page = parse_tags(path.read_text(encoding="utf-8"), Page(index=int(path.stem))).page
        assert not page.flags and page.blocks
