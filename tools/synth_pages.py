"""Synthetic scanned pages with full diacritics, for teaching the model harakat (spec §8).

    python tools/synth_pages.py work/datasets OUT_DIR --pages 2000 [--seed 0]

Pages imitate a classical Arabic book edition: running header and page number,
numbered entries ("٢٢٧ •"), fully vowelled prose (Tashkeela), two-column poems
(well-vowelled poems from Ashaar), and foot notes under a short rule with (١)
markers in the text. Blocks are added only while they fit, so the ground truth is
exactly what is printed. Pages are rasterised and aged like scans (tilt, blur,
noise, contrast). Writes ``NNNN.png`` and ``pairs.jsonl`` (instruction
``keep_diacritics``: the targets keep every diacritic).
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import random
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np
import pymupdf
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdf2epub.arabic import diacritic_ratio, format_number  # noqa: E402
from pdf2epub.parse import parse_tags  # noqa: E402
from pdf2epub.model import Page  # noqa: E402

FONTS = ROOT / "pdf2epub" / "epub" / "fonts"
FACES = [("Amiri-Regular.ttf", "Amiri-Bold.ttf"), ("NotoNaskhArabic-Regular.ttf", "NotoNaskhArabic-Bold.ttf")]
W, H = 340, 500  # points: a small octavo page, like the Cairo editions


# --------------------------------------------------------------------------- text sources


def load_sources(datasets: Path, rng: random.Random) -> tuple[list[str], list[list[str]]]:
    """(vowelled prose sentences, vowelled poems as lists of half-lines)."""
    prose: list[str] = []
    for f in sorted(glob.glob(str(datasets / "tashkeela" / "*.parquet"))):
        for text in pq.read_table(f, columns=["text"]).column("text").to_pylist():
            if diacritic_ratio(text[5000:9000]) < 0.6:
                continue
            text = re.sub(r"\(\d+/\d+\)|[A-Za-z_/\\\\:.]{6,}|\[[^\]]*\]", " ", text)
            for sent in re.split(r"(?<=[.؟!:])\s+", text):
                sent = " ".join(sent.split())
                if 40 <= len(sent) <= 400 and diacritic_ratio(sent) >= 0.6:
                    prose.append(sent)
    poems: list[list[str]] = []
    for f in sorted(glob.glob(str(datasets / "ashaar" / "*.parquet"))):
        for v in pq.read_table(f, columns=["poem verses"]).column("poem verses").to_pylist():
            halves = v if isinstance(v, list) else ast.literal_eval(v)
            halves = [" ".join(h.split()) for h in halves if h and h.strip()]
            if len(halves) >= 4 and len(halves) % 2 == 0 and diacritic_ratio(" ".join(halves)) >= 0.6 \
                    and all(len(h) <= 42 for h in halves):
                poems.append(halves)
    rng.shuffle(prose)
    rng.shuffle(poems)
    return prose, poems


# --------------------------------------------------------------------------- page content


class Book:
    """Draws content in a book-like order with running entry numbers and note markers."""

    def __init__(self, prose: list[str], poems: list[list[str]], rng: random.Random):
        self.prose, self.poems, self.rng = prose, poems, rng
        self.entry = rng.randint(1, 400)
        self.page = rng.randint(5, 400)

    def sentence(self) -> str:
        return self.prose.pop() if self.prose else "بِسْمِ اللَّهِ"

    def paragraph(self) -> str:
        return " ".join(self.sentence() for _ in range(self.rng.randint(1, 3)))

    def poem(self, n: int) -> list[tuple[str, str]]:
        halves = self.poems.pop() if self.poems else ["أَلَا لَيْتَ شِعْرِي", "هَلْ أَبِيتَنَّ لَيْلَةً"]
        pairs = list(zip(halves[0::2], halves[1::2]))
        return pairs[:n]


def make_page(book: Book, digits_style: str, rng: random.Random) -> list[dict]:
    """A list of blocks (in order) to try to fit on one page."""
    blocks: list[dict] = []
    if rng.random() < 0.2:
        blocks.append({"t": "h", "text": f"{format_number(rng.randint(1, 60), digits_style)} - " + book.sentence()[:30].rsplit(" ", 1)[0]})
    for _ in range(12):
        r = rng.random()
        if r < 0.45:
            book.entry += 1
            lead = f"{format_number(book.entry, digits_style)} • " if rng.random() < 0.5 else ""
            blocks.append({"t": "p", "text": lead + book.paragraph()})
        elif r < 0.85:
            blocks.append({"t": "poem", "verses": book.poem(rng.randint(1, 6))})
        else:
            blocks.append({"t": "p", "text": book.sentence()})
    return blocks


def _marker(n: int, style: str) -> str:
    return format_number(n, style)


def place_markers(blocks: list[dict], notes_wanted: int, style: str, rng: random.Random) -> list[tuple[int, int, int, str]]:
    """Choose where markers go: (block, verse/word position, half, sign)."""
    spots = []
    for bi, b in enumerate(blocks):
        if b["t"] == "p":
            words = len(b["text"].split())
            spots += [(bi, w, -1) for w in range(3, words, 7)]
        elif b["t"] == "poem":
            spots += [(bi, vi, 1) for vi in range(len(b["verses"]))]
    rng.shuffle(spots)
    chosen = sorted(spots[:notes_wanted])
    return [(bi, pos, half, _marker(k + 1, style)) for k, (bi, pos, half) in enumerate(chosen)]


# --------------------------------------------------------------------------- rendering


def css(face: tuple[str, str], size: float, rng: random.Random) -> str:
    lh = rng.choice([1.45, 1.55, 1.65])
    return f"""
@font-face {{font-family: f; src: url({face[0]});}}
@font-face {{font-family: f; src: url({face[1]}); font-weight: bold;}}
* {{font-family: f; direction: rtl;}}
body {{font-size: {size}pt; line-height: {lh};}}
p {{text-align: justify; margin: 0 0 3pt 0; text-indent: 12pt;}}
h3 {{text-align: center; font-size: {size + 3}pt; margin: 4pt 0;}}
table {{width: 100%; margin: 2pt 0;}}
td {{text-align: center; width: 50%;}}
.m {{font-size: {size - 3}pt;}}
.notes p {{font-size: {size - 2.5}pt; text-indent: 0; margin: 0;}}
.hdr {{text-align: center; font-size: {size - 2}pt;}}
"""


def block_html(b: dict, markers: dict[tuple[int, int, int], str]) -> tuple[str, str]:
    """(HTML, tags) for one block, with markers inserted."""
    bi = b["i"]
    if b["t"] == "h":
        return f"<h3>{escape(b['text'])}</h3>", f"<h2>{escape(b['text'])}</h2>"
    if b["t"] == "p":
        words = b["text"].split()
        hw, tw = [], []
        for wi, w in enumerate(words):
            hw.append(escape(w))
            tw.append(escape(w))
            sign = markers.get((bi, wi, -1))
            if sign:
                hw[-1] += f'<span class="m">({sign})</span>'
                tw[-1] += f'<ref n="{sign}"/>'
        return f"<p>{' '.join(hw)}</p>", f"<p>{' '.join(tw)}</p>"
    rows, verses = [], []
    for vi, (s, a) in enumerate(b["verses"]):
        sign = markers.get((bi, vi, 1))
        hm = f'<span class="m">({sign})</span>' if sign else ""
        tm = f'<ref n="{sign}"/>' if sign else ""
        rows.append(f"<tr><td>{escape(s)}</td><td>{escape(a)}{hm}</td></tr>")
        verses.append(f"<v><s>{escape(s)}</s><a>{escape(a)}{tm}</a></v>")
    return f"<table>{''.join(rows)}</table>", f"<poem>{''.join(verses)}</poem>"


def fits(html: str, style: str, rect: pymupdf.Rect, archive: pymupdf.Archive) -> bool:
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    spare, scale = page.insert_htmlbox(rect, html, css=style, archive=archive, scale_low=1)
    doc.close()
    return spare >= 0 and scale >= 0.999


def render(index: int, book: Book, notes_src: Book, rng: random.Random, archive: pymupdf.Archive,
           out: Path, dpi: int) -> dict | None:
    face = rng.choice(FACES)
    size = rng.choice([11.0, 11.5, 12.0, 12.5])
    style = css(face, size, rng)
    digits = "arabic-indic"
    margin = rng.choice([24, 28, 32])
    header = notes_src.sentence()[:28].rsplit(" ", 1)[0]
    book.page += 1
    pagenum = format_number(book.page, digits)
    candidates = make_page(book, digits, rng)
    for k, b in enumerate(candidates):
        b["i"] = k
    n_notes = rng.choice([0, 1, 2, 2, 3, 4])
    marks = place_markers(candidates, n_notes, digits, rng)
    notes_text = [notes_src.sentence()[: rng.randint(60, 220)] for _ in marks]
    markers = {(bi, pos, half): sign for bi, pos, half, sign in marks}
    # Notes first (they take the foot), then as many blocks as fit above them.
    notes_html = '<div class="notes">' + "".join(
        f"<p>({sign}) {escape(text)}</p>" for (_bi, _p, _h, sign), text in zip(marks, notes_text)) + "</div>"
    body_top, foot = 40, H - 30
    if marks:
        doc = pymupdf.open()
        probe = doc.new_page(width=W, height=H)
        spare, _ = probe.insert_htmlbox(pymupdf.Rect(margin, 0, W - margin, 300), notes_html, css=style, archive=archive)
        doc.close()
        notes_h = 300 - spare
        notes_top = foot - notes_h
    else:
        notes_top = foot
    rect = pymupdf.Rect(margin, body_top, W - margin, notes_top - 10)
    kept: list[dict] = []
    for b in candidates:
        html = "".join(block_html(x, markers)[0] for x in kept + [b])
        if fits(html, style, rect, archive):
            kept.append(b)
        else:
            break
    if not kept:
        return None
    kept_ids = {b["i"] for b in kept}
    used = [(m, t) for m, t in zip(marks, notes_text) if m[0] in kept_ids]
    markers = {(bi, pos, half): sign for (bi, pos, half, sign), _t in used}
    # Renumber the markers that survive, in order.
    renum = {}
    for k, ((bi, pos, half, sign), _t) in enumerate(used, 1):
        renum[(bi, pos, half)] = format_number(k, digits)
    markers = renum
    notes_html = '<div class="notes">' + "".join(
        f"<p>({markers[(bi, pos, half)]}) {escape(text)}</p>" for (bi, pos, half, _s), text in used) + "</div>"
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    page.insert_htmlbox(pymupdf.Rect(margin, 12, W - margin, 32),
                        f'<p class="hdr">{escape(header)}</p>', css=style, archive=archive)
    page.insert_htmlbox(pymupdf.Rect(W - margin - 40, 12, W - margin, 32) if rng.random() < 0.5
                        else pymupdf.Rect(margin, 12, margin + 40, 32),
                        f'<p class="hdr">{pagenum}</p>', css=style, archive=archive)
    body_html = "".join(block_html(b, markers)[0] for b in kept)
    page.insert_htmlbox(rect, body_html, css=style, archive=archive)
    if used:
        page.draw_line((W - margin - 90, notes_top - 4), (W - margin, notes_top - 4), width=0.5)
        page.insert_htmlbox(pymupdf.Rect(margin, notes_top, W - margin, foot + 20), notes_html, css=style, archive=archive)
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    doc.close()
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    img = age(img, rng)
    path = out / f"{index:05d}.png"
    cv2.imwrite(str(path), img)
    tags = [f"<header>{escape(header)}</header>"] + [block_html(b, markers)[1] for b in kept]
    if used:
        tags.append('<hashiya pos="bottom">\n' + "\n".join(
            f'<note n="{markers[(bi, pos, half)]}">{escape(text)}</note>' for (bi, pos, half, _s), text in used)
            + "\n</hashiya>")
    tags.append(f"<pagenum>{pagenum}</pagenum>")
    target = "\n".join(tags) + "\n"
    result = parse_tags(target, Page(index=1))
    if not result.ok or result.page.flags:
        return None
    return {"image": str(path), "target": target, "instruction": "keep_diacritics"}


def age(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Make a clean render look like a scan of an old printed page."""
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    angle = rng.uniform(-1.0, 1.0)
    m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=255)
    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 0.9))
    if rng.random() < 0.4:  # ink spread or thinning
        k = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)
    img = img.astype(np.int16) + nrng.normal(0, rng.uniform(3, 12), img.shape)
    img = img - rng.uniform(0, 25)  # paper tone
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 90)])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--pages", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    prose, poems = load_sources(args.datasets, rng)
    print(f"sources: {len(prose)} vowelled sentences, {len(poems)} vowelled poems", flush=True)
    half = len(prose) // 2
    book = Book(prose[:half], poems, rng)
    notes = Book(prose[half:], [], rng)
    archive = pymupdf.Archive(str(FONTS))
    args.out.mkdir(parents=True, exist_ok=True)
    lines = []
    i = 0
    while len(lines) < args.pages and book.prose:
        i += 1
        rec = render(i, book, notes, rng, archive, args.out, args.dpi)
        if rec:
            lines.append(json.dumps(rec, ensure_ascii=False))
        if len(lines) % 100 == 0 and rec:
            print(f"{len(lines)} pages", flush=True)
    (args.out / "pairs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} pages")


if __name__ == "__main__":
    main()
