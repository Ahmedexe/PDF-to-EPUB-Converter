"""Synthetic book pages in many typefaces, rendered from real Shamela text, on Modal (spec §8).

    .venv/bin/modal deploy train/synth_modal.py
    python -c "import modal; modal.Function.from_name('pdf2epub-synth', 'make').spawn('synth-fonts', 6000)"

1. ``fetch``: pages of Shamela books from the genres heritage books come from (tafsir, hadith,
   sira, history, biographies, adab, poetry, language, fiqh …), a few random pages per book,
   saved in the ``pdf2epub-train`` volume (``shamela-text/<book>/<n>.html``). The test books'
   editions are never taken.
2. ``render``: each synthetic page takes a run of one book's text (paragraphs, verses split at
   "...", Qur'an in ﴿﴾, foot notes with their markers, as the edition prints them, with its own
   punctuation and diacritics) and lays it out like a printed page in one of 17 open Arabic
   typefaces (mostly Naskh), one of four page sizes, with a running header, page number, rule
   style, entry numbers and sometimes margin line numbers. The render is aged like a scan
   (tilt, curl, blur, ink spread or fading, uneven light, noise, bilevel or JPEG). The target is
   exactly what is printed, in the tag format; margin line numbers are not transcribed.
3. ``make``: fetch (once) + render in parallel shards, writing a training bundle
   (``/data/<name>/images``, ``train.jsonl``) with the same prompts ``train/bundle.py`` uses.
   ``--heldout`` books are kept apart for a font-by-font test (``test.jsonl``).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import modal

FONTS_COMMIT = "bd8f81ddb5c74d5c8897b36ad88b440266245103"  # google/fonts, 2026-10-09
FONT_FILES = {
    # body faces (Naskh first: what heritage books are set in), weight for sampling
    "Amiri": ("amiri/Amiri-Regular.ttf", "amiri/Amiri-Bold.ttf", 5),
    "NotoNaskh": ("notonaskharabic/NotoNaskhArabic[wght].ttf", None, 4),
    "Scheherazade": ("scheherazadenew/ScheherazadeNew-Regular.ttf", "scheherazadenew/ScheherazadeNew-Bold.ttf", 4),
    "Lateef": ("lateef/Lateef-Regular.ttf", "lateef/Lateef-Bold.ttf", 3),
    "Markazi": ("markazitext/MarkaziText[wght].ttf", None, 3),
    "Mirza": ("mirza/Mirza-Regular.ttf", "mirza/Mirza-Bold.ttf", 3),
    "Harmattan": ("harmattan/Harmattan-Regular.ttf", "harmattan/Harmattan-Bold.ttf", 2),
    "NotoSans": ("notosansarabic/NotoSansArabic[wdth,wght].ttf", None, 1),
    "IBMPlex": ("ibmplexsansarabic/IBMPlexSansArabic-Regular.ttf", "ibmplexsansarabic/IBMPlexSansArabic-Bold.ttf", 1),
    "Cairo": ("cairo/Cairo[slnt,wght].ttf", None, 1),
    "Tajawal": ("tajawal/Tajawal-Regular.ttf", "tajawal/Tajawal-Bold.ttf", 1),
}
HEADING_FONTS = {"ReemKufi": "reemkufi/ReemKufi[wght].ttf", "ElMessiri": "elmessiri/ElMessiri[wght].ttf",
                 "ArefRuqaa": "arefruqaa/ArefRuqaa-Bold.ttf", "Kufam": "kufam/Kufam[wght].ttf",
                 "Rakkas": "rakkas/Rakkas-Regular.ttf"}
QURAN_FONT = "amiriquran/AmiriQuran-Regular.ttf"
# Genres of heritage books (Shamela category ids) and how many books to sample from each.
CATEGORIES = {3: 10, 6: 14, 7: 10, 23: 8, 24: 8, 25: 14, 26: 14, 28: 6, 29: 6, 30: 6, 31: 8, 32: 16, 34: 8,
              35: 4, 18: 6, 16: 4, 36: 6}
EXCLUDE_BOOKS = {23785, 23790}  # الشعر والشعراء and عيون الأخبار: the test books' text
PAGES_PER_BOOK = 18

_fonts = " && ".join(
    f"curl -sfgL -o '/fonts/{Path(p).name}' 'https://raw.githubusercontent.com/google/fonts/{FONTS_COMMIT}/ofl/{p}'"
    for p in sorted({x for v in FONT_FILES.values() for x in v[:2] if x} | set(HEADING_FONTS.values()) | {QURAN_FONT}))
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .run_commands("mkdir -p /fonts", _fonts)
    .pip_install("pymupdf==1.28.2", "opencv-python-headless==4.12.0.88", "numpy==2.2.6", "httpx==0.28.1",
                 "pydantic==2.11.9", "lxml==6.0.2", "rapidfuzz==3.14.6", "regex==2026.9.29", "pillow==12.3.0",
                 "jinja2==3.1.6", "click==8.5.0")
    # the whole package (prompt templates, Qur'an text), not only its .py files
    .add_local_dir("pdf2epub", "/root/pdf2epub", ignore=["**/__pycache__/**"])
    .add_local_python_source("train")
)
vol = modal.Volume.from_name("pdf2epub-train", create_if_missing=True)
app = modal.App("pdf2epub-synth")


# --------------------------------------------------------------------------- 1. text


def turath_page_html(n: int, page: dict) -> str:
    """A turath.io page in the Shamela page layout (as tools/turath_pages.py writes it)."""
    import html as htmllib

    meta = json.loads(page["meta"]) if isinstance(page.get("meta"), str) else page.get("meta") or {}
    text = re.sub(r"<span[^>]*>(.*?)</span>", r"\1", page.get("text", ""), flags=re.S)
    main, _, notes = text.partition("\n_________\n")
    marker = re.compile(r"\s*\(([٠-٩0-9]{1,3})\)")
    body = "".join("<p>" + marker.sub(lambda m: f' <span class="c4">[{m.group(1)}]</span>', htmllib.escape(ln.strip())) + "</p>"
                   for ln in main.splitlines() if ln.strip())
    foot = "".join("<p>" + re.sub(r"^\(([٠-٩0-9]{1,3})\)\s*", r"[\1] ", htmllib.escape(ln.strip())) + "</p>"
                   for ln in notes.splitlines() if ln.strip())
    return f'<div class="nass" data-page-id="{n}" data-page-num="{meta.get("page", n)}">{body}' + (f"<hr>{foot}" if foot else "") + "</div>"


@app.function(image=image, timeout=4 * 60 * 60, volumes={"/data": vol})
def fetch(categories: dict, seed: int = 0) -> dict:
    """Runs of consecutive pages from books of each genre, from turath.io (Shamela refuses requests
    from cloud addresses; turath serves the same books under the same ids). ``categories`` maps a
    Shamela category id to its book ids (train/shamela_categories.json, listed from Shamela)."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    rng = random.Random(seed)
    root = Path("/data/shamela-text")
    root.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=40, follow_redirects=True, headers={"User-Agent": "pdf2epub/0.1 (OCR training)"})

    def get(url: str) -> dict | None:
        for attempt in range(4):
            try:
                r = client.get(url)
                if r.status_code == 200 and r.text.strip():
                    time.sleep(0.3)
                    return r.json()
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(2 + 4 * attempt)
        return None

    def one_book(task: tuple[int, int]) -> int:
        cat, book = task
        info = get(f"https://api.turath.io/book?id={book}&include=indexes&ver=3")
        if not info or "indexes" not in info:
            return 0
        last = len(info["indexes"].get("page_map") or [])
        if last < 20:
            return 0
        start = rng.randint(1, last - PAGES_PER_BOOK)
        out = root / str(book)
        out.mkdir(exist_ok=True)
        got = 0
        for n in range(start, start + PAGES_PER_BOOK):
            page = get(f"https://api.turath.io/page?book_id={book}&pg={n}&ver=3")
            if page and page.get("text"):
                (out / f"{n}.html").write_text(turath_page_html(n, page), encoding="utf-8")
                got += 1
        (out / "category.txt").write_text(f"{cat}\n{info.get('meta', {}).get('name', '')}")
        return got

    tasks = []
    for cat, ids in categories.items():
        ids = [i for i in ids if int(i) not in EXCLUDE_BOOKS]
        rng.shuffle(ids)
        # more ids than needed: some are not on turath
        tasks += [(int(cat), int(b)) for b in ids[: CATEGORIES.get(int(cat), 6) * 2]]
    pages = 0
    books = 0
    with ThreadPoolExecutor(4) as pool:
        for k, got in enumerate(pool.map(one_book, tasks), 1):
            pages += got
            books += got > 0
            if k % 10 == 0:
                vol.commit()
                print(f"{k}/{len(tasks)} books tried, {books} kept, {pages} pages", flush=True)
    vol.commit()
    return {"books": books, "pages": pages}


def parse_page(html: str) -> tuple[list[dict], dict[str, str]]:
    """A Shamela page as blocks ({"t": "p"|"verse"|"h", ...} with markers as ⟦n⟧) and notes {n: text}."""
    import html as htmllib

    main, _, notes = html.partition("<hr")
    def lines(part: str) -> list[str]:
        part = re.sub(r'<a [^>]*class="btn_tag[^>]*>.*?</a>', " ", part, flags=re.S)
        part = re.sub(r'<span class="c4">\s*\[([٠-٩0-9]{1,3})\]\s*</span>', r" ⟦\1⟧", part)
        part = re.sub(r'<span class="c2">\s*«([٠-٩0-9]{1,3})»\s*</span>', r" ⟦\1⟧", part)
        out = []
        for p in re.findall(r"<p[^>]*>(.*?)</p>", part, re.S):
            for seg in re.split(r"<br\s*/?>", p):
                t = " ".join(htmllib.unescape(re.sub(r"<[^>]+>", " ", seg)).split())
                if t:
                    out.append(t)
        return out
    blocks = []
    for ln in lines(main):
        halves = ln.split("...")
        if len(halves) == 2 and all(1 <= len(h.split()) <= 9 for h in halves):
            blocks.append({"t": "verse", "s": halves[0].strip(), "a": halves[1].strip()})
        elif re.fullmatch(r"\[[^\]]{2,60}\]", ln) or (len(ln.split()) <= 6 and not ln.endswith((".", "،", ":"))):
            blocks.append({"t": "h", "text": ln.strip("[]")})
        else:
            blocks.append({"t": "p", "text": ln})
    note_map: dict[str, str] = {}
    for ln in lines(notes):
        m = re.match(r"^\(?\[?([٠-٩0-9]{1,3})\]?\)?\s*(.+)", ln)
        if m:
            note_map[m.group(1)] = m.group(2)
    return blocks, note_map


# --------------------------------------------------------------------------- 2. pages


AR = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
HONORIFICS = {"ﷺ": "صلى الله عليه وسلم", "﵁": "رضي الله عنه", "﵂": "رضي الله عنها", "﵃": "رضي الله عنهم",
              "﵄": "رضي الله عنهما", "﵀": "رحمه الله", "﷿": "عز وجل", "ﷻ": "جل جلاله", "﵇": "عليه السلام"}
SIZES = [(340, 500), (360, 540), (420, 595), (460, 680)]


def _num(n: int, western: bool) -> str:
    return str(n) if western else str(n).translate(AR)


def _esc(t: str) -> str:
    from xml.sax.saxutils import escape

    return escape(t)


def add_notes(blocks: list[dict], notes: dict[str, str], pool: list[str], rng: random.Random) -> None:
    """For a page whose text has no notes: put 1-4 markers in it and attach real foot-note texts
    taken from other pages (printed editions are full of notes; many digital texts carry none)."""
    if notes or not pool or rng.random() > 0.65:
        return
    spots = [(k, "p") for k, b in enumerate(blocks) if b["t"] == "p" and len(b["text"].split()) > 6] + \
            [(k, "a") for k, b in enumerate(blocks) if b["t"] == "verse"]
    rng.shuffle(spots)
    for j, (k, kind) in enumerate(sorted(spots[:rng.randint(1, 4)])):
        key = f"S{j}"
        notes[key] = rng.choice(pool)[: rng.randint(40, 260)].rsplit(" ", 1)[0]
        if kind == "a":
            blocks[k]["a"] += f" ⟦{key}⟧"
        else:
            words = blocks[k]["text"].split()
            at = rng.randint(2, len(words) - 1)
            blocks[k]["text"] = " ".join(words[:at] + [f"⟦{key}⟧"] + words[at:])


def add_verses(blocks: list[dict], pool: list[list[dict]], rng: random.Random) -> None:
    """Heritage books quote poetry on most pages: a page whose text has none gets a short poem
    (1-4 verses taken from verse runs elsewhere in the fetched books) between two paragraphs."""
    if not pool or any(b["t"] == "verse" for b in blocks) or rng.random() > 0.3 or len(blocks) < 2:
        return
    poem = rng.choice(pool)
    start = rng.randrange(len(poem))
    at = rng.randint(1, min(len(blocks) - 1, 4))
    blocks[at:at] = [dict(v) for v in poem[start:start + rng.randint(1, 4)]]


def render_page(blocks: list[dict], notes: dict[str, str], rng: random.Random, title: str, page_no: int) -> tuple | None:
    """(grayscale image, target tags, font name) for one page, or None if nothing fits."""
    import cv2
    import numpy as np
    import pymupdf

    from pdf2epub.model import Page
    from pdf2epub.parse import parse_tags

    names = list(FONT_FILES)
    face = rng.choices(names, weights=[FONT_FILES[n][2] for n in names])[0]
    regular, bold, _w = FONT_FILES[face]
    head_face = rng.choice(list(HEADING_FONTS)) if rng.random() < 0.35 else face
    W, H = rng.choice(SIZES)
    scale = W / 340
    size = round(rng.uniform(10.5, 13.5) * scale, 1)
    lh = rng.choice([1.4, 1.5, 1.6, 1.75])
    western = rng.random() < 0.12
    ligatures_ok = face in ("Amiri", "NotoNaskh", "Scheherazade")
    css = f"""
@font-face {{font-family: b; src: url({Path(regular).name});}}
@font-face {{font-family: b; src: url({Path(bold or regular).name}); font-weight: bold;}}
@font-face {{font-family: h; src: url({Path(HEADING_FONTS.get(head_face, regular)).name});}}
@font-face {{font-family: q; src: url({Path(QURAN_FONT).name});}}
* {{font-family: b; direction: rtl;}}
body {{font-size: {size}pt; line-height: {lh};}}
p {{text-align: {rng.choice(['justify', 'justify', 'right'])}; margin: 0 0 {rng.choice([1, 3, 5])}pt 0; text-indent: {rng.choice([0, 10, 14])}pt;}}
h3 {{font-family: h; text-align: center; font-size: {size + rng.choice([1, 2, 4])}pt; margin: 5pt 0; font-weight: bold;}}
table {{width: 100%; margin: 1pt 0;}} td {{text-align: center; width: 50%;}}
.q {{font-family: q;}}
.m {{font-size: {round(size * 0.7, 1)}pt;}}
.notes p {{font-size: {round(size * rng.uniform(0.75, 0.88), 1)}pt; text-indent: 0; margin: 0; text-align: justify;}}
.hdr {{text-align: center; font-size: {round(size * 0.85, 1)}pt;}}
"""
    archive = pymupdf.Archive("/fonts")

    def clean(t: str) -> str:
        if not ligatures_ok:
            for k, v in HONORIFICS.items():
                t = t.replace(k, f" {v} ")
        return " ".join(t.split())

    def inline(t: str, refs: dict[str, str]) -> tuple[str, str]:
        """(html, tags) for running text: markers ⟦n⟧ renumbered, Qur'an in ﴿﴾ as <quran>."""
        h_out, t_out = [], []
        for piece in re.split(r"(⟦[^⟧]+⟧|﴿[^﴾]*﴾)", clean(t)):
            if not piece:
                continue
            m = re.fullmatch(r"⟦([^⟧]+)⟧", piece)
            if m:
                if m.group(1) in notes:
                    new = refs.setdefault(m.group(1), _num(len(refs) + 1, western))
                    h_out.append(f'<span class="m">({new})</span>')
                    t_out.append(f'<ref n="{new}"/>')
                continue
            q = re.fullmatch(r"﴿([^﴾]*)﴾", piece)
            if q:
                h_out.append(f'<span class="q">﴿{_esc(q.group(1).strip())}﴾</span>')
                t_out.append(f"<quran>{_esc(q.group(1).strip())}</quran>")
                continue
            h_out.append(_esc(piece))
            t_out.append(_esc(piece))
        return "".join(h_out).strip(), "".join(t_out).strip()

    def block_pair(b: dict, refs: dict[str, str]) -> tuple[str, str]:
        if b.get("cut"):
            h, t = inline(b["text"], refs)
            return f"<p>{h}</p>", f'<p cont="next">{t}</p>'
        if b["t"] == "h":
            h, t = inline(b["text"], refs)
            return f"<h3>{h}</h3>", f"<h2>{t}</h2>"
        if b["t"] == "verse":
            hs, ts = inline(b["s"], refs)
            ha, ta = inline(b["a"], refs)
            return f"<table><tr><td>{hs}</td><td>{ha}</td></tr></table>", f"<poem><v><s>{ts}</s><a>{ta}</a></v></poem>"
        h, t = inline(b["text"], refs)
        return f"<p>{h}</p>", f"<p>{t}</p>"

    margin = rng.uniform(0.07, 0.11) * W
    top, foot = 0.08 * H, H - 0.06 * H
    numbers_left = rng.random() < 0.15  # Hārūn-style line numbers in the margin (not transcribed)

    # fit blocks one by one; notes follow the markers that made it
    kept: list[dict] = []
    best = None

    def layout(trial: list[dict]):
        """(pairs, note_items, notes_html, notes_h, body_rect) if ``trial`` fits on the page."""
        refs: dict[str, str] = {}
        pairs = [block_pair(x, refs) for x in trial]
        note_items = [(new, notes[old]) for old, new in refs.items()]
        notes_html = '<div class="notes">' + "".join(f"<p>({n}) {inline(t, {})[0]}</p>" for n, t in note_items) + "</div>"
        doc = pymupdf.open()
        pg = doc.new_page(width=W, height=H)
        notes_h = 0.0
        if note_items:
            spare, _ = pg.insert_htmlbox(pymupdf.Rect(margin, 0, W - margin, H), notes_html, css=css, archive=archive)
            notes_h = H - spare
        body_rect = pymupdf.Rect(margin, top + 0.04 * H, W - margin, foot - notes_h - (12 if note_items else 0))
        if body_rect.height < 60:
            doc.close()
            return None
        spare, sc = pg.insert_htmlbox(body_rect, "".join(h for h, _t in pairs), css=css, archive=archive, scale_low=1)
        doc.close()
        return (pairs, note_items, notes_html, notes_h, body_rect) if spare >= 0 and sc >= 0.999 else None

    for b in blocks:
        fit = layout(kept + [b])
        if fit is None:
            # fill the page: as many words of a paragraph as fit, continued on the next page
            words = b["text"].split() if b["t"] == "p" else []
            lo, hi, part = 6, len(words) - 1, None
            while lo <= hi:
                mid = (lo + hi) // 2
                trial = layout(kept + [{"t": "p", "text": " ".join(words[:mid]), "cut": True}])
                if trial:
                    lo, part = mid + 1, trial
                else:
                    hi = mid - 1
            if part:
                best = part
                kept = kept + [b]
            break
        kept = kept + [b]
        best = fit
    if not best or not kept:
        return None
    pairs, note_items, notes_html, notes_h, body_rect = best
    doc = pymupdf.open()
    pg = doc.new_page(width=W, height=H)
    pagenum = _num(page_no, western)
    header_style = rng.random()
    header = title if header_style < 0.75 else ""
    if header:
        pg.insert_htmlbox(pymupdf.Rect(margin, top - 0.04 * H, W - margin, top + 0.02 * H),
                          f'<p class="hdr">{_esc(clean(header))}</p>', css=css, archive=archive)
        if rng.random() < 0.4:
            pg.draw_line((margin, top + 0.025 * H), (W - margin, top + 0.025 * H), width=0.6)
    on_top = rng.random() < 0.75
    pn_rect = (pymupdf.Rect(W - margin - 40 * scale, top - 0.04 * H, W - margin, top + 0.02 * H) if rng.random() < 0.5
               else pymupdf.Rect(margin, top - 0.04 * H, margin + 40 * scale, top + 0.02 * H)) if (on_top and header) \
        else pymupdf.Rect(W / 2 - 40, (top - 0.04 * H) if on_top else (H - 0.05 * H), W / 2 + 40, (top + 0.02 * H) if on_top else H - 0.005 * H)
    pn_text = pagenum if rng.random() < 0.7 else f"- {pagenum} -"
    pg.insert_htmlbox(pn_rect, f'<p class="hdr">{pn_text}</p>', css=css, archive=archive)
    pg.insert_htmlbox(body_rect, "".join(h for h, _t in pairs), css=css, archive=archive)
    if note_items:
        y = foot - notes_h - 6
        style = rng.random()
        if style < 0.55:
            pg.draw_line((W - margin - 0.3 * W, y), (W - margin, y), width=0.6)
        elif style < 0.8:
            pg.draw_line((margin, y), (W - margin, y), width=0.5)
        else:
            x = W - margin
            while x > W - margin - 0.45 * W:
                pg.draw_circle((x, y), 0.45, fill=(0, 0, 0))
                x -= 3.2
        pg.insert_htmlbox(pymupdf.Rect(margin, foot - notes_h, W - margin, foot + 2), notes_html, css=css, archive=archive)
    if numbers_left:
        step = size * lh * 5
        k, y = 5, body_rect.y0 + step
        while y < body_rect.y1:
            pg.insert_htmlbox(pymupdf.Rect(margin * 0.15, y - size, margin * 0.85, y + size),
                              f'<p class="hdr">{_num(k, western)}</p>', css=css, archive=archive)
            k, y = k + 5, y + step
    dpi = rng.choice([200, 230, 260, 300])
    pix = pg.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    doc.close()
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    img = age(img, rng)
    tags = ([f"<header>{_esc(clean(header))}</header>"] if header else []) + [t for _h, t in pairs]
    if note_items:
        tags.append('<hashiya pos="bottom">\n' + "\n".join(f'<note n="{n}">{inline(t, {})[1]}</note>' for n, t in note_items)
                    + "\n</hashiya>")
    tags.append(f"<pagenum>{pagenum}</pagenum>")
    target = "\n".join(tags) + "\n"
    target = re.sub(r"</poem>\n<poem>", "", target)  # consecutive verses are one poem
    result = parse_tags(target, Page(index=1))
    if not result.ok or any(f.severity != "info" for f in result.page.flags):
        return None
    _ = cv2  # (aging uses OpenCV)
    return img, target, face


def age(img, rng: random.Random):
    """Make a clean render look like a scan: tilt, page curl, blur, ink spread or fading, uneven
    light, noise, then bilevel (like JBIG2 scans) or JPEG."""
    import cv2
    import numpy as np

    nrng = np.random.default_rng(rng.randrange(1 << 30))
    h, w = img.shape
    if rng.random() < 0.3:  # curl near the spine: rows bend a little
        amp = rng.uniform(2, 8)
        xs = np.arange(w, dtype=np.float32)
        map_x = np.tile(xs, (h, 1))
        map_y = (np.arange(h, dtype=np.float32)[:, None] + amp * (xs / w) ** 3).astype(np.float32)
        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-1.5, 1.5), 1.0)
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.0))
    r = rng.random()
    if r < 0.25:
        img = cv2.erode(img, np.ones((2, 2), np.uint8))   # ink spread
    elif r < 0.4:
        img = cv2.dilate(img, np.ones((2, 2), np.uint8))  # thin, faded type
    if rng.random() < 0.4:  # uneven light
        grad = np.linspace(rng.uniform(-30, 0), rng.uniform(0, 30), w)[None, :] + np.linspace(rng.uniform(-20, 0), rng.uniform(0, 20), h)[:, None]
        img = img.astype(np.float32) + grad
    img = img.astype(np.float32) + nrng.normal(0, rng.uniform(2, 12), img.shape) - rng.uniform(0, 30)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.15:  # low resolution scan
        f = rng.uniform(0.55, 0.8)
        img = cv2.resize(cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    r = rng.random()
    if r < 0.35:
        _t, img = cv2.threshold(img, rng.randint(110, 170), 255, cv2.THRESH_BINARY)
    elif r < 0.75:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(50, 92)])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


@app.function(image=image, cpu=4, memory=8192, timeout=3 * 60 * 60, volumes={"/data": vol})
def render(name: str, shard: int, n_pages: int, books: list[str], seed: int, split: str = "train") -> int:
    """Render ``n_pages`` pages from the given books into ``/data/<name>``."""
    import cv2

    from train.bundle import prompts

    rng = random.Random(seed * 1000 + shard)
    root = Path("/data")
    out = root / name
    pool: list[str] = []  # real foot-note texts, for pages whose text has none
    verses: list[list[dict]] = []  # runs of verses, for pages whose text quotes no poetry
    for book in books:
        for f in sorted((root / "shamela-text" / book).glob("*.html")):
            b, n = parse_page(f.read_text(encoding="utf-8"))
            pool += [t for t in n.values() if len(t) > 30]
            run: list[dict] = []
            for x in b + [{"t": "end"}]:
                if x["t"] == "verse":
                    run.append(x)
                elif run:
                    verses.append(run)
                    run = []
    (out / "images").mkdir(parents=True, exist_ok=True)
    system, user = prompts("keep_diacritics")
    lines, made, tries = [], 0, 0
    while made < n_pages and tries < n_pages * 4:
        tries += 1
        book = rng.choice(books)
        files = sorted((root / "shamela-text" / book).glob("*.html"), key=lambda p: int(p.stem))
        if len(files) < 3:
            continue
        k = rng.randrange(len(files) - 1)
        blocks, notes = [], {}
        for f in files[k:k + 2]:  # two consecutive edition pages: enough text for one printed page
            b, n = parse_page(f.read_text(encoding="utf-8"))
            blocks += b
            notes.update(n)
        if rng.random() < 0.5:  # start mid-stream sometimes
            blocks = blocks[rng.randrange(max(1, len(blocks) // 3)):]
        add_verses(blocks, verses, rng)
        add_notes(blocks, notes, pool, rng)
        title = next((b["text"] for b in blocks if b["t"] == "h"), "")[:40] or rng.choice(
            ["كتاب الأدب", "الجزء الأول", "باب في الأخبار", "تاريخ", "شرح الديوان"])
        res = render_page(blocks, notes, rng, title, rng.randint(5, 900))
        if not res:
            continue
        img, target, face = res
        rel = f"images/{split}-{shard:02d}-{made:05d}.png"
        cv2.imwrite(str(out / rel), img)
        lines.append(json.dumps({"image": rel, "system": system, "user": user, "target": target,
                                 "instruction": "keep_diacritics", "source": f"synth-fonts/{face}"}, ensure_ascii=False))
        made += 1
    (out / f"{split}-{shard:02d}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    vol.commit()
    return made


@app.function(image=image, timeout=6 * 60 * 60, volumes={"/data": vol})
def make(name: str = "synth-fonts", pages: int = 6000, shards: int = 16, heldout_books: int = 12,
         test_pages: int = 240, seed: int = 0) -> dict:
    """Fetch text (if not yet there), then render train and test pages in parallel."""
    vol.reload()
    root = Path("/data/shamela-text")
    if not root.exists() or len(list(root.iterdir())) < 50:
        raise RuntimeError("fetch the text first: fetch.remote(categories)")
    books = sorted(p.name for p in root.iterdir() if p.is_dir() and len(list(p.glob("*.html"))) >= 3)
    rng = random.Random(seed)
    rng.shuffle(books)
    held, train = books[:heldout_books], books[heldout_books:]
    per = -(-pages // shards)
    render_fn = modal.Function.from_name("pdf2epub-synth", "render")
    made = sum(render_fn.starmap([(name, s, per, train, seed, "train") for s in range(shards)]))
    tests = sum(render_fn.starmap([(name, s, -(-test_pages // 4), held, seed + 7, "test") for s in range(4)]))
    vol.reload()
    out = Path("/data") / name
    with (out / "train.jsonl").open("w", encoding="utf-8") as f:
        for part in sorted(out.glob("train-*.jsonl")):
            f.write(part.read_text(encoding="utf-8"))
    with (out / "test.jsonl").open("w", encoding="utf-8") as f:
        for part in sorted(out.glob("test-*.jsonl")):
            f.write(part.read_text(encoding="utf-8"))
    (out / "books.json").write_text(json.dumps({"train": train, "heldout": held}))
    vol.commit()
    return {"books": len(books), "train_pages": made, "test_pages": tests}


@app.function(image=image, timeout=12 * 60 * 60, volumes={"/data": vol})
def chain_ft4(fetch_calls: list[str], name: str = "synth-fonts", pages: int = 6000, run: str = "ft4",
              real: str = "bundle-v4:3000", base: str = "/models/ft2/merged", lr: float = 5e-5) -> dict:
    """Wait for running fetches, render the bundle, then start training ``run`` from ``base`` on
    the synthetic pages plus a sample of real pages (so it keeps what it learned)."""
    for call in fetch_calls:
        try:
            print("fetch:", modal.FunctionCall.from_id(call).get(), flush=True)
        except Exception as exc:  # a failed downloader leaves fewer books, not none
            print("fetch failed:", call, exc, flush=True)
    vol.reload()
    # (by name: calling a sibling function's handle from inside a deployed function failed to hydrate)
    made = modal.Function.from_name("pdf2epub-synth", "make").remote(name, pages)
    print("render:", made, flush=True)
    train = modal.Function.from_name("pdf2epub-finetune", "train")
    call = train.spawn(f"{name},{real}", run, epochs=1.0, lr=lr, rank=16, accum=8, eval_every=150, base=base)
    print("training:", call.object_id, flush=True)
    return {"render": made, "train_call": call.object_id}


@app.function(image=image, timeout=3 * 60 * 60, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("pdf2epub-api-key")])
def font_test(endpoint: str, model: str, name: str = "synth-fonts", concurrency: int = 16) -> dict:
    """Read the held-out synthetic pages (``/data/<name>/test.jsonl``) with a model server and
    score them by typeface: word and character error rates of the text (letters only, as the
    test-book scores), and diacritics per letter of output vs. page."""
    import asyncio
    import base64
    import os
    from collections import defaultdict

    import httpx
    from rapidfuzz.distance import Levenshtein

    from pdf2epub.arabic import DIACRITICS, normalize_for_match
    from pdf2epub.eval.metrics import page_text
    from pdf2epub.model import Page
    from pdf2epub.parse import parse_tags

    root = Path("/data") / name
    items = [json.loads(x) for x in (root / "test.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    key = os.environ.get("PDF2EPUB_API_KEY", "")

    async def read_all() -> list[str]:
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=900, follow_redirects=True, headers={"Authorization": f"Bearer {key}"}) as c:
            async def one(item: dict) -> str:
                data = base64.b64encode((root / item["image"]).read_bytes()).decode()
                body = {"model": model, "temperature": 0, "max_tokens": 8192, "messages": [
                    {"role": "system", "content": item["system"]},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}},
                                                 {"type": "text", "text": item["user"]}]}]}
                async with sem:
                    for attempt in range(4):
                        try:
                            r = await c.post(endpoint.rstrip("/") + "/chat/completions", json=body)
                            if r.status_code == 200:
                                return r.json()["choices"][0]["message"]["content"] or ""
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(5 + 10 * attempt)
                return ""
            return await asyncio.gather(*(one(i) for i in items))

    outputs = asyncio.run(read_all())

    def words(tags: str) -> list[str]:
        page = parse_tags(tags, Page(index=1)).page
        return normalize_for_match(page_text(page, notes=True)).split()

    def dens(t: str) -> float:
        letters = sum(1 for ch in t if "ء" <= ch <= "ي")
        return sum(1 for ch in t if ch in DIACRITICS) / max(1, letters)

    per = defaultdict(lambda: [0, 0, 0, 0, 0.0, 0.0, 0])  # word edits, words, char edits, chars, dens out, dens ref, n
    for item, out in zip(items, outputs):
        ref, hyp = words(item["target"]), words(out)
        face = item["source"].split("/")[-1]
        for k in (face, "all"):
            p = per[k]
            p[0] += Levenshtein.distance(ref, hyp); p[1] += len(ref)
            p[2] += Levenshtein.distance(" ".join(ref), " ".join(hyp)); p[3] += len(" ".join(ref))
            p[4] += dens(out); p[5] += dens(item["target"]); p[6] += 1
    return {k: {"pages": v[6], "wer": round(v[0] / max(1, v[1]), 4), "cer": round(v[2] / max(1, v[3]), 4),
                "diacritics_out": round(v[4] / max(1, v[6]), 3), "diacritics_page": round(v[5] / max(1, v[6]), 3)}
            for k, v in sorted(per.items())}
