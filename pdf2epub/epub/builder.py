"""EPUB3 packaging for Kindle (Send to Kindle) and Boox (spec §7).

One plain, standards-clean EPUB3: ``xml:lang="ar"`` and ``dir="rtl"`` on every
``<html>``, ``page-progression-direction="rtl"`` on the spine, direction only
from ``dir`` attributes, embedded open fonts, one XHTML file per chapter, a
table of contents plus a page list mapped to the printed page numbers, an NCX
for older readers, a cover and a credits page with the Tanzil attribution its
licence requires. Output is deterministic: rebuilding the same book with the
same ``modified`` timestamp gives the same bytes.
"""

from __future__ import annotations

import datetime as _dt
import io
import logging
import uuid
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from .. import __version__
from ..config import BookConfig
from ..model import Book, Chapter, Figure, Heading, PageBreak, block_containers, inline_text, walk_inlines
from ..workdir import WorkDir
from .render import page_id, render_chapter, xhtml_document

log = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
FONTS = HERE / "fonts"
CSS = HERE / "css" / "book.css"
TANZIL_LICENSE = HERE.parent / "data" / "tanzil" / "LICENSE-TANZIL.txt"
FONT_FILES = {
    "amiri": ["Amiri-Regular.ttf", "Amiri-Bold.ttf", "AmiriQuran.ttf"],
    "noto-naskh": ["NotoNaskhArabic-Regular.ttf", "NotoNaskhArabic-Bold.ttf", "AmiriQuran.ttf"],
}
FONT_FACES = {  # file -> (family, weight)
    "Amiri-Regular.ttf": ("Amiri", "normal"),
    "Amiri-Bold.ttf": ("Amiri", "bold"),
    "AmiriQuran.ttf": ("Amiri Quran", "normal"),
    "NotoNaskhArabic-Regular.ttf": ("Noto Naskh Arabic", "normal"),
    "NotoNaskhArabic-Bold.ttf": ("Noto Naskh Arabic", "bold"),
}


def stylesheet(cfg: BookConfig) -> bytes:
    """book.css with @font-face rules for exactly the fonts embedded."""
    faces = []
    if cfg.build.embed_fonts:
        for name in FONT_FILES[cfg.build.body_font]:
            family, weight = FONT_FACES[name]
            faces.append(f'@font-face {{ font-family: "{family}"; font-style: normal; font-weight: {weight}; '
                         f"src: url(../fonts/{name}); }}")
    return ("\n".join(faces) + "\n" + CSS.read_text(encoding="utf-8")).encode("utf-8")
ZIP_DATE = (2000, 1, 1, 0, 0, 0)


class BuildError(RuntimeError):
    pass


def _modified(value: str | None) -> str:
    if value:
        return value
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uses_quran(book: Book) -> bool:
    from ..model import Quran

    for ch in book.chapters:
        for blk in ch.blocks + [b for n in ch.notes for b in n.blocks]:
            for c in block_containers(blk):
                if any(isinstance(n, Quran) for n in walk_inlines(c)):
                    return True
    return False


def _credits(book: Book, cfg: BookConfig) -> str:
    tanzil = TANZIL_LICENSE.read_text(encoding="utf-8") if TANZIL_LICENSE.exists() else ""
    body_font = "Noto Naskh Arabic" if cfg.build.body_font == "noto-naskh" else "Amiri"
    body = f"""<div class="credits" epub:type="credits">
<h1>شكر وحقوق</h1>
<p>نص القرآن الكريم في هذا الكتاب مأخوذ دون تغيير من مشروع تنزيل (Tanzil Project)، ومرخص بموجب رخصة المشاع الإبداعي النسبة 3.0. الموقع: <a href="https://tanzil.net">tanzil.net</a></p>
<p>Quran text: Tanzil Project, used verbatim under the Creative Commons Attribution 3.0 licence. Source and updates: <a href="https://tanzil.net">https://tanzil.net</a></p>
<p>الخطوط: {escape(body_font)} و Amiri Quran، بموجب رخصة SIL Open Font License.</p>
<p>Fonts: {escape(body_font)} and Amiri Quran, SIL Open Font License 1.1.</p>
<p>Converted from a scanned edition with pdf2epub {escape(__version__)} using the open model {escape(cfg.ocr.model)}.</p>
<pre>{escape(tanzil)}</pre>
</div>"""
    return xhtml_document("شكر وحقوق", body)


def _cover(image_rel: str) -> str:
    return xhtml_document("الغلاف", f'<div class="cover"><img src={quoteattr(image_rel)} alt="الغلاف"/></div>')


def _toc_entries(chapters: list[Chapter], files: dict[str, str]) -> list[tuple[int, str, str]]:
    """(level, label, href) for chapters and the headings inside them."""
    out: list[tuple[int, str, str]] = []
    for ch in chapters:
        href = files[ch.id]
        out.append((ch.level, ch.title or "—", href))
        first_heading = True
        for block in ch.blocks:
            if isinstance(block, Heading):
                if first_heading and " ".join(inline_text(block.c).split()) == ch.title:
                    first_heading = False
                    continue
                first_heading = False
                label = " ".join(inline_text(block.c).split())
                if label:
                    out.append((max(block.level, ch.level + 1), label, f"{href}#{block.id}"))
    return out


def _nested_ol(entries: list[tuple[int, str, str]]) -> str:
    """Nested <ol> for (level, label, href) entries; levels may jump."""
    if not entries:
        return "<ol><li><span>—</span></li></ol>"
    out: list[str] = []
    stack: list[int] = []
    for level, label, href in entries:
        if not stack:
            out.append("<ol>")
            stack.append(level)
        elif level > stack[-1]:
            out.append("<ol>")
            stack.append(level)
        else:
            out.append("</li>")
            while len(stack) > 1 and level < stack[-1]:
                out.append("</ol></li>")
                stack.pop()
        out.append(f"<li><a href={quoteattr(href)}>{escape(label)}</a>")
    out.append("</li>")
    while len(stack) > 1:
        out.append("</ol></li>")
        stack.pop()
    out.append("</ol>")
    return "".join(out)


def _page_targets(chapters: list[Chapter], files: dict[str, str], labels: dict[int, str]) -> list[tuple[str, str]]:
    seen: set[int] = set()
    out: list[tuple[str, str]] = []
    for ch in chapters:
        for block in ch.blocks:
            for c in block_containers(block):
                for node in walk_inlines(c):
                    if isinstance(node, PageBreak) and node.page not in seen:
                        seen.add(node.page)
                        out.append((labels.get(node.page, node.label), f"{files[ch.id]}#{page_id(node.page)}"))
    return out


def _nav(book: Book, chapters: list[Chapter], files: dict[str, str], credits_href: str | None) -> str:
    toc = _nested_ol(_toc_entries(chapters, files))
    pages = _page_targets(chapters, files, book.page_labels)
    page_list = ""
    if pages:
        items = "".join(f"<li><a href={quoteattr(h)}>{escape(lbl)}</a></li>" for lbl, h in pages)
        page_list = f'<nav epub:type="page-list" id="page-list" hidden="hidden"><h2>الصفحات</h2><ol>{items}</ol></nav>'
    first = files[chapters[0].id] if chapters else None
    land = []
    if first:
        land.append(f'<li><a epub:type="bodymatter" href={quoteattr(first)}>بداية الكتاب</a></li>')
    if credits_href:
        land.append(f'<li><a epub:type="backmatter" href={quoteattr(credits_href)}>شكر وحقوق</a></li>')
    body = (f'<nav epub:type="toc" id="toc" role="doc-toc"><h1>الفهرس</h1>{toc}</nav>\n{page_list}\n'
            f'<nav epub:type="landmarks" id="landmarks" hidden="hidden"><h2>معالم</h2><ol>{"".join(land)}</ol></nav>')
    return xhtml_document("الفهرس", body, css="css/book.css")


def _ncx(book: Book, chapters: list[Chapter], files: dict[str, str]) -> str:
    entries = _toc_entries(chapters, files)
    points = []
    for n, (_level, label, href) in enumerate(entries, 1):
        points.append(f'<navPoint id="np{n}" playOrder="{n}"><navLabel><text>{escape(label)}</text></navLabel>'
                      f'<content src={quoteattr("text/" + href)}/></navPoint>')
    if not points:
        points.append('<navPoint id="np1" playOrder="1"><navLabel><text>—</text></navLabel>'
                      '<content src="nav.xhtml"/></navPoint>')
    page_targets = []
    for n, (label, href) in enumerate(_page_targets(chapters, files, book.page_labels), 1):
        page_targets.append(f'<pageTarget id="pt{n}" type="normal" value="{n}"><navLabel><text>{escape(label)}</text>'
                            f'</navLabel><content src={quoteattr("text/" + href)}/></pageTarget>')
    page_list = f"<pageList><navLabel><text>الصفحات</text></navLabel>{''.join(page_targets)}</pageList>" \
        if page_targets else ""
    return ('<?xml version="1.0" encoding="utf-8"?>\n'
            '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1" xml:lang="ar">\n'
            f'<head><meta name="dtb:uid" content={quoteattr(book.identifier)}/>'
            '<meta name="dtb:depth" content="1"/><meta name="dtb:totalPageCount" content="0"/>'
            '<meta name="dtb:maxPageNumber" content="0"/></head>\n'
            f'<docTitle><text>{escape(book.title)}</text></docTitle>\n'
            f'<navMap>{"".join(points)}</navMap>{page_list}\n</ncx>\n')


def _opf(book: Book, manifest: list[tuple[str, str, str, str]], spine: list[tuple[str, bool]], modified: str,
         cover_id: str | None) -> str:
    items = "".join(
        f'<item id={quoteattr(i)} href={quoteattr(h)} media-type={quoteattr(t)}'
        + (f" properties={quoteattr(p)}" if p else "") + "/>\n" for i, h, t, p in manifest)
    refs = "".join(f'<itemref idref={quoteattr(i)}' + ("" if linear else ' linear="no"') + "/>\n" for i, linear in spine)
    creator = f"<dc:creator>{escape(book.author)}</dc:creator>\n" if book.author else ""
    cover_meta = f'<meta name="cover" content={quoteattr(cover_id)}/>\n' if cover_id else ""
    return ('<?xml version="1.0" encoding="utf-8"?>\n'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid" '
            'xml:lang="ar" dir="rtl">\n'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
            f'<dc:identifier id="bookid">{escape(book.identifier)}</dc:identifier>\n'
            f'<dc:title>{escape(book.title)}</dc:title>\n<dc:language>{escape(book.language)}</dc:language>\n'
            f'{creator}<meta property="dcterms:modified">{modified}</meta>\n{cover_meta}'
            '</metadata>\n'
            f'<manifest>\n{items}</manifest>\n'
            f'<spine toc="ncx" page-progression-direction="rtl">\n{refs}</spine>\n</package>\n')


def _image_bytes(path: Path, max_side: int | None = None) -> tuple[bytes, str]:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise BuildError(f"cannot read image {path}")
    if max_side and max(img.shape[:2]) > max_side:
        scale = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if max_side:  # cover: JPEG keeps it small
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes(), "image/jpeg"
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes(), "image/png"


def _write_zip(out_path: Path, files: list[tuple[str, bytes]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("mimetype", date_time=ZIP_DATE)
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, b"application/epub+zip")
        for name, data in files:
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(out_path)


def _build_one(book: Book, chapters: list[Chapter], cfg: BookConfig, wd: WorkDir | None, out_path: Path, *,
               modified: str, draft: bool, cover: Path | None) -> Path:
    files: list[tuple[str, bytes]] = []
    manifest: list[tuple[str, str, str, str]] = []
    spine: list[tuple[str, bool]] = []
    container = ('<?xml version="1.0" encoding="utf-8"?>\n'
                 '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                 '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                 '</rootfiles></container>\n')
    files.append(("META-INF/container.xml", container.encode()))
    files.append(("OEBPS/css/book.css", stylesheet(cfg)))
    manifest.append(("css", "css/book.css", "text/css", ""))
    if cfg.build.embed_fonts:
        for k, name in enumerate(FONT_FILES[cfg.build.body_font], 1):
            files.append((f"OEBPS/fonts/{name}", (FONTS / name).read_bytes()))
            manifest.append((f"font{k}", f"fonts/{name}", "font/ttf", ""))
    cover_id = None
    if cover is not None and cover.exists():
        data, mime = _image_bytes(cover, max_side=1600)
        files.append(("OEBPS/images/cover.jpg", data))
        manifest.append(("cover-image", "images/cover.jpg", mime, "cover-image"))
        files.append(("OEBPS/text/cover.xhtml", _cover("../images/cover.jpg").encode()))
        manifest.append(("cover", "text/cover.xhtml", "application/xhtml+xml", ""))
        spine.append(("cover", True))
        cover_id = "cover-image"
    # Figures referenced by the chapters.
    image_map: dict[str, str] = {}
    for ch in chapters:
        for block in ch.blocks:
            if isinstance(block, Figure) and block.image and block.image not in image_map and wd is not None:
                try:
                    src = wd.abs(block.image)
                except ValueError:
                    continue
                if not src.exists():
                    continue
                name = f"fig{len(image_map) + 1:03d}.png"
                data, mime = _image_bytes(src)
                files.append((f"OEBPS/images/{name}", data))
                manifest.append((f"img{len(image_map) + 1}", f"images/{name}", mime, ""))
                image_map[block.image] = f"../images/{name}"
    chapter_files = {ch.id: f"{ch.id}.xhtml" for ch in chapters}
    for ch in chapters:
        html = render_chapter(ch, book, cfg, draft=draft, image_map=image_map)
        files.append((f"OEBPS/text/{chapter_files[ch.id]}", html.encode("utf-8")))
        manifest.append((ch.id, f"text/{chapter_files[ch.id]}", "application/xhtml+xml", ""))
        spine.append((ch.id, True))
    credits_href = "credits.xhtml"
    files.append(("OEBPS/text/credits.xhtml", _credits(book, cfg).encode("utf-8")))
    manifest.append(("credits", "text/credits.xhtml", "application/xhtml+xml", ""))
    spine.append(("credits", True))
    nav_files = {k: f"text/{v}" for k, v in chapter_files.items()}
    files.append(("OEBPS/nav.xhtml", _nav(book, chapters, nav_files, f"text/{credits_href}").encode("utf-8")))
    manifest.append(("nav", "nav.xhtml", "application/xhtml+xml", "nav"))
    files.append(("OEBPS/toc.ncx", _ncx(book, chapters, chapter_files).encode("utf-8")))
    manifest.append(("ncx", "toc.ncx", "application/x-dtbncx+xml", ""))
    files.append(("OEBPS/content.opf", _opf(book, manifest, spine, modified, cover_id).encode("utf-8")))
    _write_zip(out_path, files)
    return out_path


def build_epub(book: Book, cfg: BookConfig, wd: WorkDir | None, out_path: Path, *, modified: str | None = None,
               draft: bool = False, cover: Path | None = None) -> list[Path]:
    """Write the EPUB(s) for ``book``; returns the files written.

    ``cover`` is an image file for the cover (the pipeline passes the first page
    when ``build.cover == "first-page"``). Independent works printed in a frame
    become extra sections, or a second EPUB with ``build.frames == "separate-epub"``.
    ``draft`` marks unresolved Quran spans and unclear words instead of refusing them.
    """
    stamp = _modified(modified)
    main = [c for c in book.chapters if c.stream == "main"]
    frame = [c for c in book.chapters if c.stream != "main"]
    if not main and not frame:
        raise BuildError("the book has no content")
    written: list[Path] = []
    if cfg.build.frames == "separate-epub" and frame:
        written.append(_build_one(book, main or frame, cfg, wd, out_path, modified=stamp, draft=draft, cover=cover))
        frame_book = book.model_copy(update={
            "title": f"{book.title} — بهامشه",
            "identifier": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, 'pdf2epub:' + book.slug + '-frame')}",
        })
        frame_path = out_path.with_name(out_path.stem + "-frame.epub")
        written.append(_build_one(frame_book, frame, cfg, wd, frame_path, modified=stamp, draft=draft, cover=None))
    else:
        written.append(_build_one(book, main + frame, cfg, wd, out_path, modified=stamp, draft=draft, cover=cover))
    return written
