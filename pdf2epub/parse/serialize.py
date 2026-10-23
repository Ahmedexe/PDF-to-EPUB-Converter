"""Page -> tagged text: the inverse of :func:`pdf2epub.parse.tags.parse_tags`.

Two forms:

* ``full=False`` (default) is the training-target form: only what is printed,
  exactly the format the model is taught to emit.
* ``full=True`` is the editing form used by the review screen: it also writes
  note and ref ids, generated anchors (``<ref generated="1" note=… method=…/>``),
  region geometry, note kinds and unclear confidence, so a page survives
  ``parse_tags(serialize_page(p, full=True))`` with its links intact.

Page breaks (inserted by assembly) are never written.
"""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape, quoteattr

from ..model import (
    Figure,
    HashiyaRegion,
    Heading,
    Matn,
    Note,
    Page,
    PageBreak,
    Paragraph,
    Poem,
    Quote,
    Quran,
    QuranBlock,
    Ref,
    Separator,
    Unclear,
)


def _quote_style(text: str) -> str | None:
    from .tags import _quote_style as style

    return style(text)


def serialize_inlines(inlines: list[Any], *, full: bool = False, notes: dict[str, Note] | None = None) -> str:
    out: list[str] = []
    for node in inlines:
        if isinstance(node, str):
            out.append(escape(node))
        elif isinstance(node, Ref):
            if node.generated:
                if full and node.note:
                    method = ""
                    if notes and node.note in notes and notes[node.note].link.method:
                        method = f" method={quoteattr(notes[node.note].link.method)}"
                    out.append(f'<ref generated="1" id={quoteattr(node.id)} note={quoteattr(node.note)}{method}/>')
                continue
            ident = f" id={quoteattr(node.id)}" if full and node.id else ""
            out.append(f"<ref n={quoteattr(node.n or '')}{ident}/>")
        elif isinstance(node, Quran):
            out.append(f"<quran>{escape(node.text)}</quran>")
        elif isinstance(node, Quote):
            inner = serialize_inlines(node.c, full=full, notes=notes)
            from ..model import inline_text

            style = ""
            if node.style and node.style != _quote_style(inline_text(node.c)):
                style = f" style={quoteattr(node.style)}"
            out.append(f"<quote{style}>{inner}</quote>")
        elif isinstance(node, Matn):
            out.append(f"<matn>{serialize_inlines(node.c, full=full, notes=notes)}</matn>")
        elif isinstance(node, Unclear):
            conf = f' confidence="{node.confidence:g}"' if full and node.confidence is not None else ""
            out.append(f"<unclear{conf}>{escape(node.text)}</unclear>")
        elif isinstance(node, PageBreak):
            continue
    return "".join(out)


def _cont_attr(continued: bool, continues: bool) -> str:
    if continued and continues:
        return ' cont="both"'
    if continued:
        return ' cont="prev"'
    if continues:
        return ' cont="next"'
    return ""


def serialize_block(block: Any, *, page: Page | None = None, full: bool = False,
                    notes: dict[str, Note] | None = None) -> str:
    inl = lambda c: serialize_inlines(c, full=full, notes=notes)  # noqa: E731
    if isinstance(block, Heading):
        level = min(max(block.level, 1), 4)
        return f"<h{level}>{inl(block.c)}</h{level}>"
    if isinstance(block, Paragraph):
        return f"<p{_cont_attr(block.continued, block.continues)}>{inl(block.c)}</p>"
    if isinstance(block, QuranBlock):
        spans = [n for n in block.c if isinstance(n, Quran)]
        if not spans:
            return f"<p>{inl(block.c)}</p>"
        if len(block.c) == 1:
            return f"<quran>{escape(spans[0].text)}</quran>"
        # Several verses, or text/markers between them: one element with the
        # verses in ornate brackets, which the parser splits by bracket parity.
        parts = []
        for n in block.c:
            if isinstance(n, Quran):
                parts.append(f"\ufd3f{escape(n.text)}\ufd3e")
            else:
                parts.append(inl([n]))
        return "<quran>" + "".join(parts) + "</quran>"
    if isinstance(block, Poem):
        verses = []
        for v in block.verses:
            if v.ajuz is None:
                verses.append(f"<v><s>{inl(v.sadr)}</s></v>")
            else:
                verses.append(f"<v><s>{inl(v.sadr)}</s><a>{inl(v.ajuz)}</a></v>")
        attr = ' type="single"' if block.single else ""
        return f"<poem{attr}>" + "".join(verses) + "</poem>"
    if isinstance(block, Figure):
        box = ""
        if block.bbox and page is not None and page.width and page.height:
            x0, y0, x1, y1 = block.bbox
            rel = [x0 * 1000 / page.width, y0 * 1000 / page.height,
                   x1 * 1000 / page.width, y1 * 1000 / page.height]
            box = " box=" + quoteattr(" ".join(str(round(v)) for v in rel))
        caption = inl(block.caption)
        return f"<figure{box}>{caption}</figure>" if caption else f"<figure{box}/>"
    if isinstance(block, Separator):
        return f"<sep text={quoteattr(block.text)}/>" if block.text else "<sep/>"
    raise TypeError(f"cannot serialise {type(block).__name__}")


def serialize_note(note: Note, *, page: Page | None = None, full: bool = False,
                   notes: dict[str, Note] | None = None) -> str:
    attrs = ""
    if full and note.id:
        attrs += f" id={quoteattr(note.id)}"
    if note.marker:
        attrs += f" n={quoteattr(note.marker)}"
    if note.lemma:
        attrs += f" lemma={quoteattr(note.lemma)}"
    if note.kind == "editor" or (full and note.kind == "author"):
        attrs += f' kind="{note.kind}"'
    attrs += _cont_attr(note.cont, note.cont_next)
    parts = []
    for block in note.blocks:
        if isinstance(block, Paragraph) and len(note.blocks) == 1:
            parts.append(serialize_inlines(block.c, full=full, notes=notes))
        else:
            parts.append(serialize_block(block, page=page, full=full, notes=notes))
    return f"<note{attrs}>" + "".join(parts) + "</note>"


def serialize_region(region: HashiyaRegion, *, page: Page | None = None, full: bool = False,
                     notes: dict[str, Note] | None = None) -> str:
    attrs = f' pos="{region.region}"'
    if region.layer != 1:
        attrs += f' layer="{region.layer}"'
    if region.independent:
        attrs += ' independent="1"'
    if full and region.bbox:
        attrs += ' bbox="' + ",".join(f"{v:.1f}" for v in region.bbox) + '"'
    if full and region.rotation:
        attrs += f' rotation="{region.rotation:g}"'
    if region.independent:
        body = [serialize_block(b, page=page, full=full, notes=notes) for b in region.blocks]
    else:
        body = [serialize_note(n, page=page, full=full, notes=notes) for n in region.notes]
    return f"<hashiya{attrs}>\n" + "\n".join(body) + "\n</hashiya>"


def serialize_page(page: Page, *, meta: bool = True, full: bool = False) -> str:
    """Tagged text for a page. ``meta`` includes the running header and page number;
    ``full`` writes the editing form (see the module docstring)."""
    notes = {n.id: n for n in page.notes()} if full else None
    lines: list[str] = []
    if meta and page.running_header:
        for part in page.running_header.split(" | "):
            lines.append(f"<header>{escape(part)}</header>")
    for block in page.blocks:
        lines.append(serialize_block(block, page=page, full=full, notes=notes))
    for region in page.hashiya:
        lines.append(serialize_region(region, page=page, full=full, notes=notes))
    if meta and page.printed_page:
        lines.append(f"<pagenum>{escape(page.printed_page)}</pagenum>")
    return "\n".join(lines) + "\n"
