"""XHTML rendering of chapters, blocks, inlines and notes (spec §6.6, §7).

One chapter becomes one XHTML file. Notes are shown in one of three modes:

* ``popup``: a noteref marker; the note as ``<aside epub:type="footnote">`` at
  the end of the chapter file with a back-link (Kindle shows these as pop-ups);
* ``inline``: the notes in a smaller font right after the block holding their
  anchors;
* ``chapter-end``: an endnotes section at the end of the chapter, linked both ways.

Notes of layer 2/3 always use chapter-end style (pop-ups inside pop-ups are
unreliable on e-readers). Notes without an anchor are still shown, in a visible
list at the end of the chapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.sax.saxutils import escape, quoteattr

from ..arabic import ORNATE_CLOSE, ORNATE_OPEN, format_number
from ..config import BookConfig
from ..model import (
    Book,
    Chapter,
    Figure,
    Heading,
    Matn,
    Note,
    PageBreak,
    Paragraph,
    Poem,
    Quote,
    Quran,
    QuranBlock,
    Ref,
    Separator,
    Unclear,
    block_containers,
    inline_text,
    walk_inlines,
)
from ..quran_meta import ref_label

EDITOR_LETTERS = "أبتثجحخدذرزسشصضطظعغفقكلمنهوي"
XHTML_HEAD = ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
              '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" '
              'xml:lang="ar" lang="ar" dir="rtl">\n')


def page_id(index: int) -> str:
    return f"pg-{index:04d}"


def note_anchor(note_id: str) -> str:
    return f"n-{note_id}"


def ref_anchor(ref_id: str) -> str:
    return f"r-{ref_id}"


def xhtml_document(title: str, body: str, *, css: str = "../css/book.css", body_class: str = "") -> str:
    cls = f" class={quoteattr(body_class)}" if body_class else ""
    return (XHTML_HEAD + f"<head>\n<meta charset=\"utf-8\"/>\n<title>{escape(title)}</title>\n"
            f"<link rel=\"stylesheet\" type=\"text/css\" href={quoteattr(css)}/>\n</head>\n"
            f"<body dir=\"rtl\"{cls}>\n{body}\n</body>\n</html>\n")


def _editor_letter(n: int) -> str:
    letters = EDITOR_LETTERS
    out = ""
    while n > 0:
        n, r = divmod(n - 1, len(letters))
        out = letters[r] + out
    return out


@dataclass
class ChapterContext:
    """Per-chapter rendering state: which notes are shown, their markers, where they go."""

    chapter: Chapter
    book: Book
    cfg: BookConfig
    draft: bool = False
    markers: dict[str, str] = field(default_factory=dict)  # note id -> marker shown
    shown: dict[str, Note] = field(default_factory=dict)  # notes rendered in this chapter
    first_ref: dict[str, str] = field(default_factory=dict)  # note id -> first anchor ref id
    anchored: set[str] = field(default_factory=set)
    inline_pending: list[str] = field(default_factory=list)
    quran_notes: list[tuple[str, str]] = field(default_factory=list)  # (id, label) for quran_refs=footnote
    hidden: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        build = self.cfg.build
        author_n = editor_n = 0
        anchored_ids = set()
        for block in self.chapter.blocks:
            for container in block_containers(block):
                for node in walk_inlines(container):
                    if isinstance(node, Ref) and node.note:
                        anchored_ids.add(node.note)
        for note in self.chapter.notes:
            for block in note.blocks:
                for container in block_containers(block):
                    for node in walk_inlines(container):
                        if isinstance(node, Ref) and node.note:
                            anchored_ids.add(node.note)
        for note in self.chapter.notes:
            kind = "editor" if note.kind == "editor" else "author"
            if (kind == "editor" and build.editor_notes == "hide") or (kind == "author" and build.author_notes == "hide"):
                self.hidden.add(note.id)
                continue
            self.shown[note.id] = note
            if build.note_markers == "printed" and note.marker:
                self.markers[note.id] = note.marker
            elif kind == "editor":
                editor_n += 1
                self.markers[note.id] = _editor_letter(editor_n)
            else:
                author_n += 1
                self.markers[note.id] = format_number(author_n, self.book.digits)
            if build.note_markers == "printed" and not note.marker:
                self.markers[note.id] = "*"
        self.anchored = anchored_ids

    def mode_for(self, note: Note) -> str:
        if note.layer > 1 or note.id not in self.anchored:
            return "chapter-end"
        return self.cfg.build.notes


# --------------------------------------------------------------------------- inline


def render_inlines(nodes: list[object], ctx: ChapterContext) -> str:
    out: list[str] = []
    for i, node in enumerate(nodes):
        if isinstance(node, str):
            out.append(escape(node))
        elif isinstance(node, PageBreak):
            out.append(f'<span epub:type="pagebreak" role="doc-pagebreak" id="{page_id(node.page)}" '
                       f'aria-label={quoteattr(node.label)}></span>')
        elif isinstance(node, Ref):
            out.append(_render_ref(node, ctx))
        elif isinstance(node, Quran):
            out.append(_render_quran(node, ctx, following=_following_text(nodes, i)))
        elif isinstance(node, Quote):
            out.append(f'<span class="quote">{render_inlines(node.c, ctx)}</span>')
        elif isinstance(node, Matn):
            out.append(f'<b class="matn">{render_inlines(node.c, ctx)}</b>')
        elif isinstance(node, Unclear):
            if ctx.draft:
                out.append(f'<span class="unclear draft-flag">{escape(node.text)}</span>')
            else:
                out.append(escape(node.text))
    return "".join(out)


def _following_text(nodes: list[object], i: int) -> str:
    text = ""
    for node in nodes[i + 1:i + 3]:
        if isinstance(node, str):
            text += node
        else:
            break
    return text


def _render_ref(ref: Ref, ctx: ChapterContext) -> str:
    note_id = ref.note
    if note_id and note_id in ctx.hidden:
        return ""
    if note_id and note_id in ctx.shown:
        marker = ctx.markers[note_id]
        rid = ref_anchor(ref.id)
        ctx.first_ref.setdefault(note_id, ref.id)
        note = ctx.shown[note_id]
        if ctx.mode_for(note) == "inline":
            ctx.inline_pending.append(note_id)
            return f'<a class="noteref" id={quoteattr(rid)} href="#{note_anchor(note_id)}">{escape(marker)}</a>'
        return (f'<a epub:type="noteref" role="doc-noteref" class="noteref" id={quoteattr(rid)} '
                f'href="#{note_anchor(note_id)}">{escape(marker)}</a>')
    if ref.generated or not ref.n:
        return ""
    return f'<span class="marker">({escape(ref.n)})</span>'


def _render_quran(span: Quran, ctx: ChapterContext, *, following: str = "") -> str:
    cls = "quran"
    if ctx.draft and span.status in ("flagged", "unchecked"):
        cls += " draft-flag"
    html = f'<span class={quoteattr(cls)}>{ORNATE_OPEN}{escape(span.text)}{ORNATE_CLOSE}</span>'
    mode = ctx.cfg.build.quran_refs
    if mode == "none" or not span.ref or span.status not in ("verified", "corrected") or span.alternatives:
        return html
    if following.lstrip().startswith(("[", "(")) and ":" in following[:40]:
        return html  # the book prints its own reference
    label = ref_label(span.ref, ctx.book.digits)
    if not label:
        return html
    if mode == "label":
        return html + f' <span class="qref">{escape(label)}</span>'
    qid = f"q{len(ctx.quran_notes) + 1}-{ctx.chapter.id}"
    ctx.quran_notes.append((qid, label))
    return html + (f'<a epub:type="noteref" role="doc-noteref" class="noteref" id="r-{qid}" '
                   f'href="#n-{qid}">*</a>')


# --------------------------------------------------------------------------- blocks


def render_block(block: object, ctx: ChapterContext, *, image_map: dict[str, str] | None = None) -> str:
    bid = quoteattr(getattr(block, "id", "") or "")
    id_attr = f" id={bid}" if getattr(block, "id", "") else ""
    if isinstance(block, Heading):
        level = min(max(block.level, 1), 4)
        return f"<h{level}{id_attr}>{render_inlines(block.c, ctx)}</h{level}>"
    if isinstance(block, Paragraph):
        return f"<p{id_attr}>{render_inlines(block.c, ctx)}</p>"
    if isinstance(block, QuranBlock):
        return f'<p class="quran-block"{id_attr}>{render_inlines(block.c, ctx)}</p>'
    if isinstance(block, Poem):
        return _render_poem(block, ctx, id_attr)
    if isinstance(block, Figure):
        src = (image_map or {}).get(block.image or "")
        caption = render_inlines(block.caption, ctx)
        if not src:
            return f'<p class="figure-missing"{id_attr}>{caption}</p>' if caption else ""
        alt = quoteattr(inline_text(block.caption) or "صورة")
        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        return f'<figure class="figure"{id_attr}><img src={quoteattr(src)} alt={alt}/>{cap}</figure>'
    if isinstance(block, Separator):
        return f'<p class="sep"{id_attr}>{escape(block.text or "* * *")}</p>'
    return ""


def _render_poem(poem: Poem, ctx: ChapterContext, id_attr: str) -> str:
    single = poem.single or ctx.cfg.text.poem_columns == "single" or all(v.ajuz is None for v in poem.verses)
    if single:
        lines = "".join(f'<p class="sadr">{render_inlines(v.sadr, ctx)}</p>' for v in poem.verses)
        return f'<div class="poem single"{id_attr}>{lines}</div>'
    if ctx.cfg.build.poem_layout == "stacked":
        verses = "".join(
            f'<div class="bayt"><p class="sadr">{render_inlines(v.sadr, ctx)}</p>'
            f'<p class="ajuz">{render_inlines(v.ajuz or [], ctx)}</p></div>' for v in poem.verses)
        return f'<div class="poem stacked"{id_attr}>{verses}</div>'
    rows = "".join(
        f'<tr><td class="sadr">{render_inlines(v.sadr, ctx)}</td>'
        f'<td class="ajuz">{render_inlines(v.ajuz or [], ctx)}</td></tr>' for v in poem.verses)
    return f'<table class="poem"{id_attr}><tbody>{rows}</tbody></table>'


# --------------------------------------------------------------------------- notes


def _note_body(note: Note, ctx: ChapterContext) -> str:
    marker = ctx.markers.get(note.id, "*")
    ref_id = ctx.first_ref.get(note.id)
    back = (f'<a class="backlink" href="#{ref_anchor(ref_id)}">{escape(marker)}</a> ' if ref_id
            else f'<span class="backlink">{escape(marker)}</span> ')
    lemma = ""
    if note.lemma and note.anchor_type == "lemma":
        lemma = f'<b class="lemma">{escape(ctx.cfg.text.lemma_template.format(lemma=note.lemma))}</b> '
    parts: list[str] = []
    for k, block in enumerate(note.blocks):
        if isinstance(block, Paragraph):
            lead = back + lemma if k == 0 else ""
            parts.append(f"<p>{lead}{render_inlines(block.c, ctx)}</p>")
        else:
            if k == 0:
                parts.append(f"<p>{back}{lemma}</p>")
            parts.append(render_block(block, ctx))
    if not note.blocks:
        parts.append(f"<p>{back}{lemma}</p>")
    return "".join(parts)


def render_note(note: Note, ctx: ChapterContext, mode: str) -> str:
    nid = quoteattr(note_anchor(note.id))
    cls = "editor-note " if note.kind == "editor" else ""
    body = _note_body(note, ctx)
    if mode == "popup":
        return f'<aside epub:type="footnote" role="doc-footnote" class="{cls}footnote" id={nid}>{body}</aside>'
    if mode == "inline":
        return f'<div class="{cls}note" id={nid}>{body}</div>'
    if note.id in ctx.anchored:
        return f'<div class="{cls}endnote" epub:type="endnote" id={nid}>{body}</div>'
    return f'<div class="{cls}endnote" id={nid}>{body}</div>'  # no anchor: plain, always visible


def render_chapter(chapter: Chapter, book: Book, cfg: BookConfig, *, draft: bool = False,
                   image_map: dict[str, str] | None = None) -> str:
    ctx = ChapterContext(chapter, book, cfg, draft=draft)
    body: list[str] = [f'<section epub:type="chapter" role="doc-chapter" id={quoteattr(chapter.id)}>']
    inline_done: set[str] = set()
    for block in chapter.blocks:
        ctx.inline_pending = []
        body.append(render_block(block, ctx, image_map=image_map))
        pending = [n for n in dict.fromkeys(ctx.inline_pending) if n not in inline_done]
        if pending:
            notes = "".join(render_note(ctx.shown[n], ctx, "inline") for n in pending)
            inline_done.update(pending)
            body.append(f'<div class="notes-inline">{notes}</div>')
    body.append("</section>")
    # Render the notes' own content first, so refs inside notes register their targets.
    popup, endnotes = [], []
    for note in chapter.notes:
        if note.id not in ctx.shown or note.id in inline_done:
            continue
        mode = ctx.mode_for(note)
        if mode == "inline":  # an inline note whose anchor was in a note: chapter end
            mode = "chapter-end"
        (popup if mode == "popup" else endnotes).append((note, mode))
    if popup:
        body.append('<section class="footnotes" epub:type="footnotes">')
        body.extend(render_note(n, ctx, m) for n, m in popup)
        body.append("</section>")
    if endnotes or ctx.quran_notes:
        body.append('<section class="endnotes" epub:type="endnotes" role="doc-endnotes">')
        body.append('<h2 class="notes-title">الحواشي</h2>')
        body.extend(render_note(n, ctx, m) for n, m in endnotes)
        for qid, label in ctx.quran_notes:
            body.append(f'<div class="endnote" epub:type="endnote" id="n-{qid}"><p>'
                        f'<a class="backlink" href="#r-{qid}">*</a> {escape(label)}</p></div>')
        body.append("</section>")
    body_class = "noto" if cfg.build.body_font == "noto-naskh" else ""
    return xhtml_document(chapter.title or book.title, "\n".join(body), body_class=body_class)
