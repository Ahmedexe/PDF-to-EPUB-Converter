"""Tagged model output -> :class:`~pdf2epub.model.Page` (spec §3, step 4 of §2).

The tag set is documented in ``pdf2epub/ocr/prompts/tagspec.md``. The parser
is tolerant and never drops text silently:

1. ``preprocess`` strips code fences and reasoning blocks, repairs entities and
   unquoted attributes, then *balances* the tags (self-closes void tags, drops
   stray end tags, closes a paragraph or poem the model forgot to close).
2. The balanced text is parsed strictly with lxml; lxml's recovering parser is
   only a last resort, and a plain-text fallback keeps the text of hopeless
   output for review.
3. Elements become page-model objects. Unknown tags are unwrapped, common
   aliases (``<footnote>``, ``<sadr>`` ...) accepted, notes written inline are
   moved to the foot, loose note text is split into notes.
4. The letters in the result are counted against the letters the model wrote;
   any loss is flagged.

Every repair is recorded as a :class:`~pdf2epub.model.Flag` with ``stage="parse"``.
"""

from __future__ import annotations

import html
import html.entities
import re
from dataclasses import dataclass
from typing import Any

from lxml import etree

from ..arabic import BRACKET_PAIRS, ORNATE_CLOSE, ORNATE_OPEN, clean_text, normalize_marker, parse_number
from ..model import (
    REGION_ORDER,
    Figure,
    FlagCode,
    HashiyaRegion,
    Heading,
    Matn,
    Note,
    NoteLink,
    Page,
    Paragraph,
    Poem,
    Quote,
    Quran,
    QuranBlock,
    Ref,
    Separator,
    Unclear,
    Verse,
    assign_ids,
    block_containers,
    iter_blocks,
    normalize_inlines,
    walk_inlines,
    BlockText,
)

STAGE = "parse"

# Tag aliases the model sometimes produces instead of the documented names.
ALIASES = {
    "footnotes": "hashiya", "notes": "hashiya", "hamish": "hashiya",
    "footnote": "note", "fn": "note",
    "verse": "v", "bayt": "v", "sadr": "s", "ajuz": "a", "ajz": "a", "ajez": "a",
    "marker": "ref", "noteref": "ref",
    "hr": "sep", "separator": "sep",
    "pn": "pagenum", "page_number": "pagenum", "pagenumber": "pagenum", "folio": "pagenum",
    "running_header": "header", "runninghead": "header",
    "ayah": "quran", "aya": "quran",
    "hadith": "quote",
    "img": "figure", "image": "figure",
    "title": "h1",
}
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 4, "h6": 4}
INLINE_TAGS = {"ref", "quran", "quote", "matn", "unclear", "b", "strong", "i", "em", "span", "br",
               "sup", "small", "u", "font", "mark"}
UNWRAP_SILENT = {"span", "i", "em", "sup", "small", "u", "font", "mark", "div", "body", "html",
                 "page", "root", "text", "main", "section", "article", "doc", "document"}
VOID_TAGS = {"br", "sep", "ref", "hr", "marker", "noteref", "pb"}
# Block tags that cannot sit inside a paragraph: the balancer closes an open <p> before them.
CLOSES_P = {"p", "poem", "hashiya", "footnotes", "header", "pagenum", "pn", "figure", "h1", "h2", "h3",
            "h4", "h5", "h6", "title", "sep", "hr"}
POEM_INNER = {"v", "s", "a", "verse", "sadr", "ajuz", "ajz", "bayt", "ref", "unclear", "b", "i", "span",
              "quran", "quote", "matn", "br"}
POS_ALIASES = {
    "bottom": "bottom", "foot": "bottom", "footer": "bottom", "footnotes": "bottom",
    "margin-right": "margin-right", "right": "margin-right", "margin_right": "margin-right",
    "margin-left": "margin-left", "left": "margin-left", "margin_left": "margin-left",
    "top": "top", "header": "top", "frame": "frame", "box": "frame",
}
TRUE_VALUES = {"1", "true", "yes", "y", "on"}

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*[ \t]*\n?|\n?[ \t]*```\s*$")
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_XMLDECL_RE = re.compile(r"<\?xml[^>]*\?>")
_ENTITY_RE = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);?")
_STRAY_LT_RE = re.compile(r"<(?![A-Za-z/!?])")
_TAG_RE = re.compile(r"<[^<>]*>")
_TOKEN_RE = re.compile(r"<(/?)([A-Za-z][\w:-]*)((?:\s+[^<>]*?)?)\s*(/?)>")
_UNQUOTED_ATTR_RE = re.compile(r"(\s[\w-]+)=([^\s\"'>/][^\s\"'>]*)")
_WS_RUN = re.compile(r"\s+")
_DECOR_RE = re.compile(r"[\s*…⁂.\-ـ٭✻]+")
_LEAD_MARKER_RE = re.compile(
    r"^\s*(?:\((?P<a>[\d٠-٩۰-۹]{1,3}|[*٭]{1,3}|[ء-ي]{1,2})\)"
    r"|\[(?P<b>[\d٠-٩۰-۹]{1,3}|[*٭]{1,3})\]"
    r"|(?P<c>[\d٠-٩۰-۹]{1,3})\s*[-–.)]"
    r"|(?P<d>[*٭]{1,3}))\s*"
)
_PRINTED_REF_RE = re.compile(
    r"\s*[\[(]\s*(?:سورة\s*)?[ء-يًٰ-ْ ]{2,30}\s*[:：]\s*[\d٠-٩۰-۹]+"
    r"(?:\s*[-–]\s*[\d٠-٩۰-۹]+)?\s*[\])]\s*$"
)
_ORNATE = ORNATE_OPEN + ORNATE_CLOSE
_REF_PLACEHOLDER = ""


@dataclass
class ParseResult:
    page: Page
    ok: bool  # False when the text could not be parsed as tags at all (plain-text fallback used)
    recovered: bool  # the tags needed repair (balancing or lxml recovery)
    text_loss: float  # fraction of the letters in the input missing from the page model (0..1)


# --------------------------------------------------------------------------- preprocessing


def _escape_bare_amp(text: str) -> str:
    return re.sub(r"&(?!(#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);?)", "&amp;", text)


def _fix_entities(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name.startswith("#"):
            return m.group(0) if m.group(0).endswith(";") else m.group(0) + ";"
        if name in ("amp", "lt", "gt", "quot", "apos") and m.group(0).endswith(";"):
            return m.group(0)
        char = html.entities.html5.get(name + ";")
        if char is not None and m.group(0).endswith(";"):
            return char.replace("&", "&amp;").replace("<", "&lt;")
        return "&amp;" + m.group(0)[1:]

    return _ENTITY_RE.sub(repl, _escape_bare_amp(text))


def _quote_attrs(text: str) -> str:
    def fix_tag(m: re.Match[str]) -> str:
        return _UNQUOTED_ATTR_RE.sub(lambda a: f'{a.group(1)}="{a.group(2)}"', m.group(0))

    return re.sub(r"<[A-Za-z][^<>]*>", fix_tag, text)


def _clean_raw(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if "</think>" in text.lower():
        text = text[text.lower().rindex("</think>") + len("</think>"):]
    text = _THINK_RE.sub("", text)
    text = _FENCE_RE.sub("", text.strip())
    text = _XMLDECL_RE.sub("", text)
    text = _COMMENT_RE.sub("", text)
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text, flags=re.I)
    text = _fix_entities(text)
    text = _STRAY_LT_RE.sub("&lt;", text)
    return _quote_attrs(text).strip()


def balance(text: str) -> tuple[str, bool]:
    """Repair the tag structure; returns ``(text, changed)``.

    Self-closes void tags, drops end tags that match nothing open, closes an open
    paragraph when a block tag starts, closes an open poem when a non-poem tag
    starts, never nests notes or hashiya regions, and closes whatever is left open.
    """
    def ref_content(m: re.Match[str]) -> str:
        attrs, content = m.group(1) or "", m.group(2).strip()
        if re.search(r"\bn\s*=", attrs) or not content:
            return f"<{m.group(0)[1:4]}{attrs}/>"
        return f'<{m.group(0)[1:4]}{attrs} n="{content}"/>'

    text = re.sub(r"<ref(\s[^<>]*?)?>([^<]{0,8})</ref\s*>", ref_content, text, flags=re.I)
    out: list[str] = []
    stack: list[str] = []  # canonical names
    changed = False
    pos = 0

    def close_to(index: int) -> None:
        nonlocal changed
        while len(stack) > index:
            name = stack.pop()
            out.append(f"</{name}>")

    for m in _TOKEN_RE.finditer(text):
        out.append(text[pos:m.start()])
        pos = m.end()
        closing, raw_name, attrs, selfclose = m.group(1), m.group(2), m.group(3) or "", m.group(4)
        name = raw_name.lower()
        canon = ALIASES.get(name, name)
        if closing:
            if name in VOID_TAGS or canon in VOID_TAGS:
                changed = True
                continue
            # Find the innermost open element with this name.
            idx = next((i for i in range(len(stack) - 1, -1, -1) if stack[i] in (name, canon)), None)
            if idx is None:
                changed = True
                continue
            if idx != len(stack) - 1:
                changed = True
            close_to(idx + 1)
            stack.pop()
            out.append(f"</{name}>")
            continue
        if name in VOID_TAGS or canon in VOID_TAGS:
            if not selfclose:
                changed = True
            out.append(f"<{raw_name}{attrs}/>")
            continue
        if selfclose:
            out.append(m.group(0))
            continue
        # Implicit closes before this start tag.
        if name in CLOSES_P or canon in CLOSES_P:
            idx = next((i for i, t in enumerate(stack) if t == "p" or t in HEADINGS), None)
            if idx is not None:
                changed = True
                close_to(idx)
        if "poem" in stack and name not in POEM_INNER and canon not in POEM_INNER:
            changed = True
            close_to(stack.index("poem"))
        if canon == "note" and "note" in stack:
            changed = True
            close_to(stack.index("note"))
        if canon == "hashiya" and "hashiya" in stack:
            changed = True
            close_to(stack.index("hashiya"))
        if canon in ("header", "pagenum") and "hashiya" in stack:
            changed = True
            close_to(stack.index("hashiya"))
        if canon == "v" and "v" in stack:
            changed = True
            close_to(stack.index("v"))
        if canon in ("s", "a"):
            idx = next((i for i, t in enumerate(stack) if t in ("s", "a")), None)
            if idx is not None:
                changed = True
                close_to(idx)
        if canon in HEADINGS:
            idx = next((i for i, t in enumerate(stack) if t in HEADINGS), None)
            if idx is not None:
                changed = True
                close_to(idx)
        stack.append(name if name == canon else name)
        out.append(m.group(0))
    out.append(text[pos:])
    if stack:
        changed = True
        close_to(0)
    return "".join(out), changed


def preprocess(raw: str) -> str:
    """Clean and balance model output so it parses as XML (see module docstring)."""
    return balance(_clean_raw(raw))[0]


def _letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum())


def _loose_ratio(root: Any) -> tuple[int, int, bool]:
    """(letters outside any block tag, all letters, has any block tag) at the top level."""
    loose = _letters(root.text or "")
    has_block = False
    for child in root:
        tag = _tag(child)
        if tag in INLINE_TAGS and tag != "quran":
            loose += _letters("".join(child.itertext()))
        elif tag:
            has_block = True
        loose += _letters(child.tail or "")
    return loose, _letters("".join(root.itertext())), has_block


def _untagged(root: Any) -> bool:
    loose, total, has_block = _loose_ratio(root)
    if not total:
        return False
    return (not has_block and loose > 0) or loose > 0.10 * total


def tags_parse_ok(raw: str) -> bool:
    """True when ``raw`` parses as tags without repair and follows the tag format.

    Used by the OCR retry rule (spec §4 rule 6): a False answer triggers one
    retry at a higher resolution.
    """
    text = _clean_raw(raw)
    try:
        root = etree.fromstring(f"<root>{text}</root>".encode("utf-8"))
    except etree.XMLSyntaxError:
        return False
    return not _untagged(root)


# --------------------------------------------------------------------------- conversion


class _Ctx:
    def __init__(self, page: Page, commentary: str):
        self.page = page
        self.commentary = commentary
        self.unknown: set[str] = set()
        self.in_note = False
        self.ref_methods: dict[int, str] = {}
        self.quran_refs: list[Ref] = []

    def flag(self, code: str, message: str, severity: str = "warning", target: str | None = None) -> None:
        self.page.add_flag(code, message, severity=severity, target=target, stage=STAGE)  # type: ignore[arg-type]


def _tag(el: Any) -> str:
    if not isinstance(el.tag, str):
        return ""  # comment or processing instruction
    name = etree.QName(el).localname.lower()
    return ALIASES.get(name, name)


def _attr(el: Any, *names: str) -> str | None:
    for name in names:
        value = el.get(name)
        if value is not None:
            return value
    return None


def _clean(s: str | None) -> str:
    if not s:
        return ""
    return _WS_RUN.sub(" ", clean_text(s))


def _strip_edges(inlines: list[Any]) -> list[Any]:
    normalize_inlines(inlines)
    if inlines and isinstance(inlines[0], str):
        inlines[0] = inlines[0].lstrip()
    if inlines and isinstance(inlines[-1], str):
        inlines[-1] = inlines[-1].rstrip()
    return normalize_inlines(inlines)


def _edge_spaces(inlines: list[Any]) -> tuple[str, str]:
    lead = " " if inlines and isinstance(inlines[0], str) and inlines[0][:1].isspace() else ""
    trail = " " if inlines and isinstance(inlines[-1], str) and inlines[-1][-1:].isspace() else ""
    return lead, trail


def _quote_style(text: str) -> str | None:
    t = text.strip()
    if len(t) < 2:
        return None
    if t.startswith("((") and t.endswith("))"):
        return "(())"
    opener = t[0]
    closer = BRACKET_PAIRS.get(opener)
    if closer and t.endswith(closer):
        return opener + closer
    return None


def _quran_tokens(el: Any, ctx: _Ctx) -> str:
    """Flatten a <quran> element to text, with a placeholder per <ref> child.

    Refs are kept in ``ctx`` order; unclear readings inside a verse are flagged.
    """
    parts: list[str] = [el.text or ""]
    for child in el:
        tag = _tag(child)
        if tag == "ref":
            marker = _attr(child, "n", "num", "marker") or "".join(child.itertext())
            ctx.quran_refs.append(Ref(n=_clean(marker).strip() or None))
            parts.append(_REF_PLACEHOLDER)
        else:
            if tag == "unclear":
                ctx.flag(FlagCode.UNCLEAR_TEXT, "unclear reading inside a Quran passage")
            parts.append("".join(child.itertext()))
        parts.append(child.tail or "")
    return "".join(parts)


def _strip_enclosing(text: str) -> str:
    """Remove at most one pair of brackets enclosing the whole passage."""
    t = text.strip()
    if len(t) >= 2:
        first, last = t[0], t[-1]
        if first in _ORNATE and last in _ORNATE:
            inner = t[1:-1]
            if not any(ch in _ORNATE for ch in inner):
                return inner.strip()
        pairs = {"(": ")", "«": "»", "{": "}"}
        if first in pairs and last == pairs[first]:
            inner = t[1:-1]
            if first not in inner and last not in inner:
                return inner.strip()
    return t


def _split_quran(el: Any, ctx: _Ctx) -> list[Any]:
    """<quran> content -> Quran spans (one per bracketed verse) with any text and refs between.

    Pieces alternate between verse and commentary by bracket parity, so
    "﴿…﴾ إلى قوله ﴿…﴾" keeps "إلى قوله" as plain text, and a printed verse
    number or reference after the last bracket stays outside the verse.
    """
    ctx.quran_refs = []
    raw = _clean(_quran_tokens(el, ctx)).strip()
    refs = list(ctx.quran_refs)
    trailing = ""
    m = _PRINTED_REF_RE.search(raw)
    if m and m.start() > 0:
        trailing, raw = raw[m.start():], raw[:m.start()]
    raw = raw.strip()
    out: list[Any] = []

    def emit_text(piece: str) -> None:
        for i, chunk in enumerate(piece.split(_REF_PLACEHOLDER)):
            if i:
                out.append(refs.pop(0) if refs else Ref())
            if chunk:
                out.append(chunk)

    def emit_verse(piece: str) -> None:
        count = piece.count(_REF_PLACEHOLDER)
        text = _strip_enclosing(_clean(piece.replace(_REF_PLACEHOLDER, " "))).strip()
        # A note marker printed at the end of the passage, written as text: "… جزلٌ (٣)".
        markers = []
        while True:
            m = re.search(r"\s*[(\[]([٠-٩0-9]{1,2}|\*{1,3})[)\]]\s*$", text)
            if not m:
                break
            markers.insert(0, m.group(1))
            text = text[:m.start()].rstrip()
        if text:
            out.append(Quran(text=text))
        out.extend(Ref(n=sign) for sign in markers)
        for _ in range(count):
            out.append(refs.pop(0) if refs else Ref())

    if not any(ch in _ORNATE for ch in raw):
        emit_verse(raw)
    else:
        pieces = re.split(r"[﴾﴿]", raw)
        parity = 0 if pieces[0].strip(" " + _REF_PLACEHOLDER) else 1
        for i, piece in enumerate(pieces):
            if i % 2 == parity:
                emit_verse(piece)
            else:
                emit_text(piece)
    if trailing:
        out.append(trailing)
    return normalize_inlines(out)


def _inline_children(el: Any, ctx: _Ctx) -> list[Any]:
    """Inline content of an element: its text, converted children and their tails."""
    out: list[Any] = []
    if el.text:
        out.append(_clean(el.text))
    for child in el:
        out.extend(_inline_node(child, ctx))
        if child.tail:
            out.append(_clean(child.tail))
    return out


def _span(cls: type, children: list[Any], **kw: Any) -> list[Any]:
    lead, trail = _edge_spaces(children)
    node = cls(c=_strip_edges(children), **kw)
    return [x for x in (lead, node, trail) if x != ""]


def _inline_node(el: Any, ctx: _Ctx) -> list[Any]:
    tag = _tag(el)
    if not tag:
        return []
    if tag == "ref":
        ref = _make_ref(el, ctx)
        rest = [_inline_node(c, ctx) for c in el]
        return [ref, *[x for part in rest for x in part]]
    if tag == "quran":
        return _split_quran(el, ctx)
    if tag == "quote":
        children = _inline_children(el, ctx)
        style = _attr(el, "style") or _quote_style("".join(_texts(children)))
        return _span(Quote, children, style=style)
    if tag == "matn" or (tag in ("b", "strong") and ctx.commentary == "mixed"):
        return _span(Matn, _inline_children(el, ctx))
    if tag == "unclear":
        conf = _attr(el, "confidence", "conf", "p")
        try:
            confidence = float(conf) if conf is not None else None
        except ValueError:
            confidence = None
        return [Unclear(text=_clean("".join(el.itertext())).strip(), confidence=confidence)]
    if tag == "br":
        return [" ", *_inline_children(el, ctx)]
    if tag == "note" and not ctx.in_note:
        region = _implicit_region(ctx.page)
        note = _parse_note(el, ctx, region)
        region.notes.append(note)
        ctx.flag(FlagCode.NOTE_INLINE, "a note was written inside the text; moved to the foot of the page")
        return [Ref(n=note.marker)] if note.marker else []
    if tag in ("b", "strong") or tag in UNWRAP_SILENT:
        return _inline_children(el, ctx)
    if tag in HEADINGS or tag in ("p", "poem", "v", "s", "a", "note", "hashiya", "figure", "sep",
                                  "header", "pagenum"):
        _unknown(ctx, tag, inline=True)
        return [" ", *_inline_children(el, ctx), " "]
    _unknown(ctx, tag)
    return _inline_children(el, ctx)


def _make_ref(el: Any, ctx: _Ctx) -> Ref:
    generated = (_attr(el, "generated") or "").lower() in TRUE_VALUES
    marker = _attr(el, "n", "num", "marker")
    if marker is None and not generated:
        marker = el.text or ""
    ref = Ref(n=_clean(marker).strip() or None if marker is not None else None)
    ref.id = (_attr(el, "id") or "").strip()
    if generated:
        ref.generated = True
        ref.n = None
        ref.note = (_attr(el, "note") or "").strip() or None
        ctx.ref_methods[id(ref)] = (_attr(el, "method") or "").strip()
    return ref


def _texts(inlines: list[Any]) -> list[str]:
    out = []
    for node in inlines:
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, (Quran, Unclear)):
            out.append(node.text)
        elif isinstance(node, (Quote, Matn)):
            out.extend(_texts(node.c))
    return out


def _unknown(ctx: _Ctx, tag: str, inline: bool = False) -> None:
    key = tag + ("@inline" if inline else "")
    if key in ctx.unknown:
        return
    ctx.unknown.add(key)
    if inline and tag in ("poem", "v", "s", "a", "hashiya"):
        ctx.flag(FlagCode.UNKNOWN_TAG, f"<{tag}> inside running text; its text was kept as prose",
                 severity="warning")
        return
    where = "inside running text" if inline else "unknown tag"
    ctx.flag(FlagCode.UNKNOWN_TAG, f"<{tag}> {where}; kept its text", severity="info")


def _has_text(inlines: list[Any]) -> bool:
    return any(not isinstance(n, str) or n.strip() for n in inlines)


def _all_inline(el: Any) -> bool:
    return all(_tag(c) in INLINE_TAGS for c in el)


# --------------------------------------------------------------------------- poems


def _parse_poem(el: Any, ctx: _Ctx) -> Poem:
    poem = Poem(single=(_attr(el, "type", "kind") or "").lower() in ("single", "mashtur", "free"))
    pending: Verse | None = None

    def repair(msg: str) -> None:
        ctx.flag(FlagCode.POEM_TAG_REPAIR, msg)

    def add_half(kind: str, content: list[Any]) -> None:
        nonlocal pending
        content = _strip_edges(content)
        if kind == "s":
            pending = Verse(sadr=content)
            poem.verses.append(pending)
        elif pending is not None and pending.ajuz is None:
            pending.ajuz = content
            pending = None
        else:
            poem.verses.append(Verse(sadr=[], ajuz=content))
            pending = None

    if el.text and el.text.strip() and not _DECOR_RE.fullmatch(el.text):
        poem.verses.append(Verse(sadr=_strip_edges([_clean(el.text)])))
    for child in el:
        tag = _tag(child)
        if tag == "v":
            verse = Verse()
            loose: list[Any] = [_clean(child.text)] if child.text and not _DECOR_RE.fullmatch(child.text) else []
            for half in child:
                htag = _tag(half)
                content = _strip_edges(_inline_children(half, ctx)) if htag in ("s", "a") else None
                if htag == "s":
                    if not verse.sadr:
                        verse.sadr = content
                    elif verse.ajuz is None:
                        verse.ajuz = content
                        repair("two right-hand half-lines in one verse; the second was taken as the left-hand one")
                    else:
                        verse.ajuz.extend([" ", *content])
                        repair("three half-lines in one verse")
                elif htag == "a":
                    if verse.ajuz is None:
                        verse.ajuz = content
                    else:
                        verse.ajuz.extend([" ", *content])
                        repair("two left-hand half-lines in one verse")
                else:
                    loose.extend(_inline_node(half, ctx))
                if half.tail and half.tail.strip() and not _DECOR_RE.fullmatch(half.tail):
                    loose.append(_clean(half.tail))
            if _has_text(loose):
                loose = _strip_edges(loose)
                if not _has_text(verse.sadr):
                    verse.sadr = loose
                elif verse.ajuz is None:
                    verse.ajuz = loose
                else:
                    verse.ajuz.extend([" ", *loose])
            poem.verses.append(verse)
            pending = None
        elif tag in ("s", "a"):
            add_half(tag, _inline_children(child, ctx))
        elif tag == "ref" and poem.verses:
            last = poem.verses[-1]
            (last.ajuz if last.ajuz is not None else last.sadr).extend(_inline_node(child, ctx))
        else:
            content = _strip_edges(_inline_node(child, ctx))
            if _has_text(content):
                poem.verses.append(Verse(sadr=content))
                pending = None
        if child.tail and child.tail.strip() and not _DECOR_RE.fullmatch(child.tail):
            poem.verses.append(Verse(sadr=_strip_edges([_clean(child.tail)])))
            pending = None
    poem.verses = [v for v in poem.verses if _has_text(v.sadr) or (v.ajuz is not None and _has_text(v.ajuz))]
    return poem


# --------------------------------------------------------------------------- notes


def _note_blocks(el: Any, ctx: _Ctx) -> list[Any]:
    blocks: list[Any] = []
    current: list[Any] = [_clean(el.text)] if el.text else []

    def flush() -> None:
        nonlocal current
        if _has_text(current):
            blocks.append(Paragraph(c=_strip_edges(current)))
        current = []

    for child in el:
        tag = _tag(child)
        if tag == "poem":
            flush()
            blocks.append(_parse_poem(child, ctx))
        elif tag == "p" or tag in HEADINGS:
            flush()
            current = _inline_children(child, ctx)
            flush()
        elif tag == "figure":
            flush()
            blocks.append(_parse_figure(child, ctx))
        else:
            current.extend(_inline_node(child, ctx))
        if child.tail:
            current.append(_clean(child.tail))
    flush()
    return blocks


def _first_str(note: Note) -> tuple[list[Any], int] | None:
    if note.blocks and isinstance(note.blocks[0], Paragraph):
        c = note.blocks[0].c
        for i, node in enumerate(c):
            if isinstance(node, str):
                if node.strip():
                    return c, i
                continue
            return None
    return None


def _lift_marker(note: Note) -> None:
    """A printed marker repeated at the start of the note text becomes ``marker`` (or is removed)."""
    hit = _first_str(note)
    if hit is None:
        return
    container, i = hit
    m = _LEAD_MARKER_RE.match(container[i])
    if not m:
        return
    sign = next(g for g in (m.group("a"), m.group("b"), m.group("c"), m.group("d")) if g)
    if note.marker is None:
        note.marker = sign
    elif normalize_marker(note.marker) != normalize_marker(sign):
        return
    container[i] = container[i][m.end():]
    normalize_inlines(container)


def _strip_cont_signs(note: Note) -> None:
    """A note opening with = continues the previous page's note; one ending with = continues on."""
    if not note.blocks:
        return
    hit = _first_str(note)
    if hit is not None:
        container, i = hit
        head = container[i].lstrip()
        if head.startswith("="):
            note.cont = True
            container[i] = head[1:].lstrip()
    last = note.blocks[-1]
    if isinstance(last, Paragraph) and last.c and isinstance(last.c[-1], str):
        tail = last.c[-1].rstrip()
        if tail.endswith("="):
            note.cont_next = True
            last.c[-1] = tail[:-1].rstrip()
    for block in note.blocks:
        if isinstance(block, Paragraph):
            normalize_inlines(block.c)
    note.blocks = [b for b in note.blocks if not (isinstance(b, Paragraph) and not _has_text(b.c))]


def _finish_note(note: Note) -> Note:
    _strip_cont_signs(note)
    _lift_marker(note)
    _strip_cont_signs(note)
    note.anchor_type = "marker" if note.marker else "lemma" if note.lemma else "position"
    return note


def _parse_note(el: Any, ctx: _Ctx, region: HashiyaRegion) -> Note:
    marker = _attr(el, "n", "num", "marker")
    lemma = _attr(el, "lemma", "qawluh", "qawlihi")
    kind = (_attr(el, "kind", "type") or "").lower()
    cont = (_attr(el, "cont", "continued", "continues") or "").lower()
    was_in_note = ctx.in_note
    ctx.in_note = True
    try:
        blocks = _note_blocks(el, ctx)
    finally:
        ctx.in_note = was_in_note
    note = Note(
        id=(_attr(el, "id") or "").strip(),
        region=region.region,
        layer=region.layer,
        marker=(_clean(marker).strip() or None) if marker is not None else None,
        lemma=(_clean(lemma).strip() or None) if lemma is not None else None,
        blocks=blocks,
    )
    if kind in ("editor", "muhaqqiq", "tahqiq", "محقق"):
        note.kind = "editor"
    elif kind in ("author", "musannif"):
        note.kind = "author"
    if cont in TRUE_VALUES or cont in ("prev", "both"):
        note.cont = True
    if cont in ("next", "both"):
        note.cont_next = True
    return _finish_note(note)


def _parse_bbox(value: str | None) -> list[float] | None:
    if not value:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", value)
    if len(nums) != 4:
        return None
    return [float(x) for x in nums]


def _parse_hashiya(el: Any, ctx: _Ctx) -> HashiyaRegion:
    pos = POS_ALIASES.get((_attr(el, "pos", "position", "region") or "bottom").lower().strip(), "bottom")
    region = HashiyaRegion(region=pos)  # type: ignore[arg-type]
    region.id = (_attr(el, "id") or "").strip()
    layer = _attr(el, "layer")
    if layer and layer.strip().isdigit():
        region.layer = max(1, int(layer.strip()))
    region.independent = (_attr(el, "independent", "separate") or "").lower() in TRUE_VALUES
    rotation = _attr(el, "rotation", "rot")
    if rotation:
        try:
            region.rotation = float(rotation)
        except ValueError:
            pass
    region.bbox = _parse_bbox(_attr(el, "bbox"))
    if region.independent:
        region.blocks = _parse_blocks(el, ctx, top=False)
        return region

    pending_cont = False
    repaired = False

    def add_loose_lines(nodes: list[Any]) -> None:
        """Loose text in a region: each line is a note, or continues the previous one."""
        nonlocal pending_cont, repaired
        lines: list[list[Any]] = [[]]
        for node in nodes:
            if isinstance(node, str) and "\n" in node:
                parts = node.split("\n")
                lines[-1].append(parts[0])
                for part in parts[1:]:
                    lines.append([part])
            else:
                lines[-1].append(node)
        for line in lines:
            line = [(_clean(x) if isinstance(x, str) else x) for x in line]
            if not _has_text(line):
                continue
            text = "".join(_texts(line)).strip()
            if text == "=":
                pending_cont = True
                continue
            repaired = True
            content = Paragraph(c=_strip_edges(line))
            starts_marker = bool(_LEAD_MARKER_RE.match(text))
            if region.notes and not starts_marker and not text.startswith("="):
                last = region.notes[-1].blocks[-1] if region.notes[-1].blocks else None
                if isinstance(last, Paragraph):  # a wrapped line of the same note
                    last.c.extend([" ", *content.c])
                    normalize_inlines(last.c)
                else:
                    region.notes[-1].blocks.append(content)
                continue
            note = Note(region=region.region, layer=region.layer, blocks=[content], cont=pending_cont)
            pending_cont = False
            region.notes.append(_finish_note(note))

    loose: list[Any] = [el.text] if el.text else []
    for child in el:
        tag = _tag(child)
        if tag == "note":
            add_loose_lines(loose)
            loose = []
            note = _parse_note(child, ctx, region)
            if pending_cont:
                note.cont = True
                pending_cont = False
            if note.blocks or note.marker or note.lemma:
                region.notes.append(note)
        elif tag == "poem":
            add_loose_lines(loose)
            loose = []
            poem = _parse_poem(child, ctx)
            if region.notes:
                region.notes[-1].blocks.append(poem)
            else:
                region.notes.append(Note(region=region.region, layer=region.layer, blocks=[poem]))
        elif tag == "p":
            add_loose_lines(loose)
            loose = []
            was = ctx.in_note
            ctx.in_note = True
            add_loose_lines([*_raw_inline_children(child, ctx), "\n"])
            ctx.in_note = was
        else:
            was = ctx.in_note
            ctx.in_note = True
            loose.extend(_inline_node(child, ctx))
            ctx.in_note = was
        if child.tail:
            loose.append(child.tail)
    add_loose_lines(loose)
    if repaired:
        ctx.flag(FlagCode.UNTAGGED_TEXT, "notes were not written as <note> elements; split by line",
                 target=None)
    return region


def _raw_inline_children(el: Any, ctx: _Ctx) -> list[Any]:
    """Like _inline_children but keeps newlines in text, for line splitting."""
    out: list[Any] = []
    if el.text:
        out.append(el.text)
    for child in el:
        out.extend(_inline_node(child, ctx))
        if child.tail:
            out.append(child.tail)
    return out


# --------------------------------------------------------------------------- blocks


def _parse_figure(el: Any, ctx: _Ctx) -> Figure:
    fig = Figure(caption=_strip_edges(_inline_children(el, ctx)))
    box = _parse_bbox(_attr(el, "box", "bbox"))
    page = ctx.page
    if box and page.width and page.height:
        x0, y0, x1, y1 = box
        fig.bbox = [x0 * page.width / 1000, y0 * page.height / 1000,
                    x1 * page.width / 1000, y1 * page.height / 1000]
    elif box:
        fig.bbox = box  # 0..1000 units until resolve_figure_boxes knows the page size
    if fig.bbox is None:
        ctx.flag(FlagCode.FIGURE_WITHOUT_BOX, "figure without a usable box", severity="warning")
    return fig


def _append_to_last(blocks: list[Any], nodes: list[Any]) -> bool:
    if not blocks:
        return False
    last = blocks[-1]
    if isinstance(last, (Paragraph, Heading, QuranBlock)):
        last.c.extend(nodes)
        return True
    if isinstance(last, Poem) and last.verses:
        verse = last.verses[-1]
        (verse.ajuz if verse.ajuz is not None else verse.sadr).extend(nodes)
        return True
    return False


def _strip_trailing_ornate(loose: list[Any]) -> None:
    for i in range(len(loose) - 1, -1, -1):
        node = loose[i]
        if isinstance(node, str):
            if not node.strip():
                continue
            loose[i] = re.sub(r"\s*[﴾﴿]\s*$", "", node)
        return


def _parse_blocks(root: Any, ctx: _Ctx, *, top: bool) -> list[Any]:
    blocks: list[Any] = []
    loose: list[Any] = [_clean(root.text)] if root.text else []
    prev_raw = root.text or ""  # raw text just before the current child
    strip_lead_bracket = False

    def flush() -> None:
        nonlocal loose
        if _has_text(loose):
            blocks.append(Paragraph(c=_strip_edges(loose)))
        loose = []

    for child in root:
        tag = _tag(child)
        own_line = "\n" in prev_raw[len(prev_raw.rstrip()):]  # child starts on a new line
        if tag == "quran" and prev_raw.rstrip().endswith(("﴿", "﴾")):
            # Brackets the model left outside the tag are not text (the builder adds its own).
            _strip_trailing_ornate(loose)
            core = prev_raw.rstrip().rstrip("﴾﴿").rstrip(" \t")
            own_line = core.endswith("\n") or not core.strip()
        if not tag:
            pass
        elif tag in HEADINGS:
            flush()
            blocks.append(Heading(level=HEADINGS[tag], c=_strip_edges(_inline_children(child, ctx))))
        elif tag == "p":
            flush()
            para = Paragraph(c=_strip_edges(_inline_children(child, ctx)))
            cont = (_attr(child, "cont", "continues", "continued") or "").lower()
            para.continued = cont in ("prev", "both") or cont in TRUE_VALUES
            para.continues = cont in ("next", "both")
            if _has_text(para.c):
                blocks.append(para)
        elif tag == "quran" and _has_text(loose) and not own_line:
            loose.extend(_inline_node(child, ctx))  # a verse inside untagged running text
            strip_lead_bracket = True
        elif tag == "quran":
            flush()
            spans = _split_quran(child, ctx)
            if spans:
                blocks.append(QuranBlock(c=_strip_edges(spans)))
            strip_lead_bracket = True
        elif tag == "poem":
            flush()
            poem = _parse_poem(child, ctx)
            if poem.verses:
                blocks.append(poem)
        elif tag == "sep":
            flush()
            blocks.append(Separator(text=_clean(_attr(child, "text") or "".join(child.itertext())).strip()))
        elif tag == "figure":
            flush()
            blocks.append(_parse_figure(child, ctx))
        elif tag == "hashiya" and top:
            flush()
            ctx.page.hashiya.append(_parse_hashiya(child, ctx))
        elif tag == "note" and top:
            flush()
            region = _implicit_region(ctx.page)
            region.notes.append(_parse_note(child, ctx, region))
        elif tag == "header" and top:
            flush()
            text = _clean("".join(child.itertext())).strip()
            if text:
                ctx.page.running_header = (ctx.page.running_header + " | " + text
                                           if ctx.page.running_header else text)
        elif tag == "pagenum" and top:
            flush()
            text = _clean("".join(child.itertext())).strip()
            if text and ctx.page.printed_page is None:
                ctx.page.printed_page = text
                ctx.page.printed_page_num = parse_number(text)
            elif text and text != ctx.page.printed_page:
                # A second number is usually a reference printed in the margin (another edition's page).
                ctx.flag(FlagCode.UNKNOWN_TAG, f"second page number «{text}» ignored", severity="info")
        elif tag == "ref" and not _has_text(loose):
            if not _append_to_last(blocks, _inline_node(child, ctx)):
                loose.extend(_inline_node(child, ctx))
        elif tag in INLINE_TAGS or (tag in UNWRAP_SILENT and _all_inline(child)):
            loose.extend(_inline_node(child, ctx))
        elif tag in UNWRAP_SILENT:
            flush()
            blocks.extend(_parse_blocks(child, ctx, top=top))
        else:
            flush()
            _unknown(ctx, tag)
            blocks.extend(_parse_blocks(child, ctx, top=top))
        tail = child.tail or ""
        if strip_lead_bracket:
            tail = re.sub(r"^\s*[﴾﴿]", "", tail)
            strip_lead_bracket = False
        if tail:
            loose.append(_clean(tail))
        prev_raw = tail
    flush()
    return blocks


def _implicit_region(page: Page) -> HashiyaRegion:
    for region in page.hashiya:
        if region.region == "bottom" and not region.independent:
            return region
    region = HashiyaRegion(region="bottom")
    page.hashiya.append(region)
    return region


def _plain_fallback(text: str, page: Page) -> None:
    stripped = html.unescape(_TAG_RE.sub(" ", text))
    for chunk in re.split(r"\n\s*\n", stripped):
        line = _clean(chunk).strip()
        if line:
            page.blocks.append(Paragraph(c=[line]))


def _order_regions(page: Page) -> None:
    rank = {name: i for i, name in enumerate(REGION_ORDER)}
    page.hashiya.sort(key=lambda r: (r.layer, rank.get(r.region, 99)))


def _strip_outer_brackets_inline(page: Page) -> None:
    """Remove ornate brackets left next to an inline Quran span (the builder adds its own)."""
    for block, _owner in iter_blocks(page):
        for container in block_containers(block):
            stack = [container]
            while stack:
                lst = stack.pop()
                for i, node in enumerate(lst):
                    if isinstance(node, (Quote, Matn)):
                        stack.append(node.c)
                    if not isinstance(node, Quran):
                        continue
                    if i > 0 and isinstance(lst[i - 1], str):
                        lst[i - 1] = re.sub(r"[﴾﴿]\s*$", "", lst[i - 1])
                    if i + 1 < len(lst) and isinstance(lst[i + 1], str):
                        lst[i + 1] = re.sub(r"^\s*[﴾﴿]", "", lst[i + 1])
                normalize_inlines(lst)


def _resolve_generated_refs(page: Page, ctx: _Ctx) -> None:
    """Generated anchors written with ``<ref generated="1" note=…/>`` re-link their notes."""
    notes = {n.id: n for n in page.notes()}
    for block, _owner in iter_blocks(page):
        bt = None
        for container in block_containers(block):
            for node in walk_inlines(container):
                if isinstance(node, Ref) and node.generated and node.note in notes:
                    note = notes[node.note]
                    method = ctx.ref_methods.get(id(node)) or "lemma"
                    if bt is None:
                        bt = BlockText(block)
                    note.link = NoteLink(status="linked" if method != "position" else "guessed",
                                         method=method if method in ("marker", "lemma", "position", "manual",
                                                                     "continuation") else "lemma",
                                         page=page.index, block=block.id, ref=node.id,
                                         offset=bt.ref_offsets.get(node.id))


def _page_letters(page: Page) -> int:
    total = _letters(page.running_header or "") + _letters(page.printed_page or "")
    for block, _owner in iter_blocks(page):
        total += _letters(BlockText(block).text)
        if isinstance(block, Separator):
            total += _letters(block.text)
    return total


def parse_tags(raw: str, base: Page | None = None, *, commentary: str = "none") -> ParseResult:
    """Parse one page of model output.

    ``base`` supplies index, image, size and OCR info (and is copied, not
    modified); the result has ids, ``status="new"`` and any parse flags.
    ``commentary="mixed"`` maps ``<b>`` to matn spans (شرح ممزوج).
    """
    page = base.model_copy(deep=True) if base is not None else Page(index=0)
    page.blocks, page.hashiya, page.flags = [], [], [f for f in page.flags if f.stage != STAGE]
    page.printed_page = page.printed_page_num = page.running_header = None
    page.status = "new"
    ctx = _Ctx(page, commentary)
    cleaned = _clean_raw(raw)
    if not cleaned:
        assign_ids(page)
        return ParseResult(page, ok=True, recovered=False, text_loss=0.0)
    text, changed = balance(cleaned)
    wrapped = f"<root>{text}</root>".encode("utf-8")
    recovered = changed
    try:
        root = etree.fromstring(wrapped)
    except etree.XMLSyntaxError as exc:
        recovered = True
        root = etree.fromstring(wrapped, etree.XMLParser(recover=True, huge_tree=True))
        if root is None:
            _plain_fallback(cleaned, page)
            ctx.flag(FlagCode.PARSE_ERROR, f"output is not valid tags ({exc}); kept as plain text",
                     severity="error")
            assign_ids(page)
            return ParseResult(page, ok=False, recovered=True, text_loss=0.0)
    try:  # judge the format on the model's own structure, before any repair
        format_root = etree.fromstring(f"<root>{cleaned}</root>".encode("utf-8")) if changed else root
    except etree.XMLSyntaxError:
        format_root = root
    if _untagged(format_root):
        ctx.flag(FlagCode.UNTAGGED_TEXT, "much of the output is text outside the tag format")
    if recovered:
        ctx.flag(FlagCode.PARSE_RECOVERED, "the tags were malformed and had to be repaired")
    page.blocks = _parse_blocks(root, ctx, top=True)
    _strip_outer_brackets_inline(page)
    _order_regions(page)
    assign_ids(page)
    _resolve_generated_refs(page, ctx)
    expected = _letters(html.unescape(_TAG_RE.sub(" ", cleaned)))
    got = _page_letters(page)
    loss = max(0.0, 1.0 - got / expected) if expected else 0.0
    n_markers = len(page.notes()) + sum(1 for b, _o in iter_blocks(page) for c in block_containers(b)
                                        for x in walk_inlines(c) if isinstance(x, Ref))
    if expected >= 50 and loss > 0.01 + 3 * n_markers / expected:
        ctx.flag(FlagCode.TEXT_LOST, f"{loss:.0%} of the text the model wrote is missing from the page",
                 severity="error")
    return ParseResult(page, ok=True, recovered=recovered, text_loss=loss)


def resolve_figure_boxes(page: Page) -> None:
    """Convert figure boxes left in 0..1000 units into pixels once the page size is known.

    Only for pages parsed without a size: ``parse_tags`` already converts boxes
    when the base page has ``width`` and ``height``.
    """
    if not (page.width and page.height):
        return
    for block, _owner in iter_blocks(page):
        if isinstance(block, Figure) and block.bbox and max(block.bbox) <= 1000.0:
            x0, y0, x1, y1 = block.bbox
            block.bbox = [x0 * page.width / 1000, y0 * page.height / 1000,
                          x1 * page.width / 1000, y1 * page.height / 1000]
