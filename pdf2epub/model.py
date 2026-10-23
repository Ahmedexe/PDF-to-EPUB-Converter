"""The page model: the one intermediate format between OCR and EPUB (spec §3 and §6.2).

The model never writes EPUB directly. It writes light tagged text, which
``pdf2epub.parse`` turns into a :class:`Page`; every later stage reads and
corrects pages, and ``pdf2epub.assemble`` joins them into a :class:`Book`.

Inline content
--------------
Text-bearing fields (paragraph content, heading content, poem half-lines,
captions, quote and matn children) are *inline lists*: Python lists whose items
are plain ``str`` runs or inline objects (:class:`Ref`, :class:`Quran`,
:class:`Quote`, :class:`Matn`, :class:`Unclear`, :class:`PageBreak`).

Character offsets are never stored as the source of truth. They are computed on
demand by :class:`BlockText`, which flattens a block's inline lists into one
plain string and maps offsets back to nodes. ``NoteLink.offset`` is
informational only and is recomputed whenever links are resolved.

Mapping to the spec's block table (§3)
--------------------------------------
heading, paragraph, quran, poem, figure  -> block classes below
quote, matn, unclear                     -> inline spans (they occur inside text)
hashiya, note                            -> Page.hashiya regions holding Note objects
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

RegionPos = Literal["bottom", "margin-right", "margin-left", "top", "frame"]
NoteKind = Literal["author", "editor", "unknown"]
AnchorType = Literal["marker", "lemma", "position"]
LinkStatus = Literal["unlinked", "linked", "guessed", "flagged"]
LinkMethod = Literal["marker", "lemma", "position", "manual", "continuation"]
QuranStatus = Literal["unchecked", "verified", "corrected", "flagged", "accepted"]  # accepted = approved by a reviewer as printed
PageStatus = Literal["new", "clean", "flagged", "reviewed"]
Severity = Literal["info", "warning", "error"]

# Reading order of hashiya regions after the main text (spec §6.3 step 4).
REGION_ORDER: tuple[str, ...] = ("bottom", "margin-right", "margin-left", "top", "frame")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- inline


class Ref(_Model):
    """A note anchor in running text.

    Printed markers come from ``<ref n="..."/>`` in the model output. The linker
    inserts *generated* refs for lemma (قوله) and position notes, at the end of
    the words the note explains, because the printed text has no marker there.
    """

    t: Literal["ref"] = "ref"
    id: str = ""  # "p0037-r1"; unique within the book
    n: str | None = None  # printed marker exactly as printed ("١", "*", "أ"); None if generated
    generated: bool = False
    note: str | None = None  # id of the linked Note, set by checks.hashiya


class QuranLoc(_Model):
    """Where a quoted Quran passage sits in the Tanzil text (inclusive range).

    Word numbers are 1-based positions within an aya, counted in ``variant``'s
    own word segmentation (Uthmani and simple spelling segment some words
    differently).
    """

    variant: str
    sura: int
    aya: int
    word: int
    end_sura: int
    end_aya: int
    end_word: int


class Quran(_Model):
    """A Quran passage, block-level or inside running text (spec §5.1).

    ``text`` never includes the ornate brackets: the EPUB builder adds U+FD3F
    before and U+FD3E after, in logical order.
    """

    t: Literal["quran"] = "quran"
    text: str = ""  # current text; the Tanzil text once verified or corrected
    ocr_text: str | None = None  # the text as read, kept whenever ``text`` was replaced
    ref: str | None = None  # "2:155", "2:155-157" or "2:286-3:2"
    loc: QuranLoc | None = None
    status: QuranStatus = "unchecked"
    score: float | None = None  # similarity of ocr text to the chosen Tanzil passage (0..1)
    variant: str | None = None  # Tanzil variant the replacement text was taken from
    alternatives: list[str] = Field(default_factory=list)  # other refs with an identical text


class Quote(_Model):
    """Other bracketed text such as hadith in « » or (( )) (spec §5.3).

    The brackets stay in the text exactly as printed; ``style`` records the
    bracket pair (for example "«»" or "(())") for styling and later checks.
    """

    t: Literal["quote"] = "quote"
    style: str | None = None
    c: list[Inline] = Field(default_factory=list)


class Matn(_Model):
    """Main-text words woven into a commentary (شرح ممزوج). Printed parentheses stay in the text."""

    t: Literal["matn"] = "matn"
    c: list[Inline] = Field(default_factory=list)


class Unclear(_Model):
    """Words the model could not read with confidence; ``text`` is its tentative reading."""

    t: Literal["unclear"] = "unclear"
    text: str = ""
    confidence: float | None = None


class PageBreak(_Model):
    """Where a printed page begins. Inserted only by ``assemble``; drives the EPUB page list."""

    t: Literal["pb"] = "pb"
    page: int  # page index (1-based position in the book)
    label: str  # printed page number as printed, or a fallback label


InlineObj = Annotated[Union[Ref, Quran, Quote, Matn, Unclear, PageBreak], Field(discriminator="t")]
Inline = Union[str, InlineObj]
InlineList = list[Inline]


# --------------------------------------------------------------------------- blocks


class Heading(_Model):
    type: Literal["heading"] = "heading"
    id: str = ""
    level: int = 2  # 1..4; feeds the EPUB table of contents
    c: list[Inline] = Field(default_factory=list)


class Paragraph(_Model):
    type: Literal["paragraph"] = "paragraph"
    id: str = ""
    c: list[Inline] = Field(default_factory=list)
    continues: bool = False  # runs onto the next page
    continued: bool = False  # continues from the previous page


class QuranBlock(_Model):
    """A displayed Quran passage: one or more Quran spans, plus any refs or text printed with it."""

    type: Literal["quran"] = "quran"
    id: str = ""
    c: list[Inline] = Field(default_factory=list)


class Verse(_Model):
    """One بيت: right-hand half-line (صدر) first, then the left-hand half-line (عجز)."""

    sadr: list[Inline] = Field(default_factory=list)
    ajuz: list[Inline] | None = None  # None for single-column verses


class Poem(_Model):
    type: Literal["poem"] = "poem"
    id: str = ""
    verses: list[Verse] = Field(default_factory=list)
    single: bool = False  # single-column (مشطور or modern poetry): sadr only
    continues: bool = False
    continued: bool = False


class Figure(_Model):
    type: Literal["figure"] = "figure"
    id: str = ""
    bbox: list[float] | None = None  # [x0, y0, x1, y1] in page-image pixels
    image: str | None = None  # cropped image, path relative to the book's work folder
    caption: list[Inline] = Field(default_factory=list)


class Separator(_Model):
    """A decorative section divider (for example * * *), kept out of the text."""

    type: Literal["separator"] = "separator"
    id: str = ""
    text: str = ""  # as printed; the builder may substitute its own ornament


Block = Annotated[
    Union[Heading, Paragraph, QuranBlock, Poem, Figure, Separator], Field(discriminator="type")
]


# --------------------------------------------------------------------------- hashiya


class NoteLink(_Model):
    """The resolved anchor of a note (spec §6.2 ``link``)."""

    status: LinkStatus = "unlinked"
    method: LinkMethod | None = None
    page: int | None = None  # page index holding the anchor (a lemma may match the previous page)
    block: str | None = None  # id of the block holding the anchor
    ref: str | None = None  # id of the anchor Ref
    offset: int | None = None  # informational: char offset of the anchor in BlockText(block).text
    score: float | None = None  # lemma match score (0..1)
    # Manual links (review screen) also record the anchored words, so they can be
    # re-placed after the page is re-parsed: the end of the ``occurrence``-th match
    # of ``anchor`` in page ``page``'s main text.
    anchor: str | None = None
    occurrence: int = 1


class Note(_Model):
    """One حاشية note (spec §6.2)."""

    id: str = ""  # "p0037-n2"
    kind: NoteKind = "unknown"
    region: RegionPos = "bottom"
    anchor_type: AnchorType = "position"
    marker: str | None = None  # printed marker, for marker notes
    lemma: str | None = None  # quoted main-text words after قوله, for lemma notes
    blocks: list[Block] = Field(default_factory=list)  # note content (Quran, poems, quotes allowed)
    layer: int = 1  # 1 = on the main text, 2 = on the commentary, 3 = on the حاشية
    cont: bool = False  # opens as a continuation of the previous page's last note
    cont_next: bool = False  # ends with the = sign: continues on the next page
    continues_from: str | None = None  # id of the previous page's note it continues
    continues_to: str | None = None  # id of the next page's note continuing it
    link: NoteLink = Field(default_factory=NoteLink)


class HashiyaRegion(_Model):
    """One printed region of notes: foot of page, a margin, the top, or a frame."""

    id: str = ""  # "p0037-h1"
    region: RegionPos = "bottom"
    rotation: float = 0.0  # degrees, see LayoutRegion.rotation
    bbox: list[float] | None = None  # [x0, y0, x1, y1] in page-image pixels, when known
    layer: int = 1
    independent: bool = False  # a separate work printed بهامشه; its text is in ``blocks``
    notes: list[Note] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)  # only for independent regions


# --------------------------------------------------------------------------- page


class Flag(_Model):
    """A reason a page needs human review (warning/error) or an informational note (info)."""

    code: str  # one of FlagCode
    message: str = ""
    severity: Severity = "warning"
    target: str | None = None  # id of the block, note, ref or region concerned
    stage: str = ""  # parse | ocr | layout | quran | poetry | hashiya | health | review


class FlagCode:
    """Registry of flag codes, so checks, the review screen and eval agree on names."""

    PARSE_ERROR = "parse_error"  # tags did not parse even after retry
    PARSE_RECOVERED = "parse_recovered"  # parsed only with lxml recovery; text may be lost
    UNKNOWN_TAG = "unknown_tag"
    OCR_FAILED = "ocr_failed"
    EMPTY_OUTPUT = "empty_output"  # no text, although the page has ink
    TRUNCATED = "truncated"  # finish_reason == "length"
    REPETITION = "repetition"  # a line or phrase loops
    WORDCOUNT_MISMATCH = "wordcount_mismatch"  # differs sharply from Tesseract's count
    UNCLEAR_TEXT = "unclear_text"
    LAYOUT_FAILED = "layout_failed"
    FIGURE_WITHOUT_BOX = "figure_without_box"
    QURAN_UNMATCHED = "quran_unmatched"
    QURAN_AMBIGUOUS = "quran_ambiguous"
    QURAN_SHORT = "quran_short"
    POEM_MISSING_HALF = "poem_missing_half"
    POEM_HALF_LENGTH = "poem_half_length"
    POEM_RHYME = "poem_rhyme"
    POEM_METER = "poem_meter"
    REF_WITHOUT_NOTE = "ref_without_note"
    NOTE_WITHOUT_REF = "note_without_ref"
    NOTE_NUMBERING_GAP = "note_numbering_gap"
    NOTE_ORDER = "note_order"
    LEMMA_NOT_FOUND = "lemma_not_found"
    LEMMA_LOW_SCORE = "lemma_low_score"
    POSITION_GUESS = "position_guess"
    CONTINUATION_ORPHAN = "continuation_orphan"
    UNTAGGED_TEXT = "untagged_text"  # output ignored the tag format (loose text, notes as <p>)
    TEXT_LOST = "text_lost"  # the page model holds less text than the model wrote
    NOTE_INLINE = "note_inline"  # a <note> written inside running text, moved to the foot
    HEADER_IN_TEXT = "header_in_text"  # a running header or page number tagged as text
    POEM_SEPARATOR = "poem_separator"  # decorative marks read as part of a half-line
    POEM_TAG_REPAIR = "poem_tag_repair"  # half-line tags had to be repaired
    QURAN_UNCHECKED = "quran_unchecked"  # a verse no check resolved (blocks a final build)
    MISSING_NOTES = "missing_notes"  # the image shows a foot-note rule but no notes were read
    MARKER_IN_TEXT = "marker_in_text"  # "(١)" left in the text with no note to pair it with


class OcrInfo(_Model):
    backend: str = ""
    model: str = ""
    prompt_version: str = ""
    calls: int = 0  # model calls made for this page (one per region when split)
    attempts: int = 1  # 2 when the higher-resolution retry was used
    image_px: list[int] | None = None  # [width, height] of the image sent
    elapsed_s: float | None = None
    finish_reasons: list[str] = Field(default_factory=list)  # "stop" | "length" | "error" | "blank"
    cached: bool = False
    layout_source: str | None = None  # PageLayout.source when the layout step ran
    layout_error: str | None = None  # why the layout answer was unusable


class Page(_Model):
    """One page of the book, as stored in ``pages/NNNN.json``."""

    schema_version: int = SCHEMA_VERSION
    index: int  # 1-based position in the book, after two-page spreads are split
    pdf_page: int = 0  # 1-based page number in the source PDF
    side: Literal["full", "right", "left"] = "full"
    image: str = ""  # page image, relative to the work folder ("pages/0037.png")
    width: int = 0
    height: int = 0
    printed_page: str | None = None  # page number as printed, e.g. "٣٧"
    printed_page_num: int | None = None
    running_header: str | None = None  # recognised and dropped from the text
    blocks: list[Block] = Field(default_factory=list)  # main text, in reading order
    hashiya: list[HashiyaRegion] = Field(default_factory=list)  # in REGION_ORDER
    flags: list[Flag] = Field(default_factory=list)
    status: PageStatus = "new"
    ocr: OcrInfo | None = None
    main_bbox: list[float] | None = None  # main-text region box, when the layout step ran
    blank: bool = False  # ingest found no ink on the page

    @property
    def prefix(self) -> str:
        return page_prefix(self.index)

    def notes(self) -> list[Note]:
        return [n for r in self.hashiya for n in r.notes]

    def add_flag(
        self,
        code: str,
        message: str = "",
        *,
        severity: Severity = "warning",
        target: str | None = None,
        stage: str = "",
    ) -> Flag:
        flag = Flag(code=code, message=message, severity=severity, target=target, stage=stage)
        self.flags.append(flag)
        return flag

    def clear_flags(self, stage: str) -> None:
        """Drop the flags raised by one stage, so re-running that stage is idempotent."""
        self.flags = [f for f in self.flags if f.stage != stage]


# --------------------------------------------------------------------------- layout


class LayoutRegion(_Model):
    """A region found by the layout step (spec §6.3), stored in ``pages/NNNN.regions.json``.

    ``rotation`` is the angle in degrees, counter-clockwise, by which the printed
    text is turned away from upright; ``layout.crop`` rotates the crop clockwise
    by this angle to make it upright. For Arabic: 90 when the tops of the letters
    point left (the lines then read top-to-bottom), 270 when they point right
    (lines read bottom-to-top), otherwise the slant angle.
    """

    id: str  # "main", "h1", "h2", "header", "pagenum", "fig1", "frame1"
    kind: Literal["main", "hashiya", "header", "page_number", "figure", "frame"]
    pos: RegionPos | None = None  # for hashiya regions
    bbox: list[float]  # [x0, y0, x1, y1] in page-image pixels
    rotation: float = 0.0
    layer: int = 1
    independent: bool = False  # for kind == "frame": a separate work, not notes
    image: str | None = None  # upright crop, relative to the work folder
    text: str | None = None  # printed text of header / page_number regions, when the layout call read it


class PageLayout(_Model):
    index: int
    width: int
    height: int
    source: Literal["vlm", "cv", "manual", "none"] = "none"
    split: bool = False  # True when OCR should read regions separately
    regions: list[LayoutRegion] = Field(default_factory=list)
    error: str | None = None  # why the model's layout answer was not used


# --------------------------------------------------------------------------- manifest


class PageEntry(_Model):
    """One page image produced by ingest."""

    index: int
    pdf_page: int
    side: Literal["full", "right", "left"] = "full"
    image: str  # relative to the work folder
    width: int
    height: int
    sha256: str
    source: Literal["render", "extract"] = "render"
    dpi: int | None = None
    deskew_angle: float = 0.0
    blank: bool = False


class Manifest(_Model):
    schema_version: int = SCHEMA_VERSION
    source_pdf: str = ""
    pdf_sha256: str = ""
    pdf_page_count: int = 0
    pages: list[PageEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- book


class Chapter(_Model):
    id: str  # "ch001"
    stream: str = "main"  # "main", or the name of an independent work printed in a frame
    level: int = 1
    title: str = ""  # plain-text title for the table of contents
    blocks: list[Block] = Field(default_factory=list)  # starts with its Heading when it has one
    notes: list[Note] = Field(default_factory=list)  # notes anchored here (all layers), anchor order
    pages: list[int] = Field(default_factory=list)  # page indices this chapter covers


class Book(_Model):
    schema_version: int = SCHEMA_VERSION
    slug: str
    title: str
    author: str | None = None
    language: str = "ar"
    identifier: str  # "urn:uuid:..."
    digits: Literal["arabic-indic", "western"] = "arabic-indic"
    quran_variant: str | None = None
    chapters: list[Chapter] = Field(default_factory=list)
    page_labels: dict[int, str] = Field(default_factory=dict)  # page index -> printed label


for _cls in (Quote, Matn, Heading, Paragraph, QuranBlock, Verse, Poem, Figure, Note, HashiyaRegion,
             Page, Chapter, Book):
    _cls.model_rebuild()


# --------------------------------------------------------------------------- ids


def page_prefix(index: int) -> str:
    """Id prefix for everything on a page: ``p0037``."""
    return f"p{index:04d}"


def assign_ids(page: Page) -> None:
    """Give every block, region, note and ref on the page an id, in reading order.

    Blocks: p0037-b1..; regions: p0037-h1..; notes: p0037-n1.. (across regions);
    blocks inside notes: <note id>-b1..; refs: p0037-r1.. .

    Ids are stable: a node that already has a well-formed, unique id for this
    page keeps it, so links into the page (from other pages too) stay valid.
    Nodes without one are numbered after the highest existing number. Links
    and flag targets on the page are remapped when an id had to change.
    """
    pre = page.prefix
    remap: dict[str, str] = {}

    def make_counter(kind: str, existing: list[str]) -> Callable[[str], str]:
        head = f"{pre}-{kind}"
        used = set()
        top = 0
        for value in existing:
            tail = value[len(head):] if value.startswith(head) else ""
            if tail.isdigit():
                top = max(top, int(tail))
        state = {"top": top, "used": used}

        def give(old: str) -> str:
            tail = old[len(head):] if old.startswith(head) else ""
            if tail.isdigit() and old not in state["used"]:
                state["used"].add(old)
                return old
            state["top"] += 1
            new = f"{head}{state['top']}"
            state["used"].add(new)
            if old:
                remap[old] = new
            return new

        return give

    def all_refs() -> list[Ref]:
        out = []
        for block, _owner in iter_blocks(page):
            for container in block_containers(block):
                out.extend(n for n in walk_inlines(container) if isinstance(n, Ref))
        return out

    block_ids = [b.id for b in page.blocks]
    give_block = make_counter("b", block_ids)
    for block in page.blocks:
        block.id = give_block(block.id)
    give_region = make_counter("h", [r.id for r in page.hashiya])
    give_note = make_counter("n", [n.id for r in page.hashiya for n in r.notes])
    for region in page.hashiya:
        region.id = give_region(region.id)
        for j, block in enumerate(region.blocks, 1):
            new = f"{region.id}-b{j}"
            if block.id and block.id != new:
                remap[block.id] = new
            block.id = new
        for note in region.notes:
            note.id = give_note(note.id)
            for j, block in enumerate(note.blocks, 1):
                new = f"{note.id}-b{j}"
                if block.id and block.id != new:
                    remap[block.id] = new
                block.id = new
    refs = all_refs()
    give_ref = make_counter("r", [r.id for r in refs])
    for ref in refs:
        ref.id = give_ref(ref.id)
    if not remap:
        return
    for note in page.notes():
        for attr in ("ref", "block"):
            value = getattr(note.link, attr)
            if value in remap:
                setattr(note.link, attr, remap[value])
        for attr in ("continues_from", "continues_to"):
            value = getattr(note, attr)
            if value in remap:
                setattr(note, attr, remap[value])
    for ref in refs:
        if ref.note in remap:
            ref.note = remap[ref.note]
    for flag in page.flags:
        if flag.target in remap:
            flag.target = remap[flag.target]


def next_ref_id(page: Page) -> str:
    """A ref id not yet used on the page."""
    pre = page.prefix + "-r"
    used = 0
    for block, _owner in iter_blocks(page):
        for container in block_containers(block):
            for node in walk_inlines(container):
                if isinstance(node, Ref) and node.id.startswith(pre):
                    tail = node.id[len(pre):]
                    if tail.isdigit():
                        used = max(used, int(tail))
    return f"{pre}{used + 1}"


# --------------------------------------------------------------------------- traversal


def iter_blocks(page: Page, *, notes: bool = True) -> Iterator[tuple[Any, Note | HashiyaRegion | None]]:
    """Yield ``(block, owner)``: main-text blocks (owner None), independent-region blocks
    (owner the region) and, when ``notes``, note blocks (owner the Note)."""
    for block in page.blocks:
        yield block, None
    for region in page.hashiya:
        for block in region.blocks:
            yield block, region
        if notes:
            for note in region.notes:
                for block in note.blocks:
                    yield block, note


def block_containers(block: Any) -> list[list[Any]]:
    """The inline lists of a block, in reading order.

    Heading, Paragraph, QuranBlock: [c]; Poem: [sadr, ajuz] per verse (ajuz
    omitted when None); Figure: [caption]; Separator: [].
    """
    if isinstance(block, (Heading, Paragraph, QuranBlock)):
        return [block.c]
    if isinstance(block, Poem):
        out: list[list[Any]] = []
        for verse in block.verses:
            out.append(verse.sadr)
            if verse.ajuz is not None:
                out.append(verse.ajuz)
        return out
    if isinstance(block, Figure):
        return [block.caption]
    return []


def walk_inlines(inlines: list[Any]) -> Iterator[Any]:
    """Depth-first walk over inline nodes, including the children of Quote and Matn."""
    for node in inlines:
        yield node
        if isinstance(node, (Quote, Matn)):
            yield from walk_inlines(node.c)


def walk_inline_slots(inlines: list[Any]) -> Iterator[tuple[list[Any], int, Any]]:
    """Like :func:`walk_inlines` but yields ``(container, index, node)`` so callers can replace nodes."""
    for i, node in enumerate(inlines):
        yield inlines, i, node
        if isinstance(node, (Quote, Matn)):
            yield from walk_inline_slots(node.c)


def iter_quran(page: Page) -> Iterator[tuple[Any, Quran]]:
    """Yield ``(block, quran_span)`` for every Quran span on the page, main text and notes."""
    for block, _owner in iter_blocks(page):
        for container in block_containers(block):
            for node in walk_inlines(container):
                if isinstance(node, Quran):
                    yield block, node


def iter_refs(blocks: list[Any]) -> Iterator[tuple[Any, Ref]]:
    """Yield ``(block, ref)`` for every Ref in the given blocks, in reading order."""
    for block in blocks:
        for container in block_containers(block):
            for node in walk_inlines(container):
                if isinstance(node, Ref):
                    yield block, node


def inline_text(inlines: list[Any]) -> str:
    """Plain text of an inline list. Refs and page breaks contribute nothing."""
    parts: list[str] = []
    for node in inlines:
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, (Quran, Unclear)):
            parts.append(node.text)
        elif isinstance(node, (Quote, Matn)):
            parts.append(inline_text(node.c))
    return "".join(parts)


def block_text(block: Any) -> str:
    """Plain text of a block; the same string :class:`BlockText` indexes."""
    return BlockText(block).text


# --------------------------------------------------------------------------- offsets


HALF_SEP = "\t"  # between صدر and عجز in BlockText
VERSE_SEP = "\n"  # between verses, and between other containers


class _Seg:
    __slots__ = ("start", "end", "container", "index", "atomic")

    def __init__(self, start: int, end: int, container: list[Any], index: int, atomic: bool):
        self.start, self.end = start, end
        self.container, self.index, self.atomic = container, index, atomic


class BlockText:
    """A block's text flattened into one string, with offsets mapped back to inline nodes.

    The text is the concatenation of the block's containers (see
    :func:`block_containers`) joined by HALF_SEP between the halves of a verse
    and VERSE_SEP elsewhere. ``str`` runs can be split; Quran and Unclear spans
    are atomic (an offset inside one resolves to its end).

    A BlockText is a snapshot: rebuild it after any edit to the block.
    """

    def __init__(self, block: Any):
        self.block = block
        self.containers = block_containers(block)
        self.segs: list[_Seg] = []
        self.ref_offsets: dict[str, int] = {}
        self.container_starts: list[int] = []
        parts: list[str] = []
        pos = 0
        poem = isinstance(block, Poem)
        half_flags: list[bool] = []
        if poem:
            for verse in block.verses:
                half_flags.append(False)
                if verse.ajuz is not None:
                    half_flags.append(True)
        for ci, container in enumerate(self.containers):
            if ci > 0:
                sep = HALF_SEP if poem and half_flags[ci] else VERSE_SEP
                parts.append(sep)
                pos += len(sep)
            self.container_starts.append(pos)
            pos = self._walk(container, pos, parts)
        self.text = "".join(parts)

    def _walk(self, container: list[Any], pos: int, parts: list[str]) -> int:
        for i, node in enumerate(container):
            if isinstance(node, str):
                if node:
                    self.segs.append(_Seg(pos, pos + len(node), container, i, False))
                    parts.append(node)
                    pos += len(node)
            elif isinstance(node, (Quran, Unclear)):
                if node.text:
                    self.segs.append(_Seg(pos, pos + len(node.text), container, i, True))
                    parts.append(node.text)
                    pos += len(node.text)
            elif isinstance(node, (Quote, Matn)):
                pos = self._walk(node.c, pos, parts)
            elif isinstance(node, Ref):
                if node.id:
                    self.ref_offsets[node.id] = pos
        return pos

    def node_at(self, offset: int) -> tuple[list[Any], int, int] | None:
        """``(container, index, char_offset_in_node)`` of the node holding ``offset``.

        An offset on a boundary resolves to the end of the preceding node.
        Returns None when the block has no text.
        """
        best = None
        for seg in self.segs:
            if seg.start < offset <= seg.end or (offset == 0 and seg.start == 0):
                best = seg
                break
            if seg.end <= offset:
                best = seg
        if best is None:
            return None
        return best.container, best.index, max(0, min(offset, best.end) - best.start)

    def insert_at(self, offset: int, node: Any) -> bool:
        """Insert an inline node (usually a generated Ref) at a text offset.

        A ``str`` run is split at the offset; an offset inside an atomic span
        places the node right after that span. With no text at all the node is
        appended to the first container. Returns False if the block has no
        containers. The BlockText is stale afterwards.
        """
        if not self.containers:
            return False
        hit = self.node_at(offset)
        if hit is None:
            self.containers[0].append(node)
            return True
        container, index, k = hit
        current = container[index]
        if isinstance(current, str) and 0 < k < len(current):
            container[index:index + 1] = [current[:k], node, current[k:]]
        elif isinstance(current, str) and k == 0:
            container.insert(index, node)
        else:
            container.insert(index + 1, node)
        return True

    def wrap_range(self, start: int, end: int, make: Callable[[list[Any]], Any]) -> Any | None:
        """Replace the nodes covering text ``[start, end)`` with ``make(nodes)``.

        Both ends must fall in ``str`` runs (or on node boundaries) of the same
        container list; runs are split at the ends. ``make`` receives the
        covered nodes (which may include Refs or spans lying wholly inside) and
        returns the replacement node. Returns that node, or None when the range
        crosses containers or cuts through an atomic span.
        """
        if start >= end:
            return None
        first = last = None
        for seg in self.segs:
            if first is None and seg.start <= start < seg.end:
                first = seg
            if seg.start < end <= seg.end:
                last = seg
        if first is None or last is None or first.container is not last.container:
            return None
        if (first.atomic and start != first.start) or (last.atomic and end != last.end):
            return None
        container = first.container
        i0, i1 = first.index, last.index
        head = tail = None
        nodes = container[i0:i1 + 1]
        if isinstance(nodes[-1], str):
            k = end - last.start
            nodes[-1], tail = nodes[-1][:k], nodes[-1][k:]
        if isinstance(nodes[0], str):
            k = start - first.start
            head, nodes[0] = nodes[0][:k], nodes[0][k:]
        covered = [n for n in nodes if not (isinstance(n, str) and n == "")]
        span = make(covered)
        replacement = [x for x in (head, span, tail) if not (isinstance(x, str) and x == "") and x is not None]
        container[i0:i1 + 1] = replacement
        return span


def normalize_inlines(inlines: list[Any]) -> list[Any]:
    """Merge adjacent ``str`` runs and drop empty ones, recursively. Returns the same list."""
    out: list[Any] = []
    for node in inlines:
        if isinstance(node, str):
            if not node:
                continue
            if out and isinstance(out[-1], str):
                out[-1] += node
                continue
        elif isinstance(node, (Quote, Matn)):
            normalize_inlines(node.c)
        out.append(node)
    inlines[:] = out
    return inlines


# --------------------------------------------------------------------------- io


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def dump_model(model: BaseModel) -> str:
    return model.model_dump_json(indent=1) + "\n"


def save_model(model: BaseModel, path: Path | str) -> None:
    """Write a model as JSON, atomically (a crash never leaves half a file)."""
    _atomic_write(Path(path), dump_model(model))


def load_page(path: Path | str) -> Page:
    return Page.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_book(path: Path | str) -> Book:
    return Book.model_validate_json(Path(path).read_text(encoding="utf-8"))


def page_from_dict(data: dict[str, Any]) -> Page:
    return Page.model_validate(data)


def to_jsonable(model: BaseModel) -> dict[str, Any]:
    return json.loads(model.model_dump_json())
