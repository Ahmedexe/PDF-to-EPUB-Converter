"""Join pages into a book (spec §2 step 7, §5.2 page breaks).

* Every page contributes a ``PageBreak`` at the start of its first text block,
  inside a joined paragraph or poem when the page starts mid-way; the EPUB page
  list is built from them, so citations to the paper edition still work.
* A paragraph cut by a page break is joined, as is a poem continued on the
  next page (a verse split at the break is re-joined).
* Chapters start at headings; long runs without headings are split so e-ink
  readers open them fast.
"""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Any

from ..arabic import detect_digit_style, format_number
from ..config import BookConfig
from ..model import (
    Book,
    Chapter,
    Figure,
    Heading,
    PageBreak,
    Page,
    Paragraph,
    Poem,
    Quran,
    QuranBlock,
    block_containers,
    block_text,
    inline_text,
    iter_blocks,
    normalize_inlines,
    walk_inlines,
)
from .notes import attach_orphans, chapter_notes, join_continued_notes

TERMINAL = (".", "!", "؟", "?", ":", "…", "»", "\"", ")", "]", "﴾", "۔", "*")

Stream = list[tuple[Any, int]]


def page_label(page: Page, pages: list[Page]) -> str:
    """The printed page number, or one inferred from the nearest numbered neighbour."""
    if page.printed_page:
        return page.printed_page
    ordered = sorted(pages, key=lambda p: p.index)
    style = detect_digit_style(" ".join(p.printed_page or "" for p in ordered)) or "arabic-indic"
    best = None
    for other in ordered:
        if other.printed_page_num is None:
            continue
        dist = abs(other.index - page.index)
        if best is None or dist < best[0]:
            best = (dist, other)
    if best is not None:
        value = best[1].printed_page_num + (page.index - best[1].index)
        if value > 0:
            return format_number(value, style)
    return format_number(page.index, style)


def _first_container(block: Any) -> list[Any] | None:
    if isinstance(block, (Heading, Paragraph, QuranBlock)):
        return block.c
    if isinstance(block, Poem) and block.verses:
        return block.verses[0].sadr
    if isinstance(block, Figure):
        return block.caption
    return None


def _ends_open(para: Paragraph) -> bool:
    text = inline_text(para.c).rstrip()
    return bool(text) and not text.endswith(TERMINAL)


def join_pages(pages: list[Page], cfg: BookConfig) -> tuple[Stream, Stream]:
    """(main stream, independent-work stream) of ``(block, page index)``, pages joined.

    Blocks are copies; the pages are not modified.
    """
    pages = sorted(pages, key=lambda p: p.index)
    main: Stream = []
    frame: Stream = []
    pending: list[PageBreak] = []
    prev_index: int | None = None
    for page in pages:
        pb = PageBreak(page=page.index, label=page_label(page, pages))
        blocks = [b.model_copy(deep=True) for b in page.blocks]
        breaks = pending + [pb]
        pending = []
        placed = False
        for block in blocks:
            container = _first_container(block)
            if container is not None:
                container[0:0] = breaks
                placed = True
                break
        if not placed:
            pending = breaks
        adjacent = prev_index is not None and page.index == prev_index + 1
        for k, block in enumerate(blocks):
            if k == 0 and adjacent and main:
                last, _ = main[-1]
                if isinstance(last, Paragraph) and isinstance(block, Paragraph) and \
                        (last.continues or block.continued or _ends_open(last)):
                    sep = [] if inline_text(last.c).endswith(" ") else [" "]
                    last.c.extend(sep + block.c)
                    normalize_inlines(last.c)
                    last.continues = block.continues
                    continue
                if isinstance(last, Poem) and isinstance(block, Poem):
                    verses = block.verses
                    if last.verses and verses and last.verses[-1].ajuz is None and \
                            not inline_text(verses[0].sadr).strip() and verses[0].ajuz is not None:
                        lead = [n for n in verses[0].sadr if isinstance(n, PageBreak)]
                        last.verses[-1].ajuz = lead + verses[0].ajuz
                        verses = verses[1:]
                    last.verses.extend(verses)
                    last.single = last.single and block.single
                    continue
            main.append((block, page.index))
        for region in page.hashiya:
            if region.independent:
                for block in region.blocks:
                    frame.append((block.model_copy(deep=True), page.index))
        prev_index = page.index
    if pending and main:
        container = _first_container(main[-1][0])
        if container is not None:
            container.extend(pending)
    return main, frame


def _title(block: Heading) -> str:
    return " ".join(inline_text(block.c).split())


def split_chapters(stream: Stream, cfg: BookConfig, *, book_title: str = "", stream_name: str = "main",
                   start: int = 1) -> list[Chapter]:
    levels = [b.level for b, _ in stream if isinstance(b, Heading)]
    split_level = cfg.build.split_level
    if split_level is None and levels:
        # The top level that occurs more than once: a lone h1 is usually the book's title.
        repeated = sorted(lvl for lvl in set(levels) if levels.count(lvl) > 1)
        split_level = repeated[0] if repeated else min(levels)
    chapters: list[Chapter] = []
    current: Chapter | None = None

    def new(title: str, level: int) -> Chapter:
        ch = Chapter(id=f"ch{start + len(chapters):03d}", stream=stream_name, level=level, title=title)
        chapters.append(ch)
        return ch

    for block, page in stream:
        is_split = isinstance(block, Heading) and split_level is not None and block.level <= split_level
        too_long = current is not None and current.pages and page not in current.pages and \
            len(current.pages) >= cfg.build.max_chapter_pages
        if current is None or is_split or too_long:
            if is_split:
                title, level = _title(block) or book_title, block.level
            else:
                prev = chapters[-1].title if chapters else ""
                title = f"{prev} (تابع)" if prev else (book_title or "الكتاب")
                level = chapters[-1].level if chapters else 1
            current = new(title or book_title or "الكتاب", level)
        current.blocks.append(block)
        if page not in current.pages:
            current.pages.append(page)
    return chapters


def _digits(pages: list[Page], cfg: BookConfig) -> str:
    if cfg.text.digits != "auto":
        return cfg.text.digits
    sample = " ".join(p.printed_page or "" for p in pages)
    sample += " ".join(block_text(b) for p in pages[:50] for b, _o in iter_blocks(p))
    return detect_digit_style(sample) or "arabic-indic"


def _quran_variant(pages: list[Page]) -> str | None:
    counts: Counter[str] = Counter()
    for page in pages:
        for block, _o in iter_blocks(page):
            for container in block_containers(block):
                for node in walk_inlines(container):
                    if isinstance(node, Quran) and node.variant:
                        counts[node.variant] += 1
    return counts.most_common(1)[0][0] if counts else None


def assemble_book(pages: list[Page], cfg: BookConfig, *, slug: str, title: str, author: str | None = None,
                  identifier: str | None = None) -> Book:
    """Join pages into a :class:`Book` with chapters and their notes."""
    pages = sorted(pages, key=lambda p: p.index)
    main, frame = join_pages(pages, cfg)
    chapters = split_chapters(main, cfg, book_title=title)
    frame_chapters = split_chapters(frame, cfg, book_title=f"{title} — بهامشه", stream_name="frame",
                                    start=len(chapters) + 1) if frame else []
    notes, home, _alias = join_continued_notes(pages)
    claimed: set[str] = set()
    for ch in chapters + frame_chapters:
        ch.notes = chapter_notes(ch, notes, claimed)
    attach_orphans(chapters + frame_chapters, notes, home, claimed)
    return Book(
        slug=slug, title=title, author=author,
        identifier=identifier or f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, 'pdf2epub:' + slug)}",
        digits=_digits(pages, cfg),  # type: ignore[arg-type]
        quran_variant=_quran_variant(pages),
        chapters=chapters + frame_chapters,
        page_labels={p.index: page_label(p, pages) for p in pages},
    )
