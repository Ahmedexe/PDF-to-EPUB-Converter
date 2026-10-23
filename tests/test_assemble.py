"""Joining pages into a book (spec §2 step 7)."""

from __future__ import annotations

from conftest import fixture_page, fixture_sequence
from pdf2epub.assemble import assemble_book, join_pages, page_label, split_chapters
from pdf2epub.checks import run_checks
from pdf2epub.config import BookConfig
from pdf2epub.model import PageBreak, Page, Poem, Ref, block_containers, block_text, walk_inlines
from pdf2epub.parse import parse_tags


def cfg(**build) -> BookConfig:
    c = BookConfig(slug="a")
    for k, v in build.items():
        setattr(c.build, k, v)
    return c


def checked_sequence():
    pages = fixture_sequence()
    run_checks(pages, None, cfg(), tesseract=False)
    return pages


def breaks(blocks):
    out = []
    for b in blocks:
        for c in block_containers(b):
            out.extend(n.label for n in walk_inlines(c) if isinstance(n, PageBreak))
    return out


def test_paragraph_and_poem_joined_with_page_breaks():
    pages = checked_sequence()
    main, frame = join_pages(pages, cfg())
    kinds = [b.type for b, _ in main]
    assert kinds == ["heading", "heading", "paragraph", "poem", "heading", "paragraph"]
    para = main[2][0]
    assert "الدين موروثة" in block_text(para)
    assert breaks([b for b, _ in main]) == ["١٢", "١٣", "١٤"]
    poem = main[3][0]
    assert len(poem.verses) == 3
    assert not frame
    # The pages themselves are untouched.
    assert not any(isinstance(n, PageBreak) for n in pages[0].blocks[2].c)


def test_heading_at_top_prevents_join_and_terminal_punctuation():
    p1 = parse_tags("<p>نهاية الفقرة.</p>", Page(index=1)).page
    p2 = parse_tags("<p>فقرة جديدة</p>", Page(index=2)).page
    main, _ = join_pages([p1, p2], cfg())
    assert len(main) == 2
    p1 = parse_tags("<p>فقرة بلا نهاية</p>", Page(index=1)).page
    p2 = parse_tags("<h2>باب</h2><p>فقرة</p>", Page(index=2)).page
    main, _ = join_pages([p1, p2], cfg())
    assert [b.type for b, _ in main] == ["paragraph", "heading", "paragraph"]
    # A gap in the page list means no join.
    p3 = parse_tags("<p>تتمة</p>", Page(index=4)).page
    main, _ = join_pages([p1, p3], cfg())
    assert len(main) == 2


def test_verse_split_at_page_break_is_rejoined():
    p1 = parse_tags("<poem><v><s>على قدر أهل العزم تأتي العزائم</s></v></poem>", Page(index=1)).page
    p2 = parse_tags("<poem><v><a>وتأتي على قدر الكرام المكارم</a></v></poem>", Page(index=2)).page
    main, _ = join_pages([p1, p2], cfg())
    poem = main[0][0]
    assert isinstance(poem, Poem) and len(poem.verses) == 1
    assert "المكارم" in block_text(poem) and breaks([poem]) == ["١", "٢"]


def test_page_labels_inferred():
    pages = [Page(index=1, printed_page="٩", printed_page_num=9), Page(index=2), Page(index=3)]
    assert page_label(pages[1], pages) == "١٠" and page_label(pages[2], pages) == "١١"
    assert page_label(Page(index=7), [Page(index=7)]) == "٧"


def test_chapters_split_and_titles():
    raw = ["<h2>باب الأول</h2><p>أ</p>", "<p>ب</p><h2>باب الثاني</h2><p>ج</p>", "<h3>فصل</h3><p>د</p>"]
    pages = [parse_tags(r, Page(index=i)).page for i, r in enumerate(raw, 1)]
    chapters = split_chapters(join_pages(pages, cfg())[0], cfg(), book_title="ك")
    assert [c.title for c in chapters] == ["باب الأول", "باب الثاني"]
    assert chapters[1].pages == [2, 3]
    untitled = parse_tags("<p>مقدمة بلا عنوان</p>", Page(index=1)).page
    assert split_chapters(join_pages([untitled], cfg())[0], cfg(), book_title="ك")[0].title == "ك"
    long = [parse_tags(f"<p>صفحة {i}.</p>", Page(index=i)).page for i in range(1, 8)]
    chs = split_chapters(join_pages(long, cfg())[0], cfg(max_chapter_pages=3), book_title="ك")
    assert len(chs) == 3 and chs[1].title == "ك (تابع)"


def test_book_notes_per_chapter_and_continuations_merged():
    pages = checked_sequence()
    book = assemble_book(pages, cfg(), slug="a", title="كتاب")
    notes = [n for ch in book.chapters for n in ch.notes]
    ids = [n.id for n in notes]
    assert "p0002-n1" not in ids  # the continuation part is merged into p0001-n1
    first = next(n for n in notes if n.id == "p0001-n1")
    assert "نصف الإيمان" in block_text(first.blocks[0])
    assert book.page_labels == {1: "١٢", 2: "١٣", 3: "١٤"}
    assert book.identifier.startswith("urn:uuid:") and book.digits == "arabic-indic"
    # Every anchor in the text points at a note of the same chapter.
    for ch in book.chapters:
        ch_ids = {n.id for n in ch.notes}
        for b in ch.blocks:
            for c in block_containers(b):
                for node in walk_inlines(c):
                    if isinstance(node, Ref) and node.note:
                        assert node.note in ch_ids


def test_unlinked_notes_are_kept():
    page = parse_tags('<p>نص</p><hashiya><note n="9">حاشية بلا علامة</note></hashiya>', Page(index=1)).page
    run_checks([page], None, cfg(), tesseract=False)
    book = assemble_book([page], cfg(), slug="a", title="ك")
    assert [n.id for n in book.chapters[0].notes] == [page.notes()[0].id]


def test_independent_frame_stream():
    page = fixture_page("margin", 1)
    book = assemble_book([page], cfg(), slug="a", title="ك")
    streams = [c.stream for c in book.chapters]
    assert streams == ["main", "frame"]
    assert block_text(book.chapters[1].blocks[0]) == "متن الكتاب"
