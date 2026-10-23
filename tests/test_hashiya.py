"""Note linking (spec §6.4-6.5)."""

from __future__ import annotations

from conftest import fixture_page, fixture_sequence
from pdf2epub.checks.hashiya import extract_lemma, link_notes
from pdf2epub.config import BookConfig
from pdf2epub.model import BlockText, NoteLink, Page, Ref, block_text
from pdf2epub.parse import parse_tags, serialize_page


def cfg(**text) -> BookConfig:
    c = BookConfig(slug="h")
    for k, v in text.items():
        setattr(c.text, k, v)
    return c


def anchor_word(page: Page, ref_id: str) -> str:
    """The word just before an anchor."""
    for block in page.blocks:
        bt = BlockText(block)
        if ref_id in bt.ref_offsets:
            return bt.text[: bt.ref_offsets[ref_id]].split()[-1]
    raise AssertionError(f"{ref_id} not found")


def codes(page: Page) -> set[str]:
    return {f.code for f in page.flags}


def test_marker_and_lemma_on_one_page():
    page = fixture_page("basic", 1)
    link_notes([page], cfg())
    n1, n2 = page.notes()
    assert (n1.link.status, n1.link.method) == ("linked", "marker")
    assert (n2.link.status, n2.link.method) == ("linked", "lemma")
    assert anchor_word(page, n2.link.ref) == "ضياء"
    assert not [f for f in page.flags if f.severity != "info"]


def test_lookback_and_continuation_across_pages():
    pages = fixture_sequence()
    link_notes(pages, cfg())
    p1, p2, p3 = pages
    cont, imama, shafii = p2.notes()
    assert cont.continues_from == p1.notes()[0].id and p1.notes()[0].continues_to == cont.id
    assert cont.link.method == "continuation"
    # "الإمامة في الدين" is printed on page 1: the anchor goes there.
    assert imama.link.page == 1 and anchor_word(p1, imama.link.ref) == "الدين"
    assert shafii.link.page == 2 and anchor_word(p2, shafii.link.ref) == "الشافعي"
    assert all(not [f for f in p.flags if f.severity != "info"] for p in pages)


def test_relinking_is_idempotent():
    pages = fixture_sequence()
    link_notes(pages, cfg())
    first = [serialize_page(p, full=True) for p in pages]
    link_notes(pages, cfg())
    assert [serialize_page(p, full=True) for p in pages] == first


def test_repeated_lemma_follows_note_order():
    raw = ('<p>قال الصبر ضياء، والصبر نور، والصبر ضياء للمؤمن</p>'
           '<hashiya><note lemma="والصبر نور">أ</note><note lemma="والصبر ضياء">ب</note></hashiya>')
    page = parse_tags(raw, Page(index=1)).page
    link_notes([page], cfg())
    a, b = page.notes()
    off = {r: o for blk in page.blocks for r, o in BlockText(blk).ref_offsets.items()}
    # The second note must anchor on the occurrence *after* the first note's anchor.
    assert off[b.link.ref] > off[a.link.ref]


def test_qawluh_opening_becomes_lemma_and_is_removed():
    raw = '<p>قال المصنف: والصبر نور</p><hashiya><note>قوله: (والصبر نور) أي يهدي.</note></hashiya>'
    page = parse_tags(raw, Page(index=1)).page
    note = page.notes()[0]
    assert extract_lemma(note) == "والصبر نور"
    link_notes([page], cfg())
    assert note.lemma == "والصبر نور" and note.anchor_type == "lemma"
    assert block_text(note.blocks[0]) == "أي يهدي."
    # A lemma given as attribute and repeated in the text is not printed twice.
    raw = '<p>والصبر نور</p><hashiya><note lemma="والصبر نور">قوله (والصبر نور) إلخ: أي يهدي.</note></hashiya>'
    page = parse_tags(raw, Page(index=1)).page
    link_notes([page], cfg())
    assert block_text(page.notes()[0].blocks[0]) == "أي يهدي."


def test_unbracketed_lemma_with_etc():
    raw = '<p>ثم ذكر الصبر على البلاء</p><hashiya><note>قوله الصبر إلخ: أي المذكور.</note></hashiya>'
    page = parse_tags(raw, Page(index=1)).page
    link_notes([page], cfg())
    note = page.notes()[0]
    assert note.lemma == "الصبر" and note.link.status == "linked"


def test_fuzzy_lemma_and_not_found():
    raw = ('<p>والصبر ضياء للمؤمنين</p><hashiya><note lemma="والصبر ضيا">أ</note>'
           '<note lemma="كلمة غير موجودة أصلا">ب</note></hashiya>')
    page = parse_tags(raw, Page(index=1)).page
    link_notes([page], cfg())
    a, b = page.notes()
    assert a.link.status == "linked" and a.link.score < 1.0
    assert b.link.status == "flagged" and "lemma_not_found" in codes(page)


def test_plain_text_marker_is_converted():
    raw = '<p>نص فيه علامة (١) وأخرى</p><hashiya><note n="١">أ</note></hashiya>'
    page = parse_tags(raw, Page(index=1)).page
    link_notes([page], cfg())
    assert block_text(page.blocks[0]) == "نص فيه علامة وأخرى"
    assert page.notes()[0].link.status == "linked"


def test_unpaired_markers_and_numbering_gap():
    raw = ('<p>أ<ref n="1"/> ب<ref n="2"/> ج<ref n="9"/></p>'
           '<hashiya><note n="1">x</note><note n="2">y</note><note n="4">z</note></hashiya>')
    page = parse_tags(raw, Page(index=1)).page
    link_notes([page], cfg())
    assert {"ref_without_note", "note_without_ref", "note_numbering_gap"} <= codes(page)


def test_numbering_runs_on_from_previous_page():
    p1 = parse_tags('<p>أ<ref n="1"/> ب<ref n="2"/></p><hashiya><note n="1">x</note><note n="2">y</note></hashiya>',
                    Page(index=1)).page
    p2 = parse_tags('<p>ج<ref n="3"/></p><hashiya><note n="3">z</note></hashiya>', Page(index=2)).page
    link_notes([p1, p2], cfg())
    assert "note_numbering_gap" not in codes(p2)


def test_repeated_symbol_markers_pair_in_order():
    raw = '<p>أ<ref n="*"/> ب<ref n="*"/></p><hashiya><note n="*">x</note><note n="*">y</note></hashiya>'
    page = parse_tags(raw, Page(index=1)).page
    link_notes([page], cfg())
    x, y = page.notes()
    assert anchor_word(page, x.link.ref) == "أ" and anchor_word(page, y.link.ref) == "ب"


def test_editor_kind_from_marker_style():
    raw = '<p>أ<ref n="1"/> ب<ref n="أ"/></p><hashiya><note n="1">x</note><note n="أ">y</note></hashiya>'
    page = parse_tags(raw, Page(index=1)).page
    link_notes([page], cfg(editor_markers="letters"))
    assert [n.kind for n in page.notes()] == ["author", "editor"]


def test_margin_position_note_is_guessed():
    page = fixture_page("margin", 1, width=1000, height=1400)
    link_notes([page], cfg())
    margin_note = page.notes()[0]
    assert margin_note.link.status == "guessed" and margin_note.link.method == "position"
    assert "position_guess" in codes(page)


def test_layer_two_notes_link_into_layer_one():
    raw = ('<p>المتن<ref n="1"/></p>'
           '<hashiya pos="bottom"><note n="1">شرح فيه كلمة<ref n="أ"/></note></hashiya>'
           '<hashiya pos="frame" layer="2"><note n="أ">تقرير على الشرح</note></hashiya>')
    page = parse_tags(raw, Page(index=1)).page
    link_notes([page], cfg())
    sharh, taqrir = page.notes()
    assert sharh.link.status == "linked" and taqrir.link.status == "linked"
    assert taqrir.link.block == sharh.blocks[0].id


def test_manual_link_survives_relink_and_reparse():
    page = fixture_page("basic", 1)
    link_notes([page], cfg())
    lemma_note = page.notes()[1]
    # The reviewer moves the anchor to "وسلم" (second paragraph).
    para = page.blocks[3]
    for _b in page.blocks:
        for c in [_b.c] if hasattr(_b, "c") else []:
            c[:] = [n for n in c if not (isinstance(n, Ref) and n.generated)]
    bt = BlockText(para)
    end = bt.text.index("وسلم") + len("وسلم")
    ref = Ref(id="p0001-r9", generated=True, note=lemma_note.id)
    bt.insert_at(end, ref)
    lemma_note.link = NoteLink(status="linked", method="manual", page=1, block=para.id, ref=ref.id,
                               anchor="صلى الله عليه وسلم", occurrence=1)
    link_notes([page], cfg())
    assert anchor_word(page, lemma_note.link.ref) == "وسلم"
    # After a tags round trip without the generated ref, the anchor is re-placed from its words.
    again = parse_tags(serialize_page(page), Page(index=1)).page
    again.notes()[1].link = lemma_note.link.model_copy()
    link_notes([again], cfg())
    assert anchor_word(again, again.notes()[1].link.ref) == "وسلم"


def test_reviewed_pages_are_not_relinked():
    pages = fixture_sequence()
    link_notes(pages, cfg())
    pages[1].status = "reviewed"
    before = serialize_page(pages[1], full=True)
    flags = list(pages[1].flags)
    link_notes(pages, cfg())
    assert serialize_page(pages[1], full=True) == before and pages[1].flags == flags


def test_orphan_continuation():
    page = parse_tags('<p>نص</p><hashiya><note cont="prev">بقية حاشية</note></hashiya>', Page(index=5)).page
    link_notes([page], cfg())
    assert "continuation_orphan" in codes(page)


def test_gap_in_page_list_means_no_neighbour():
    p1 = fixture_page("seq_0001", 1)
    p3 = fixture_page("seq_0002", 3)  # page 2 missing
    link_notes([p1, p3], cfg())
    cont = p3.notes()[0]
    assert cont.continues_from is None
    lemma = p3.notes()[1]
    assert lemma.link.page != 1  # no lookback across the gap
