"""Tests for the shared core: model, arabic, config, workdir, parse."""

from __future__ import annotations

import pytest

from conftest import fixture_page, fixture_sequence, fixture_tags
from pdf2epub.arabic import (
    ORNATE_CLOSE,
    ORNATE_OPEN,
    clean_text,
    detect_digit_style,
    format_number,
    has_uthmani_features,
    normalize_for_match,
    normalize_for_score,
    normalize_marker,
    normalize_with_map,
    parse_number,
    strip_diacritics,
    to_rasm,
)
from pdf2epub.config import BookConfig, apply_overrides, slugify
from pdf2epub.model import (
    BlockText,
    Matn,
    Note,
    Page,
    Paragraph,
    Poem,
    Quote,
    Quran,
    QuranBlock,
    Ref,
    Verse,
    assign_ids,
    block_text,
    load_page,
    next_ref_id,
    save_model,
)
from pdf2epub.parse import parse_tags, serialize_page, tags_parse_ok
from pdf2epub.workdir import WorkDir


# --------------------------------------------------------------------------- arabic


def test_normalize_for_match_unifies_spelling():
    assert normalize_for_match("ٱلْحَمْدُ لِلَّهِ") == "الحمد لله"
    assert normalize_for_match("إِلَى الْقُرَى") == "الي القري"
    assert normalize_for_match("الصَّلَاةِ") == normalize_for_match("الصلاه")
    assert normalize_for_match("ٱلْعَٰلَمِينَ", dagger_alef="alef") == normalize_for_match("الْعَالَمِينَ")
    assert normalize_for_match("الصبـــر") == "الصبر"
    assert normalize_for_match("قال: «الصبر»، (ضياء)!") == "قال الصبر ضياء"


def test_normalize_with_map_points_back_to_source():
    src = "قال: «والصبرُ ضياءٌ»"
    norm, idx = normalize_with_map(src)
    assert norm == "قال والصبر ضياء"
    assert len(idx) == len(norm) + 1 and idx[-1] == len(src)
    start = norm.index("والصبر")
    end = start + len("والصبر")
    # The source span covering the normalised word starts at its first letter.
    assert src[idx[start]] == "و"
    assert src[idx[end - 1]] == "ر"


def test_rasm_merges_dotted_letters():
    assert to_rasm("بيت") == to_rasm("نبت")
    assert to_rasm("فق") == to_rasm("قف")


def test_digits_and_markers():
    assert parse_number("(١٢)") == 12
    assert parse_number("[3]") == 3
    assert parse_number("۱۵") == 15
    assert parse_number("*") is None
    assert format_number(155, "arabic-indic") == "١٥٥"
    assert detect_digit_style("ص ٣٧ و١٢") == "arabic-indic"
    assert detect_digit_style("no digits") is None
    assert normalize_marker("(١)") == normalize_marker("1") == "1"
    assert normalize_marker("**") == "**"


def test_clean_text_keeps_letters_and_honorifics():
    assert clean_text("‏ﻻ إله إلا الله ﷺ") == "لا إله إلا الله ﷺ"
    assert clean_text(f"{ORNATE_OPEN}x{ORNATE_CLOSE}") == f"{ORNATE_OPEN}x{ORNATE_CLOSE}"
    assert strip_diacritics("الصَّبْرُ") == "الصبر"
    assert has_uthmani_features("ٱلْعَٰلَمِينَ") and not has_uthmani_features("الْعَالَمِينَ")


def test_normalize_for_score():
    assert normalize_for_score("الصَّـبْرُ   ضِياء", keep_diacritics=False) == "الصبر ضياء"
    assert normalize_for_score("الصَّـبْرُ   ضِياء", keep_diacritics=True) == "الصَّبْرُ ضِياء"


# --------------------------------------------------------------------------- model


def test_page_json_roundtrip(tmp_path):
    page = fixture_page("basic", index=37)
    path = tmp_path / "p.json"
    save_model(page, path)
    assert load_page(path) == page


def test_assign_ids_and_link_remap():
    page = Page(index=5, blocks=[Paragraph(c=["نص", Ref(id="old-r", n="1", note="old-n")])])
    note = Note(id="old-n", marker="1", anchor_type="marker", blocks=[Paragraph(c=["حاشية"])])
    note.link.ref = "old-r"
    from pdf2epub.model import HashiyaRegion

    page.hashiya.append(HashiyaRegion(notes=[note]))
    assign_ids(page)
    ref = page.blocks[0].c[1]
    assert ref.id == "p0005-r1" and ref.note == "p0005-n1"
    assert page.hashiya[0].notes[0].link.ref == "p0005-r1"
    assert next_ref_id(page) == "p0005-r2"


def test_blocktext_insert_and_wrap():
    para = Paragraph(c=["قال تعالى: ", Quran(text="إنما يوفى"), " وقال: ", Quote(c=["«والصبر ضياء»"])])
    bt = BlockText(para)
    assert bt.text == "قال تعالى: إنما يوفى وقال: «والصبر ضياء»"
    end = bt.text.index("والصبر") + len("والصبر")
    assert bt.insert_at(end, Ref(id="g", generated=True))
    assert isinstance(para.c[3], Quote) and isinstance(para.c[3].c[1], Ref)
    # Inside an atomic Quran span, the node goes after the span.
    bt = BlockText(para)
    assert bt.insert_at(bt.text.index("يوفى"), Ref(id="g2", generated=True))
    assert isinstance(para.c[1], Quran) and para.c[2].id == "g2"
    bt = BlockText(para)
    s = bt.text.index("وقال")
    span = bt.wrap_range(s, s + 4, lambda nodes: Matn(c=nodes))
    assert isinstance(span, Matn) and span.c == ["وقال"]
    assert block_text(para) == "قال تعالى: إنما يوفى وقال: «والصبر ضياء»"


def test_blocktext_poem_offsets():
    poem = Poem(verses=[Verse(sadr=["أ ب"], ajuz=["ج د"]), Verse(sadr=["هـ"], ajuz=None)])
    bt = BlockText(poem)
    assert bt.text == "أ ب\tج د\nهـ"
    assert bt.insert_at(len("أ ب\tج د"), Ref(id="x", generated=True))
    assert poem.verses[0].ajuz[-1].id == "x"


def test_wrap_range_refuses_crossing_containers():
    poem = Poem(verses=[Verse(sadr=["أ ب"], ajuz=["ج د"])])
    bt = BlockText(poem)
    assert bt.wrap_range(1, 6, lambda n: Matn(c=n)) is None


# --------------------------------------------------------------------------- parse


def test_parse_basic_fixture():
    page = fixture_page("basic", index=37)
    kinds = [b.type for b in page.blocks]
    assert kinds == ["heading", "paragraph", "quran", "paragraph", "poem"]
    assert page.running_header == "كتاب الصبر"
    assert page.printed_page == "٣٧" and page.printed_page_num == 37
    quran = page.blocks[2]
    assert isinstance(quran, QuranBlock) and isinstance(quran.c[0], Quran)
    assert quran.c[0].text.startswith("إِنَّمَا")
    para = page.blocks[3]
    assert isinstance(para.c[1], Quote) and para.c[1].style == "«»"
    assert isinstance(para.c[2], Ref) and para.c[2].n == "١"
    poem = page.blocks[4]
    assert len(poem.verses) == 2 and poem.verses[0].ajuz
    notes = page.notes()
    assert [n.anchor_type for n in notes] == ["marker", "lemma"]
    assert notes[1].lemma == "والصبر ضياء"
    assert not page.flags


def test_parse_commentary_and_inline_quran():
    page = fixture_page("commentary", index=5)
    para = page.blocks[0]
    assert isinstance(para.c[0], Matn) and para.c[0].c == ["(الحمد لله)"]
    qurans = [n for n in para.c if isinstance(n, Quran)]
    assert qurans and qurans[0].text == "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"  # brackets stripped
    quote = [n for n in page.blocks[1].c if isinstance(n, Quote)][0]
    assert quote.style == "(())"


def test_parse_mixed_bold_as_matn():
    page = parse_tags("<p><b>(قال المصنف)</b> أي</p>", Page(index=1), commentary="mixed").page
    assert isinstance(page.blocks[0].c[0], Matn)
    page = parse_tags("<p><b>(قال المصنف)</b> أي</p>", Page(index=1)).page
    assert page.blocks[0].c == ["(قال المصنف) أي"]


def test_parse_messy_is_recovered_and_flagged():
    raw = fixture_tags("messy")
    assert not tags_parse_ok(raw)
    result = parse_tags(raw, Page(index=2))
    assert result.ok and result.recovered
    codes = {f.code for f in result.page.flags}
    assert "parse_recovered" in codes
    assert result.page.blocks[0].type == "heading"


def test_parse_continuations():
    pages = fixture_sequence()
    assert pages[0].notes()[0].cont_next is True
    assert pages[1].blocks[0].continued is True
    first_note = pages[1].notes()[0]
    assert first_note.cont is True and first_note.anchor_type == "position"
    assert block_text(first_note.blocks[0]).startswith("وأقسامه")


def test_parse_untagged_outer_text_and_top_level_ref():
    page = parse_tags("قال الشاعر\n<quran>الم</quran><ref n=\"2\"/>\n<poem><s>صدر</s><a>عجز</a></poem>",
                      Page(index=3)).page
    assert page.blocks[0].type == "paragraph"
    assert page.blocks[1].type == "quran" and isinstance(page.blocks[1].c[-1], Ref)
    assert page.blocks[2].verses[0].ajuz == ["عجز"]


def test_parse_multiple_verses_in_one_quran_tag():
    page = parse_tags("<quran>﴿الم﴾ ﴿ذلك الكتاب﴾</quran>", Page(index=1)).page
    spans = [n for n in page.blocks[0].c if isinstance(n, Quran)]
    assert [s.text for s in spans] == ["الم", "ذلك الكتاب"]


def test_parse_entities_fences_and_think():
    raw = "<think>reasoning</think>\n```xml\n<p>أ &amp; ب &nbsp;ج &laquo;د&raquo; & هـ</p>\n```"
    page = parse_tags(raw, Page(index=1)).page
    assert block_text(page.blocks[0]) == "أ & ب ج «د» & هـ"


def test_parse_garbage_falls_back_to_text():
    result = parse_tags("<<<< not tags at all >>>>\n\nسطر عربي", Page(index=1))
    assert result.page.blocks  # text kept for review
    assert block_text(result.page.blocks[-1]).endswith("سطر عربي")


def test_parse_empty():
    result = parse_tags("   ", Page(index=1))
    assert result.ok and not result.page.blocks


@pytest.mark.parametrize("name", ["basic", "commentary", "seq_0001", "seq_0002", "seq_0003", "margin"])
@pytest.mark.parametrize("full", [False, True])
def test_serialize_roundtrip(name, full):
    page = fixture_page(name, index=9)
    again = parse_tags(serialize_page(page, full=full), Page(index=9)).page
    exclude = {"flags"}
    if not full:
        exclude |= {"hashiya"}  # geometry and kinds are only kept in the full form
        assert [n.model_dump(exclude={"region"}) for n in again.notes()] == \
            [n.model_dump(exclude={"region"}) for n in page.notes()]
    assert again.model_dump(exclude=exclude) == page.model_dump(exclude=exclude)


def test_full_serialization_keeps_generated_links():
    page = fixture_page("basic", index=4)
    note = page.notes()[1]
    para = page.blocks[3]
    bt = BlockText(para)
    end = bt.text.index("ضياء") + len("ضياء")
    ref = Ref(id="p0004-r9", generated=True, note=note.id)
    bt.insert_at(end, ref)
    note.link.status, note.link.method, note.link.ref = "linked", "manual", ref.id
    again = parse_tags(serialize_page(page, full=True), Page(index=4)).page
    note2 = again.notes()[1]
    assert note2.id == note.id and note2.link.ref == "p0004-r9" and note2.link.method == "manual"
    assert "generated" not in serialize_page(page)


def test_parse_repairs():
    cases = {
        '<p>سطر أول<br>سطر ثان</p><p>فقرة</p>': ["paragraph", "paragraph"],
        '<p>أ ب ج</p></quran><p>د هـ و</p>': ["paragraph", "paragraph"],
        '<p>فقرة أولى\n<p>فقرة ثانية</p>': ["paragraph", "paragraph"],
        '<poem><v><s>صدر</s><a>عجز</a></v>\n<p>نثر بعد القصيدة</p>': ["poem", "paragraph"],
        '<p>قال الشاعر: <poem><v><s>صدر</s><a>عجز</a></v></poem> ثم قال</p>': ["paragraph", "poem", "paragraph"],
    }
    for raw, kinds in cases.items():
        result = parse_tags(raw, Page(index=1))
        assert [b.type for b in result.page.blocks] == kinds, raw
        assert "parse_recovered" in {f.code for f in result.page.flags}
        assert result.text_loss < 0.02


def test_quran_brackets_and_parity():
    page = parse_tags('<p>قال: <quran>﴿إِنَّا أَعْطَيْنَاكَ الْكَوْثَرَ﴾ إلى قوله ﴿إِنَّ شَانِئَكَ هُوَ الْأَبْتَرُ﴾</quran></p>',
                      Page(index=1)).page
    kinds = [type(n).__name__ for n in page.blocks[0].c]
    assert kinds == ["str", "Quran", "str", "Quran"] and page.blocks[0].c[2] == " إلى قوله "
    page = parse_tags('<p>قال تعالى:</p>\n﴿<quran>الْحَمْدُ لِلَّهِ</quran>﴾\n<p>بعده</p>', Page(index=1)).page
    assert [b.type for b in page.blocks] == ["paragraph", "quran", "paragraph"]
    page = parse_tags('<p>قال ﴿<quran>الم</quran>﴾ ثم</p>', Page(index=1)).page
    assert block_text(page.blocks[0]) == "قال الم ثم"
    page = parse_tags('<quran>﴿الحمد لله رب العالمين﴾ (٢)</quran>', Page(index=1)).page
    assert page.blocks[0].c[0].text == "الحمد لله رب العالمين" and page.blocks[0].c[1].strip() == "(٢)"
    page = parse_tags('<quran>وبشر الصابرين [البقرة: ١٥٥]</quran>', Page(index=1)).page
    assert page.blocks[0].c[0].text == "وبشر الصابرين"
    page = parse_tags('<quran>وبشر الصابرين<ref n="1"/></quran>', Page(index=1)).page
    assert isinstance(page.blocks[0].c[-1], Ref)


def test_inline_note_moves_to_foot():
    page = parse_tags('<p>وهذا كلام حسن<footnote n="2">انظر الديوان.</footnote> وجميل</p>', Page(index=1)).page
    assert block_text(page.blocks[0]) == "وهذا كلام حسن وجميل"
    assert page.notes()[0].marker == "2"
    assert "note_inline" in {f.code for f in page.flags}


def test_span_edge_spaces_and_poem_repair():
    page = parse_tags('<p>وقال<quote> «الصبر ضياء» </quote>ثم <matn>قال المصنف </matn>رحمه الله</p>', Page(index=1)).page
    assert block_text(page.blocks[0]) == "وقال «الصبر ضياء» ثم قال المصنف رحمه الله"
    page = parse_tags('<poem><v><s>صدر البيت</s><s>عجز البيت</s></v></poem>', Page(index=1)).page
    assert page.blocks[0].verses[0].ajuz == ["عجز البيت"] and not page.blocks[0].single
    page = parse_tags('<poem><v><s>صدر</s></v><v><s>صدر ثان</s></v></poem>', Page(index=1)).page
    assert not page.blocks[0].single  # only type="single" makes a poem single-column


def test_loose_notes_and_markers():
    page = parse_tags('<hashiya pos="bottom"><p>(١) رواه مسلم.</p><p>(٢) رواه البخاري\nفي صحيحه.</p></hashiya>',
                      Page(index=1)).page
    notes = page.notes()
    assert [n.marker for n in notes] == ["١", "٢"]
    assert block_text(notes[1].blocks[0]) == "رواه البخاري في صحيحه."
    page = parse_tags('<hashiya><note n="١">(١) رواه مسلم.</note></hashiya>', Page(index=1)).page
    assert block_text(page.notes()[0].blocks[0]) == "رواه مسلم."
    page = parse_tags('<hashiya>=<note>وأقسامه.</note></hashiya>', Page(index=1)).page
    assert len(page.notes()) == 1 and page.notes()[0].cont


def test_untagged_output_is_flagged():
    from pdf2epub.parse import tags_parse_ok

    raw = "هذا نص بلا وسوم\nوسطر آخر ٣٧"
    assert "untagged_text" in {f.code for f in parse_tags(raw, Page(index=1)).page.flags}
    assert not tags_parse_ok(raw)
    assert tags_parse_ok(fixture_tags("basic"))


def test_ids_are_stable_across_reassignment():
    page = fixture_page("basic", index=3)
    ids = [b.id for b in page.blocks]
    page.blocks.insert(0, Paragraph(c=["جديد"]))
    assign_ids(page)
    assert [b.id for b in page.blocks[1:]] == ids and page.blocks[0].id == "p0003-b6"


def test_serialize_omits_generated_refs():
    page = Page(index=1, blocks=[Paragraph(c=["أ", Ref(n=None, generated=True), " ب"])])
    assert "<ref" not in serialize_page(page)


# --------------------------------------------------------------------------- config, workdir


def test_config_overrides_and_slug():
    cfg = BookConfig(slug="x")
    cfg2 = apply_overrides(cfg, {"build.notes": "inline", "title": "كتاب", "ocr.endpoint": None})
    assert cfg2.build.notes == "inline" and cfg2.title == "كتاب" and cfg.build.notes == "popup"
    with pytest.raises(KeyError):
        apply_overrides(cfg, {"build.nope": 1})
    assert slugify("/tmp/Sharh Ibn Aqil.pdf") == "sharh-ibn-aqil"
    assert slugify("شرح.pdf") == "شرح"


def test_workdir_paths_and_best_page(tmp_path):
    wd = WorkDir(tmp_path / "b").ensure()
    assert wd.page_image(37).name == "0037.png"
    page = fixture_page("basic", index=37)
    wd.save_page(page, "checked")
    assert wd.best_page(37) == page
    reviewed = page.model_copy(update={"status": "reviewed"})
    wd.save_page(reviewed, "reviewed")
    assert wd.best_page(37).status == "reviewed"
    with pytest.raises(ValueError):
        wd.abs("../../etc/passwd")
    assert wd.rel(wd.page_image(1)) == "pages/0001.png"
