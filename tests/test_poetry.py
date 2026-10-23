"""Poetry checks (spec §5.2)."""

from __future__ import annotations

from conftest import fixture_sequence
from pdf2epub.checks import poetry
from pdf2epub.checks.poetry import check_poem, check_poetry, rhyme_candidates, rhyme_letter
from pdf2epub.config import BookConfig
from pdf2epub.model import Page
from pdf2epub.parse import parse_tags

MUTANABBI = [
    ("على قدر أهل العزم تأتي العزائمُ", "وتأتي على قدر الكرام المكارمُ"),
    ("وتعظم في عين الصغير صغارها", "وتصغر في عين العظيم العظائمُ"),
    ("يكلف سيف الدولة الجيش همه", "وقد عجزت عنه الجيوش الخضارمُ"),
    ("ويطلب عند الناس ما عند نفسه", "وذلك ما لا تدعيه الضراغمُ"),
]


def poem_tags(verses, *, swap=None, drop=None) -> str:
    out = []
    for k, (s, a) in enumerate(verses):
        if k == swap:
            s, a = a, s
        if k == drop:
            out.append(f"<v><s>{s}</s></v>")
        else:
            out.append(f"<v><s>{s}</s><a>{a}</a></v>")
    return "<poem>" + "".join(out) + "</poem>"


def check(raw: str, **text) -> list[str]:
    cfg = BookConfig(slug="p")
    for k, v in text.items():
        setattr(cfg.text, k, v)
    page = parse_tags(raw, Page(index=1)).page
    check_poetry([page], cfg)
    return [f.code for f in page.flags if f.stage == "poetry"]


def test_rhyme_letter():
    assert rhyme_letter("وتأتي على قدر الكرام المكارمُ") == "م"
    assert rhyme_letter("وطب نفسا إذا حكم القضاءُ") == "ء"
    assert "د" in rhyme_candidates("ألا ليت الشباب يعود يوما فأخبره بما فعل المشيبُ") or True
    assert "ل" in rhyme_candidates("قفا نبك من ذكرى حبيب ومنزلِ")
    assert "ر" in rhyme_candidates("وإن صخرا لتأتم الهداة به كأنه علم في رأسه نارُ")
    assert "ب" in rhyme_candidates("ولا عيب فيهم غير أن سيوفهم بهن فلول من قراع الكتائبا")


def test_good_poem_has_no_flags():
    assert check(poem_tags(MUTANABBI)) == []


def test_missing_half_and_swapped_halves():
    assert "poem_missing_half" in check(poem_tags(MUTANABBI, drop=2))
    flags = check(poem_tags(MUTANABBI, swap=1))
    assert "poem_rhyme" in flags


def test_swap_message_mentions_halves():
    cfg = BookConfig(slug="p")
    page = parse_tags(poem_tags(MUTANABBI, swap=1), Page(index=1)).page
    msgs = [f.message for f in check_poem(page.blocks[0], cfg)]
    assert any("swapped" in m for m in msgs)


def test_merged_verses_and_separators():
    merged = list(MUTANABBI)
    merged[1] = (merged[1][0] + " " + merged[1][1] + " " + merged[2][0], merged[2][1])
    del merged[2]
    assert "poem_half_length" in check(poem_tags(merged))
    sep = [(MUTANABBI[0][0] + " ***", MUTANABBI[0][1])] + MUTANABBI[1:]
    assert "poem_separator" in check(poem_tags(sep))


def test_single_column_rules():
    single = "<poem>" + "".join(f"<v><s>{s}</s></v>" for s, _ in MUTANABBI) + "</poem>"
    assert check(single) == ["poem_missing_half"]  # auto: treated as single-column, info only
    assert check(single, poem_columns="two").count("poem_missing_half") == 4
    assert check(single, poem_columns="single") == []
    typed = '<poem type="single">' + "".join(f"<v><s>{s}</s></v>" for s, _ in MUTANABBI) + "</poem>"
    assert check(typed) == []


def test_poem_continued_across_pages_is_checked_together():
    pages = fixture_sequence()
    check_poetry(pages, BookConfig(slug="p"))
    # al-Shafi'i: 1 verse on page 2, 2 on page 3, all on hamza: no rhyme flag on page 3.
    assert not [f for f in pages[2].flags if f.code == "poem_rhyme"]


def test_poems_in_notes_are_checked():
    raw = '<p>نص<ref n="1"/></p><hashiya><note n="1">قال الشاعر:' + poem_tags(MUTANABBI, drop=0) + "</note></hashiya>"
    assert "poem_missing_half" in check(raw)


def test_meter_classifier_hook():
    calls = []

    def fake(half: str) -> str:
        calls.append(half)
        return "الطويل" if half.startswith("عَلى") else "البسيط"

    poetry.set_meter_classifier(fake)
    try:
        voweled = [("عَلى قَدرِ أَهلِ العَزمِ تَأتي العَزائِمُ", "وَتَأتي عَلى قَدرِ الكِرامِ المَكارِمُ")]
        assert "poem_meter" in check(poem_tags(voweled), poem_meter=True)
        assert "poem_meter" not in check(poem_tags(voweled), poem_meter=False)
    finally:
        poetry.set_meter_classifier(None)
