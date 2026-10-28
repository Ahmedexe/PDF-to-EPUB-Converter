"""Rules for building training targets from a digital edition (tools/build_training_pairs.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_training_pairs import (  # noqa: E402
    _edition_tokens, _is_verse, confirmed_pagenum, edition_main, edition_notes, renumber,
)

from pdf2epub.checks.flags import find_repetition  # noqa: E402


def test_verse_needs_two_short_halves():
    assert _is_verse("ألا ليت شعرى عن عوارضتى قنى ... لطول الليالى هل تغيرتا بعدى")
    assert not _is_verse("قال ... ثم قال ... وانصرف")  # several ellipses: prose
    long_half = " ".join(["كلمة"] * 20)
    assert not _is_verse(f"{long_half} ... وقال البيهقى")


def test_prose_ellipsis_stays_prose():
    toks = _edition_tokens("<p>" + " ".join(["كلمة"] * 20) + " ... وقال البيهقى كان يرى</p>")
    assert all(t.half == -1 for t in toks)
    assert edition_main(toks).startswith("<p>")


def test_c2_guillemet_markers_become_refs():
    toks = _edition_tokens('<p>هل تغيرتا بعدى <span class="c2">«٣»</span> وبعد</p>')
    assert toks[2].refs == ["٣"]
    assert "<ref" not in edition_main(_edition_tokens('<p>عن <span class="c2">(بمن عدى)</span> قال</p>'))


def test_notes_split_at_line_breaks():
    toks = _edition_tokens("<p>[١] «شابة» جبل بنجد.<br />[٢] «حزم نبايع» : جبل.</p>", notes=True)
    tags = edition_notes(toks)
    assert '<note n="١">' in tags and '<note n="٢">' in tags


def test_renumber_in_page_order():
    main, notes = renumber('<p>أ<ref n="٤"/> ب<ref n="١"/> ج<ref n="*"/></p>',
                           '<hashiya pos="bottom">\n<note cont="prev">x</note>\n<note n="٤">y</note>\n'
                           '<note n="١">z</note>\n</hashiya>')
    assert main == '<p>أ<ref n="١"/> ب<ref n="٢"/> ج<ref n="*"/></p>'
    assert '<note n="١">y</note>' in notes and '<note n="٢">z</note>' in notes
    # markers and notes that do not pair up: the notes are dropped
    assert renumber('<p>أ<ref n="١"/></p>', '<hashiya pos="bottom">\n<note n="١">y</note>\n'
                    '<note n="٢">z</note>\n</hashiya>')[1] is None


def test_page_number_must_fit_the_sequence():
    cands = {1: {11}, 2: {12, 400}, 3: {31}, 4: {14}, 9: {77}}
    assert confirmed_pagenum(2, cands) == 12
    assert confirmed_pagenum(3, cands) is None  # 31 has no neighbour
    assert confirmed_pagenum(9, cands) is None


def test_long_loops_are_found():
    phrase = " ".join(f"كلمة{k}" for k in range(35))
    assert find_repetition("بداية " + " ".join([phrase] * 4), 4)


# --------------------------------------------------------------------------- sources and detection

import json  # noqa: E402

import numpy as np  # noqa: E402

from pdf2epub.layout.detect import _dotted_rule_y  # noqa: E402


def _dots(mask: np.ndarray, y: int, x0: int, x1: int, step: int = 12) -> None:
    for x in range(x0, x1, step):
        mask[y:y + 3, x:x + 3] = 255


def test_dotted_rule_found_once_and_not_on_dot_leaders():
    mask = np.zeros((1000, 700), np.uint8)
    _dots(mask, 760, 300, 650)                       # one dotted separator above the notes
    assert abs(_dotted_rule_y(mask) - 761) <= 2
    for y in (500, 560, 620, 680):                   # a contents page: a leader on every line
        _dots(mask, y, 150, 600)
    assert _dotted_rule_y(mask) is None


def test_turath_page_layout():
    from turath_pages import page_html

    page = {"meta": json.dumps({"page": 7}),
            "text": '<span data-type="title" id=toc-2>الحديث الأول: [النية]</span>\nعَنْ عُمَرَ قَالَ (١).\n_________\n(١) البخاري (١).'}
    out = page_html(7, page)
    assert 'data-page-num="7"' in out
    assert "<p>الحديث الأول: [النية]</p>" in out                  # printed brackets stay
    assert '<span class="c4">[١]</span>' in out and "<hr><p>[١] البخاري (١).</p>" in out


def test_openiti_pages(tmp_path):
    import openiti_pages

    src = tmp_path / "t.txt"
    src.write_text("#META#Header#End#\n### | الجزء الأول\n# كلام أول ms0001\n~~تتمة PageV01P003 الثاني\n"
                   "# صدر البيت %~% عجز البيت PageV01P004\n", encoding="utf-8")
    import sys as _sys
    argv, _sys.argv = _sys.argv, ["openiti_pages.py", str(src), str(tmp_path / "out")]
    try:
        openiti_pages.main()
    finally:
        _sys.argv = argv
    first = (tmp_path / "out" / "1.html").read_text(encoding="utf-8")
    second = (tmp_path / "out" / "2.html").read_text(encoding="utf-8")
    assert 'data-page-num="3"' in first and "<p>كلام أول تتمة</p>" in first
    assert "صدر البيت ... عجز البيت" in second


def test_coverage_problem(tmp_path):
    from build_training_pairs import coverage_problem

    from pdf2epub.workdir import WorkDir

    wd = WorkDir(tmp_path)
    wd.pages_dir.mkdir(parents=True)
    words = " ".join(f"كلمة{k}" for k in range(60))
    wd.page_tags(1).write_text(f"<p>{words}</p>", encoding="utf-8")
    assert coverage_problem(wd, 1, f"<p>{words}</p>", region=False) is None
    assert coverage_problem(wd, 1, "<p>كلمة1 كلمة2</p>", region=False) == "target covers too little of the page"
    reordered = " ".join(reversed(words.split()))                 # verse read column by column
    assert coverage_problem(wd, 1, f"<p>{reordered}</p>", region=False) is None


def test_default_model_endpoint(tmp_path, monkeypatch):
    from pdf2epub.ocr.models import DEFAULT_MODEL, key_for, recorded_endpoint

    assert key_for(DEFAULT_MODEL) == "ft-ft4" and key_for("Qwen/Qwen3-VL-8B-Instruct") == "qwen3-vl-8b"
    monkeypatch.setenv("PDF2EPUB_CONFIG", str(tmp_path))
    (tmp_path / "endpoints.json").write_text('{"ft-ft4": "https://x.modal.run/v1"}')
    assert recorded_endpoint(DEFAULT_MODEL) == "https://x.modal.run/v1"


def test_rescore_corrects_only_doubtful_words():
    from pdf2epub import rescore as rs

    # "فالوا" read with a doubtful first token whose runner-up gives "قالوا"; "نعم" is sure.
    tokens = [{"t": "<p>", "lp": 0.0, "alt": []},
              {"t": "ف", "lp": -0.9, "alt": [["ف", -0.9], ["ق", -1.1]]}, {"t": "الوا", "lp": 0.0, "alt": []},
              {"t": " نعم", "lp": 0.0, "alt": [[" نعم", 0.0], [" بعم", -6.0]]},
              {"t": "</p>", "lp": 0.0, "alt": []}]
    words = rs.words_of(tokens)
    assert [w.norm for w in words] == ["فالوا", "نعم"] and "قالوا" in words[0].cands
    jobs = rs.windows(words, doubt=0.99)
    assert {i for i, _c, _w in jobs} == {0}                         # the sure word is not reconsidered
    lm = {"فالوا نعم": -9.0, "قالوا نعم": -4.0}                      # log10: "قالوا" fits far better
    scores = {(i, c): lm[w] for i, c, w in jobs}
    changes = rs.choose(words, scores, weight=0.5, margin=0.5, doubt=0.99)
    assert [c[1] for c in changes] == ["قالوا"]
    assert rs.apply("".join(t["t"] for t in tokens), words, changes) == "<p>قالوا نعم</p>"
    # a candidate that only cuts the word short is never offered
    trunc = [{"t": "صخ", "lp": -0.5, "alt": [["صخ", -0.5], ["صخر", -0.6]]}, {"t": "ره", "lp": 0.0, "alt": []}]
    assert all(len(c) >= 4 for c in rs.words_of(trunc)[0].cands)
