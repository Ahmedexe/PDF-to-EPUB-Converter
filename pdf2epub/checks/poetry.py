"""Poetry checks: structure, rhyme, optional meter (spec §5.2). Poems are never rewritten.

Typical OCR failures are half-lines swapped (read left to right), two verses
merged into one, a verse split at a page break, and decorative separators read
as text. Each poem is checked for:

* structure: both halves present (unless single-column), similar lengths,
  no separator marks inside a half;
* rhyme: in a classical poem every عجز ends on the same rhyme letter (الروي);
  a verse that breaks the pattern is flagged, and a sadr carrying the rhyme
  instead suggests swapped halves;
* meter (optional): when a classifier is available, halves of one verse
  classified as different meters are flagged.

A page that opens with a poem (no heading first) right after a page ending with
a poem continues it, so the two parts are checked as one poem.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Protocol

from ..arabic import HARAKAT, LETTER_UNIFY, clean_text, is_arabic_letter, strip_diacritics
from ..config import BookConfig
from ..model import Flag, FlagCode, Page, Poem, Verse, inline_text, iter_blocks

log = logging.getLogger(__name__)
STAGE = "poetry"

_SEPARATOR_RE = re.compile(r"(?:^|\s)(?:\*{1,3}|\.{3}|…|-{2,}|ـ{3,}|٭)(?:\s|$)|^\s*[*…٭]|[*…٭]\s*$")
_WASL = set("اويىه")  # long vowels / ha of ishbaʿ and wasl that may follow the rawi
_TANWIN_FATH = "ً"


class MeterClassifier(Protocol):
    def __call__(self, half: str) -> str | None: ...


_meter: MeterClassifier | None | bool = False  # False = not looked up yet


def set_meter_classifier(classifier: MeterClassifier | None) -> None:
    """Install a meter classifier (``half-line text -> meter name or None``); None disables it."""
    global _meter
    _meter = classifier


def _get_meter() -> MeterClassifier | None:
    global _meter
    if _meter is False:
        _meter = None
        try:  # optional dependency: https://github.com/arbml/qawafi
            import qawafi  # type: ignore[import-not-found]

            fn = getattr(qawafi, "predict_meter", None) or getattr(qawafi, "classify_meter", None)
            if callable(fn):
                _meter = fn
        except Exception:  # pragma: no cover - not installed in the default environment
            _meter = None
    return _meter  # type: ignore[return-value]


def _letters_only(text: str) -> str:
    return "".join(ch for ch in strip_diacritics(clean_text(text)) if is_arabic_letter(ch))


def rhyme_candidates(half: str) -> set[str]:
    """Possible rawi letters of a half-line: the last letter, and the one before a final
    long vowel or ha of wasl (ishbaʿ), with alef, ة and hamza seats unified."""
    text = clean_text(half).strip()
    text = re.sub(r"[^\w\sً-ْٰ]+$", "", text).strip()
    words = text.split()
    if not words:
        return set()
    word = words[-1]
    tanwin_alef = word.endswith("ا") and _TANWIN_FATH in word[-3:]
    letters = [LETTER_UNIFY.get(ch, ch) for ch in strip_diacritics(word) if is_arabic_letter(ch)]
    letters = ["ء" if ch in "أإؤئ" else ch for ch in letters]
    if not letters:
        return set()
    if tanwin_alef and len(letters) > 1:
        letters = letters[:-1]
    out = {letters[-1]}
    if len(letters) > 1 and letters[-1] in _WASL:
        out.add(letters[-2])
        if len(letters) > 2 and letters[-1] == "ا" and letters[-2] == "ه":  # ـها
            out.add(letters[-3])
    return out


def rhyme_letter(half: str) -> str | None:
    """The single most likely rawi of a half-line (None if there are no letters)."""
    cands = rhyme_candidates(half)
    if not cands:
        return None
    letters = _letters_only(half)
    last = letters[-1] if letters else None
    last = LETTER_UNIFY.get(last, last) if last else None
    if last in cands and last not in _WASL:
        return last
    return sorted(cands - _WASL)[0] if cands - _WASL else next(iter(cands))


def _half_text(half: list[object] | None) -> str:
    return inline_text(half or []).strip()


def _single(poem: Poem, cfg: BookConfig) -> bool:
    if poem.single or cfg.text.poem_columns == "single":
        return True
    if cfg.text.poem_columns == "auto":
        return bool(poem.verses) and all(v.ajuz is None for v in poem.verses)
    return False


def check_poem(poem: Poem, cfg: BookConfig, *, context: list[Verse] | None = None) -> list[Flag]:
    """Flags for one poem. ``context`` holds verses of the same poem from the previous page,
    used for the rhyme pattern but not flagged here."""
    flags: list[Flag] = []
    single = _single(poem, cfg)
    if single and not poem.single and cfg.text.poem_columns == "auto" and len(poem.verses) > 1:
        flags.append(Flag(code=FlagCode.POEM_MISSING_HALF, severity="info", target=poem.id, stage=STAGE,
                          message="no verse has a left-hand half-line: treated as a single-column poem"))
    ratio = cfg.checks.poem_half_ratio
    for k, verse in enumerate(poem.verses, 1):
        sadr, ajuz = _half_text(verse.sadr), _half_text(verse.ajuz)
        for name, text in (("right", sadr), ("left", ajuz)):
            if text and _SEPARATOR_RE.search(text):
                flags.append(Flag(code=FlagCode.POEM_SEPARATOR, target=poem.id, stage=STAGE,
                                  message=f"verse {k}: decorative marks read as part of the {name} half-line"))
        if single:
            continue
        if verse.ajuz is None or not ajuz or not sadr:
            missing = "left-hand" if (verse.ajuz is None or not ajuz) else "right-hand"
            flags.append(Flag(code=FlagCode.POEM_MISSING_HALF, target=poem.id, stage=STAGE,
                              message=f"verse {k} has no {missing} half-line"))
            continue
        ls, la = len(_letters_only(sadr)), len(_letters_only(ajuz))
        if min(ls, la) and max(ls, la) / min(ls, la) > ratio:
            flags.append(Flag(code=FlagCode.POEM_HALF_LENGTH, target=poem.id, stage=STAGE,
                              message=f"verse {k}: half-lines of very different length ({ls} and {la} letters)"
                                      " — merged or split verses?"))
    if single:
        return flags
    verses = [v for v in (context or []) + poem.verses if v.ajuz is not None and _half_text(v.ajuz)]
    if len(verses) >= 3:
        counts: Counter[str] = Counter()
        for v in verses:
            counts.update(rhyme_candidates(_half_text(v.ajuz)))
        rawi, n = counts.most_common(1)[0] if counts else (None, 0)
        if rawi is not None and n >= max(2, int(0.6 * len(verses))):
            offset = len(verses) - len([v for v in poem.verses if v.ajuz is not None and _half_text(v.ajuz)])
            own = [v for v in poem.verses if v.ajuz is not None and _half_text(v.ajuz)]
            for k, verse in enumerate(own, 1):
                if rawi in rhyme_candidates(_half_text(verse.ajuz)):
                    continue
                swapped = rawi in rhyme_candidates(_half_text(verse.sadr))
                hint = " — the halves look swapped" if swapped else ""
                flags.append(Flag(code=FlagCode.POEM_RHYME, target=poem.id, stage=STAGE,
                                  message=f"verse {k}: does not end on the poem's rhyme letter «{rawi}»{hint}"))
            del offset
    meter = _get_meter() if cfg.text.poem_meter else None
    if meter is not None:
        for k, verse in enumerate(poem.verses, 1):
            sadr, ajuz = _half_text(verse.sadr), _half_text(verse.ajuz)
            if not (sadr and ajuz) or not any(ch in HARAKAT for ch in sadr + ajuz):
                continue
            try:
                m1, m2 = meter(sadr), meter(ajuz)
            except Exception:  # a classifier error never blocks a page
                log.debug("meter classifier failed", exc_info=True)
                continue
            if m1 and m2 and m1 != m2:
                flags.append(Flag(code=FlagCode.POEM_METER, target=poem.id, stage=STAGE,
                                  message=f"verse {k}: halves classified as {m1} and {m2}"))
    return flags


def _continued_context(pages: list[Page], page: Page, poem: Poem) -> list[Verse] | None:
    """Verses of the previous page's last poem when ``poem`` opens ``page`` and continues it."""
    if poem.continued:
        pass
    elif not page.blocks or page.blocks[0] is not poem:
        return None
    prev = next((p for p in pages if p.index == page.index - 1), None)
    if prev is None or not prev.blocks or not isinstance(prev.blocks[-1], Poem):
        return None
    return list(prev.blocks[-1].verses)


def check_poetry(pages: list[Page], cfg: BookConfig, *, only: set[int] | None = None) -> None:
    """Check every poem (main text and notes) on the editable pages."""
    pages = sorted(pages, key=lambda p: p.index)
    for page in pages:
        if page.status == "reviewed" or (only is not None and page.index not in only):
            continue
        page.clear_flags(STAGE)
        for block, _owner in iter_blocks(page):
            if isinstance(block, Poem):
                context = _continued_context(pages, page, block) if _owner is None else None
                page.flags.extend(check_poem(block, cfg, context=context))
