"""Arabic text utilities shared by parsing, checks, assembly, EPUB building and scoring.

Two kinds of normalisation live here and must not be confused:

* :func:`clean_text` is applied by the parser to everything the model writes.
  It only removes encoding noise (bidi controls, presentation-form glyphs,
  odd whitespace) and never changes letters, diacritics or punctuation.
* :func:`normalize_for_match` / :func:`normalize_with_map` are for *matching*
  (Quran verses against Tanzil, قوله lemmas against the main text). They strip
  diacritics and Quranic marks, unify alef forms, ى and ي, ة and ه, remove
  tatweel and punctuation (spec §5.1 step 2). Their output is never shown.

:func:`normalize_for_score` is the scoring normalisation of spec §9.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

# --------------------------------------------------------------------------- character classes

TATWEEL = "ـ"
ORNATE_OPEN = "﴿"  # ﴿ precedes a verse in logical order
ORNATE_CLOSE = "﴾"  # ﴾ follows a verse in logical order
SUPERSCRIPT_ALEF = "ٰ"

BIDI_CONTROLS = frozenset("‎‏؜‪‫‬‭‮⁦⁧⁨⁩")
ZERO_WIDTH = frozenset("​‌‍⁠﻿")


def _ranges(*pairs: tuple[int, int]) -> frozenset[str]:
    return frozenset(chr(c) for a, b in pairs for c in range(a, b + 1))


# Harakat, shadda, sukun, maddah/hamza combining marks, honorific small signs.
HARAKAT = _ranges((0x064B, 0x065F), (0x0610, 0x061A)) | {SUPERSCRIPT_ALEF}
# Quranic annotation: pause marks, small letters, end of ayah, rub el hizb, sajdah, extended marks.
QURANIC_MARKS = _ranges((0x06D6, 0x06ED), (0x08D3, 0x08FF))
PAUSE_MARKS = _ranges((0x06D6, 0x06DC))
UTHMANI_SIGNS = _ranges((0x06DF, 0x06E8), (0x06EA, 0x06ED)) | {"ٱ"}
DIACRITICS = HARAKAT | QURANIC_MARKS

ALEF_FORMS = {"آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا",
              "ٲ": "ا", "ٳ": "ا", "ٵ": "ا"}
LETTER_UNIFY = {
    **ALEF_FORMS,
    "ى": "ي",  # ى -> ي
    "ی": "ي",  # Farsi yeh -> ي
    "ې": "ي",
    "ة": "ه",  # ة -> ه
    "ۃ": "ه",
    "ہ": "ه",  # heh goal
    "ھ": "ه",  # heh doachashmee
    "ە": "ه",
    "ک": "ك",  # keheh -> ك
    "ڪ": "ك",
}
HAMZA_UNIFY = {"ؤ": "و", "ئ": "ي", "ء": ""}

# Undotted skeleton (rasm), position-insensitive. Applied after LETTER_UNIFY.
RASM = {}
for _letters, _base in (
    ("بتثنيئ", "ٮ"), ("جحخ", "ح"), ("دذ", "د"), ("رز", "ر"), ("سش", "س"),
    ("صض", "ص"), ("طظ", "ط"), ("عغ", "ع"), ("فق", "ڡ"), ("ؤ", "و"),
):
    for _ch in _letters:
        RASM[_ch] = _base
RASM["ء"] = ""

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EXTENDED_INDIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_DIGIT_VALUE = {d: i for i, d in enumerate(ARABIC_INDIC_DIGITS)}
_DIGIT_VALUE.update({d: i for i, d in enumerate(EXTENDED_INDIC_DIGITS)})
_DIGIT_VALUE.update({str(i): i for i in range(10)})

# Bracket pairs as printed, opening character first in logical order.
BRACKET_PAIRS: dict[str, str] = {
    "«": "»", "‹": "›", "“": "”", "„": "“", '"': '"', "(": ")", "[": "]", "{": "}",
    "﴿": "﴾", "﴾": "﴿", "⟨": "⟩", "<": ">",
}

ARABIC_PUNCT = "،؛؟۔٪٫٬٭"


def _is_punct(ch: str) -> bool:
    if ch in ARABIC_PUNCT:
        return True
    cat = unicodedata.category(ch)
    return cat[0] in "PS" and ch not in "ـ"


# --------------------------------------------------------------------------- cleaning (parser)

_PRESENTATION_KEEP = _ranges((0xFDF0, 0xFDFF))  # ﷺ ﷲ ﷻ ﷽ and other honorific ligatures stay
_WS_RE = re.compile(r"[ \t\r\f\v   -   　]+")


def fold_presentation_forms(s: str) -> str:
    """Replace Arabic presentation-form glyphs (U+FB50–U+FDCF, U+FE70–U+FEFC) by plain letters.

    Honorific ligatures such as ﷺ and ﷲ (U+FDF0–U+FDFF) and the ornate
    parentheses are kept as printed.
    """
    out = []
    for ch in s:
        cp = ord(ch)
        if (0xFB50 <= cp <= 0xFDCF or 0xFE70 <= cp <= 0xFEFC) and ch not in _PRESENTATION_KEEP \
                and ch not in (ORNATE_OPEN, ORNATE_CLOSE):
            out.append(unicodedata.normalize("NFKC", ch))
        else:
            out.append(ch)
    return "".join(out)


def strip_bidi(s: str) -> str:
    return "".join(ch for ch in s if ch not in BIDI_CONTROLS)


def clean_text(s: str) -> str:
    """Encoding clean-up applied to all model output: NFC, no bidi controls, no
    presentation forms, unified horizontal whitespace. Letters, diacritics and
    punctuation are left exactly as written. Newlines are kept."""
    s = unicodedata.normalize("NFC", s)
    s = strip_bidi(s)
    s = s.replace("﻿", "").replace("​", "")
    s = fold_presentation_forms(s)
    s = unicodedata.normalize("NFC", s)
    return _WS_RE.sub(" ", s)


def collapse_spaces(s: str) -> str:
    return " ".join(s.split())


# --------------------------------------------------------------------------- matching

def strip_tatweel(s: str) -> str:
    return s.replace(TATWEEL, "")


def strip_diacritics(s: str) -> str:
    """Remove harakat, superscript alef, honorific small signs and Quranic marks."""
    return "".join(ch for ch in s if ch not in DIACRITICS)


def unify_letters(s: str, *, hamza: bool = False) -> str:
    """Unify alef forms (أ إ آ ٱ -> ا), ى/ی -> ي, ة -> ه, ک -> ك; optionally hamza seats."""
    table = dict(LETTER_UNIFY)
    if hamza:
        table.update(HAMZA_UNIFY)
    return "".join(table.get(ch, ch) for ch in s)


def to_rasm(s: str) -> str:
    """Reduce matching-normalised text to its undotted skeleton (dotted letters merge)."""
    return "".join(RASM.get(ch, ch) for ch in s)


def normalize_with_map(
    s: str,
    *,
    hamza: bool = False,
    rasm: bool = False,
    dagger_alef: Literal["strip", "alef"] = "strip",
    keep_digits: bool = True,
) -> tuple[str, list[int]]:
    """Matching normalisation that remembers where each output character came from.

    Returns ``(norm, idx)`` where ``norm[i]`` derives from ``s[idx[i]]`` and
    ``idx`` has one extra final entry equal to ``len(s)``. Punctuation,
    brackets and whitespace runs become single spaces (mapped to the first
    character of the run); leading and trailing spaces are dropped.

    ``s`` should already be :func:`clean_text`-ed (NFC, no presentation forms).
    ``dagger_alef="alef"`` turns the superscript alef into a full alef, which
    brings Uthmani spelling (ٱلْعَٰلَمِينَ) closer to standard spelling (العالمين).
    """
    table = dict(LETTER_UNIFY)
    if hamza:
        table.update(HAMZA_UNIFY)
    out: list[str] = []
    idx: list[int] = []
    pending_space = -1
    for i, ch in enumerate(s):
        if ch == SUPERSCRIPT_ALEF and dagger_alef == "alef":
            # ىٰ (alef maksura + dagger alef) is spelled ا in standard orthography.
            if out and i > 0 and s[i - 1] == "ى" and idx and idx[-1] == i - 1:
                out[-1] = "ا"
                continue
            rep = "ا"
        elif ch in DIACRITICS or ch == TATWEEL or ch in ZERO_WIDTH or ch in BIDI_CONTROLS:
            continue
        elif ch.isspace() or _is_punct(ch) or (not keep_digits and ch in _DIGIT_VALUE):
            if pending_space < 0:
                pending_space = i
            continue
        else:
            rep = table.get(ch, ch)
            if rasm:
                rep = "".join(RASM.get(c, c) for c in rep)
        if not rep:
            continue
        if pending_space >= 0:
            if out:
                out.append(" ")
                idx.append(pending_space)
            pending_space = -1
        for c in rep:
            out.append(c)
            idx.append(i)
    idx.append(len(s))
    return "".join(out), idx


def normalize_for_match(s: str, **opts) -> str:
    """Normalised text for matching (see :func:`normalize_with_map` for options)."""
    return normalize_with_map(s, **opts)[0]


def match_words(s: str, **opts) -> list[str]:
    """Matching-normalised words."""
    return normalize_for_match(s, **opts).split()


# --------------------------------------------------------------------------- scoring

def normalize_for_score(s: str, *, keep_diacritics: bool) -> str:
    """Spec §9: Unicode NFC, tatweel removed, spaces unified; diacritics optionally removed."""
    s = clean_text(s)
    s = strip_tatweel(s)
    s = "".join(ch for ch in s if ch not in ZERO_WIDTH)
    if not keep_diacritics:
        s = strip_diacritics(s)
    s = unicodedata.normalize("NFC", s)
    return collapse_spaces(s)


# --------------------------------------------------------------------------- features

def has_diacritics(s: str) -> bool:
    return any(ch in HARAKAT for ch in s)


def diacritic_ratio(s: str) -> float:
    """Diacritics per Arabic letter (0 for undiacritised text, ~0.6-1.0 for fully voweled text)."""
    letters = sum(1 for ch in s if "ء" <= ch <= "ي" or ch == "ٱ")
    if not letters:
        return 0.0
    return sum(1 for ch in s if ch in HARAKAT) / letters


def has_pause_marks(s: str) -> bool:
    return any(ch in PAUSE_MARKS for ch in s)


def has_uthmani_features(s: str) -> bool:
    """True when the text uses Uthmani script features (alef wasla, small high letters)."""
    return any(ch in UTHMANI_SIGNS for ch in s)


def is_arabic_letter(ch: str) -> bool:
    return ("ء" <= ch <= "ي") or ("ٱ" <= ch <= "ۓ") or ch in "ەۮۯ"


# --------------------------------------------------------------------------- digits

DigitStyle = Literal["arabic-indic", "western"]


def parse_number(s: str) -> int | None:
    """Integer value of a string of digits in any style (١٢, ۱۲, 12), ignoring brackets
    and spaces around it; None if it is not purely a number."""
    core = s.strip().strip("()[]{}<>«»﴾﴿-–—.,:؛،*").strip()
    if not core or any(ch not in _DIGIT_VALUE for ch in core):
        return None
    value = 0
    for ch in core:
        value = value * 10 + _DIGIT_VALUE[ch]
    return value


def format_number(n: int, style: DigitStyle) -> str:
    text = str(n)
    if style == "arabic-indic":
        return "".join(ARABIC_INDIC_DIGITS[int(c)] for c in text)
    return text


def detect_digit_style(text: str) -> DigitStyle | None:
    """The digit style most used in ``text``, or None if it has no digits."""
    indic = sum(1 for ch in text if ch in ARABIC_INDIC_DIGITS or ch in EXTENDED_INDIC_DIGITS)
    western = sum(1 for ch in text if "0" <= ch <= "9")
    if not indic and not western:
        return None
    return "arabic-indic" if indic >= western else "western"


# --------------------------------------------------------------------------- markers

_MARKER_STRIP = "()[]{}<>«»﴾﴿  "


def normalize_marker(marker: str) -> str:
    """Canonical form of a printed note marker so that (١), [1] and ١ pair up.

    Numbers become their Western digits ("1"); symbols and letters are kept
    with brackets and spaces removed ("*", "**", "أ" -> "ا" unified).
    """
    core = clean_text(marker).strip(_MARKER_STRIP)
    value = parse_number(core)
    if value is not None:
        return str(value)
    core = strip_diacritics(core)
    core = "".join(LETTER_UNIFY.get(ch, ch) for ch in core)
    return core.replace("٭", "*").replace("✻", "*").replace("⁎", "*")


def source_span(s: str, idx: list[int], start: int, end: int) -> tuple[int, int]:
    """Source offsets ``[a, b)`` of the normalised range ``norm[start:end]``.

    ``idx`` comes from :func:`normalize_with_map` on ``s``. The end is moved past
    any combining marks (harakat, tanwin, dagger alef) and tatweel that belong to
    the last matched letter, so an anchor inserted at ``b`` never separates a
    letter from its diacritics; it stays before following punctuation.
    """
    if end <= start:
        return idx[start], idx[start]
    a = idx[start]
    b = idx[end - 1] + 1
    while b < len(s) and (unicodedata.combining(s[b]) or s[b] == TATWEEL or s[b] in DIACRITICS):
        b += 1
    return a, b
