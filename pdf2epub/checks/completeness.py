"""Completeness check: every printed line is read once, nothing is added (spec §6.5).

A page model can end early, skip a line in the middle, or write text the page does not have.
None of that breaks the tags, so the other checks miss it. This check finds the printed text
lines on the page image (main text, above the foot-note rule), reads each line on its own (a crop
of one line cannot be half-skipped), and aligns those line readings with the page reading:

* a printed line that matches nothing in the page reading is a skipped line, and its reading is
  inserted where it belongs (after the words its neighbour matched);
* a run of page-reading words that no printed line matches is possible extra text; an exact
  repeat is removed, anything else is flagged.

Line readings are cached in ``pages/NNNN.lines.json``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..arabic import normalize_for_match

LINE_SYSTEM = (
    "You are an OCR engine for scanned Arabic books. The image is ONE printed line cut from a page. "
    "Transcribe it exactly as printed: the same letters, dots, hamzas and diacritics, punctuation and digits. "
    "Never correct, complete or modernise. Output only the text of the line, with no tags, no quotes and no "
    "comments. If the line holds a verse in two halves, write the right-hand half, then ' ... ', then the "
    "left-hand half. Note markers stay as printed, e.g. (١)."
)
LINE_USER = "Transcribe this line."


@dataclass
class Line:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in the page image
    halves: tuple[tuple[int, int], tuple[int, int]] | None = None  # (x0, x1) of right and left half (verse)
    text: str = ""


@dataclass
class Completeness:
    lines: list[Line] = field(default_factory=list)
    missing: list[tuple[int, int]] = field(default_factory=list)   # (line index, page-word index to insert after)
    extra: list[tuple[int, int]] = field(default_factory=list)     # (first, last+1) page-word runs nothing matched


# --------------------------------------------------------------------------- finding lines


def _ink(gray: np.ndarray) -> np.ndarray:
    _t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _letter_height(mask: np.ndarray) -> int:
    """Typical height of a letter group: the median height of the ink's connected pieces."""
    n, _l, stats, _ = cv2.connectedComponentsWithStats(mask)
    pieces = [(stats[i][4], stats[i][3]) for i in range(1, n) if stats[i][4] >= 20 and stats[i][3] < 0.2 * mask.shape[0]]
    if not pieces:
        return 20
    # Letters, not dots and vowel marks: only the pieces larger than the median piece.
    cut = np.median([a for a, _h in pieces])
    return int(np.median([h for a, h in pieces if a >= cut]))


def _peaks(smooth: np.ndarray, floor: float, min_gap: float) -> list[int]:
    peaks: list[int] = []
    for y in range(1, len(smooth) - 1):
        if smooth[y] >= floor and smooth[y] >= smooth[y - 1] and smooth[y] > smooth[y + 1]:
            if peaks and y - peaks[-1] < min_gap:
                if smooth[y] > smooth[peaks[-1]]:
                    peaks[-1] = y
                continue
            peaks.append(y)
    return peaks


def text_lines(gray: np.ndarray, top: int, bottom: int, x0: int, x1: int) -> list[tuple[int, int]]:
    """(y0, y1) of each text line between ``top`` and ``bottom``, columns ``x0``–``x1``.

    The ink profile is smoothed at the typical letter height; its peaks are the lines. The line
    spacing is the median distance between peaks, and peaks closer than 0.6 of it are one line.
    Each line runs from the quietest row before it to the quietest row after it, so diacritics
    above and below the letters stay with their line.
    """
    mask = _ink(gray[top:bottom, x0:x1])
    prof = (mask > 0).sum(axis=1).astype(float)
    if prof.max() <= 0:
        return []
    ch = _letter_height(mask)
    k = max(3, ch) | 1
    smooth = np.convolve(prof, np.ones(k) / k, mode="same")
    floor = 0.12 * np.percentile(smooth[smooth > 0], 90)
    first = _peaks(smooth, floor, 1.2 * ch)
    pitch = int(np.median(np.diff(first))) if len(first) > 2 else 3 * ch
    peaks = _peaks(smooth, floor, 0.6 * pitch)
    if not peaks:
        return []
    cuts = [max(0, peaks[0] - pitch // 2)]
    for a, b in zip(peaks, peaks[1:]):
        cuts.append(a + int(np.argmin(smooth[a:b])))
    cuts.append(min(len(smooth), peaks[-1] + pitch // 2))
    out = []
    for y0, y1 in zip(cuts, cuts[1:]):
        rows = np.where(prof[y0:y1] > 0)[0]
        if len(rows) and (rows[-1] - rows[0]) >= 0.25 * pitch:
            out.append((top + y0, top + y1))
    return out


def verse_halves(gray: np.ndarray, box: tuple[int, int, int, int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """For a verse line printed in two halves: the x-ranges of the right and left half (a wide
    empty gap near the middle), else None."""
    x0, y0, x1, y1 = box
    cols = (_ink(gray[y0:y1, x0:x1]) > 0).sum(axis=0) > 0
    w = len(cols)
    best, run, start = (0, 0), 0, 0
    for x in range(int(0.3 * w), int(0.7 * w)):
        if not cols[x]:
            if run == 0:
                start = x
            run += 1
            if run > best[0]:
                best = (run, start)
        else:
            run = 0
    if best[0] < 0.06 * w:
        return None
    left, right = cols[: best[1]], cols[best[1] + best[0]:]
    # Both halves must carry text across much of their width (a short prose line has one side empty).
    if left.sum() < 0.25 * w or right.sum() < 0.25 * w:
        return None
    gap0, gap1 = x0 + best[1], x0 + best[1] + best[0]
    return (gap1, x1), (x0, gap0)


def page_lines(gray: np.ndarray) -> list[Line]:
    """Printed lines of the main text: above the foot-note rule, within the text column, without
    the running header / page number line at the top."""
    from ..layout.detect import _blocks, _main_block, _rule_y, _text_area

    h, w = gray.shape
    blocks = _blocks(gray)
    main = _main_block(blocks)
    if main is None:
        return []
    rule = _rule_y(gray, main, first=True)
    area = _text_area(gray) or (main[0], main[2])
    x0, x1 = max(0, area[0] - 10), min(w, area[1] + 10)
    # All wide blocks, not only the largest (a boxed heading can be the largest block).
    wide = [b for b in blocks if (b[2] - b[0]) >= 0.25 * w and (rule is None or b[1] < rule)]
    top = min(b[1] for b in wide) if wide else main[1]
    bottom = rule - 4 if rule else (max(b[3] for b in wide) if wide else main[3])
    lines = [Line((x0, a, x1, b)) for a, b in text_lines(gray, top, bottom, x0, x1)]
    for ln in lines:
        ln.halves = verse_halves(gray, ln.box)
    return lines


# --------------------------------------------------------------------------- matching


_RASM = str.maketrans({"ب": "ٮ", "ت": "ٮ", "ث": "ٮ", "ن": "ٮ", "ي": "ى", "ج": "ح", "خ": "ح", "ذ": "د", "ز": "ر",
                       "ش": "س", "ض": "ص", "ظ": "ط", "غ": "ع", "ف": "ڡ", "ق": "ڡ", "ة": "ه", "ؤ": "و", "ئ": "ى"})


def key(word: str) -> str:
    """A word as two readings of the same print should agree on: letters only, dots ignored."""
    w = normalize_for_match(re.sub(r"[(\[][٠-٩0-9]{1,3}[)\]]", "", word))
    return w.translate(_RASM)


def match(line_words: list[list[str]], page_words: list[str], *, min_cover: float = 0.35,
          min_words: int = 3, extra_run: int = 6) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Align line readings with the page reading.

    Returns (missing, extra): ``missing`` holds (line index, page-word index to insert after,
    -1 for the start) for lines of at least ``min_words`` words of which under ``min_cover``
    match the page; ``extra`` holds runs of at least ``extra_run`` page words matched by no line.
    """
    from rapidfuzz.distance import Levenshtein

    flat, owner = [], []
    for li, words in enumerate(line_words):
        for w in words:
            if key(w):
                flat.append(key(w))
                owner.append(li)
    page = [key(w) for w in page_words]
    covered_line = [0] * len(line_words)
    total_line = [0] * len(line_words)
    for li in owner:
        total_line[li] += 1
    page_hit = [False] * len(page)
    last_page_for_line: dict[int, int] = {}
    for op in Levenshtein.opcodes(flat, page):
        if op.tag in ("equal", "replace"):
            for a, b in zip(range(op.src_start, op.src_end), range(op.dest_start, op.dest_end)):
                if op.tag == "equal" or Levenshtein.normalized_distance(flat[a], page[b]) <= 0.34:
                    covered_line[owner[a]] += 1
                    page_hit[b] = True
                    last_page_for_line[owner[a]] = b
    missing = []
    for li in range(len(line_words)):
        if total_line[li] >= min_words and covered_line[li] < min_cover * total_line[li]:
            before = [last_page_for_line[j] for j in range(li) if j in last_page_for_line]
            missing.append((li, max(before) if before else -1))
    extra, run = [], None
    for k, hit in enumerate(page_hit + [True]):
        if not hit and page[k] if k < len(page) else False:
            run = k if run is None else run
        elif run is not None:
            if k - run >= extra_run:
                extra.append((run, k))
            run = None
    return missing, extra


# --------------------------------------------------------------------------- the page reading


def page_words(tags: str, *, main_only: bool = True) -> list[tuple[str, int, int]]:
    """Words of a page reading with their (start, end) in the tags, without tag names. By
    default only the main text (no header, page number or notes); with ``main_only=False``
    everything the reading holds, which is what a printed line is matched against."""
    cut = tags.find("<hashiya")
    end = cut if (cut >= 0 and main_only) else len(tags)
    meta = r"<header>.*?</header>|<pagenum>.*?</pagenum>|" if main_only else ""
    skip = [m.span() for m in re.finditer(meta + r"<[^>]+>", tags[:end], re.S)]
    out = []
    for m in re.finditer(r"[^\s<>]+", tags[:end]):
        a = m.start()
        if any(s <= a < e for s, e in skip):
            continue
        if key(m.group(0)):
            out.append((m.group(0), a, m.end()))
    return out


def plausible(text: str, typical_words: float) -> bool:
    """A line reading worth trusting: no loop, and not far longer than a printed line can be."""
    from .flags import find_repetition

    n = len(text.split())
    return bool(text.strip()) and not find_repetition(text, 3) and n <= max(25, 2.5 * typical_words)


def _markers_as_refs(text: str) -> str:
    from xml.sax.saxutils import escape

    return re.sub(r"\(([٠-٩0-9]{1,3})\)", r'<ref n="\1"/>', escape(text))


def insert_lines(tags: str, words: list[tuple[str, int, int]], inserts: list[tuple[int, str, tuple[str, str] | None]]) -> str:
    """Insert skipped lines into a page reading. Each insert is (page-word index to insert after,
    -1 for the start; the line's text; its two halves for a verse line, or None). Inside a poem a
    verse line becomes a new verse after the current one and any other line a paragraph after the
    poem; elsewhere the words join the text in place."""
    edits: list[tuple[int, str]] = []
    for after, text, halves in inserts:
        pos = words[after][2] if after >= 0 and words else (words[0][1] if words else 0)
        opener = max((m for m in re.finditer(r"<(p|s|a|h\d|quran)\b[^>]*>", tags[:pos])), key=lambda m: m.start(), default=None)
        in_poem = opener is not None and opener.group(1) in ("s", "a") and "</poem>" not in tags[opener.start():pos]
        if in_poem and not halves:
            close = tags.find("</poem>", pos)  # a prose line after the poem's last verse
            at = close + len("</poem>") if close >= 0 else pos
            edits.append((at, f"\n<p>{_markers_as_refs(text.strip())}</p>"))
        elif in_poem:
            close = tags.find("</v>", pos)
            at = close + len("</v>") if close >= 0 else pos
            right, left = halves if halves else (text, "")
            verse = f"<v><s>{_markers_as_refs(right.strip())}</s>" + (f"<a>{_markers_as_refs(left.strip())}</a>" if left.strip() else "") + "</v>"
            edits.append((at, verse))
        elif after < 0:
            edits.append((pos, _markers_as_refs(text.strip()) + " "))
        else:
            edits.append((pos, " " + _markers_as_refs(text.strip())))
    for at, piece in sorted(edits, key=lambda e: -e[0]):
        tags = tags[:at] + piece + tags[at:]
    return tags
