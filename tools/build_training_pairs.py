"""Training pairs from scanned pages + a page-matched digital text of the same edition.

    python tools/build_training_pairs.py [--mode edition|corrected] SHAMELA_PAGES_DIR RUN_DIR OUT_DIR

``edition`` (default): letters-only targets built from the edition's own lines for each page
(prose, headings, verses split at the "..." half-line boundary, markers, notes), with the
instruction "no_diacritics". ``corrected``: the model's output with its words corrected
(described below), which keeps the model's (unreliable) diacritics.

For every page read by a model run, the model's structure (paragraphs, poems, notes,
markers) is kept and its words are corrected from the digital text:

* the page is located in the edition's text (main and notes separately) with word
  4-grams that occur only once;
* model words and edition words are aligned; a wrong word is replaced by the edition's,
  keeping the model's diacritics letter by letter when only letters/dots differ (same
  letter count), otherwise the edition's undiacritised word is used and counted;
* words the model skipped are inserted, and words it invented are removed (never at
  the page's edges, where the edition's page boundaries may differ);
* note markers the edition records at a word ("[١]") are added where the model left
  them out; lines wrongly tagged as Quran (flagged by the Quran check) become text.

Pages that align poorly are skipped. Writes ``NNNN.json`` (image, target tags, stats)
and ``pairs.jsonl``. The edition (Shamela) keeps only shadda, so corrected words may
lose harakat: ``undiacritised_words`` counts them, for weighting or review.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import html  # noqa: E402

from rapidfuzz.distance import Levenshtein  # noqa: E402

from pdf2epub.arabic import DIACRITICS, TATWEEL, normalize_for_match  # noqa: E402
from pdf2epub.model import (  # noqa: E402
    Matn, Page, Quote, Quran, Ref, Unclear, block_containers, normalize_inlines,
)
from pdf2epub.parse import parse_tags, serialize_page  # noqa: E402
from pdf2epub.workdir import WorkDir  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_tesseract import split_paths  # noqa: E402

EDGE = 8  # words at each end of a page whose additions/removals are not trusted
MAX_EDIT_RATE = 0.35


# --------------------------------------------------------------------------- the edition


@dataclass
class Tok:
    word: str
    norm: str
    refs: list[str] = field(default_factory=list)  # markers printed right after this word
    entry: int = -1  # notes: which note line this word belongs to
    lead: str | None = None  # notes: the marker opening this word's note line (on its first word)
    line: int = -1  # which edition line the word is on
    half: int = -1  # verse lines: 0 = right-hand half-line (صدر), 1 = left-hand (عجز); prose: -1


_ENTRY = [0, 0]  # note entry counter, line counter


def _is_verse(line: str) -> bool:
    """A verse line is "صدر ... عجز": one "..." between two short halves. Prose also uses
    "..." (an ellipsis), so a long half or several of them mean prose."""
    halves = line.split("...")
    if len(halves) != 2:
        return False
    sizes = [len(re.sub(r"\[[^\]]*\]|⟦[^⟧]*⟧", " ", h).split()) for h in halves]
    return all(1 <= n <= 12 for n in sizes)


def _edition_tokens(part: str, notes: bool = False) -> list[Tok]:
    """Words of one page part (main or notes) with the markers that follow them."""
    toks: list[Tok] = []
    lines = re.findall(r"<p[^>]*>(.*?)</p>", part, re.S)
    if notes:  # several notes may share a paragraph, one per <br> line: "...<br />[٢] ..."
        lines = [seg for ln in lines for seg in re.split(r"<br\s*/?>", ln)]
    for ln in lines:
        ln = re.sub(r"<a [^>]*class=\"btn_tag[^>]*>.*?</a>", " ", ln, flags=re.S)
        # Markers are bracketed numbers or symbols; other bracketed words are the editor's
        # insertions, printed in brackets, and stay in the text.
        ln = re.sub(r'<span class="c4">\s*\[([٠-٩0-9]{1,3}|\*{1,3})\]\s*</span>', r" ⟦\1⟧ ", ln)
        # Some parts of an edition print markers as «١» (class c2, which also marks other insertions).
        ln = re.sub(r'<span class="c2">\s*«([٠-٩0-9]{1,3})»\s*</span>', r" ⟦\1⟧ ", ln)
        ln = html.unescape(re.sub(r"<[^>]+>", " ", ln))
        lead = re.match(r"^\s*\[([٠-٩0-9]+)\]\s*", ln)
        ln = re.sub(r"^\s*\[[٠-٩0-9]+\]\s*", "", ln)
        verse = _is_verse(ln)
        ln = ln.replace("...", " ⟪HALF⟫ ") if verse else ln.replace("...", " ")
        first = True
        if notes and (lead or not toks):
            _ENTRY[0] += 1
        _ENTRY[1] += 1
        half = 0 if verse else -1
        for w in ln.split():
            if w == "⟪HALF⟫":
                half = 1
                continue
            m = re.fullmatch(r"⟦(.*)⟧", w)
            if m:
                if toks:
                    toks[-1].refs.append(m.group(1))
                continue
            norm = normalize_for_match(w)
            if not norm and toks and all("\ufd3e" <= ch <= "\ufdff" for ch in w):
                toks[-1].word += " " + w  # an honorific ligature (ﷺ, ﵁, ﵄): printed, so kept with its word
                continue
            if norm:
                tok = Tok(w, norm, entry=_ENTRY[0] if notes else -1, line=_ENTRY[1], half=half)
                if notes and first and lead:
                    tok.lead = lead.group(1)
                first = False
                toks.append(tok)
    return toks


def load_edition(pages_dir: Path) -> tuple[list[Tok], list[Tok]]:
    main: list[Tok] = []
    notes: list[Tok] = []
    pages = []
    for f in pages_dir.glob("*.html"):
        raw = f.read_text(encoding="utf-8")
        m = re.search(r'<div class="nass[^"]*"[^>]*data-page-id="(\d+)"[^>]*>(.*?)</div>', raw, re.S)
        if m:
            pages.append((int(m.group(1)), m.group(2)))
    for _pid, body in sorted(pages):
        a, _, b = body.partition("<hr>")
        main.extend(_edition_tokens(a))
        notes.extend(_edition_tokens(b, notes=True))
    return main, notes


def unique_grams(toks: list[Tok]) -> dict[tuple[str, ...], int]:
    norm = [t.norm for t in toks]
    counts = Counter(tuple(norm[i:i + 4]) for i in range(len(norm) - 3))
    return {tuple(norm[i:i + 4]): i for i in range(len(norm) - 3) if counts[tuple(norm[i:i + 4])] == 1}


def locate(query: list[str], toks: list[Tok], grams: dict[tuple[str, ...], int]) -> tuple[int, int] | None:
    hits = [(j, grams[tuple(query[j:j + 4])]) for j in range(len(query) - 3) if tuple(query[j:j + 4]) in grams]
    if len(hits) < 3:
        return None
    offsets = sorted(p - j for j, p in hits)
    mid = offsets[len(offsets) // 2]
    good = [(j, p) for j, p in hits if abs((p - j) - mid) <= 40]
    start = max(0, min(p - j for j, p in good[:3]))
    end = min(len(toks), max(p + (len(query) - j) for j, p in good[-3:]))
    return start, end


# --------------------------------------------------------------------------- word slots in a page


@dataclass
class Slot:
    container: list
    node: int  # index of the str node in the container
    token: int  # index of the word among the node's tokens


def _flatten_spans(container: list) -> None:
    """Flagged Quran spans (poetry mistaken for Quran) and unclear readings become plain text."""
    out = []
    for node in container:
        if isinstance(node, Quran) and node.status == "flagged":
            out.append(" " + (node.ocr_text or node.text) + " ")
        elif isinstance(node, Unclear):
            out.append(node.text)
        elif isinstance(node, (Quote, Matn)):
            _flatten_spans(node.c)
            out.append(node)
        else:
            out.append(node)
    container[:] = out
    normalize_inlines(container)


def _containers(page: Page, notes: bool) -> list[list]:
    if notes:
        return [c for n in page.notes() for b in n.blocks for c in block_containers(b)]
    return [c for b in page.blocks for c in block_containers(b)]


def _tokens(text: str) -> list[str]:
    return re.split(r"(\s+)", text)


def collect_slots(containers: list[list]) -> tuple[list[Slot], dict[int, list[str]]]:
    slots: list[Slot] = []
    tokens: dict[int, list[str]] = {}
    for container in containers:
        stack = [container]
        while stack:
            c = stack.pop(0)
            for k, node in enumerate(c):
                if isinstance(node, (Quote, Matn)):
                    stack.append(node.c)
                elif isinstance(node, str):
                    toks = _tokens(node)
                    tokens[id(c) * 1000 + k] = toks
                    for t, tok in enumerate(toks):
                        if tok.strip() and normalize_for_match(tok):
                            slots.append(Slot(c, k, t))
    return slots, tokens


def transfer_diacritics(model_word: str, edition_word: str) -> tuple[str, bool]:
    """The edition's letters with the model's diacritics, when both have the same letters count."""
    def split(w: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for ch in w:
            if ch in DIACRITICS or ch == TATWEEL:
                if out:
                    out[-1] = (out[-1][0], out[-1][1] + ch)
            else:
                out.append((ch, ""))
        return out
    m, e = split(model_word), split(edition_word)
    if len(m) != len(e):
        return edition_word, any(ch in DIACRITICS for ch in model_word)
    rebuilt = []
    for (_ml, md), (el, ed) in zip(m, e):
        marks = md or ed
        if "ّ" in ed and "ّ" not in marks:  # the edition's shadda wins
            marks = "ّ" + marks
        rebuilt.append(el + marks)
    return "".join(rebuilt), False


def vowelled(model_text: str, span: list[Tok]) -> dict[int, str]:
    """Edition words (by position in ``span``) with the diacritics the model read on the page:
    the model's words are aligned with the edition's, and each matched pair with the same
    letters takes the model's marks (and keeps the edition's shadda). Words the model missed
    keep the edition's own marks."""
    from rapidfuzz import fuzz

    model_words = [w for w in model_text.split() if normalize_for_match(w)]
    model_norm = [normalize_for_match(w) for w in model_words]
    ed_norm = [t.norm for t in span]
    voc: dict[int, str] = {}
    for op in Levenshtein.opcodes(model_norm, ed_norm):
        if op.tag not in ("equal", "replace") or op.src_end - op.src_start != op.dest_end - op.dest_start:
            continue
        for a, b in zip(range(op.src_start, op.src_end), range(op.dest_start, op.dest_end)):
            if op.tag == "equal" or fuzz.ratio(model_norm[a], ed_norm[b]) >= 60:
                word, _ = transfer_diacritics(model_words[a], span[b].word)
                voc[b] = word
    return voc


def correct(containers: list[list], span: list[Tok], stats: Counter, *, remove: bool = True) -> None:
    """Correct the model's words in ``containers`` from the edition ``span``: replace misread
    words, insert missed ones, and (with ``remove``) drop words the edition does not have."""
    for c in containers:
        _flatten_spans(c)
    slots, tokens = collect_slots(containers)
    model_norm = [normalize_for_match(tokens[id(s.container) * 1000 + s.node][s.token]) for s in slots]
    edition_norm = [t.norm for t in span]
    stats["words"] += len(edition_norm)
    after: dict[int, list[str]] = {}  # slot index -> words to insert after it
    drop: set[int] = set()
    refs_after: dict[int, list[str]] = {}
    ops = Levenshtein.editops(edition_norm, model_norm)
    changed = {o.dest_pos for o in ops if o.tag == "replace"}
    # Equal words: copy markers. Replaced words: correct them.
    mapping = {}
    for blk in Levenshtein.opcodes(edition_norm, model_norm):
        if blk.tag == "equal":
            for d in range(blk.src_end - blk.src_start):
                mapping[blk.dest_start + d] = blk.src_start + d
    for o in ops:
        if o.tag == "replace":
            mapping[o.dest_pos] = o.src_pos
    for si, ei in mapping.items():
        slot = slots[si]
        toks = tokens[id(slot.container) * 1000 + slot.node]
        if si in changed:
            new, lost = transfer_diacritics(toks[slot.token], span[ei].word)
            toks[slot.token] = new
            stats["replaced"] += 1
            stats["undiacritised_words"] += int(lost)
        if span[ei].refs:
            refs_after[si] = span[ei].refs
    for o in ops:
        if o.tag == "delete" and EDGE <= o.src_pos < len(span) - EDGE and slots:
            target = max(0, min(o.dest_pos - 1, len(slots) - 1))
            after.setdefault(target, []).append(span[o.src_pos].word)
            if span[o.src_pos].refs:
                refs_after.setdefault(target, []).extend(span[o.src_pos].refs)
            stats["inserted"] += 1
            stats["undiacritised_words"] += 1
        elif remove and o.tag == "insert" and EDGE <= o.dest_pos < len(slots) - EDGE:
            drop.add(o.dest_pos)
            stats["removed"] += 1
    for si in drop:
        slot = slots[si]
        tokens[id(slot.container) * 1000 + slot.node][slot.token] = ""
    for si, words in after.items():
        slot = slots[si]
        toks = tokens[id(slot.container) * 1000 + slot.node]
        toks[slot.token] = (toks[slot.token] + " " + " ".join(words)).strip()
    # Markers: add a Ref after a word when the model has none nearby in that container.
    existing = {id(c): {n.n for n in c if isinstance(n, Ref)} for c in containers}
    ref_points: dict[tuple[int, int], list[tuple[int, str]]] = {}
    for si, signs in refs_after.items():
        slot = slots[si]
        for sign in signs:
            if sign not in existing.get(id(slot.container), set()):
                ref_points.setdefault((id(slot.container), slot.node), []).append((slot.token, sign))
                stats["markers_added"] += 1
    # Rebuild the containers' str nodes.
    seen = set()
    for slot in slots:
        key = (id(slot.container), slot.node)
        if key in seen:
            continue
        seen.add(key)
    for container in containers:
        stack = [container]
        while stack:
            c = stack.pop(0)
            rebuilt: list = []
            for k, node in enumerate(c):
                if isinstance(node, (Quote, Matn)):
                    stack.append(node.c)
                    rebuilt.append(node)
                    continue
                if not isinstance(node, str):
                    rebuilt.append(node)
                    continue
                toks = tokens.get(id(c) * 1000 + k, [node])
                points = sorted(ref_points.get((id(c), k), []))
                if not points:
                    rebuilt.append("".join(toks))
                    continue
                start = 0
                for t, sign in points:
                    rebuilt.append("".join(toks[start:t + 1]))
                    rebuilt.append(Ref(n=sign))
                    start = t + 1
                rebuilt.append("".join(toks[start:]))
            c[:] = [x for x in rebuilt if not (isinstance(x, str) and x == "")]
            normalize_inlines(c)
            for i, x in enumerate(c):
                if isinstance(x, str):
                    c[i] = re.sub(r"\s{2,}", " ", x)


def refine(span: tuple[int, int], query: list[str], toks: list[Tok], reach: int = 25) -> tuple[int, int]:
    """Trim the span to the words the page shows: a local alignment (Smith-Waterman) of the page's
    words with the edition around the span, where similar words count as matches (Tesseract
    misreads many). Words not on the page (a page number or header in the query, the
    neighbouring pages in the edition) fall outside the best-scoring stretch."""
    from rapidfuzz import fuzz

    lo, hi = max(0, span[0] - reach), min(len(toks), span[1] + reach)
    window = [t.norm for t in toks[lo:hi]]
    if not query or not window:
        return span
    m = len(window)
    prev = [0] * (m + 1)
    prev_start = [0] * (m + 1)
    best, best_end, best_start = 0, span[1] - lo, span[0] - lo
    for q in query:
        cur = [0] * (m + 1)
        cur_start = [0] * (m + 1)
        for d in range(1, m + 1):
            w = window[d - 1]
            sim = 2 if q == w else (1 if fuzz.ratio(q, w) >= 70 else -1)
            options = ((prev[d - 1] + sim, prev_start[d - 1] if prev[d - 1] > 0 else d - 1),
                       (prev[d] - 1, prev_start[d]), (cur[d - 1] - 1, cur_start[d - 1]))
            score, start = max(options, key=lambda o: o[0])
            if score > 0:
                cur[d], cur_start[d] = score, start
                if score > best:
                    best, best_end, best_start = score, d, start
        prev, prev_start = cur, cur_start
    return lo + best_start, lo + best_end


_HEADING_RE = re.compile(r"^[٠-٩0-9]{1,3}\s*[-–]\s*\S")


def _plain(word: str) -> str:
    """A word without diacritics or tatweel (targets for the letters-only instruction)."""
    return "".join(ch for ch in word if ch not in DIACRITICS and ch != TATWEEL)


def edition_main(span: list[Tok], voc: dict[int, str] | None = None) -> str:
    """Tags for a stretch of the edition's main text: prose lines, headings, verses, markers.
    ``voc`` gives words with diacritics by position in the span (see ``vowelled``); without it
    the words are letters only."""
    from xml.sax.saxutils import escape

    pos = {id(t): k for k, t in enumerate(span)}

    def words(toks: list[Tok]) -> str:
        out = []
        for t in toks:
            w = voc.get(pos[id(t)], t.word) if voc is not None else _plain(t.word)
            out.append(escape(w) + "".join(f'<ref n="{escape(r)}"/>' for r in t.refs))
        return " ".join(out)

    lines: list[list[Tok]] = []
    for tok in span:
        if not lines or tok.line != lines[-1][0].line:
            lines.append([tok])
        else:
            lines[-1].append(tok)
    out: list[str] = []
    verses: list[str] = []
    for line in lines:
        if line[0].half >= 0 or any(t.half == 1 for t in line):
            right = [t for t in line if t.half == 0]
            left = [t for t in line if t.half == 1]
            verses.append(f"<v><s>{words(right)}</s>" + (f"<a>{words(left)}</a>" if left else "") + "</v>")
            continue
        if verses:
            out.append("<poem>" + "".join(verses) + "</poem>")
            verses = []
        text = re.sub(r"^([٠-٩0-9]+)\*\s*", r"\1 • ", words(line))  # the edition's "٢٢٧*" is printed "٢٢٧ •"
        plain = " ".join(_plain(t.word) for t in line)
        if _HEADING_RE.match(plain) and len(line) <= 7 and "*" not in plain:
            out.append(f"<h2>{text}</h2>")
        else:
            out.append(f"<p>{text}</p>")
    if verses:
        out.append("<poem>" + "".join(verses) + "</poem>")
    return "\n".join(out)


def edition_notes(span: list[Tok], voc: dict[int, str] | None = None) -> str:
    from xml.sax.saxutils import escape

    pos = {id(t): k for k, t in enumerate(span)}

    def word(t: Tok) -> str:
        return escape(voc.get(pos[id(t)], t.word) if voc is not None else _plain(t.word))

    groups: list[list[Tok]] = []
    for tok in span:
        if not groups or tok.entry != groups[-1][0].entry:
            groups.append([tok])
        else:
            groups[-1].append(tok)
    notes = []
    for k, group in enumerate(groups):
        text = " ".join(word(t) for t in group)
        marker = group[0].lead
        if marker:
            notes.append(f'<note n="{escape(marker)}">{text}</note>')
        else:
            notes.append(f'<note cont="prev">{text}</note>' if k == 0 else f"<note>{text}</note>")
    return '<hashiya pos="bottom">\n' + "\n".join(notes) + "\n</hashiya>" if notes else ""


def rebuild_notes(page: Page, span: list[Tok]) -> int:
    """Replace the page's foot notes by the edition's (markers and text) for this span."""
    from pdf2epub.model import HashiyaRegion, Note, Paragraph

    groups: list[list[Tok]] = []
    for tok in span:
        if not groups or tok.entry != groups[-1][0].entry:
            groups.append([tok])
        else:
            groups[-1].append(tok)
    notes = []
    for k, group in enumerate(groups):
        marker = group[0].lead
        text = " ".join(t.word for t in group)
        note = Note(marker=marker, anchor_type="marker" if marker else "position",
                    cont=marker is None and k == 0, blocks=[Paragraph(c=[text])])
        notes.append(note)
    region = next((r for r in page.hashiya if r.region == "bottom" and not r.independent), None)
    if region is None:
        region = HashiyaRegion(region="bottom")
        page.hashiya.append(region)
    region.notes = notes
    return len(notes)


# --------------------------------------------------------------------------- main


def _query_text(wd: WorkDir, i: int, notes: bool) -> tuple[str, Page | None]:
    """Text used to locate a page: the model's output when there is one, else Tesseract's."""
    if wd.page_tags(i).exists():
        page = parse_tags(wd.page_tags(i).read_text(encoding="utf-8"), Page(index=i)).page
        containers = _containers(page, notes=notes)
        text = " ".join(" ".join(x if isinstance(x, str) else getattr(x, "text", "") for x in c) for c in containers)
        return text, page
    main_file, notes_file = split_paths(wd, i)
    if main_file.exists():  # tools/split_tesseract.py: main text and notes read apart
        return (notes_file if notes else main_file).read_text(encoding="utf-8"), None
    tess = wd.page_tesseract(i)
    if not tess.exists():
        return "", None
    raw = tess.read_text(encoding="utf-8")
    return raw, None  # main and notes together: each stream only matches its own part


def _clean_query(wd: WorkDir, i: int) -> bool:
    """Whether the page's query text holds one part only (so its span can be trimmed to it)."""
    return wd.page_tags(i).exists() or split_paths(wd, i)[0].exists()


_AR = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def renumber(main: str, notes: str | None) -> tuple[str, str | None]:
    """Number markers ١, ٢, ٣ … in page order, as printed (the edition numbers them by its own
    pages, which break elsewhere). Notes follow their markers in order; when the numbered notes
    and the markers do not pair up one to one, the notes are dropped (None)."""
    count = iter(range(1, 1000))
    main = re.sub(r'<ref n="[٠-٩0-9]+"/>', lambda m: f'<ref n="{str(next(count)).translate(_AR)}"/>', main)
    refs = len(re.findall(r'<ref n="[٠-٩]+"/>', main))
    if notes is None:
        return main, None
    if len(re.findall(r'<note n="[٠-٩0-9]+">', notes)) != refs:
        return main, None
    count = iter(range(1, 1000))
    notes = re.sub(r'<note n="[٠-٩0-9]+">', lambda m: f'<note n="{str(next(count)).translate(_AR)}">', notes)
    return main, notes


_BIDI = dict.fromkeys(map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\ufeff"))


def _clean_header(text: str) -> str:
    return " ".join(_plain(text.translate(_BIDI)).replace("ی", "ي").replace("ک", "ك").split())


def header_canon(wd: WorkDir) -> list[str]:
    """Running headers the model read on at least three pages (spelled as most often read):
    a header seen once is more likely a misreading than the book's title or chapter."""
    from rapidfuzz import fuzz

    counts: Counter = Counter()
    for i in wd.page_indices():
        if wd.page_tags(i).exists():
            page = parse_tags(wd.page_tags(i).read_text(encoding="utf-8"), Page(index=i)).page
            if page.running_header:
                counts[_clean_header(page.running_header)] += 1
    groups: list[list[str]] = []
    for h, _ in counts.most_common():
        for g in groups:
            if fuzz.ratio(normalize_for_match(h), normalize_for_match(g[0])) >= 85:
                g.append(h)
                break
        else:
            groups.append([h])
    return [g[0] for g in groups if sum(counts[h] for h in g) >= 3 and g[0]]


def confirmed_header(header: str | None, canon: list[str]) -> tuple[str | None, bool]:
    """(header to write, usable): a header must be one of the book's recurring headers."""
    from rapidfuzz import fuzz

    if not header:
        return None, True
    h = normalize_for_match(_clean_header(header))
    best = max(canon, key=lambda c: fuzz.ratio(h, normalize_for_match(c)), default=None)
    if best is not None and fuzz.ratio(h, normalize_for_match(best)) >= 85:
        return best, True
    return None, False


def _num(text: str | None) -> int | None:
    if not text or not re.fullmatch(r"[٠-٩0-9]{1,4}", text.strip()):
        return None
    return int(text.strip().translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))


def _tess_lines(wd: WorkDir, i: int) -> list[str]:
    tess = wd.page_tesseract(i)
    if not tess.exists():
        return []
    return [ln.strip() for ln in tess.read_text(encoding="utf-8").splitlines() if ln.strip()]


def page_number_candidates(wd: WorkDir) -> dict[int, set[int]]:
    """Numbers that may be the printed page number, by page index: what the model tagged as page
    number or header, and the model's reading of the top and bottom strips (tools/read_pagenums.py)
    or else lone numbers on Tesseract's first and last lines."""
    cands: dict[int, set[int]] = {}
    strips_file = wd.root / "pagenums.json"  # tools/read_pagenums.py: the model's reading of each strip
    strips = json.loads(strips_file.read_text(encoding="utf-8")) if strips_file.exists() else None
    for i in wd.page_indices():
        found: set[int] = set()
        if wd.page_tags(i).exists():
            page = parse_tags(wd.page_tags(i).read_text(encoding="utf-8"), Page(index=i)).page
            for text in (page.printed_page, page.running_header):
                if _num(text) is not None:
                    found.add(_num(text))
        if strips is not None:
            # The model sometimes reads the digits right to left (٥٤ for ٤٥): both are candidates
            # and the page sequence decides.
            for text in (strips.get(str(i)) or {}).values():
                for variant in (text, text[::-1]):
                    if _num(variant) is not None:
                        found.add(_num(variant))
            cands[i] = found
            continue
        lines = _tess_lines(wd, i)
        for ln in lines[:1] + lines[-1:]:
            if len(ln.split()) <= 7:
                found.update(n for n in map(_num, ln.split()) if n is not None)
        cands[i] = found
    return cands


def confirmed_pagenum(i: int, cands: dict[int, set[int]]) -> int | None:
    """The candidate that fits the sequence best: the most pages within five on either side read
    n-1, n+1, … (at least one). Small numbers (note or verse numbers) rarely line up as well as
    the page numbers do."""
    best, support = None, 0
    for n in cands.get(i, ()):
        k = sum(1 for d in range(-5, 6) if d and n + d in cands.get(i + d, ()))
        if k > support or (k == support and best is not None and n > best):
            best, support = n, k
    return best if support else None

def _arabic_digits(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))


def _spans_of(run: WorkDir, toks: list[Tok], grams: dict) -> list[tuple[int, int]]:
    spans = []
    for i in run.page_indices():
        text, _ = _query_text(run, i, notes=False)
        query = [w for w in normalize_for_match(text).split() if w]
        span = locate(query, toks, grams) if len(query) >= 6 else None
        if span:
            spans.append(span)
    return spans


def _crop_above_rule(image: Path, out: Path) -> bool:
    """The page cut just above the foot-note rule (main text only)."""
    import cv2

    from pdf2epub.layout.detect import _blocks, _main_block, _rule_y

    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return False
    main = _main_block(_blocks(gray))
    rule = _rule_y(gray, main, first=True) if main else None
    if rule is None:
        return False
    cv2.imwrite(str(out), gray[: max(10, rule - 4)])
    return True


def _pagenum_on_top(wd: WorkDir, i: int, n: int) -> bool:
    """Whether page number ``n`` is printed at the top (so it stays in a crop above the notes)."""
    strips_file = wd.root / "pagenums.json"
    if not strips_file.exists():
        return True  # books read by the model: page numbers print at the top in these editions
    strips = json.loads(strips_file.read_text(encoding="utf-8")).get(str(i)) or {}
    return _num(strips.get("top")) == n


def _disagreement(wd: WorkDir, i: int, main_tags: str) -> tuple[float, float] | None:
    """(share of the model's words with nothing like them in the target, share of the target's
    words with nothing like them in the model's reading), letters only and in any order; None
    without a model reading. Neither is a clean signal on its own (the model drops lines and
    mixes notes into the main text), but for another edition a high second share means the
    edition has words this print lacks (its editor's headings, variant readings)."""
    from pdf2epub.bench import _lenient
    from pdf2epub.checks.flags import find_repetition
    from pdf2epub.eval.metrics import page_text

    if not wd.page_tags(i).exists():
        return None
    tags = wd.page_tags(i).read_text(encoding="utf-8")
    if find_repetition(tags, 4):
        return None  # the model looped: its reading says nothing about the target
    model = parse_tags(tags, Page(index=i)).page
    target = parse_tags(main_tags, Page(index=i)).page
    ref, hyp = _lenient(page_text(target)).split(), _lenient(page_text(model)).split()
    # Extra words regardless of order (the model often reads verse column by column): a word
    # with nothing like it anywhere in the target.
    from rapidfuzz import fuzz, process

    def unseen(words: list[str], other: list[str]) -> float:
        vocab = sorted(set(other))
        miss = sum(1 for w in words if w not in vocab
                   and not process.extractOne(w, vocab, scorer=fuzz.ratio, score_cutoff=75))
        return miss / max(1, len(words))

    return unseen(hyp, ref), unseen(ref, hyp)


def coverage_problem(wd: WorkDir, i: int, target: str, region: bool) -> str | None:
    """Why a target does not cover the page, or None. The model's reading of the page is the
    yardstick (its errors are a few words in a hundred): a whole-page target must hold at least
    60% as many words, and at most 25% of the model's words may be missing from it (in any
    order: the model may read verse column by column); a region target must hold at least half
    the model's main-text words (the model sometimes reads notes as main text). Pages located in
    the wrong place, index pages and cut-off spans fail this."""
    from rapidfuzz import fuzz, process

    from pdf2epub.eval.metrics import page_text

    if not wd.page_tags(i).exists():
        return None
    model = parse_tags(wd.page_tags(i).read_text(encoding="utf-8"), Page(index=i)).page
    tgt = parse_tags(target, Page(index=i)).page
    norm = lambda t: [w for w in (normalize_for_match(x) for x in t.split()) if w]  # noqa: E731
    if region:
        m, t = norm(page_text(model)), norm(page_text(tgt))
        return "target covers too little of the page" if len(t) < 0.5 * len(m) else None
    m, t = norm(page_text(model, notes=True)), norm(page_text(tgt, notes=True))
    if len(t) < 0.6 * len(m):
        return "target covers too little of the page"
    vocab = sorted(set(t))
    missing = sum(1 for w in m if w not in vocab and not process.extractOne(w, vocab, scorer=fuzz.ratio, score_cutoff=75))
    return "page text missing from the target" if missing > 0.25 * max(1, len(m)) else None


def main_edition(pages_dir: Path, run_dir: Path, out: Path, *, main_only: bool = False,
                 exclude: list[Path] | None = None, max_extra: float = 1.0, max_missing: float = 1.0,
                 diacritics: str = "model") -> None:
    """Letters-only targets built from the edition's own lines (structure from the edition).

    A page whose notes are in the edition becomes a whole-page pair. A page with a foot-note
    rule but no notes in the edition (or ``main_only``: a different edition, whose notes differ)
    becomes a pair for the region above the rule. Every page needs a page number that fits the
    sequence, and keeps it when it is inside the image.
    """
    from xml.sax.saxutils import escape

    main_toks, note_toks = load_edition(pages_dir)
    grams = {"main": unique_grams(main_toks), "notes": unique_grams(note_toks)}
    banned = [sp for ex in (exclude or []) for sp in _spans_of(WorkDir(ex), main_toks, grams["main"])]
    wd = WorkDir(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    kept = excluded = 0
    why: Counter = Counter()
    modes: Counter = Counter()
    canon = header_canon(wd)
    nums = page_number_candidates(wd)
    print(f"recurring headers: {canon[:12]}")
    lines = []
    for entry in wd.load_manifest().pages:
        i = entry.index
        if entry.blank:
            continue
        header = None
        if wd.page_tags(i).exists():
            header = parse_tags(wd.page_tags(i).read_text(encoding="utf-8"), Page(index=i)).page.running_header
        if _num(header) is not None:
            header = None  # a page number read as a header
        found: dict[str, str] = {}
        overlap = False
        for key, toks in (("main", main_toks), ("notes", note_toks)):
            if key == "notes" and main_only:
                continue
            text, _ = _query_text(wd, i, notes=key == "notes")
            query = [w for w in normalize_for_match(text).split() if w]
            if len(query) < 6:
                continue
            span = locate(query, toks, grams[key])
            if span is None:
                continue
            if key == "main" and any(a < span[1] + 30 and span[0] - 30 < b for a, b in banned):
                overlap = True
                break
            if _clean_query(wd, i):
                span = refine(span, query, toks)
            elif key == "notes":
                continue  # notes located from a whole-page reading have no reliable edges
            piece = toks[span[0]:span[1]]
            if diacritics == "edition":
                voc: dict[int, str] | None = {}  # the edition's own words, diacritics and all
            elif diacritics == "model":
                voc = vowelled(text, piece) if wd.page_tags(i).exists() else None
                if voc is None:
                    continue  # diacritics come from the model's reading
            else:
                voc = None
            found[key] = edition_main(piece, voc) if key == "main" else edition_notes(piece, voc)
        if overlap:
            excluded += 1
            continue
        if "main" not in found:
            why["main text not found in the edition"] += 1
            continue
        found["main"], notes = renumber(found["main"], found.get("notes"))
        if notes is None:
            found.pop("notes", None)
        else:
            found["notes"] = notes
        pagenum = confirmed_pagenum(i, nums)
        if pagenum is None:
            why["no page number that fits the sequence"] += 1
            continue
        image = wd.abs(entry.image)
        crop = out / f"{i:04d}.main.png"
        region = (main_only or "notes" not in found) and _crop_above_rule(image, crop)
        if main_only and not region:
            why["no foot-note rule to crop at"] += 1
            continue
        if not region and "notes" not in found and "<ref " in found["main"]:
            why["markers but no notes"] += 1  # notes printed, but neither found nor cut off
            continue
        parts = [found["main"]]
        if main_only:
            # Another edition: its markers are not where this print has them, so the pair asks for
            # the words only (no markers, header or page number).
            # The edition's bracketed lines are its editor's headings, not in this print.
            text = re.sub(r'<ref n="[^"]*"/>', "", found["main"])
            text = re.sub(r"<p>\[[^\]<]*\]</p>\n?", "", text)
            parts = [re.sub(r"\s*«[٠-٩0-9]{1,3}»", "", text)]
            header = None
        if not region and "notes" in found:
            parts.append(found["notes"])
        if not main_only and (not region or _pagenum_on_top(wd, i, pagenum)):
            parts.append(f"<pagenum>{_arabic_digits(pagenum)}</pagenum>")
        if header:
            canon_header, _ = confirmed_header(header, canon)
            if canon_header:
                parts.insert(0, f"<header>{escape(canon_header)}</header>")
            elif normalize_for_match(_clean_header(header)) not in normalize_for_match(found["main"])[:300]:
                why["header not confirmed"] += 1
                continue
            # else: a heading of the main text that the model took for a header
        target = "\n".join(parts) + "\n"
        main_part = parts[1] if parts[0].startswith("<header>") else parts[0]
        disagree = _disagreement(wd, i, main_part)
        if disagree is not None and (disagree[0] > max_extra or disagree[1] > max_missing):
            why["model reading too far from the target"] += 1
            continue
        if not parse_tags(target, Page(index=i)).ok:
            why["target does not parse"] += 1
            continue
        problem = coverage_problem(wd, i, target, bool(region))
        if problem:
            why[problem] += 1
            continue
        instruction = "keep_diacritics" if diacritics != "none" else "no_diacritics"
        if region:
            image = crop
            instruction = ("text_only_main_region" if main_only
                           else "main_region" if diacritics != "none" else "no_diacritics_main_region")
        modes[instruction] += 1
        record = {"index": i, "image": str(image), "tags": target, "instruction": instruction,
                  "source": f"edition text from {pages_dir.name}, structure from the edition, "
                            + {"model": "diacritics from the model's reading", "edition": "diacritics from the edition",
                               "none": "diacritics removed"}[diacritics]}
        (out / f"{i:04d}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        lines.append(json.dumps({"image": str(image), "target": target, "instruction": instruction}, ensure_ascii=False))
        kept += 1
    (out / "pairs.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"kept {kept} pages {dict(modes)}; excluded (overlap the test set) {excluded}; left out: {dict(why)}")


def main() -> None:
    if sys.argv[1] == "--mode":
        mode, args = sys.argv[2], sys.argv[3:]
    else:
        mode, args = "edition", sys.argv[1:]
    if mode == "edition":
        flags = args[3:]
        exclude = [Path(flags[k + 1]) for k, f in enumerate(flags) if f == "--exclude"]
        def opt(name: str, default: float) -> float:
            return float(flags[flags.index(name) + 1]) if name in flags else default

        main_edition(Path(args[0]), Path(args[1]), Path(args[2]), main_only="--main-only" in flags, exclude=exclude,
                     max_extra=opt("--max-extra", 1.0), max_missing=opt("--max-missing", 1.0),
                     diacritics="none" if "--no-diacritics" in flags
                     else flags[flags.index("--diacritics") + 1] if "--diacritics" in flags else "model")
        return
    main_corrected(Path(args[0]), Path(args[1]), Path(args[2]))


def main_corrected(pages_dir: Path, run_dir: Path, out: Path) -> None:
    """The model's reading with its words corrected from the edition: for editions whose digital
    text lacks punctuation, diacritics or notes (OpenITI texts), so structure, punctuation and
    diacritics stay the model's. Pages with notes become pairs for the region above the foot-note
    rule (the notes would stay uncorrected); page numbers must fit the sequence."""
    from pdf2epub.checks.flags import find_repetition

    main_toks, _note_toks = load_edition(pages_dir)
    grams = unique_grams(main_toks)
    wd = WorkDir(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    nums = page_number_candidates(wd)
    canon = header_canon(wd)
    kept = 0
    why: Counter = Counter()
    modes: Counter = Counter()
    lines = []
    for entry in wd.load_manifest().pages:
        i = entry.index
        if entry.blank or not wd.page_tags(i).exists():
            continue
        raw = wd.page_tags(i).read_text(encoding="utf-8")
        if find_repetition(raw, 3):
            why["the model looped"] += 1
            continue
        page = parse_tags(raw, Page(index=i, width=entry.width, height=entry.height)).page
        # The editor's headings are not in a bare text of the work: leave them as read.
        containers = [c for b in page.blocks if getattr(b, "type", "") != "heading" for c in block_containers(b)]
        text = " ".join(" ".join(x if isinstance(x, str) else getattr(x, "text", "") for x in c) for c in containers)
        query = [w for w in normalize_for_match(text).split() if w]
        span = locate(query, main_toks, grams) if len(query) >= 6 else None
        if span is None:
            why["main text not found in the edition"] += 1
            continue
        span = refine(span, query, main_toks)
        stats: Counter = Counter()
        # No removals: a word the bare text lacks is as likely the editor's (in brackets, a
        # heading, a gloss) as the model's invention.
        correct(containers, main_toks[span[0]:span[1]], stats, remove=False)
        if stats["words"] and (stats["replaced"] + stats["inserted"]) / stats["words"] > MAX_EDIT_RATE:
            why["too many corrections"] += 1
            continue
        pagenum = confirmed_pagenum(i, nums)
        if pagenum is None:
            why["no page number that fits the sequence"] += 1
            continue
        if page.running_header:
            canon_header, _ = confirmed_header(page.running_header, canon)
            if canon_header:
                page.running_header = canon_header
            elif _num(page.running_header) is None:
                why["header not confirmed"] += 1
                continue
            else:
                page.running_header = None
        image = wd.abs(entry.image)
        crop = out / f"{i:04d}.main.png"
        has_notes = bool(page.hashiya) or any(True for _ in page.notes())
        instruction = "keep_diacritics"
        if has_notes or _crop_above_rule(image, crop):
            if not crop.exists() and not _crop_above_rule(image, crop):
                why["notes but no foot-note rule to crop at"] += 1
                continue
            page.hashiya = []
            image, instruction = crop, "main_region"
            on_top = _pagenum_on_top(wd, i, pagenum)
            page.printed_page = _arabic_digits(pagenum) if on_top else None
        else:
            page.printed_page = _arabic_digits(pagenum)
        target = serialize_page(page)
        if not parse_tags(target, Page(index=i)).ok:
            why["target does not parse"] += 1
            continue
        problem = coverage_problem(wd, i, target, instruction == "main_region")
        if problem:
            why[problem] += 1
            continue
        modes[instruction] += 1
        record = {"index": i, "image": str(image), "tags": target, "instruction": instruction, "stats": dict(stats),
                  "source": f"model reading with its words corrected from {pages_dir.name}"}
        (out / f"{i:04d}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        lines.append(json.dumps({"image": str(image), "target": target, "instruction": instruction}, ensure_ascii=False))
        kept += 1
    (out / "pairs.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"kept {kept} pages {dict(modes)}; left out: {dict(why)}")


if __name__ == "__main__":
    main()
