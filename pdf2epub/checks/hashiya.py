"""Hashiya (حاشية): link every note to the words it explains (spec §6.4-6.5).

Links are resolved in this order and every note ends up linked, guessed or
flagged:

1. by marker: printed refs and notes pair per page by their sign, in order;
   a marker the model left as plain text ("(١)") is turned into a ref;
2. by lemma (قوله): the quoted words are searched in this page's text, then
   the previous page's; all lemma notes of a page are aligned together so
   their anchors follow note order; a generated ref goes after the words;
3. by position: a margin note with neither links to the main-text line level
   with it (status "guessed");
4. continuations: a note opening a page as the rest of the previous page's
   last note is joined to it;
5. layers: notes on the commentary link into the notes of the layer above.

Manual links made in the review screen are never moved. Reviewed pages are
read-only except for generated refs anchoring notes of other pages.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from ..arabic import (
    normalize_for_match,
    normalize_marker,
    normalize_with_map,
    parse_number,
    source_span,
)
from ..config import BookConfig
from ..model import (
    BlockText,
    FlagCode,
    Note,
    NoteLink,
    Page,
    Paragraph,
    Ref,
    block_containers,
    iter_blocks,
    next_ref_id,
    normalize_inlines,
    walk_inline_slots,
)

log = logging.getLogger(__name__)
STAGE = "hashiya"

_QAWLUH_RE = re.compile(r"^\s*و?قوله\s*(?:تعالى|عليه السلام|صلى الله عليه وسلم|ﷺ|رحمه الله)?\s*[:：]?\s*")
_OPENERS = {"(": ")", "«": "»", "[": "]", "{": "}", '"': '"', "“": "”", "﴿": "﴾", "﴾": "﴿"}
_ETC_RE = re.compile(r"\s*(?:إلخ|الخ|إلى آخره|الى آخره|إلخ\.|ا\.?هـ)\s*$")
_TERMINAL = tuple(".!؟?:…»)]")


# --------------------------------------------------------------------------- helpers


def _note_text(note: Note) -> str:
    return " ".join(BlockText(b).text for b in note.blocks)


def _all_notes(pages: list[Page]) -> dict[str, tuple[Page, Note]]:
    return {n.id: (p, n) for p in pages for n in p.notes()}


def _iter_refs_in(blocks: list[object]):
    for block in blocks:
        for container in block_containers(block):
            for c, i, node in walk_inline_slots(container):
                if isinstance(node, Ref):
                    yield block, c, i, node


def _target_blocks(page: Page, layer: int) -> list[object]:
    if layer <= 1:
        return list(page.blocks)
    return [b for n in page.notes() if n.layer == layer - 1 for b in n.blocks]


def _is_editable(page: Page, only: set[int] | None) -> bool:
    return page.status != "reviewed" and (only is None or page.index in only)


def _marker_kind(marker: str) -> str:
    core = normalize_marker(marker)
    if core.isdigit():
        return "numbers"
    if all(ch in "*†‡§#" for ch in core):
        return "symbols"
    return "letters"


def _set_link(note: Note, page: Page, block: object, ref: Ref, method: str, status: str = "linked",
              score: float | None = None) -> None:
    bt = BlockText(block)
    note.link = NoteLink(status=status, method=method, page=page.index, block=getattr(block, "id", None),
                         ref=ref.id, offset=bt.ref_offsets.get(ref.id), score=score,
                         anchor=note.link.anchor if note.link.method == "manual" else None)
    ref.note = note.id


# --------------------------------------------------------------------------- lemma opening


def _split_lemma_opening(text: str) -> tuple[str | None, int]:
    """Lemma and the length of the "قوله: (…)" opening at the start of ``text``."""
    m = _QAWLUH_RE.match(text)
    if not m or not text[: m.end()].strip():
        return None, 0
    rest = text[m.end():]
    pos = m.end()
    if rest[:1] in _OPENERS:
        closer = _OPENERS[rest[0]]
        end = rest.find(closer, 1)
        if end > 0:
            lemma = rest[1:end]
            pos += end + 1
            tail = text[pos:]
            etc = re.match(r"\s*(?:إلخ|الخ|إلى آخره)\.?", tail)
            if etc:
                pos += etc.end()
            colon = re.match(r"\s*[:：\-–]\s*", text[pos:])
            if colon:
                pos += colon.end()
            return _ETC_RE.sub("", lemma).strip() or None, pos
    # Unbracketed: up to a colon, or "إلخ", or the explanation's first word (أي / يعني).
    stop = re.search(r"\s*[:：]\s*|\s+(?:إلخ|الخ)\b\.?\s*|\s+(?=(?:أي|يعني|المراد)\b)", rest)
    if stop:
        lemma = rest[: stop.start()]
        return lemma.strip() or None, pos + stop.end()
    words = rest.split()
    lemma = " ".join(words[:4])
    return lemma or None, pos + len(rest[: len(lemma)])


def extract_lemma(note: Note) -> str | None:
    """The قوله lemma at the start of a note's text, when the model gave none."""
    if not note.blocks:
        return None
    lemma, _ = _split_lemma_opening(_note_text(note))
    return lemma


def _strip_lemma_opening(note: Note) -> None:
    """Set ``lemma`` from a "قوله: (…)" opening and remove the opening from the text,
    so the EPUB's bold opening is not printed twice."""
    if not note.blocks or not isinstance(note.blocks[0], Paragraph):
        return
    para = note.blocks[0]
    first = next((i for i, n in enumerate(para.c) if not (isinstance(n, str) and not n.strip())), None)
    if first is None or not isinstance(para.c[first], str):
        return
    head = para.c[first]
    lemma, length = _split_lemma_opening(head)
    if lemma is None:
        return
    if note.lemma is None:
        note.lemma = lemma
    elif normalize_for_match(note.lemma) != normalize_for_match(lemma):
        return  # the text opens with a different quotation; keep it
    para.c[first] = head[length:].lstrip(" :：-–")
    normalize_inlines(para.c)
    if not note.marker:
        note.anchor_type = "lemma"


# --------------------------------------------------------------------------- text index for lemmas


@dataclass
class _Seg:
    order: int
    page: Page
    block: object
    text: str
    norm: str
    idx: list[int]


def _segments(pages: list[Page], layer: int, current: Page, lookback: int) -> list[_Seg]:
    by_index = {p.index: p for p in pages}
    chain = []
    for k in range(lookback, 0, -1):
        prev = by_index.get(current.index - k)
        if prev is not None and all(by_index.get(current.index - j) is not None for j in range(1, k)):
            chain.append(prev)
    chain.append(current)
    segs: list[_Seg] = []
    for page in chain:
        for block in _target_blocks(page, layer):
            text = BlockText(block).text
            norm, idx = normalize_with_map(text)
            segs.append(_Seg(len(segs), page, block, text, norm, idx))
    return segs


def _word_bounded(norm: str, start: int, end: int) -> bool:
    return (start == 0 or norm[start - 1] == " ") and (end == len(norm) or norm[end] == " ")


def _candidates(lemma_norm: str, segs: list[_Seg], threshold: float, current: Page) -> list[tuple[float, int, int, int]]:
    """(score, segment order, norm start, norm end) for each place the lemma matches."""
    out: list[tuple[float, int, int, int]] = []
    if not lemma_norm:
        return out
    for seg in segs:
        start = seg.norm.find(lemma_norm)
        while start >= 0:
            end = start + len(lemma_norm)
            if _word_bounded(seg.norm, start, end):
                out.append((1.0 + (0.001 if seg.page is current else 0.0), seg.order, start, end))
            start = seg.norm.find(lemma_norm, start + 1)
    if out:
        return out
    for seg in segs:
        if not seg.norm:
            continue
        al = fuzz.partial_ratio_alignment(lemma_norm, seg.norm, score_cutoff=threshold * 100 - 15)
        if al is None:
            continue
        s, e = al.dest_start, al.dest_end
        # Snap to word boundaries.
        while s > 0 and seg.norm[s - 1] != " ":
            s -= 1
        while e < len(seg.norm) and seg.norm[e] != " ":
            e += 1
        score = fuzz.ratio(lemma_norm, seg.norm[s:e]) / 100.0
        out.append((score, seg.order, s, e))
    return out


def _align(cands: list[list[tuple[float, int, int, int]]], threshold: float) -> list[tuple[float, int, int, int] | None]:
    """Choose one candidate per note so positions never go backwards, maximising the score.

    A note may take no candidate (score 0). O(n * c^2), with few notes per page.
    """
    n = len(cands)
    # best[i][j]: (total, back pointer) for note i choosing candidate j (j == -1: none)
    best: list[dict[int, tuple[float, tuple[int, int] | None]]] = []
    for i in range(n):
        row: dict[int, tuple[float, tuple[int, int] | None]] = {}
        options = list(enumerate(cands[i])) + [(-1, None)]
        for j, cand in options:
            gain = cand[0] if cand is not None and cand[0] >= threshold else (cand[0] * 0.5 if cand else 0.0)
            if i == 0:
                row[j] = (gain, None)
                continue
            top: tuple[float, tuple[int, int] | None] = (-1e9, None)
            for pj, (ptotal, _bp) in best[i - 1].items():
                prev_pos = _pos_of(cands, best, i - 1, pj)
                if cand is not None and prev_pos is not None and (cand[1], cand[2]) < prev_pos:
                    continue
                if ptotal > top[0]:
                    top = (ptotal, (i - 1, pj))
            if top[0] > -1e9:
                row[j] = (top[0] + gain, top[1])
        best.append(row)
    if not best:
        return []
    j = max(best[-1], key=lambda k: best[-1][k][0])
    chosen: list[tuple[float, int, int, int] | None] = [None] * n
    i = n - 1
    while i >= 0:
        chosen[i] = cands[i][j] if j >= 0 else None
        bp = best[i][j][1]
        if bp is None:
            break
        i, j = bp
    return chosen


def _pos_of(cands, best, i: int, j: int) -> tuple[int, int] | None:
    """Position of the last chosen candidate up to note i (following back pointers through 'none')."""
    while i >= 0:
        if j >= 0:
            c = cands[i][j]
            return (c[1], c[2])
        bp = best[i][j][1]
        if bp is None:
            return None
        i, j = bp
    return None


def _insert_anchor(seg: _Seg, norm_start: int, norm_end: int, note: Note) -> Ref:
    a, b = source_span(seg.text, seg.idx, norm_start, norm_end)
    ref = Ref(id=next_ref_id(seg.page), generated=True, note=note.id)
    BlockText(seg.block).insert_at(b, ref)
    return ref


# --------------------------------------------------------------------------- steps


def _cleanup(pages: list[Page], relink: set[str], notes: dict[str, tuple[Page, Note]]) -> None:
    """Delete generated refs that are stale or belong to notes about to be relinked."""
    for page in pages:
        for block, _owner in iter_blocks(page):
            for container in block_containers(block):
                stack = [container]
                while stack:
                    lst = stack.pop()
                    keep = []
                    for node in lst:
                        if hasattr(node, "c") and isinstance(getattr(node, "c"), list) and not isinstance(node, Ref):
                            stack.append(node.c)
                        if isinstance(node, Ref) and node.generated:
                            owner = notes.get(node.note or "")
                            manual = owner is not None and owner[1].link.method == "manual" and owner[1].link.ref == node.id
                            stale = owner is None or owner[1].link.ref != node.id
                            if not manual and (stale or node.note in relink):
                                continue
                        keep.append(node)
                    lst[:] = keep
                    normalize_inlines(lst)
        # Printed refs pointing at notes being relinked lose that pointer.
        for _b, _c, _i, ref in _iter_refs_in([b for b, _o in iter_blocks(page)]):
            if not ref.generated and ref.note in relink:
                ref.note = None


def _resolve(pages: list[Page], ref_id: str | None, page_index: int | None) -> bool:
    if not ref_id:
        return False
    for page in pages:
        if page_index is not None and page.index != page_index:
            continue
        for _b, _c, _i, ref in _iter_refs_in([b for b, _o in iter_blocks(page)]):
            if ref.id == ref_id:
                return True
    return False


def assign_kinds(page: Page, cfg: BookConfig) -> None:
    for note in page.notes():
        if note.kind != "unknown":
            continue
        rule = cfg.text.editor_markers
        if note.marker and rule != "none" and _marker_kind(note.marker) == rule:
            note.kind = "editor"
        else:
            note.kind = cfg.text.note_kind_default


def pair_markers(page: Page, layer: int = 1) -> None:
    """Pair printed refs with marker notes of one layer on one page (spec §6.4 step 1)."""
    notes = [n for n in page.notes() if n.layer == layer and n.marker and n.link.method != "manual"
             and not n.cont]
    blocks = _target_blocks(page, layer)
    refs = [(b, r) for b, _c, _i, r in _iter_refs_in(blocks) if not r.generated and r.n]
    by_key: dict[str, list[tuple[object, Ref]]] = {}
    for block, ref in refs:
        by_key.setdefault(normalize_marker(ref.n or ""), []).append((block, ref))
    unpaired: list[Note] = []
    for note in notes:
        queue = by_key.get(normalize_marker(note.marker or ""))
        while queue and queue[0][1].note not in (None, note.id):
            queue.pop(0)
        if queue:
            block, ref = queue.pop(0)
            _set_link(note, page, block, ref, "marker")
        else:
            unpaired.append(note)
    for note in unpaired:
        if not _marker_from_text(page, note, blocks):
            note.link = NoteLink(status="flagged", method="marker")
            page.add_flag(FlagCode.NOTE_WITHOUT_REF, f"no marker ({note.marker}) for this note in the text",
                          target=note.id, stage=STAGE)
    for queue in by_key.values():
        for _block, ref in queue:
            if ref.note is None:
                page.add_flag(FlagCode.REF_WITHOUT_NOTE, f"marker ({ref.n}) has no note on this page",
                              target=ref.id, stage=STAGE)


def _marker_from_text(page: Page, note: Note, blocks: list[object]) -> bool:
    """A marker the model left as plain text, e.g. "(١)", becomes a Ref for the note."""
    value = parse_number(note.marker or "")
    signs = {note.marker or ""}
    if value is not None:
        signs |= {str(value), "".join("٠١٢٣٤٥٦٧٨٩"[int(d)] for d in str(value))}
    pattern = re.compile(r"[(\[](?:" + "|".join(re.escape(s) for s in signs if s) + r")[)\]]")
    for block in blocks:
        bt = BlockText(block)
        m = pattern.search(bt.text)
        if not m:
            continue
        ref = Ref(id=next_ref_id(page), n=note.marker)
        start = m.start() - 1 if m.start() > 0 and bt.text[m.start() - 1] == " " else m.start()
        if bt.wrap_range(start, m.end(), lambda nodes: ref) is None and \
                BlockText(block).wrap_range(m.start(), m.end(), lambda nodes: ref) is None:
            continue
        _set_link(note, page, block, ref, "marker")
        page.add_flag(FlagCode.NOTE_WITHOUT_REF, f"marker ({note.marker}) was plain text; linked",
                      severity="info", target=note.id, stage=STAGE)
        return True
    return False


def link_lemmas(pages: list[Page], i: int, cfg: BookConfig, layer: int = 1) -> None:
    """Link the lemma notes of ``pages[i]`` (spec §6.4 step 2)."""
    page = pages[i]
    notes = [n for n in page.notes() if n.layer == layer and n.anchor_type == "lemma"
             and n.link.method != "manual" and not n.cont]
    if not notes:
        return
    lookback = cfg.checks.lemma_lookback_pages if layer == 1 else 0
    segs = _segments(pages, layer, page, lookback)
    threshold = cfg.checks.lemma_threshold
    lemmas = [normalize_for_match(re.sub(r"\s*(?:إلخ|الخ)\.?\s*$", "", n.lemma or "")) for n in notes]
    cands = [_candidates(lem, segs, threshold, page) for lem in lemmas]
    chosen = _align(cands, threshold)
    for note, cand in zip(notes, chosen):
        if cand is None:
            note.link = NoteLink(status="flagged", method="lemma")
            page.add_flag(FlagCode.LEMMA_NOT_FOUND, f"«{note.lemma}» not found in the text",
                          target=note.id, stage=STAGE)
            continue
        score, order, s, e = cand
        seg = segs[order]
        if score < threshold:
            note.link = NoteLink(status="flagged", method="lemma", score=round(score, 3))
            page.add_flag(FlagCode.LEMMA_LOW_SCORE, f"«{note.lemma}» matched only weakly ({score:.2f})",
                          target=note.id, stage=STAGE)
            continue
        ref = _insert_anchor(seg, s, e, note)
        _set_link(note, seg.page, seg.block, ref, "lemma", score=round(min(score, 1.0), 3))


def link_positions(page: Page, layer: int = 1) -> None:
    """Margin notes with neither marker nor lemma link to the main-text line level with them."""
    for region in page.hashiya:
        notes = [n for n in region.notes if n.layer == layer and n.anchor_type == "position"
                 and not n.cont and n.link.method != "manual" and n.link.status == "unlinked"]
        if not notes:
            continue
        if region.region == "bottom":
            for note in notes:
                if note.continues_from:
                    continue
                note.link = NoteLink(status="flagged", method="position")
                page.add_flag(FlagCode.NOTE_WITHOUT_REF, "a note with no marker and no lemma",
                              target=note.id, stage=STAGE)
            continue
        blocks = _target_blocks(page, layer)
        if not region.bbox or not page.height or not blocks:
            for note in notes:
                note.link = NoteLink(status="flagged", method="position")
                page.add_flag(FlagCode.POSITION_GUESS, "margin note without position information",
                              target=note.id, stage=STAGE)
            continue
        top, bottom = (page.main_bbox[1], page.main_bbox[3]) if page.main_bbox else (0.0, float(page.height))
        texts = [BlockText(b) for b in blocks]
        total = sum(len(t.text) for t in texts) or 1
        y0, y1 = region.bbox[1], region.bbox[3]
        for k, note in enumerate(notes):
            y = y0 + (k + 0.5) / len(notes) * (y1 - y0)
            frac = min(1.0, max(0.0, (y - top) / max(bottom - top, 1.0)))
            target = int(frac * total)
            acc = 0
            for block, bt in zip(blocks, texts):
                if acc + len(bt.text) >= target:
                    off = target - acc
                    nxt = bt.text.find(" ", off)
                    off = len(bt.text) if nxt < 0 else nxt
                    ref = Ref(id=next_ref_id(page), generated=True, note=note.id)
                    bt.insert_at(off, ref)
                    _set_link(note, page, block, ref, "position", status="guessed")
                    page.add_flag(FlagCode.POSITION_GUESS, "linked to the line level with the note (check)",
                                  severity="info", target=note.id, stage=STAGE)
                    break
                acc += len(bt.text) + 1
            texts = [BlockText(b) for b in blocks]


def link_continuations(pages: list[Page], only: set[int] | None = None) -> None:
    """Join notes that continue the previous page's last note (spec §6.4 step 4)."""
    by_index = {p.index: p for p in pages}
    for page in pages:
        if not _is_editable(page, only):
            continue
        prev = by_index.get(page.index - 1)
        for region in page.hashiya:
            if region.independent or not region.notes:
                continue
            first = region.notes[0]
            prev_note = None
            if prev is not None:
                same = [r for r in prev.hashiya if r.region == region.region and r.notes and not r.independent]
                pool = same[-1].notes if same else [n for r in prev.hashiya for n in r.notes]
                prev_note = pool[-1] if pool else None
            implicit = (not first.marker and not first.lemma and first.anchor_type == "position"
                        and region.region == "bottom" and prev_note is not None
                        and (prev_note.cont_next or not _note_text(prev_note).rstrip().endswith(_TERMINAL)))
            if not (first.cont or implicit):
                continue
            if prev_note is None:
                first.link = NoteLink(status="flagged", method="continuation")
                page.add_flag(FlagCode.CONTINUATION_ORPHAN, "note continues a note from a missing previous page",
                              target=first.id, stage=STAGE)
                continue
            first.cont = True
            first.continues_from = prev_note.id
            prev_note.continues_to = first.id
            first.link = NoteLink(status="linked", method="continuation", page=prev.index if prev else None)


def _check_numbering(pages: list[Page], page: Page) -> None:
    by_index = {p.index: p for p in pages}
    for region in page.hashiya:
        nums = [(n, parse_number(n.marker or "")) for n in region.notes if n.marker and n.kind != "editor"]
        nums = [(n, v) for n, v in nums if v is not None]
        if not nums:
            continue
        values = [v for _n, v in nums]
        prev = by_index.get(page.index - 1)
        prev_last = None
        if prev is not None:
            pv = [parse_number(n.marker or "") for r in prev.hashiya if r.region == region.region
                  for n in r.notes if n.marker]
            pv = [v for v in pv if v is not None]
            prev_last = pv[-1] if pv else None
        start_ok = values[0] == 1 or (prev_last is not None and values[0] == prev_last + 1)
        if not start_ok or any(b != a + 1 for a, b in zip(values, values[1:])):
            page.add_flag(FlagCode.NOTE_NUMBERING_GAP, f"note numbers run {values}: a note may be missing",
                          target=region.id, stage=STAGE)


def _check_order(page: Page) -> None:
    """Anchors must follow note order; anchors a few characters apart count as the same place."""
    where: dict[str, tuple[int, int]] = {}
    for k, block in enumerate(page.blocks):
        for ref_id, off in BlockText(block).ref_offsets.items():
            where[ref_id] = (k, off)
    for region in page.hashiya:
        seq = [where[n.link.ref] for n in region.notes
               if n.layer == 1 and n.link.page == page.index and n.link.ref in where]
        for (b1, o1), (b2, o2) in zip(seq, seq[1:]):
            if b2 < b1 or (b2 == b1 and o2 < o1 - 3):
                page.add_flag(FlagCode.NOTE_ORDER, "anchors in the text are not in the order of the notes",
                              target=region.id, stage=STAGE)
                break


def _replace_manual(pages: list[Page], page: Page, note: Note) -> None:
    """Re-place a manual link whose anchor ref disappeared (e.g. after a re-parse)."""
    if _resolve(pages, note.link.ref, note.link.page):
        return
    anchor = normalize_for_match(note.link.anchor or "")
    target = next((p for p in pages if p.index == note.link.page), page)
    if anchor:
        count = 0
        for block in target.blocks:
            bt = BlockText(block)
            norm, idx = normalize_with_map(bt.text)
            start = norm.find(anchor)
            while start >= 0:
                if _word_bounded(norm, start, start + len(anchor)):
                    count += 1
                    if count == max(1, note.link.occurrence):
                        _a, b = source_span(bt.text, idx, start, start + len(anchor))
                        ref = Ref(id=next_ref_id(target), generated=True, note=note.id)
                        bt.insert_at(b, ref)
                        anchor_text, occ = note.link.anchor, note.link.occurrence
                        _set_link(note, target, block, ref, "manual")
                        note.link.anchor, note.link.occurrence = anchor_text, occ
                        return
                start = norm.find(anchor, start + 1)
    note.link = NoteLink(status="flagged", method="manual", anchor=note.link.anchor,
                         occurrence=note.link.occurrence)
    page.add_flag(FlagCode.LEMMA_NOT_FOUND, "the words of a manual link were not found again",
                  target=note.id, stage=STAGE)


def merge_wrapped_lines(page: Page) -> int:
    """Merge notes that are really a wrapped line of the note above.

    Models sometimes turn the second line of a long foot note into a new قوله note whose
    "lemma" is simply that line's first words. Such a note (lemma equals the start of its own
    text, no قوله opening, no marker) is appended to the previous note of its region.
    """
    merged = 0
    for region in page.hashiya:
        keep: list[Note] = []
        for note in region.notes:
            text = normalize_for_match(_note_text(note))
            lemma = normalize_for_match(note.lemma or "")
            wrapped = (lemma and not note.marker and text.startswith(lemma)
                       and not _QAWLUH_RE.match(_note_text(note)) and keep and keep[-1].blocks)
            if wrapped:
                prev = keep[-1]
                last = prev.blocks[-1]
                first = note.blocks[0] if note.blocks else None
                if isinstance(last, Paragraph) and isinstance(first, Paragraph):
                    last.c.extend([" ", *first.c])
                    normalize_inlines(last.c)
                    prev.blocks.extend(note.blocks[1:])
                else:
                    prev.blocks.extend(note.blocks)
                merged += 1
                continue
            keep.append(note)
        region.notes = keep
    return merged


def link_notes(pages: list[Page], cfg: BookConfig, *, only: set[int] | None = None) -> None:
    """Resolve every note link on the editable pages (see the module docstring)."""
    pages = sorted(pages, key=lambda p: p.index)
    notes = _all_notes(pages)
    editable = [p for p in pages if _is_editable(p, only)]
    relink = {n.id for p in editable for n in p.notes() if n.link.method != "manual"}
    for page in editable:
        page.clear_flags(STAGE)
        if merge_wrapped_lines(page):
            notes = _all_notes(pages)
        assign_kinds(page, cfg)
        for note in page.notes():
            _strip_lemma_opening(note)
            note.anchor_type = "marker" if note.marker else "lemma" if note.lemma else "position"
            if note.link.method != "manual":
                note.link = NoteLink()
                note.continues_from = None
            # continues_to is set by the next page's pass
        for note in page.notes():
            if note.continues_to and note.continues_to not in notes:
                note.continues_to = None
    _cleanup(pages, relink, notes)
    # Stale incoming links on read-only pages (their anchor vanished) are flagged, not moved.
    for page in pages:
        if page in editable:
            continue
        for note in page.notes():
            if note.link.status in ("linked", "guessed") and note.link.method not in ("continuation",) \
                    and not _resolve(pages, note.link.ref, note.link.page):
                log.info("note %s on reviewed page has lost its anchor", note.id)
    link_continuations(pages, only)
    index_of = {p.index: k for k, p in enumerate(pages)}
    for page in editable:
        layers = sorted({n.layer for n in page.notes()}) or [1]
        for layer in layers:
            for note in page.notes():
                if note.layer == layer and note.link.method == "manual":
                    _replace_manual(pages, page, note)
            pair_markers(page, layer)
            link_lemmas(pages, index_of[page.index], cfg, layer)
            link_positions(page, layer)
        _check_numbering(pages, page)
        _check_order(page)
