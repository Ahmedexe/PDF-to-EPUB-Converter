"""Scores for one page and a whole test set (spec §9).

All text is normalised the same way before scoring (NFC, tatweel removed,
spaces unified); CER is reported with diacritics removed (the main target) and
kept (reported separately, since VLMs lose much more accuracy on tashkeel).
"""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from rapidfuzz.distance import Levenshtein

from ..arabic import normalize_for_match, normalize_for_score, normalize_marker
from ..model import BlockText, Note, Page, Poem, Quran, block_containers, inline_text, iter_blocks, walk_inlines


def _dist(ref: str, hyp: str) -> int:
    return Levenshtein.distance(ref, hyp)


def cer(ref: str, hyp: str) -> float:
    """Character error rate (edits / reference length)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return _dist(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return Levenshtein.distance(r, h) / len(r)


def page_text(page: Page, *, notes: bool = False, main: bool = True) -> str:
    """Text in reading order: main blocks, then (optionally) notes. Header and page number excluded."""
    parts = []
    if main:
        parts.extend(BlockText(b).text for b in page.blocks)
        parts.extend(BlockText(b).text for r in page.hashiya for b in r.blocks)
    if notes:
        for note in page.notes():
            parts.append(" ".join(BlockText(b).text for b in note.blocks))
    return "\n".join(p for p in parts if p)


def _norm(text: str, diacritics: bool) -> str:
    return normalize_for_score(text, keep_diacritics=diacritics)


def _spans(page: Page) -> list[Quran]:
    return [n for b, _o in iter_blocks(page) for c in block_containers(b) for n in walk_inlines(c)
            if isinstance(n, Quran)]


def _anchor_context(page: Page, note_id: str) -> str | None:
    """The three (normalised) words before the anchor of ``note_id`` in the main text."""
    for block in page.blocks:
        bt = BlockText(block)
        for c in block_containers(block):
            for node in walk_inlines(c):
                if getattr(node, "note", None) == note_id and getattr(node, "id", None) in bt.ref_offsets:
                    before = bt.text[: bt.ref_offsets[node.id]]
                    return " ".join(normalize_for_match(before).split()[-3:])
    return None


def _match_notes(truth: list[Note], pred: list[Note]) -> list[tuple[Note, Note]]:
    pairs: list[tuple[Note, Note]] = []
    used: set[str] = set()
    for t in truth:
        t_text = normalize_for_match(" ".join(BlockText(b).text for b in t.blocks))
        best, best_score = None, 0.0
        for p in pred:
            if p.id in used:
                continue
            p_text = normalize_for_match(" ".join(BlockText(b).text for b in p.blocks))
            score = 1 - cer(t_text, p_text) if t_text else 0.0
            if t.marker and p.marker and normalize_marker(t.marker) == normalize_marker(p.marker):
                score += 0.5
            if t.lemma and p.lemma and normalize_for_match(t.lemma) == normalize_for_match(p.lemma):
                score += 0.5
            if score > best_score:
                best, best_score = p, score
        if best is not None and best_score >= 0.6:
            used.add(best.id)
            pairs.append((t, best))
    return pairs


@dataclass
class PageScores:
    id: str = ""
    chars: int = 0
    edits_nodia: int = 0
    chars_dia: int = 0
    edits_dia: int = 0
    words: int = 0
    word_edits: int = 0
    note_chars: int = 0
    note_edits: int = 0
    whole_chars: int = 0
    whole_edits: int = 0
    quran_total: int = 0
    quran_wrong: int = 0
    quran_auto: int = 0
    poem_verses: int = 0
    poem_correct: int = 0
    blocks_total: int = 0
    blocks_correct: int = 0
    notes_truth: int = 0
    notes_found: int = 0
    links_total: int = 0
    links_correct: int = 0
    regions_page: int = 0  # 1 when the page has hashiya regions in the truth
    regions_ok: int = 0
    has_error: int = 0
    flagged: int = 0

    @property
    def cer(self) -> float:
        return self.edits_nodia / self.chars if self.chars else 0.0


def score_page(truth: Page, pred: Page, *, page_id: str = "", reliable_links: set[str] | None = None) -> PageScores:
    s = PageScores(id=page_id or str(truth.index))
    t_main, p_main = page_text(truth), page_text(pred)
    for diacritics in (False, True):
        tr, pr = _norm(t_main, diacritics), _norm(p_main, diacritics)
        if diacritics:
            s.chars_dia, s.edits_dia = len(tr), _dist(tr, pr)
        else:
            s.chars, s.edits_nodia = len(tr), _dist(tr, pr)
            s.words, s.word_edits = len(tr.split()), Levenshtein.distance(tr.split(), pr.split())
    tn, pn = _norm(page_text(truth, notes=True, main=False), False), _norm(page_text(pred, notes=True, main=False), False)
    s.note_chars, s.note_edits = len(tn), _dist(tn, pn)
    tw, pw = _norm(page_text(truth, notes=True), False), _norm(page_text(pred, notes=True), False)
    s.whole_chars, s.whole_edits = len(tw), _dist(tw, pw)
    # Quran: align spans by text.
    t_spans, p_spans = _spans(truth), _spans(pred)
    used: set[int] = set()
    for t in t_spans:
        s.quran_total += 1
        best, best_score = None, 0.0
        for k, p in enumerate(p_spans):
            if k in used:
                continue
            score = 1 - cer(normalize_for_match(t.text), normalize_for_match(p.ocr_text or p.text))
            if score > best_score:
                best, best_score = k, score
        if best is None or best_score < 0.5:
            continue
        used.add(best)
        p = p_spans[best]
        if p.status in ("verified", "corrected"):
            s.quran_auto += 1
            if t.ref and p.ref and t.ref != p.ref and t.ref not in p.alternatives:
                s.quran_wrong += 1
    # Poems: verse by verse.
    t_poems = [b for b, _o in iter_blocks(truth, notes=False) if isinstance(b, Poem)]
    p_poems = [b for b, _o in iter_blocks(pred, notes=False) if isinstance(b, Poem)]
    for k, tp in enumerate(t_poems):
        pp = p_poems[k] if k < len(p_poems) else None
        for j, tv in enumerate(tp.verses):
            s.poem_verses += 1
            if pp is None or j >= len(pp.verses):
                continue
            pv = pp.verses[j]
            sadr_ok = cer(_norm(inline_text(tv.sadr), False), _norm(inline_text(pv.sadr), False)) < 0.1
            ajuz_ok = cer(_norm(inline_text(tv.ajuz or []), False), _norm(inline_text(pv.ajuz or []), False)) < 0.1
            s.poem_correct += int(sadr_ok and ajuz_ok)
    # Block types, plus notes as a pseudo-block type.
    tt = [b.type for b in truth.blocks] + ["note"] * len(truth.notes())
    pt = [b.type for b in pred.blocks] + ["note"] * len(pred.notes())
    s.blocks_total = max(len(tt), 1) if tt else 0
    s.blocks_correct = sum(m.size for m in difflib.SequenceMatcher(None, tt, pt, autojunk=False).get_matching_blocks())
    # Notes and links.
    pairs = _match_notes(truth.notes(), pred.notes())
    s.notes_truth, s.notes_found = len(truth.notes()), len(pairs)
    for t, p in pairs:
        if reliable_links is not None and t.id not in reliable_links:
            continue
        t_ctx = _anchor_context(truth, t.id)
        if t_ctx is None:
            continue
        s.links_total += 1
        p_ctx = _anchor_context(pred, p.id) if p.link.page in (None, pred.index) else None
        s.links_correct += int(p_ctx == t_ctx)
    s.links_total += sum(1 for t in truth.notes() if t not in [x for x, _ in pairs]
                         and _anchor_context(truth, t.id) is not None
                         and (reliable_links is None or t.id in reliable_links))
    regions = {r.region for r in truth.hashiya if r.notes or r.blocks}
    if regions:
        s.regions_page = 1
        s.regions_ok = int(regions <= {r.region for r in pred.hashiya if r.notes or r.blocks})
    s.flagged = int(pred.status == "flagged")
    s.has_error = int(s.cer > 0.03 or s.quran_wrong > 0 or s.links_correct < s.links_total
                      or s.notes_found < s.notes_truth)
    return s


@dataclass
class SetScores:
    pages: list[PageScores] = field(default_factory=list)

    def total(self, name: str) -> int:
        return sum(getattr(p, name) for p in self.pages)

    def rate(self, num: str, den: str) -> float | None:
        d = self.total(den)
        return self.total(num) / d if d else None

    def summary(self) -> dict[str, Any]:
        missed = sum(1 for p in self.pages if p.has_error and not p.flagged)
        return {
            "pages": len(self.pages),
            "cer_no_diacritics": self.rate("edits_nodia", "chars"),
            "cer_with_diacritics": self.rate("edits_dia", "chars_dia"),
            "wer": self.rate("word_edits", "words"),
            "cer_notes": self.rate("note_edits", "note_chars"),
            "cer_whole_page": self.rate("whole_edits", "whole_chars"),
            "quran_wrong_verse_rate": self.rate("quran_wrong", "quran_auto"),
            "quran_auto_resolved_rate": self.rate("quran_auto", "quran_total"),
            "poem_verse_accuracy": self.rate("poem_correct", "poem_verses"),
            "block_type_accuracy": self.rate("blocks_correct", "blocks_total"),
            "note_recall": self.rate("notes_found", "notes_truth"),
            "note_link_accuracy": self.rate("links_correct", "links_total"),
            "region_detection": self.rate("regions_ok", "regions_page"),
            "missed_error_rate": missed / len(self.pages) if self.pages else None,
            "flagged_pages": self.total("flagged"),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary(), "pages": [asdict(p) for p in self.pages]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SetScores":
        names = {f.name for f in fields(PageScores)}
        return cls([PageScores(**{k: v for k, v in p.items() if k in names}) for p in data.get("pages", [])])


def score_set(pairs: list[tuple[Page, Page]]) -> SetScores:
    return SetScores([score_page(t, p) for t, p in pairs])


TARGETS = {  # spec §9 version-1 targets
    "cer_no_diacritics": ("<", 0.03),
    "quran_wrong_verse_rate": ("<=", 0.0),
    "poem_verse_accuracy": (">", 0.95),
    "block_type_accuracy": (">", 0.95),
    "note_recall": (">", 0.98),
    "note_link_accuracy": (">", 0.95),
    "region_detection": (">", 0.97),
    "missed_error_rate": ("<", 0.05),
}


def review_minutes_per_100_pages(log_lines: list[dict[str, Any]], pages: int) -> float | None:
    """Minutes of human correction per 100 pages, from ``logs/review.jsonl``."""
    if not pages:
        return None
    seconds = sum(float(r.get("seconds") or 0) for r in log_lines)
    return seconds / 60 / pages * 100
