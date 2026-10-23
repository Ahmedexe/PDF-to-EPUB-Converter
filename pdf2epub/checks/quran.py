"""Quran verses: find, match against Tanzil, replace or flag (spec §5.1).

Quran text is never trusted to OCR alone. Every passage is matched against the
verified Tanzil text and either replaced by it (``verified`` / ``corrected``) or
flagged for review. No span leaves :func:`check_quran` as ``unchecked``.

Matching works on one long word sequence per spelling *family* (Uthmani:
uthmani, uthmani-plain, uthmani-min; simple: simple, simple-plain,
simple-clean; within a family the variants segment words identically), so a
partial verse and a quote spanning several verses are found the same way:

1. normalise the OCR text (diacritics, marks, alef forms, ى/ي, ة/ه, tatweel,
   verse numbers and ornaments removed);
2. vote for candidate start positions with a word n-gram index (loose
   spelling keys, then the undotted skeleton for OCR dot errors);
3. score candidate windows with rapidfuzz and refine their edges;
4. exact match (in either family) = verified, similarity ≥ threshold =
   corrected, otherwise flagged. Spans shorter than ``quran_min_words`` are
   only resolved when every location they match gives the same words.

The replacement is exactly the quoted words of the book's Tanzil variant
(whole Tanzil tokens, never edited characters). Tanzil prepends the basmala
to verse 1 of every sura except 1 and 9; the index keeps that prefix apart so
quotes of verse 1 without it still match, and a basmala alone matches 1:1.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from ..arabic import (
    ORNATE_CLOSE,
    ORNATE_OPEN,
    clean_text,
    diacritic_ratio,
    has_pause_marks,
    has_uthmani_features,
    normalize_for_match,
    parse_number,
    strip_tatweel,
    to_rasm,
)
from ..config import BookConfig
from ..model import (
    BlockText,
    FlagCode,
    Page,
    Quran,
    QuranLoc,
    Ref,
    block_containers,
    iter_blocks,
    walk_inline_slots,
)
from ..quran_meta import format_ref, suras

log = logging.getLogger(__name__)
STAGE = "quran"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tanzil"
VARIANTS = ("uthmani", "uthmani-plain", "uthmani-min", "simple", "simple-plain", "simple-clean")
FAMILY = {"uthmani": "uthmani", "uthmani-plain": "uthmani", "uthmani-min": "uthmani",
          "simple": "simple", "simple-plain": "simple", "simple-clean": "simple"}
BASMALA_KEY = ("بسم", "الله", "الرحمن", "الرحيم")

_VERSE_NUM_RE = re.compile(r"[(\[﴿﴾{]?\s*[٠-٩0-9۰-۹]{1,3}\s*[)\]﴿﴾}]?|۝\s*[٠-٩0-9۰-۹]{1,3}")
_PRINTED_REF_RE = re.compile(
    r"^\s*[\[(]\s*(?:سورة\s*)?(?P<name>[ء-يً-ْٰ ]{2,30}?)\s*[:：،,]?\s*(?:الآية\s*)?"
    r"(?P<aya>[\d٠-٩۰-۹]{1,3})"
)


def _std(word: str) -> str:
    return normalize_for_match(word, keep_digits=False)


def _loose(word: str) -> str:
    return normalize_for_match(word, dagger_alef="alef", hamza=True, keep_digits=False)


@dataclass
class QuranMatch:
    status: Literal["verified", "corrected", "flagged"]
    score: float
    ref: str | None = None
    loc: QuranLoc | None = None
    text: str | None = None  # replacement text in the chosen variant (quoted part only)
    variant: str | None = None
    alternatives: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class _Family:
    name: str
    # Parallel per-word lists for the whole Quran (basmala prefixes excluded).
    sura: list[int] = field(default_factory=list)
    aya: list[int] = field(default_factory=list)
    word: list[int] = field(default_factory=list)  # 1-based within the aya
    std: list[str] = field(default_factory=list)
    loose: list[str] = field(default_factory=list)
    rasm: list[str] = field(default_factory=list)
    # aya -> (first global word index, word count)
    aya_span: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    grams: dict[int, dict[tuple[str, ...], list[int]]] = field(default_factory=dict)
    rasm_grams: dict[tuple[str, ...], list[int]] = field(default_factory=dict)


class QuranIndex:
    """The Tanzil text of all six variants, indexed for matching."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        # variant -> aya -> list of tokens, each (surface, is_word); basmala prefix kept apart
        self.tokens: dict[str, dict[tuple[int, int], list[tuple[str, bool]]]] = {}
        self.basmala: dict[str, str] = {}
        self.families: dict[str, _Family] = {}
        for variant in VARIANTS:
            self._load_variant(variant)
        for family in ("uthmani", "simple"):
            self.families[family] = self._build_family(family)
        self._align_cache: dict[tuple[str, str, int, int], list[int]] = {}

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, data_dir: Path | None = None) -> "QuranIndex":
        return _load_index(str(Path(data_dir or DATA_DIR).resolve()))

    def _load_variant(self, variant: str) -> None:
        path = self.data_dir / f"quran-{variant}.txt"
        ayas: dict[tuple[int, int], list[tuple[str, bool]]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            s, a, text = line.split("|", 2)
            key = (int(s), int(a))
            toks = [(t, bool(_std(t))) for t in text.split()]
            if key[1] == 1 and key[0] not in (1, 9):
                words = [i for i, (t, w) in enumerate(toks) if w]
                if len(words) > 4 and tuple(_std(toks[i][0]) for i in words[:4]) == BASMALA_KEY:
                    cut = words[3] + 1
                    self.basmala.setdefault(variant, " ".join(t for t, _ in toks[:cut]))
                    toks = toks[cut:]
            ayas[key] = toks
        self.tokens[variant] = ayas

    def _build_family(self, family: str) -> _Family:
        fam = _Family(family)
        base = family  # "uthmani" or "simple" carries the family's segmentation
        for key in sorted(self.tokens[base]):
            words = [t for t, is_word in self.tokens[base][key] if is_word]
            fam.aya_span[key] = (len(fam.std), len(words))
            for i, w in enumerate(words, 1):
                fam.sura.append(key[0])
                fam.aya.append(key[1])
                fam.word.append(i)
                std = _std(w)
                loose = _loose(w)
                fam.std.append(std)
                fam.loose.append(loose)
                fam.rasm.append(to_rasm(loose))
        for n in (1, 2, 3):
            grams: dict[tuple[str, ...], list[int]] = defaultdict(list)
            for i in range(len(fam.loose) - n + 1):
                grams[tuple(fam.loose[i:i + n])].append(i)
            fam.grams[n] = grams
        rg: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i in range(len(fam.rasm) - 1):
            rg[(fam.rasm[i], fam.rasm[i + 1])].append(i)
        fam.rasm_grams = rg
        return fam

    # ------------------------------------------------------------------ lookups

    def sura_name(self, sura: int) -> str:
        return suras(str(self.data_dir))[sura - 1].name

    def verse_text(self, sura: int, aya: int, variant: str) -> str:
        return " ".join(t for t, _ in self.tokens[variant][(sura, aya)])

    def _words_text(self, variant: str, key: tuple[int, int], w0: int, w1: int) -> str:
        """Surface text of words w0..w1 (1-based, inclusive) of an aya, with the marks between them."""
        toks = self.tokens[variant][key]
        out, n = [], 0
        for t, is_word in toks:
            if is_word:
                n += 1
            if w0 <= n <= w1 and (is_word or n < w1):
                out.append(t)
        return " ".join(out)

    def _map_range(self, family: str, target: str, key: tuple[int, int], w: int) -> tuple[int, int]:
        """1-based word range in ``target``'s segmentation for word ``w`` of ``family``'s.

        Within a family the segmentation is identical. Across families (يٰأيها vs
        يا أيها) the words of the aya are aligned once with Levenshtein opcodes;
        target words with no counterpart attach to the neighbouring source word.
        """
        tfam = FAMILY[target]
        if tfam == family:
            return w, w
        cache_key = (family, tfam, *key)
        if cache_key not in self._align_cache:
            src = [_loose(t) for t, is_w in self.tokens[family][key] if is_w]
            dst = [_loose(t) for t, is_w in self.tokens[tfam][key] if is_w]
            ranges: list[list[int]] = [[-1, -1] for _ in src]
            pending_lo: int | None = None
            for tag, i1, i2, j1, j2 in Levenshtein.opcodes(src, dst):
                if tag in ("equal", "replace"):
                    ns, nt = i2 - i1, j2 - j1
                    for k in range(i1, i2):
                        lo = j1 + ((k - i1) * nt) // ns
                        hi = j1 + -(-((k - i1 + 1) * nt) // ns) - 1
                        ranges[k] = [lo, max(lo, hi)]
                    if pending_lo is not None:
                        ranges[i1][0] = pending_lo
                        pending_lo = None
                elif tag == "insert":
                    if i1 > 0 and ranges[i1 - 1][1] >= 0:
                        ranges[i1 - 1][1] = j2 - 1
                    else:
                        pending_lo = j1
                else:  # delete
                    for k in range(i1, i2):
                        pos = min(j1, max(len(dst) - 1, 0))
                        ranges[k] = [pos, pos]
            self._align_cache[cache_key] = [(lo + 1, hi + 1) for lo, hi in ranges]
        ranges_t = self._align_cache[cache_key]
        return ranges_t[min(max(w, 1), len(ranges_t)) - 1] if ranges_t else (1, 1)

    def passage_text(self, family: str, start: int, end: int, variant: str,
                     separators: list[str] | None = None) -> str:
        """Text of global words ``start..end`` (inclusive) of ``family`` in ``variant``.

        ``separators`` (printed verse numbers) are re-inserted after each aya
        when their count fits the aya boundaries.
        """
        fam = self.families[family]
        parts: list[tuple[tuple[int, int], int, int]] = []
        i = start
        while i <= end:
            key = (fam.sura[i], fam.aya[i])
            first, count = fam.aya_span[key]
            last = min(end, first + count - 1)
            w0 = self._map_range(family, variant, key, fam.word[i])[0]
            w1 = self._map_range(family, variant, key, fam.word[last])[1]
            parts.append((key, w0, max(w0, w1)))
            i = last + 1
        texts = [self._words_text(variant, key, w0, w1) for key, w0, w1 in parts]
        if separators:
            if len(separators) == len(texts):
                return " ".join(f"{t} {s}" for t, s in zip(texts, separators))
            if len(separators) == len(texts) - 1:
                return " ".join(t + (f" {separators[k]}" if k < len(separators) else "") for k, t in enumerate(texts))
        return " ".join(texts)

    def loc(self, family: str, start: int, end: int, variant: str) -> QuranLoc:
        fam = self.families[family]
        k0, k1 = (fam.sura[start], fam.aya[start]), (fam.sura[end], fam.aya[end])
        return QuranLoc(variant=variant, sura=k0[0], aya=k0[1],
                        word=self._map_range(family, variant, k0, fam.word[start])[0],
                        end_sura=k1[0], end_aya=k1[1],
                        end_word=self._map_range(family, variant, k1, fam.word[end])[1])

    def ref(self, family: str, start: int, end: int) -> str:
        fam = self.families[family]
        return format_ref(fam.sura[start], fam.aya[start], fam.sura[end], fam.aya[end])

    # ------------------------------------------------------------------ matching

    def _candidates(self, fam: _Family, q: list[str], qr: list[str]) -> list[int]:
        k = len(q)
        votes: Counter[int] = Counter()
        for n in (3, 2, 1):
            if k < n or (n == 1 and k > 3):
                continue
            grams = fam.grams[n]
            for j in range(k - n + 1):
                for pos in grams.get(tuple(q[j:j + n]), ()):
                    votes[pos - j] += 1
            if votes:
                break
        if not votes and k >= 2:
            for j in range(k - 1):
                for pos in fam.rasm_grams.get((qr[j], qr[j + 1]), ()):
                    votes[pos - j] += 1
        if not votes:
            return []
        best = votes.most_common(1)[0][1]
        floor = max(1, int(best * 0.5))
        return [pos for pos, v in votes.most_common(40) if v >= floor]

    def _score(self, fam: _Family, qs: str, ql: str, start: int, end: int) -> float:
        ws = " ".join(fam.std[start:end + 1])
        if ws == qs:
            return 1.0
        wl = " ".join(fam.loose[start:end + 1])
        return max(fuzz.ratio(qs, ws), fuzz.ratio(ql, wl) - 1.0) / 100.0

    def _best_window(self, fam: _Family, q: list[str], qs: str, ql: str, start: int) -> tuple[float, int, int]:
        k = len(q)
        n = len(fam.std)
        best = (-1.0, start, start)
        for ds in range(-2, 3):
            s = start + ds
            if s < 0 or s >= n:
                continue
            for dk in range(-3, 4):
                e = s + k - 1 + dk
                if e < s or e >= n:
                    continue
                score = self._score(fam, qs, ql, s, e)
                if score > best[0] or (score == best[0] and (e - s) == k - 1):
                    best = (score, s, e)
        return best

    def match(self, text: str, *, style: str = "auto", threshold: float = 0.90, min_words: int = 3,
              hint: tuple[int, int] | None = None) -> QuranMatch:
        """Match one OCR'd passage. ``style`` is the Tanzil variant to replace with
        (``auto`` = the family that matched, diacritics as in the OCR text);
        ``hint`` is a (sura, aya) printed next to the passage, used to choose
        between identical candidates."""
        separators = [m.group(0).strip() for m in _VERSE_NUM_RE.finditer(text)
                      if parse_number(m.group(0)) is not None]
        raw = _VERSE_NUM_RE.sub(" ", text)
        raw = raw.replace(ORNATE_OPEN, " ").replace(ORNATE_CLOSE, " ")
        q = [w for w in (_loose(t) for t in raw.split()) if w]
        qstd = [w for w in (_std(t) for t in raw.split()) if w]
        if not q:
            return QuranMatch("flagged", 0.0, reason="no letters to match")
        qs, ql = " ".join(qstd), " ".join(q)
        qr = [to_rasm(w) for w in q]
        if style == "auto":
            style = detect_quran_style([text])
        found: list[tuple[float, str, int, int]] = []
        for fname, fam in self.families.items():
            if len(q) < min_words:
                for pos in fam.grams[len(q)].get(tuple(q), ()) if len(q) <= 3 else ():
                    found.append((self._score(fam, qs, ql, pos, pos + len(q) - 1), fname, pos, pos + len(q) - 1))
                continue
            for start in self._candidates(fam, q, qr):
                score, s, e = self._best_window(fam, q, qs, ql, start)
                found.append((score, fname, s, e))
        if not found:
            return QuranMatch("flagged", 0.0, reason="no candidate passage")
        target_family = FAMILY[style]
        found.sort(key=lambda x: (-x[0], x[1] != target_family, x[2]))
        top = found[0][0]
        best = [f for f in found if f[0] >= top - 1e-9]
        # Collapse the same passage found in both families to the target family's view.
        refs_seen: dict[str, tuple[float, str, int, int]] = {}
        for f in best:
            r = self.ref(f[1], f[2], f[3])
            if r not in refs_seen or (f[1] == target_family and refs_seen[r][1] != target_family):
                refs_seen[r] = f
        choices = list(refs_seen.values())
        if hint is not None:
            hinted = [f for f in choices if self.families[f[1]].sura[f[2]] == hint[0]
                      and self.families[f[1]].aya[f[2]] <= hint[1] <= self.families[f[1]].aya[f[3]]]
            if hinted:
                choices = hinted + [f for f in choices if f not in hinted]
        score, fname, s, e = choices[0]
        texts = {self.passage_text(f[1], f[2], f[3], style) for f in choices}
        alternatives = [self.ref(f[1], f[2], f[3]) for f in choices[1:]]
        result = QuranMatch(
            status="flagged", score=round(score, 4), ref=self.ref(fname, s, e),
            loc=self.loc(fname, s, e, style), variant=style, alternatives=alternatives,
        )
        short = len(q) < min_words
        if len(texts) > 1:
            # Identical normalised words but different Tanzil text: pick by the OCR diacritics.
            ocr = strip_tatweel(clean_text(text))
            ranked = sorted(choices, key=lambda f: -fuzz.ratio(ocr, self.passage_text(f[1], f[2], f[3], style)))
            first = self.passage_text(ranked[0][1], ranked[0][2], ranked[0][3], style)
            if short and fuzz.ratio(ocr, first) < 100:
                result.reason = "short passage occurs in several places with different text"
                return result
            score, fname, s, e = ranked[0]
            result.ref, result.loc = self.ref(fname, s, e), self.loc(fname, s, e, style)
            result.alternatives = [self.ref(f[1], f[2], f[3]) for f in ranked[1:]]
        if short and score < 1.0:
            result.reason = "short passage without an exact match"
            return result
        if score >= 1.0:
            result.status = "verified"
        elif score >= threshold:
            result.status = "corrected"
        else:
            result.reason = f"best match {result.ref} scores only {score:.2f}"
            return result
        result.text = self.passage_text(fname, s, e, style, separators=separators or None)
        return result


@lru_cache(maxsize=4)
def _load_index(data_dir: str) -> QuranIndex:
    """Build the index, or load it from a pickle cache keyed by the data files' contents."""
    import hashlib
    import os
    import pickle

    h = hashlib.sha256(b"quran-index-v1")
    for variant in VARIANTS:
        h.update((Path(data_dir) / f"quran-{variant}.txt").read_bytes())
    cache_root = Path(os.environ.get("PDF2EPUB_CACHE") or Path.home() / ".cache" / "pdf2epub")
    cache = cache_root / f"quran-index-{h.hexdigest()[:16]}.pickle"
    try:
        with cache.open("rb") as fh:
            index = pickle.load(fh)
        index.data_dir = Path(data_dir)
        return index
    except Exception:  # missing or unreadable cache: rebuild
        pass
    index = QuranIndex(Path(data_dir))
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache)
    except OSError:
        log.debug("could not write the Quran index cache", exc_info=True)
    return index


# --------------------------------------------------------------------------- style


def detect_quran_style(texts: list[str]) -> str:
    """The Tanzil variant that matches how a book prints its verses."""
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return "simple-plain"
    uthmani = sum(1 for t in texts if has_uthmani_features(t))
    marks = sum(1 for t in texts if has_pause_marks(t))
    ratios = [diacritic_ratio(t) for t in texts]
    voweled = statistics.median(ratios) >= 0.25
    if uthmani >= max(1, 0.3 * len(texts)):
        return "uthmani" if marks >= max(1, 0.2 * len(texts)) else "uthmani-plain"
    if voweled:
        return "simple" if marks >= max(1, 0.2 * len(texts)) else "simple-plain"
    return "simple-clean"


def resolve_style(pages: list[Page], cfg: BookConfig) -> str:
    if cfg.text.quran_style != "auto":
        return cfg.text.quran_style
    if cfg.text.quran_style_detected:
        return cfg.text.quran_style_detected
    texts = []
    for page in pages:
        for block, _owner in iter_blocks(page):
            for container in block_containers(block):
                for _c, _i, node in walk_inline_slots(container):
                    if isinstance(node, Quran):
                        texts.append(node.ocr_text if node.ocr_text is not None else node.text)
    return detect_quran_style(texts)


# --------------------------------------------------------------------------- untagged verses


def _bracket_patterns(brackets: list[str]) -> list[re.Pattern[str]]:
    out = []
    for pair in brackets:
        if len(pair) != 2:
            continue
        o, c = re.escape(pair[0]), re.escape(pair[1])
        out.append(re.compile(f"{o}([^{o}{c}\\n]{{2,}}?){c}"))
    return out


def tag_untagged_verses(page: Page, brackets: list[str]) -> int:
    """Wrap text between ornate brackets that the model left untagged in Quran spans.

    The brackets themselves are removed (the builder adds its own). Refs inside
    the brackets are kept, right after the new span. Returns the number wrapped;
    a pair that cannot be wrapped (it crosses other markup) raises
    ``quran_unmatched`` on its block.
    """
    patterns = _bracket_patterns(brackets)
    count = 0
    for block, _owner in iter_blocks(page):
        for _attempt in range(50):
            bt = BlockText(block)
            hit = None
            for pat in patterns:
                for m in pat.finditer(bt.text):
                    hit = m
                    break
                if hit:
                    break
            if hit is None:
                break
            refs: list[Ref] = []

            def make(nodes: list[object]) -> Quran:
                text = "".join(n if isinstance(n, str) else getattr(n, "text", "") for n in nodes
                               if not isinstance(n, Ref))
                refs.extend(n for n in nodes if isinstance(n, Ref))
                return Quran(text=clean_text(text).strip(ORNATE_OPEN + ORNATE_CLOSE + " "))

            span = bt.wrap_range(hit.start(), hit.end(), make)
            if span is None:
                page.add_flag(FlagCode.QURAN_UNMATCHED, "text in ornate brackets could not be isolated as a verse",
                              target=block.id, stage=STAGE)
                break
            for container in block_containers(block):
                for c, i, node in walk_inline_slots(container):
                    if node is span:
                        c[i + 1:i + 1] = refs
                        break
            count += 1
    return count


# --------------------------------------------------------------------------- book check


def _hint_after(container: list[object], index: int, idx: QuranIndex) -> tuple[int, int] | None:
    """A printed reference right after a span, like "[البقرة: ١٥٥]" -> (2, 155)."""
    following = ""
    for node in container[index + 1:index + 3]:
        if isinstance(node, str):
            following += node
        else:
            break
    m = _PRINTED_REF_RE.match(following)
    if not m:
        return None
    name = normalize_for_match(m.group("name"))
    aya = parse_number(m.group("aya"))
    if aya is None:
        return None
    for sura in suras(str(idx.data_dir)):
        if normalize_for_match(sura.name) == name or normalize_for_match("سورة " + sura.name) == name:
            return sura.index, aya
    return None


def check_quran(pages: list[Page], cfg: BookConfig, *, log_path: Path | None = None,
                only: set[int] | None = None, index: QuranIndex | None = None) -> str:
    """Verify, correct or flag every Quran span on the given pages. Returns the variant used.

    Reviewed pages, pages outside ``only`` and ``accepted`` spans are left as they are.
    Matching always starts from the OCR text, so re-running is idempotent.
    """
    idx = index or QuranIndex.load()
    style = resolve_style(pages, cfg)
    records: list[dict[str, object]] = []
    for page in pages:
        if page.status == "reviewed" or (only is not None and page.index not in only):
            continue
        page.clear_flags(STAGE)
        tag_untagged_verses(page, cfg.text.quran_brackets)
        for block, _owner in iter_blocks(page):
            for container in block_containers(block):
                for c, i, node in list(walk_inline_slots(container)):
                    if not isinstance(node, Quran) or node.status == "accepted":
                        continue
                    if node.ocr_text is None:
                        node.ocr_text = node.text
                    m = idx.match(node.ocr_text, style=style, threshold=cfg.checks.quran_threshold,
                                  min_words=cfg.checks.quran_min_words, hint=_hint_after(c, i, idx))
                    node.status = m.status
                    node.score = m.score
                    node.ref, node.loc, node.variant = m.ref, m.loc, m.variant
                    node.alternatives = m.alternatives
                    if m.status in ("verified", "corrected") and m.text:
                        node.text = m.text
                    else:
                        node.text = node.ocr_text
                    if m.status == "flagged":
                        n_words = len(node.ocr_text.split())
                        code = FlagCode.QURAN_SHORT if n_words < cfg.checks.quran_min_words else FlagCode.QURAN_UNMATCHED
                        page.add_flag(code, f"Quran passage not resolved: {m.reason}", target=block.id, stage=STAGE)
                    elif m.alternatives:
                        page.add_flag(FlagCode.QURAN_AMBIGUOUS,
                                      f"identical text also at {', '.join(m.alternatives[:5])}",
                                      severity="info", target=block.id, stage=STAGE)
                    records.append({"page": page.index, "block": block.id, "status": m.status, "ref": m.ref,
                                    "score": m.score, "ocr_text": node.ocr_text, "text": node.text,
                                    "variant": m.variant, "alternatives": m.alternatives})
    if log_path is not None and records:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return style
