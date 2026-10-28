"""Correct doubtful words with a classical-Arabic language model (after OCR, before the checks).

The page model is asked for each token's top alternatives (``ocr.logprobs``). For every word,
the candidates are the words a swap of one token for one of its alternatives gives: the model
itself considered them. Each candidate has an image score (how much less likely the model found
it than what it wrote) and a language-model gain: how much better it fits four words of context
on either side, scored by a 5-gram model of 600 million words of classical Arabic (train/lm_modal.py,
the deployed ``Scorer`` on Modal). A word changes only when the model was not sure of it and a
candidate's combined score beats the word's own by a margin; candidates that would cut the word
short (a swap at a token boundary) are never considered.

On the held-out half of the test book this cut word errors by about 30% (5.1% to 3.6%), with
about ten right corrections for each wrong one (tools/rescore.py). Each change is recorded in
``pages/NNNN.rescore.json`` and shown as an info flag.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .arabic import normalize_for_match

LN10 = math.log(10)
CONTEXT = 4
_SKIP = re.compile(r"<header>.*?</header>|<pagenum>.*?</pagenum>|<[^>]+>", re.S)


def lenient(word: str) -> str:
    """A word as the scorer compares it: letters only, note markers dropped."""
    return " ".join(normalize_for_match(re.sub(r"[(\[][٠-٩0-9]{1,3}[)\]]", " ", word)).split())


@dataclass
class Word:
    raw: str
    start: int
    end: int
    norm: str
    low: float  # lowest token probability in the word
    cands: dict[str, tuple[str, float]] = field(default_factory=dict)  # norm -> (raw, image score)


def words_of(tokens: list[dict]) -> list[Word]:
    """Main-text words of a reading (not header, page number or notes) with their candidates.
    ``tokens`` are ``{"t": text, "lp": logprob, "alt": [[text, logprob], ...]}``."""
    text = "".join(t["t"] for t in tokens)
    starts, pos = [], 0
    for t in tokens:
        starts.append(pos)
        pos += len(t["t"])
    cut = text.find("<hashiya")
    end = cut if cut >= 0 else len(text)
    skip = [m.span() for m in _SKIP.finditer(text[:end])]
    out = []
    k0 = 0
    for m in re.finditer(r"[^\s<>]+", text[:end]):
        a, b = m.span()
        if any(s <= a < e for s, e in skip):
            continue
        norm = lenient(m.group(0))
        if not norm or " " in norm:
            continue
        w = Word(m.group(0), a, b, norm, 1.0)
        while k0 < len(tokens) and starts[k0] + len(tokens[k0]["t"]) <= a:
            k0 += 1
        k = k0
        while k < len(tokens) and starts[k] < b:
            t, ta = tokens[k], starts[k]
            tb = ta + len(t["t"])
            w.low = min(w.low, math.exp(t["lp"]))
            for alt, lp in t["alt"]:
                if alt == t["t"]:
                    continue
                new = (text[a:ta] + alt + text[tb:b]) if ta >= a else (alt + text[tb:b])[a - ta:]
                cand = lenient(new).replace(" ", "")
                # (an alternative that starts or ends a word elsewhere would split this one: skip it)
                if (cand and cand != norm and abs(len(cand) - len(norm)) <= 1 and not re.search(r"\s", new.strip())
                        and not (len(cand) < len(norm) and norm.startswith(cand))):
                    old = w.cands.get(cand)
                    if old is None or lp - t["lp"] > old[1]:
                        w.cands[cand] = (new.strip(), lp - t["lp"])
            k += 1
        out.append(w)
    return out


def windows(words: list[Word], doubt: float) -> list[tuple[int, str, str]]:
    """(word index, candidate norm or "" for the word itself, window text) to score."""
    seq = [w.norm for w in words]
    jobs = []
    for i, w in enumerate(words):
        if not w.cands or w.low > doubt:
            continue
        left, right = seq[max(0, i - CONTEXT):i], seq[i + 1:i + 1 + CONTEXT]
        jobs.append((i, "", " ".join(left + [w.norm] + right)))
        for cand in w.cands:
            jobs.append((i, cand, " ".join(left + [cand] + right)))
    return jobs


def choose(words: list[Word], scores: dict[tuple[int, str], float], *, weight: float, margin: float,
           doubt: float) -> list[tuple[int, str, str, float]]:
    """(word index, new raw text, new norm, gain) for every word that changes."""
    changes = []
    for i, w in enumerate(words):
        if not w.cands or w.low > doubt or (i, "") not in scores:
            continue
        base = scores[(i, "")]
        best, best_gain = None, margin
        for cand, (raw, img) in w.cands.items():
            if (i, cand) not in scores:
                continue
            gain = img + weight * (scores[(i, cand)] - base) * LN10
            if gain > best_gain:
                best, best_gain = (raw, cand), gain
        if best:
            changes.append((i, best[0], best[1], round(best_gain, 3)))
    return changes


def apply(tags: str, words: list[Word], changes: list[tuple[int, str, str, float]]) -> str:
    for i, raw, _norm, _gain in sorted(changes, key=lambda c: -words[c[0]].start):
        w = words[i]
        tags = tags[:w.start] + raw + tags[w.end:]
    return tags


def add_line_candidates(words: list[Word], line_texts: list[str], score: float) -> int:
    """Add the line-by-line reading's words as candidates where it disagrees with the page reading
    on an aligned word (a second view: it gets right about 2% of words the page reading gets
    wrong). Such a word counts as doubtful; the candidate's image score is ``score``. Returns the
    number of candidates added."""
    from rapidfuzz.distance import Levenshtein

    raw = [w for t in line_texts for w in t.split()]
    line = [(lenient(w), w) for w in raw]
    line = [(n, w) for n, w in line if n and " " not in n]
    seq = [w.norm for w in words]
    added = 0
    for op in Levenshtein.opcodes(seq, [n for n, _w in line]):
        if op.tag != "replace" or op.src_end - op.src_start != op.dest_end - op.dest_start:
            continue
        for a, b in zip(range(op.src_start, op.src_end), range(op.dest_start, op.dest_end)):
            w, (norm, raw_word) = words[a], line[b]
            if norm == w.norm or abs(len(norm) - len(w.norm)) > 2:
                continue
            old = w.cands.get(norm)
            w.cands[norm] = (old[0], max(old[1], score)) if old else (re.sub(r"[(\[][٠-٩0-9]{1,3}[)\]]", "", raw_word), score)
            w.low = min(w.low, 0.5)
            added += 1
    return added
