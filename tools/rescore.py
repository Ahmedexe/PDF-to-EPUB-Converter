"""Rescore doubtful words with a classical-Arabic language model; measure it on the test book.

    python tools/rescore.py [--oracle work/oracle-ft-ft2] [--ref sample/test/reference]

Uses the readings tools/oracle_alternatives.py saved with the model's token alternatives. For each
main-text word, the candidates are the words one swap of a token for an alternative the model
itself considered would give, each with its image score (the alternative's log-probability minus
the chosen token's). Every candidate is scored by the language model (train/lm_modal.py, the
deployed ``Scorer``) in a window of four words either side, and gets a bonus for how often the
book's reading uses it. A word changes only when a candidate's combined score beats the word's
own by a margin. The weights are tuned on odd test pages and reported on even ones.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from pdf2epub.bench import _lenient  # noqa: E402

LN10 = math.log(10)
CONTEXT = 4


def words_with_scores(tokens: list[dict]) -> list[tuple[str, dict[str, float], float]]:
    """Main-text words in order: (word, {candidate: image score}, lowest token probability)."""
    import re

    text = "".join(t["t"] for t in tokens)
    starts, pos = [], 0
    for t in tokens:
        starts.append(pos)
        pos += len(t["t"])
    cut = text.find("<hashiya")
    end = cut if cut >= 0 else len(text)
    skip = [(m.start(), m.end()) for m in re.finditer(r"<header>.*?</header>|<pagenum>.*?</pagenum>|<[^>]+>", text[:end], re.S)]
    out = []
    for m in re.finditer(r"[^\s<>]+", text[:end]):
        a, b = m.span()
        if any(s <= a < e for s, e in skip):
            continue
        word = _lenient(m.group(0))
        if not word:
            continue
        cands: dict[str, float] = {}
        low = 0.0
        for k, t in enumerate(tokens):
            ta, tb = starts[k], starts[k] + len(t["t"])
            if tb <= a or ta >= b:
                continue
            low = min(low, t["lp"])
            for alt, lp in t["alt"]:
                if alt == t["t"]:
                    continue
                new = text[a:ta] + alt + text[tb:b] if ta >= a else (alt + text[tb:b])[max(0, a - ta):]
                cand = _lenient(new).replace(" ", "")
                # A swap at a token boundary can cut the word short (the rest was never written for
                # the alternative): keep only candidates of about the word's length, not its stem.
                if (cand and cand != word and abs(len(cand) - len(word)) <= 1
                        and not (len(cand) < len(word) and word.startswith(cand))):
                    cands[cand] = max(cands.get(cand, -99.0), lp - t["lp"])
        out.append((word, cands, math.exp(low)))
    return out


def load(oracle: Path, ref_dir: Path) -> list[tuple[int, list[str], list]]:
    pages = []
    for f in sorted(ref_dir.glob("*.txt")):
        p = oracle / f"{int(f.stem):04d}.json"
        if p.exists():
            ref = _lenient(f.read_text(encoding="utf-8").partition("\n---\n")[0]).split()
            pages.append((int(f.stem), ref, words_with_scores(json.loads(p.read_text(encoding="utf-8"))["tokens"])))
    return pages


def lm_deltas(pages: list, max_doubt: float) -> dict[tuple[int, int, str], float]:
    """Language-model gain (natural log) of each candidate over the word read, in context."""
    import modal

    scorer = modal.Cls.from_name("pdf2epub-lm", "Scorer")()
    jobs, windows = [], []
    for pi, (_i, _ref, words) in enumerate(pages):
        seq = [w for w, _c, _p in words]
        for wi, (w, cands, low) in enumerate(words):
            if not cands or low > max_doubt:
                continue
            left, right = seq[max(0, wi - CONTEXT):wi], seq[wi + 1:wi + 1 + CONTEXT]
            for cand in [w, *cands]:
                jobs.append((pi, wi, cand))
                windows.append(" ".join(left + [cand] + right))
    scores: list[float] = []
    for k in range(0, len(windows), 5000):
        scores.extend(scorer.score.remote(windows[k:k + 5000]))
    base: dict[tuple[int, int], float] = {}
    for (pi, wi, cand), s in zip(jobs, scores):
        if cand == pages[pi][2][wi][0]:
            base[(pi, wi)] = s
    return {(pi, wi, cand): (s - base[(pi, wi)]) * LN10 for (pi, wi, cand), s in zip(jobs, scores) if cand != pages[pi][2][wi][0]}


def apply(pages: list, deltas: dict, lam: float, beta: float, margin: float, max_doubt: float) -> list[list[str]]:
    book = Counter(w for _i, _r, words in pages for w, _c, p in words if p > 0.9)
    out = []
    for pi, (_i, _ref, words) in enumerate(pages):
        seq = []
        for wi, (w, cands, low) in enumerate(words):
            best, best_score = w, beta * math.log1p(book[w])
            if cands and low <= max_doubt:
                for cand, img in cands.items():
                    d = deltas.get((pi, wi, cand))
                    if d is None:
                        continue
                    score = img + lam * d + beta * math.log1p(book[cand])
                    if score > best_score + margin:
                        best, best_score = cand, score
            seq.append(best)
        out.append(seq)
    return out


def wer(pages: list, hyps: list[list[str]], which: set[int]) -> tuple[float, int]:
    edits = total = 0
    for (i, ref, _w), hyp in zip(pages, hyps):
        if i in which:
            edits += Levenshtein.distance(ref, hyp)
            total += len(ref)
    return edits / max(1, total), edits


def main() -> None:
    oracle = Path(sys.argv[sys.argv.index("--oracle") + 1]) if "--oracle" in sys.argv else Path("work/oracle-ft-ft2")
    ref = Path(sys.argv[sys.argv.index("--ref") + 1]) if "--ref" in sys.argv else Path("sample/test/reference")
    pages = load(oracle, ref)
    if "--fixed" in sys.argv:  # apply chosen weights to every page, no tuning (another book)
        lam, beta, margin, doubt = map(float, sys.argv[sys.argv.index("--fixed") + 1].split(","))
        allp = {i for i, _r, _w in pages}
        plain = [[w for w, _c, _p in words] for _i, _r, words in pages]
        hyps = apply(pages, lm_deltas(pages, max_doubt=0.999), lam, beta, margin, doubt)
        print(f"{len(pages)} pages: WER as read {wer(pages, plain, allp)[0]:.2%} -> rescored {wer(pages, hyps, allp)[0]:.2%}")
        good = bad = 0
        for (_i, ref_w, _w), a, b in zip(pages, plain, hyps):
            refset = set(ref_w)
            for x, y in zip(a, b):
                if x != y:
                    good += y in refset and x not in refset
                    bad += x in refset and y not in refset
        print(f"changes: {good} clearly right, {bad} clearly wrong")
        return
    tune = {i for i, _r, _w in pages if i % 2}
    test = {i for i, _r, _w in pages if not i % 2}
    plain = [[w for w, _c, _p in words] for _i, _r, words in pages]
    deltas = lm_deltas(pages, max_doubt=0.999)
    print(f"{len(pages)} pages; {len(deltas)} candidate scores")
    print(f"as read:   tune WER {wer(pages, plain, tune)[0]:.2%} | test WER {wer(pages, plain, test)[0]:.2%}")
    grid = itertools.product([0.5, 1.0, 1.5, 2.0, 3.0], [0.0, 0.3, 0.6, 1.0], [0.0, 0.5, 1.0, 2.0], [0.5, 0.9, 0.99])
    best = min(grid, key=lambda g: wer(pages, apply(pages, deltas, *g), tune)[1])
    lam, beta, margin, doubt = best
    hyps = apply(pages, deltas, lam, beta, margin, doubt)
    changed = sum(1 for a, b in zip(plain, hyps) for x, y in zip(a, b) if x != y)
    print(f"best on tune pages: language weight {lam}, book bonus {beta}, margin {margin}, only words under {doubt:.0%} sure")
    print(f"rescored:  tune WER {wer(pages, hyps, tune)[0]:.2%} | test WER {wer(pages, hyps, test)[0]:.2%} | words changed {changed}")
    fixed = broke = 0
    for (i, ref_w, _w), a, b in zip(pages, plain, hyps):
        if i not in test:
            continue
        fixed += max(0, Levenshtein.distance(ref_w, a) - Levenshtein.distance(ref_w, b))
        broke += max(0, Levenshtein.distance(ref_w, b) - Levenshtein.distance(ref_w, a))
    print(f"test pages: net edits fixed {fixed}, pages made worse by {broke} edits")


if __name__ == "__main__":
    main()
