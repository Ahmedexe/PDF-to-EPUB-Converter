"""Tune the rescoring weights for a model, with the converter's own code, and report the result.

    python tools/tune_rescore.py WORK_DIR [--ref sample/test/reference] [--vocab vocab.tsv]

WORK_DIR holds a model's readings of a test book with their token alternatives
(``pages/NNNN.tokens.json``) and line-by-line readings (``pages/NNNN.lines.json``), as
``pdf2epub convert`` writes them with ``rescore.enabled`` off. Every page's candidates are scored
once by the language model (the deployed ``Scorer``); the weights (language weight, margin, doubt
threshold, line-reading score) are chosen on odd pages and reported on even pages, as errors
that hurt reading: wrong real words, non-words (with ``--vocab``, the classical vocabulary from
the ``pdf2epub-lm`` volume), garbled runs, missing and extra words. Spelling variants (hamza seats,
ى/ي, ة/ه) are counted apart as harmless. Reference text printed on another page (runs at a page
edge that no line reading has) is not counted.
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from pdf2epub import rescore as rs  # noqa: E402
from pdf2epub.bench import _lenient  # noqa: E402
from pdf2epub.checks.completeness import key  # noqa: E402

SPELL = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ؤ": "و", "ئ": "ي", "ى": "ي", "ة": "ه"})


def classify(ref: list[str], hyp: list[str], printed: set[str], vocab: set[str] | None, c: Counter) -> None:
    def on_page(ws: list[str]) -> float:
        return sum(1 for w in ws if key(w) in printed) / max(1, len(ws))

    for op in Levenshtein.opcodes(ref, hyp):
        r, h = ref[op.src_start:op.src_end], hyp[op.dest_start:op.dest_end]
        if op.tag == "equal":
            continue
        if op.tag == "delete":
            if not (len(r) >= 4 and on_page(r) < 0.5):
                c["missing words"] += len(r)
            continue
        if op.tag == "insert":
            if not (len(h) >= 4 and on_page(h) >= 0.5):
                c["extra words"] += len(h)
            continue
        if len(r) != len(h):
            spell = "".join(r).translate(SPELL).replace("ء", "") == "".join(h).translate(SPELL).replace("ء", "")
            c["spelling variant (harmless)" if spell else "garbled run"] += max(len(r), len(h))
            continue
        for a, b in zip(r, h):
            if a.translate(SPELL).replace("ء", "") == b.translate(SPELL).replace("ء", ""):
                c["spelling variant (harmless)"] += 1
            elif vocab is not None and b not in vocab:
                c["non-word"] += 1
            else:
                c["wrong word"] += 1


def main() -> None:
    import modal

    work = Path(sys.argv[1])
    ref_dir = Path(sys.argv[sys.argv.index("--ref") + 1]) if "--ref" in sys.argv else Path("sample/test/reference")
    vocab = None
    if "--vocab" in sys.argv:
        vocab = {ln.split("\t")[0] for ln in open(sys.argv[sys.argv.index("--vocab") + 1], encoding="utf-8")
                 if int(ln.rstrip("\n").split("\t")[1]) >= 3}
    pages = []
    for f in sorted(ref_dir.glob("*.txt")):
        i = int(f.stem)
        tok = work / "pages" / f"{i:04d}.tokens.json"
        lines = work / "pages" / f"{i:04d}.lines.json"
        if not tok.exists():
            continue
        words = rs.words_of(json.loads(tok.read_text(encoding="utf-8")))
        texts = [x["text"] for x in json.loads(lines.read_text(encoding="utf-8"))["lines"]] if lines.exists() else []
        ref = _lenient(f.read_text(encoding="utf-8").partition("\n---\n")[0]).split()
        pages.append((i, ref, words, texts, {key(w) for t in texts for w in t.split()}))
    print(f"{len(pages)} pages")

    scorer = modal.Cls.from_name("pdf2epub-lm", "Scorer")()

    def scored(line_score: float | None, doubt: float):
        import copy

        out = []
        all_jobs = []
        for _i, _ref, words, texts, _p in pages:
            ws = copy.deepcopy(words)
            if line_score is not None and texts:
                rs.add_line_candidates(ws, texts, line_score)
            jobs = rs.windows(ws, doubt)
            out.append((ws, jobs))
            all_jobs += [w for _a, _b, w in jobs]
        flat = []
        for k in range(0, len(all_jobs), 5000):
            flat += scorer.score.remote(all_jobs[k:k + 5000])
        pos, res = 0, []
        for ws, jobs in out:
            res.append((ws, {(a, b): flat[pos + k] for k, (a, b, _w) in enumerate(jobs)}))
            pos += len(jobs)
        return res

    def hyps(scored_pages, weight: float, margin: float, doubt: float) -> list[list[str]]:
        out = []
        for ws, scores in scored_pages:
            changes = {i: norm for i, _raw, norm, _g in rs.choose(ws, scores, weight=weight, margin=margin, doubt=doubt)}
            out.append([changes.get(k, w.norm) for k, w in enumerate(ws)])
        return out

    def report(h: list[list[str]], which: set[int]) -> tuple[float, Counter, int]:
        c, tot = Counter(), 0
        for (i, ref, _w, _t, printed), hyp in zip(pages, h):
            if i in which:
                tot += len(ref)
                classify(ref, hyp, printed, vocab, c)
        return sum(v for k, v in c.items() if "harmless" not in k) / max(1, tot), c, tot

    tune = {i for i, *_ in pages if i % 2}
    test = {i for i, *_ in pages if not i % 2}
    results = {}
    for line_score in (None, -3.0, -2.0, -1.5, -1.0):
        for doubt in (0.99, 0.999):
            sp = scored(line_score, doubt)
            for weight, margin in itertools.product((0.3, 0.5, 0.8), (0.3, 0.5, 1.0)):
                results[(line_score, doubt, weight, margin)] = (report(hyps(sp, weight, margin, doubt), tune)[0], sp)
    plain = [[w.norm for w in words] for _i, _r, words, _t, _p in pages]
    best_no_lines = min((k for k in results if k[0] is None), key=lambda k: results[k][0])
    best = min(results, key=lambda k: results[k][0])
    for name, k in (("as read", None), ("rescoring", best_no_lines), ("rescoring + line reading", best)):
        h = plain if k is None else hyps(results[k][1], k[2], k[3], k[1])
        s, c, tot = report(h, test)
        detail = ", ".join(f"{n} {v / tot:.2%}" for n, v in c.most_common())
        print(f"{name:26} errors that hurt reading (even pages): {s:.2%}   [{detail}]")
    print("best weights:", dict(zip(["line_score", "doubt", "weight", "margin"], best)))


if __name__ == "__main__":
    main()
