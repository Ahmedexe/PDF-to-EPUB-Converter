"""score_reference.py REF_DIR RUN [RUN ...]: per-page median WER (main, notes), missing words and diacritics per letter of bench runs (work/<RUN>) against reference texts (NNNN.txt: main, ---, notes)."""
import statistics, sys
from pathlib import Path
from rapidfuzz.distance import Levenshtein
from pdf2epub.arabic import DIACRITICS
from pdf2epub.bench import _lenient
from pdf2epub.eval.metrics import page_text
from pdf2epub.workdir import WorkDir

ref_dir = Path(sys.argv[1])
refs = {int(f.stem): f.read_text(encoding="utf-8").partition("\n---\n") for f in sorted(ref_dir.glob("*.txt"))}
for slug in sys.argv[2:]:
    wd = WorkDir(Path("work") / slug)
    m, n, dele, tot, letters, dia, bare = [], [], 0, 0, 0, 0, 0
    for i, (rm, _, rn) in refs.items():
        p = wd.best_page(i)
        if p is None:
            continue
        r, h = _lenient(rm).split(), _lenient(page_text(p)).split()
        ops = Levenshtein.editops(r, h)
        m.append(len(ops) / max(1, len(r))); dele += sum(o.tag == "delete" for o in ops); tot += len(r)
        if rn.strip():
            r2, h2 = _lenient(rn).split(), _lenient(page_text(p, notes=True, main=False)).split()
            n.append(Levenshtein.distance(r2, h2) / max(1, len(r2)))
        t = page_text(p, notes=True)
        l = sum(1 for c in t if "ء" <= c <= "ي"); d = sum(1 for c in t if c in DIACRITICS)
        letters += l; dia += d; bare += d < 0.01 * max(l, 1)
    notes = f"{statistics.median(n):5.1%}" if n else "  —  "
    print(f"{slug:28} main WER {statistics.median(m):5.1%} | notes WER {notes} | missing words {dele / max(1, tot):5.1%}"
          f" | diacritics/letter {dia / max(1, letters):.2f} | pages without diacritics {bare}/{len(m)}")
