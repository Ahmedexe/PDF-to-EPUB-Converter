"""Per-page reference texts for a test set, from a digital edition whose pages break differently.

    python tools/build_reference.py SHAMELA_PAGES_DIR work/bench-qwen3-vl-8b-v1 sample/test/reference

The edition's main text and notes are each joined into one stream. Each test page is
located in the stream with word 4-grams that occur only once in it, taken from one
anchor run (any reasonable model). Every model is then scored against the same
stretch, so boundary error affects all of them equally. Writes ``NNNN.txt``: main
text, a line ``---``, then the notes.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shamela_reference import page_text  # noqa: E402

from pdf2epub.arabic import normalize_for_match  # noqa: E402
from pdf2epub.eval.metrics import page_text as model_page_text  # noqa: E402
from pdf2epub.workdir import WorkDir  # noqa: E402


def stream(texts: list[str]) -> tuple[list[str], list[str]]:
    """(original words, normalised words) of the joined texts."""
    words = " ".join(texts).split()
    return words, [normalize_for_match(w) for w in words]


def locate(query: list[str], norm: list[str], grams: dict[tuple[str, ...], int]) -> tuple[int, int] | None:
    hits = []
    for j in range(len(query) - 3):
        pos = grams.get(tuple(query[j:j + 4]))
        if pos is not None:
            hits.append((j, pos))
    if len(hits) < 3:
        return None
    offsets = sorted(pos - j for j, pos in hits)
    mid = offsets[len(offsets) // 2]
    good = [(j, pos) for j, pos in hits if abs((pos - j) - mid) <= 40]
    start = max(0, min(pos - j for j, pos in good[:3]))
    end = min(len(norm), max(pos + (len(query) - j) for j, pos in good[-3:]))
    return start, end


def main() -> None:
    pages_dir, run_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    printed = {}
    for f in pages_dir.glob("*.html"):
        num, main_text, notes = page_text(f.read_text(encoding="utf-8"))
        if num is not None:
            printed[num] = (main_text, notes)
    order = sorted(printed)
    main_words, main_norm = stream([printed[n][0] for n in order])
    note_words, note_norm = stream([printed[n][1] for n in order])
    index = {}
    for words, norm, key in ((main_words, main_norm, "main"), (note_words, note_norm, "notes")):
        counts = Counter(tuple(norm[i:i + 4]) for i in range(len(norm) - 3))
        index[key] = {tuple(norm[i:i + 4]): i for i in range(len(norm) - 3) if counts[tuple(norm[i:i + 4])] == 1}
    wd = WorkDir(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for i in wd.page_indices():
        page = wd.best_page(i)
        if page is None:
            continue
        parts = []
        for key, words, norm, text in (("main", main_words, main_norm, model_page_text(page)),
                                       ("notes", note_words, note_norm, model_page_text(page, notes=True, main=False))):
            q = [w for w in normalize_for_match(text).split() if w]
            span = locate(q, norm, index[key])
            parts.append(" ".join(words[span[0]:span[1]]) if span else "")
        if parts[0]:
            (out / f"{i:04d}.txt").write_text(f"{parts[0]}\n---\n{parts[1]}\n", encoding="utf-8")
            done += 1
    print(f"located {done} of {len(wd.page_indices())} pages")


if __name__ == "__main__":
    main()
