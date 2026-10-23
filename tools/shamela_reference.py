"""Reference texts for a test set from a page-matched al-Maktaba al-Shamela edition.

    python tools/shamela_reference.py PAGES_DIR OUT_DIR --first-printed 151 --count 100

PAGES_DIR holds downloaded Shamela pages (``https://shamela.ws/book/<id>/<n>`` saved as
``<n>.html``). Each page carries its printed page number (``data-page-num``); test page
``i`` is printed page ``first-printed + i - 1``. Writes ``OUT_DIR/NNNN.txt``: the main
text, a line ``---``, then the foot notes. Note markers ([١]) are dropped. Shamela keeps
only shadda, so these references score letters and words, not diacritics.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


def page_text(raw: str) -> tuple[int | None, str, str]:
    m = re.search(r'<div class="nass[^"]*"[^>]*data-page-num="(\d+)"[^>]*>(.*?)</div>', raw, re.S)
    if not m:
        return None, "", ""
    num, body = int(m.group(1)), m.group(2)
    main, _, notes = body.partition("<hr>")

    def clean(part: str) -> str:
        part = re.sub(r'<span class="c4">\s*\[(?:[٠-٩0-9]{1,3}|\*{1,3})\]\s*</span>', " ", part)  # note markers
        part = re.sub(r"<a [^>]*class=\"btn_tag[^>]*>.*?</a>", " ", part, flags=re.S)
        lines = [html.unescape(re.sub(r"<[^>]+>", " ", p)) for p in re.findall(r"<p[^>]*>(.*?)</p>", part, re.S)]
        lines = [re.sub(r"\s*\.\.\.\s*", "  ", ln) for ln in lines]  # "sadr ... ajuz" -> two half-lines
        lines = [re.sub(r"^\s*\[[٠-٩0-9]+\]\s*", "", ln) for ln in lines]  # marker at the start of a note
        return "\n".join(" ".join(ln.split()) for ln in lines if ln.strip())

    return num, clean(main), clean(notes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--first-printed", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    args = ap.parse_args()
    by_printed: dict[int, tuple[str, str]] = {}
    for f in args.pages.glob("*.html"):
        num, main_text, notes = page_text(f.read_text(encoding="utf-8"))
        if num is not None:
            by_printed[num] = (main_text, notes)
    args.out.mkdir(parents=True, exist_ok=True)
    missing = []
    for i in range(1, args.count + 1):
        printed = args.first_printed + i - 1
        if printed not in by_printed:
            missing.append(printed)
            continue
        main_text, notes = by_printed[printed]
        (args.out / f"{i:04d}.txt").write_text(f"{main_text}\n---\n{notes}\n", encoding="utf-8")
    print(f"wrote {args.count - len(missing)} references; missing printed pages: {missing}")


if __name__ == "__main__":
    main()
