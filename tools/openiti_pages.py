"""Split an OpenITI text into page files in the Shamela layout tools/build_training_pairs.py reads.

    python tools/openiti_pages.py TEXT.txt OUT_DIR [--volume 1]

OpenITI texts (github.com/OpenITI) mark the source edition's pages with ``PageVxxPyyy`` after
the page's last word, paragraphs with ``# ``, wrapped lines with ``~~``, half-lines with
``%~%``, headings with ``### |``, and milestones with ``msNNNN``. Each page becomes
``OUT_DIR/<n>.html`` (``n`` counts pages through the text) with one ``<p>`` per paragraph and
``data-page-num`` set to the printed page. They have no foot notes and, mostly, no punctuation
or diacritics, so they serve to correct a model's letters (``--mode corrected``), not as targets.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    only = sys.argv[sys.argv.index("--volume") + 1] if "--volume" in sys.argv else None
    out.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8").split("#META#Header#End#", 1)[-1]
    text = re.sub(r"\s*\n~~", " ", text)  # join wrapped lines (before milestones, which can end a line)
    text = re.sub(r"[ \t]*ms\d+[ \t]*", " ", text)
    text = text.replace("~~", " ")
    text = re.sub(r"^### \|+[^\n]*", "", text, flags=re.M)  # volume / section headings (not printed as such)
    n = 0
    for m in re.finditer(r"(.*?)PageV(\d+)P(\d+)", text, re.S):
        body, vol, page = m.group(1), int(m.group(2)), int(m.group(3))
        n += 1
        if only and vol != int(only):
            continue
        paras = []
        for para in re.split(r"\n#\s*", "\n" + body):
            para = " ".join(para.split())
            if not para:
                continue
            para = " ".join(para.replace("%~%", " ... ").split())  # verse: "sadr ... ajuz", as in Shamela pages
            paras.append(f"<p>{html.escape(para)}</p>")
        (out / f"{n}.html").write_text(
            f'<div class="nass" data-page-id="{n}" data-page-num="{page}" data-volume="{vol}">{"".join(paras)}</div>\n',
            encoding="utf-8")
    print(f"{n} pages in {out}")


if __name__ == "__main__":
    main()
