"""Download a turath.io book as page files in the Shamela layout tools/build_training_pairs.py reads.

    python tools/turath_pages.py BOOK_ID OUT_DIR

turath.io (``https://app.turath.io/book/<id>``) serves each page as text: lines, a ``_________``
line before the foot notes, markers written ``(١)``. Each page becomes ``OUT_DIR/<n>.html`` with
``<div class="nass" data-page-id=… data-page-num=…>``, one ``<p>`` per line, ``<hr>`` before the
notes, markers as ``<span class="c4">[١]</span>`` and notes starting ``[١]``, as Shamela pages do.
The raw API answers are kept beside them as ``<n>.json``.
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

API = "https://api.turath.io"
MARKER = re.compile(r"\s*\(([٠-٩0-9]{1,3})\)")


def fetch(url: str) -> dict:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pdf2epub/0.1 (training-data builder)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"could not fetch {url}")


def page_html(n: int, page: dict) -> str:
    meta = json.loads(page["meta"]) if isinstance(page.get("meta"), str) else page.get("meta") or {}
    text = re.sub(r"<span[^>]*>(.*?)</span>", r"\1", page.get("text", ""), flags=re.S)
    main, _, notes = text.partition("\n_________\n")
    lines = []
    for ln in main.splitlines():
        if ln.strip():
            body = html.escape(ln.strip())
            lines.append("<p>" + MARKER.sub(lambda m: f' <span class="c4">[{m.group(1)}]</span>', body) + "</p>")
    note_lines = []
    for ln in notes.splitlines():
        if ln.strip():
            body = html.escape(ln.strip())
            note_lines.append("<p>" + re.sub(r"^\(([٠-٩0-9]{1,3})\)\s*", r"[\1] ", body) + "</p>")
    num = meta.get("page", n)
    body = "".join(lines) + ("<hr>" + "".join(note_lines) if note_lines else "")
    return f'<div class="nass" data-page-id="{n}" data-page-num="{num}">{body}</div>\n'


def main() -> None:
    book, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 3:
        pages: int | None = int(sys.argv[3])
    else:
        info = fetch(f"{API}/book?id={book}&include=indexes&ver=3")
        pages = len(info["indexes"].get("page_map") or []) or None
    n = 1
    while pages is None or n <= pages:
        raw = out / f"{n}.json"
        if not raw.exists():
            page = fetch(f"{API}/page?book_id={book}&pg={n}&ver=3")
            if not page.get("text"):
                break
            raw.write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.5)
        page = json.loads(raw.read_text(encoding="utf-8"))
        (out / f"{n}.html").write_text(page_html(n, page), encoding="utf-8")
        n += 1
    print(f"{n - 1} pages in {out}")


if __name__ == "__main__":
    main()
