"""Ingest a scanned book and run Tesseract on every page (in parallel), for building training pairs.

    python tools/prepare_book.py BOOK.pdf work/train-NAME [--jobs 10]

Tesseract output is cached per image (shared cache), so re-runs are instant.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf2epub.checks.flags import tesseract_text  # noqa: E402
from pdf2epub.config import BookConfig  # noqa: E402
from pdf2epub.ingest.render import ingest_pdf  # noqa: E402
from pdf2epub.workdir import WorkDir  # noqa: E402


def _one(args: tuple[str, int]) -> int:
    root, index = args
    wd = WorkDir(root)
    entry = next(e for e in wd.load_manifest().pages if e.index == index)
    text = tesseract_text(wd, entry, BookConfig(slug=wd.root.name))
    return len((text or "").split())


def main() -> None:
    pdf, root = Path(sys.argv[1]), Path(sys.argv[2])
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 10
    wd = WorkDir(root)
    cfg = BookConfig(slug=root.name)
    manifest = ingest_pdf(pdf, wd, cfg, progress=lambda n, t: print(f"ingest {n}/{t}", flush=True) if n % 100 == 0 or n == t else None)
    todo = [(str(root), e.index) for e in manifest.pages if not e.blank]
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for _ in pool.map(_one, todo, chunksize=4):
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"tesseract {done}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
