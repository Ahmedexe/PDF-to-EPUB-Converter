"""Tesseract on each page's main text and notes separately (split at the foot-note rule).

    python tools/split_tesseract.py WORK_DIR [--jobs 10]

Writes ``pages/NNNN.tess.main.txt`` and ``pages/NNNN.tess.notes.txt`` (empty when the page has
no rule). tools/build_training_pairs.py locates each part in the edition with its own text,
so a page's span no longer runs on into the notes or the next page.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf2epub.config import BookConfig, default_tessdata_dir  # noqa: E402
from pdf2epub.workdir import WorkDir  # noqa: E402


def split_paths(wd: WorkDir, i: int) -> tuple[Path, Path]:
    base = wd.page_tesseract(i)
    return base.with_name(f"{i:04d}.tess.main.txt"), base.with_name(f"{i:04d}.tess.notes.txt")


def _one(args: tuple[str, int, str]) -> int:
    import cv2
    import pytesseract

    from pdf2epub.layout.detect import _blocks, _main_block, _rule_y

    root, index, image = args
    wd = WorkDir(root)
    main_out, notes_out = split_paths(wd, index)
    if main_out.exists() and notes_out.exists():
        return 0
    gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    main = _main_block(_blocks(gray))
    rule = _rule_y(gray, main) if main else None
    cfg = BookConfig(slug="x").checks
    config = f'--tessdata-dir "{cfg.tessdata_dir or default_tessdata_dir()}" --psm 6'
    parts = [gray] if rule is None else [gray[: max(10, rule - 4)], gray[rule + 4:]]
    texts = [pytesseract.image_to_string(p, lang=cfg.tesseract_lang, config=config) for p in parts]
    main_out.write_text(texts[0], encoding="utf-8")
    notes_out.write_text(texts[1] if len(texts) > 1 else "", encoding="utf-8")
    return 1


def main() -> None:
    root = Path(sys.argv[1])
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 10
    wd = WorkDir(root)
    todo = [(str(root), e.index, str(wd.abs(e.image))) for e in wd.load_manifest().pages if not e.blank]
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for _ in pool.map(_one, todo, chunksize=4):
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"split tesseract {done}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
