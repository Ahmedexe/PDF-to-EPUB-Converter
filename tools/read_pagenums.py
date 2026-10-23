"""Read printed page numbers with the OCR model from narrow strips (top and bottom of each page).

    python tools/read_pagenums.py WORK_DIR [--model qwen3-vl-8b] [--concurrency 32]

Tesseract misreads Arabic-Indic page numbers too often to build training targets from.
The strips are small, so this costs little GPU time. Writes ``WORK_DIR/pagenums.json``:
``{"<page index>": {"top": "...", "bottom": "..."}}`` with the model's answers
(digits, or "" when it saw none); tools/build_training_pairs.py then keeps only
numbers that fit the page sequence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdf2epub.bench import api_key, endpoints, wait_ready  # noqa: E402
from pdf2epub.config import OcrSettings  # noqa: E402
from pdf2epub.ocr.backends import OpenAIBackend  # noqa: E402
from pdf2epub.ocr.models import get  # noqa: E402
from pdf2epub.workdir import WorkDir  # noqa: E402

SYSTEM = ("You read page numbers on scanned Arabic book pages. The image is a strip cut from the top or "
          "bottom of a page. If it shows a printed page number (usually alone at the top or bottom; "
          "Arabic-Indic digits like ١٢٣ or Western digits), answer with that number exactly as printed and "
          "nothing else. Ignore numbers inside running text, verse or note numbers. If there is no page "
          "number, answer NONE.")
USER = "The page number in this strip:"


def strips(image: Path) -> dict[str, bytes]:
    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    h, _ = gray.shape
    out = {}
    for name, part in (("top", gray[: int(h * 0.11)]), ("bottom", gray[int(h * 0.91):])):
        ok, buf = cv2.imencode(".png", part)
        if ok:
            out[name] = buf.tobytes()
    return out


def clean(answer: str) -> str:
    m = re.search(r"[٠-٩0-9]{1,4}", answer)
    return m.group(0) if m and "NONE" not in answer.upper() else ""


async def run(wd: WorkDir, model: str, concurrency: int) -> None:
    out_file = wd.root / "pagenums.json"
    done = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else {}
    url = endpoints().get(model)
    if not url:
        raise SystemExit(f"no endpoint for {model}: run  pdf2epub bench deploy {model}")
    import os
    os.environ.setdefault("PDF2EPUB_API_KEY", api_key() or "")
    settings = OcrSettings(endpoint=url, model=get(model).hf_id, timeout_s=300, retries=4)
    backend = OpenAIBackend(settings)
    wait_ready(url)
    sem = asyncio.Semaphore(concurrency)
    manifest = wd.load_manifest()

    async def one(i: int, image: Path) -> None:
        if str(i) in done:
            return
        res = {}
        for name, data in strips(image).items():
            async with sem:
                c = await backend.complete(data, SYSTEM, USER, request_id=f"{i:04d}.{name}", max_tokens=12,
                                           temperature=0.0, extra={})
            res[name] = clean(c.text)
        done[str(i)] = res
        if len(done) % 100 == 0:
            print(f"{len(done)} pages", flush=True)
            out_file.write_text(json.dumps(done, ensure_ascii=False, indent=0), encoding="utf-8")

    await asyncio.gather(*(one(e.index, wd.abs(e.image)) for e in manifest.pages if not e.blank))
    await backend.aclose()
    out_file.write_text(json.dumps(done, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"wrote {out_file} ({len(done)} pages)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir", type=Path)
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    asyncio.run(run(WorkDir(args.work_dir), args.model, args.concurrency))


if __name__ == "__main__":
    main()
