"""Read a training book with the base model (OCR + parse, no checks), to locate its pages in an edition.

    python tools/model_read.py BOOK.pdf work/read-NAME [--model qwen3-vl-8b] [--concurrency 64]

The model's reading is far closer to the print than Tesseract's on vowelled verse, so
tools/build_training_pairs.py can trim each page's span to the words it shows. Uses the
endpoint recorded by ``pdf2epub bench deploy``; a high concurrency lets Modal start more
than one container.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf2epub import pipeline  # noqa: E402
from pdf2epub.bench import api_key, endpoints, wait_ready  # noqa: E402
from pdf2epub.ocr.models import get  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("work", type=Path)
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--pagenums-from", type=Path, help="copy pagenums.json from this work folder (same PDF)")
    args = ap.parse_args()
    cand = get(args.model)
    url = endpoints().get(args.model)
    if not url:
        raise SystemExit(f"no endpoint for {args.model}: run  pdf2epub bench deploy {args.model}")
    os.environ.setdefault("PDF2EPUB_API_KEY", api_key() or "")
    overrides = {"ocr.backend": "openai", "ocr.endpoint": url, "ocr.model": cand.hf_id,
                 "ocr.prompt_preset": cand.preset, "ocr.extra_body": dict(cand.extra_body),
                 "ocr.concurrency": args.concurrency, "checks.tesseract": False, "title": args.work.name}
    wd, cfg = pipeline.setup_book(args.pdf, args.work.parent, overrides, args.work.name)
    manifest = pipeline.ingest_pdf(args.pdf, wd, cfg)
    if args.pagenums_from and (args.pagenums_from / "pagenums.json").exists():
        shutil.copy(args.pagenums_from / "pagenums.json", wd.root / "pagenums.json")
    wait_ready(url)
    todo = [e for e in manifest.pages if not e.blank]
    print(f"Reading {len(todo)} pages with {args.model}…", flush=True)
    results = pipeline.ocr_pages(wd, cfg, todo, echo=lambda m: print(m, flush=True))
    for entry in todo:
        if entry.index in results:
            pipeline.parse_page(entry, results[entry.index], wd, cfg)
    print(f"done: {sum(1 for r in results.values() if not r.failed)} read, "
          f"{sum(1 for r in results.values() if r.failed)} failed", flush=True)


if __name__ == "__main__":
    main()
