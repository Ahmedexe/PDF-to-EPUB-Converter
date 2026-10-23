"""The per-book work folder: every stage caches its output here (spec §10).

Layout of ``<root>/<slug>/``::

    book.json                 BookConfig (settings for this book)
    manifest.json             Manifest: page images produced by ingest
    pages/0037.png            cleaned page image sent to OCR
    pages/0037.src.png        colour source of the page, for figure crops (optional)
    pages/0037.regions.json   PageLayout from the layout step
    pages/0037.h1.png         upright crop of a region (one per region read separately)
    pages/0037.tags           accepted raw model output (all regions merged)
    pages/0037.ocr.json       OcrInfo plus every attempt's raw text, for debugging
    pages/0037.tess.txt       Tesseract text for the word-count cross-check
    pages/0037.parsed.json    Page straight from the parser
    pages/0037.json           Page after the checks (what assembly reads)
    pages/0037.reviewed.json  Page approved or corrected in the review screen (wins)
    figures/0037-b4.png       cropped figures
    cache/ocr/<key>.json      OCR response cache keyed by image hash, model and prompt
    training/0037.json        corrections saved as training data
    logs/quran.jsonl          every Quran replacement (spec §13: "every replacement logged")
    logs/review.jsonl         review actions with timestamps (review-time metric)
    out/<slug>.epub           built book
    out/book.json             assembled Book model
    out/epubcheck.json        EPUBCheck report

A crash or a re-run picks up where it stopped, and one page can be redone alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import BookConfig
from .model import Manifest, Page, _atomic_write, load_page, save_model


class WorkDir:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    # ---- folders
    @property
    def pages_dir(self) -> Path:
        return self.root / "pages"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def ocr_cache_dir(self) -> Path:
        return self.root / "cache" / "ocr"

    @property
    def training_dir(self) -> Path:
        return self.root / "training"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    def ensure(self) -> "WorkDir":
        for d in (self.pages_dir, self.figures_dir, self.ocr_cache_dir, self.training_dir,
                  self.logs_dir, self.out_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # ---- files
    @property
    def config_path(self) -> Path:
        return self.root / "book.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @staticmethod
    def stem(index: int) -> str:
        return f"{index:04d}"

    def page_image(self, i: int) -> Path:
        return self.pages_dir / f"{self.stem(i)}.png"

    def page_source_image(self, i: int) -> Path:
        return self.pages_dir / f"{self.stem(i)}.src.png"

    def page_regions(self, i: int) -> Path:
        return self.pages_dir / f"{self.stem(i)}.regions.json"

    def region_image(self, i: int, region_id: str) -> Path:
        return self.pages_dir / f"{self.stem(i)}.{region_id}.png"

    def page_tags(self, i: int) -> Path:
        return self.pages_dir / f"{self.stem(i)}.tags"

    def page_ocr_info(self, i: int) -> Path:
        return self.pages_dir / f"{self.stem(i)}.ocr.json"

    def page_tesseract(self, i: int) -> Path:
        return self.pages_dir / f"{self.stem(i)}.tess.txt"

    def page_parsed(self, i: int) -> Path:
        return self.pages_dir / f"{self.stem(i)}.parsed.json"

    def page_checked(self, i: int) -> Path:
        return self.pages_dir / f"{self.stem(i)}.json"

    def page_reviewed(self, i: int) -> Path:
        return self.pages_dir / f"{self.stem(i)}.reviewed.json"

    def figure_image(self, block_id: str) -> Path:
        return self.figures_dir / f"{block_id.removeprefix('p')}.png"

    def training_record(self, i: int) -> Path:
        return self.training_dir / f"{self.stem(i)}.json"

    @property
    def quran_log(self) -> Path:
        return self.logs_dir / "quran.jsonl"

    @property
    def review_log(self) -> Path:
        return self.logs_dir / "review.jsonl"

    def epub_path(self, slug: str, suffix: str = "") -> Path:
        return self.out_dir / f"{slug}{suffix}.epub"

    @property
    def book_model_path(self) -> Path:
        return self.out_dir / "book.json"

    @property
    def epubcheck_report(self) -> Path:
        return self.out_dir / "epubcheck.json"

    def rel(self, path: Path) -> str:
        """Path relative to the work folder, as stored in models."""
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def abs(self, rel: str) -> Path:
        """Resolve a model-stored relative path; refuses paths escaping the work folder."""
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"path escapes work folder: {rel}")
        return path

    # ---- config and manifest
    def load_config(self) -> BookConfig:
        return BookConfig.model_validate_json(self.config_path.read_text(encoding="utf-8"))

    def save_config(self, cfg: BookConfig) -> None:
        save_model(cfg, self.config_path)

    def load_manifest(self) -> Manifest:
        if not self.manifest_path.exists():
            return Manifest()
        return Manifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))

    def save_manifest(self, manifest: Manifest) -> None:
        save_model(manifest, self.manifest_path)

    # ---- pages
    def page_indices(self) -> list[int]:
        return [p.index for p in self.load_manifest().pages]

    def best_page(self, i: int) -> Page | None:
        """The most trusted version of a page: reviewed, else checked, else parsed."""
        for path in (self.page_reviewed(i), self.page_checked(i), self.page_parsed(i)):
            if path.exists():
                return load_page(path)
        return None

    def save_page(self, page: Page, kind: str) -> Path:
        """Save a page as ``parsed``, ``checked`` or ``reviewed``."""
        path = {"parsed": self.page_parsed, "checked": self.page_checked,
                "reviewed": self.page_reviewed}[kind](page.index)
        save_model(page, path)
        return path

    # ---- small helpers
    def write_text(self, path: Path, text: str) -> None:
        _atomic_write(path, text)

    def write_json(self, path: Path, data: Any) -> None:
        _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=1) + "\n")

    def append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def book_workdir(work_root: Path | str, slug: str) -> WorkDir:
    return WorkDir(Path(work_root) / slug)
