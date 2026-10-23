"""PDF -> page images (spec §2 step 1, §10).

For each PDF page: take the original scan untouched when the page is one
raster image covering it, otherwise render at the configured DPI; deskew,
crop scanner borders, and split two-page spreads (right half first, as the
book reads right to left). Book page indices are fixed by the first ingest and
kept in the manifest, because every later file is keyed by them.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from ..config import BookConfig
from ..model import Manifest, PageEntry
from ..workdir import WorkDir
from .cleanup import crop_borders, deskew, find_spread_split, is_blank, is_color, rotate, split_spread, to_gray

log = logging.getLogger(__name__)


class IngestError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pixmap_to_array(pix: pymupdf.Pixmap) -> np.ndarray:
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    if pix.colorspace is not None and pix.colorspace.n not in (1, 3):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        return arr[:, :, 0].copy()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def page_image(doc: pymupdf.Document, page: pymupdf.Page, cfg: BookConfig) -> tuple[np.ndarray, str, int]:
    """(image, "extract" | "render", dpi) for one PDF page."""
    if cfg.ingest.extract and page.rotation == 0:
        images = page.get_images(full=True)
        if len(images) == 1 and not page.get_text("text").strip():
            xref = images[0][0]
            try:
                rects = page.get_image_rects(xref, transform=True)
            except Exception:
                rects = []
            if len(rects) == 1:
                rect, matrix = rects[0]
                area = rect.width * rect.height
                upright = abs(matrix.b) < 1e-6 and abs(matrix.c) < 1e-6 and matrix.a > 0 and matrix.d > 0
                if upright and area >= 0.9 * page.rect.width * page.rect.height:
                    try:
                        pix = pymupdf.Pixmap(doc, xref)
                        img = _pixmap_to_array(pix)
                        dpi = int(round(img.shape[1] / (rect.width / 72.0)))
                        return img, "extract", dpi
                    except Exception:
                        log.debug("could not extract the image of page %d", page.number + 1, exc_info=True)
    pix = page.get_pixmap(dpi=cfg.ingest.dpi, colorspace=pymupdf.csRGB, alpha=False)
    return _pixmap_to_array(pix), "render", cfg.ingest.dpi


def _clean(img: np.ndarray, cfg: BookConfig) -> tuple[list[tuple[str, np.ndarray]], float]:
    angle = 0.0
    if cfg.ingest.deskew:
        img, angle = deskew(img)
    halves: list[tuple[str, np.ndarray]] = [("full", img)]
    if cfg.ingest.spreads != "no":
        x = find_spread_split(to_gray(img))
        if x is None and cfg.ingest.spreads == "yes" and img.shape[1] > img.shape[0]:
            x = img.shape[1] // 2
        if x is not None:
            right, left = split_spread(img, x)
            halves = [("right", right), ("left", left)]
    if cfg.ingest.crop:
        halves = [(side, crop_borders(h)[0]) for side, h in halves]
    return halves, angle


def _write(img: np.ndarray, path: Path, gray: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = to_gray(img) if gray else img
    tmp = path.with_suffix(".tmp.png")
    if not cv2.imwrite(str(tmp), out):
        raise IngestError(f"could not write {path}")
    tmp.replace(path)


def _clear_downstream(wd: WorkDir, index: int) -> None:
    for path in wd.pages_dir.glob(f"{wd.stem(index)}.*"):
        path.unlink(missing_ok=True)


def ingest_pdf(pdf_path: Path, wd: WorkDir, cfg: BookConfig, *, pdf_pages: set[int] | None = None,
               force: bool = False, progress: Callable[[int, int], None] | None = None) -> Manifest:
    """Write ``pages/NNNN.png`` for the PDF and return the saved manifest (see module docstring).

    ``force`` re-renders the selected pages and deletes everything derived from the
    indices it reassigns. Without ``force``, a run that would change existing
    indices raises :class:`IngestError`.
    """
    pdf_path = Path(pdf_path)
    wd.ensure()
    pdf_hash = _sha256(pdf_path)
    old = wd.load_manifest()
    if old.pages and old.pdf_sha256 and old.pdf_sha256 != pdf_hash and not force:
        raise IngestError("this work folder belongs to a different PDF (use --force to start over)")
    if force or old.pdf_sha256 != pdf_hash:
        old = Manifest()
    by_pdf: dict[int, list[PageEntry]] = {}
    for entry in old.pages:
        by_pdf.setdefault(entry.pdf_page, []).append(entry)

    doc = pymupdf.open(str(pdf_path))
    total = doc.page_count
    wanted = sorted(p for p in (pdf_pages or range(1, total + 1)) if 1 <= p <= total)
    new_after = max(by_pdf) if by_pdf else 0
    if by_pdf and any(p not in by_pdf and p < new_after for p in wanted):
        raise IngestError("pages before the last ingested page were never ingested; their indices would "
                          "shift later pages (use --force to re-ingest from scratch)")
    next_index = max((e.index for e in old.pages), default=0) + 1
    entries = {e.index: e for e in old.pages}
    for n, pno in enumerate(wanted, 1):
        existing = by_pdf.get(pno, [])
        if existing and all(wd.page_image(e.index).exists() and _sha256(wd.page_image(e.index)) == e.sha256
                            for e in existing):
            if progress:
                progress(n, len(wanted))
            continue
        page = doc[pno - 1]
        img, source, dpi = page_image(doc, page, cfg)
        halves, angle = _clean(img, cfg)
        if existing and len(existing) != len(halves):
            if not force:
                raise IngestError(f"PDF page {pno} now splits into {len(halves)} page(s) instead of "
                                  f"{len(existing)} (use --force)")
        indices = [e.index for e in sorted(existing, key=lambda e: e.index)] if existing and \
            len(existing) == len(halves) else []
        if not indices:
            indices = list(range(next_index, next_index + len(halves)))
            next_index += len(halves)
        for (side, half), index in zip(halves, indices):
            _clear_downstream(wd, index)
            path = wd.page_image(index)
            _write(half, path, cfg.ingest.grayscale)
            if cfg.ingest.grayscale and is_color(half):
                _write(half, wd.page_source_image(index), gray=False)
            gray = to_gray(half)
            entries[index] = PageEntry(
                index=index, pdf_page=pno, side=side, image=wd.rel(path), width=int(half.shape[1]),
                height=int(half.shape[0]), sha256=_sha256(path), source=source, dpi=dpi,
                deskew_angle=angle, blank=is_blank(gray),
            )
        if progress:
            progress(n, len(wanted))
    manifest = Manifest(source_pdf=str(pdf_path.resolve()), pdf_sha256=pdf_hash, pdf_page_count=total,
                        pages=[entries[i] for i in sorted(entries)])
    wd.save_manifest(manifest)
    return manifest


__all__ = ["IngestError", "ingest_pdf", "page_image", "rotate"]
