"""Crop regions and turn them upright; reading order of regions (spec §6.3 steps 2 and 4)."""

from __future__ import annotations

import cv2
import numpy as np

from ..model import REGION_ORDER, LayoutRegion, PageEntry, PageLayout, save_model
from ..workdir import WorkDir

READ_KINDS = ("main", "hashiya", "frame")


def crop_region(img: np.ndarray, region: LayoutRegion, pad: int = 12) -> np.ndarray:
    """Crop ``region`` with padding (clamped to the page), then rotate it clockwise by
    ``region.rotation`` so its text is upright."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = region.bbox
    x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
    x1, y1 = min(w, int(round(x1)) + pad), min(h, int(round(y1)) + pad)
    crop = img[y0:y1, x0:x1]
    angle = region.rotation % 360
    if crop.size == 0 or angle == 0:
        return crop
    exact = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if round(angle) in exact and abs(angle - round(angle)) < 1e-6:
        return cv2.rotate(crop, exact[round(angle)])
    ch, cw = crop.shape[:2]
    m = cv2.getRotationMatrix2D((cw / 2, ch / 2), -angle, 1.0)  # negative = clockwise
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(ch * sin + cw * cos + 0.5), int(ch * cos + cw * sin + 0.5)
    m[0, 2] += nw / 2 - cw / 2
    m[1, 2] += nh / 2 - ch / 2
    border = 255 if crop.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(crop, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=border)


def reading_order(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    """Main text first (frames right-to-left, top-to-bottom), then hashiya regions: bottom,
    right margin, left margin, top, frame. Header and page-number regions are not read."""
    mains = sorted([r for r in regions if r.kind == "main"], key=lambda r: (r.bbox[1], -r.bbox[2]))
    frames = sorted([r for r in regions if r.kind == "frame"], key=lambda r: (r.bbox[1], -r.bbox[2]))
    rank = {name: i for i, name in enumerate(REGION_ORDER)}
    notes = sorted([r for r in regions if r.kind == "hashiya"],
                   key=lambda r: (r.layer, rank.get(r.pos or "bottom", 99), r.bbox[1], -r.bbox[2]))
    return mains + notes + frames


def write_region_crops(entry: PageEntry, layout: PageLayout, wd: WorkDir, pad: int = 12) -> PageLayout:
    """Write ``pages/NNNN.<id>.png`` for every region that is read separately, set
    ``region.image`` and save the layout."""
    img = cv2.imread(str(wd.abs(entry.image)), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(entry.image)
    for region in layout.regions:
        if region.kind not in READ_KINDS:
            continue
        crop = crop_region(img, region, pad)
        path = wd.region_image(entry.index, region.id)
        cv2.imwrite(str(path), crop)
        region.image = wd.rel(path)
    save_model(layout, wd.page_regions(entry.index))
    return layout
