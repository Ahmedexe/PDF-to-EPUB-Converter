"""Page image clean-up with OpenCV: deskew, crop borders, split spreads (spec §10).

All functions take and return ``uint8`` numpy arrays (grayscale or BGR) and
never cut printed text: cropping keeps every ink pixel, including margin notes.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask (255 = ink) with specks removed."""
    _thr, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if mask.mean() > 127:  # light text on dark: Otsu split the other way
        mask = 255 - mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask


def _strip_dark_edges(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Box inside the dark scanner border (rows/columns that are mostly black)."""
    h, w = gray.shape
    dark = gray < 60
    rows = dark.mean(axis=1)
    cols = dark.mean(axis=0)
    y0 = 0
    while y0 < h // 4 and rows[y0] > 0.6:
        y0 += 1
    y1 = h
    while y1 > h - h // 4 and rows[y1 - 1] > 0.6:
        y1 -= 1
    x0 = 0
    while x0 < w // 4 and cols[x0] > 0.6:
        x0 += 1
    x1 = w
    while x1 > w - w // 4 and cols[x1 - 1] > 0.6:
        x1 -= 1
    return x0, y0, x1, y1


def is_blank(gray: np.ndarray) -> bool:
    """True when the page has (almost) no ink once scanner borders are ignored."""
    gray = to_gray(gray)
    x0, y0, x1, y1 = _strip_dark_edges(gray)
    inner = gray[y0:y1, x0:x1]
    if inner.size == 0:
        return True
    if int(inner.max()) - int(inner.min()) < 40:
        return True
    mask = _ink_mask(inner)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    area = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 6)
    return area / inner.size < 0.002


def estimate_skew(gray: np.ndarray, max_angle: float = 5.0) -> float:
    """Skew in degrees (positive = text turned counter-clockwise), by projection profiles.

    Text lines give the sharpest horizontal projection when the page is level;
    a coarse search then a fine one over ±``max_angle``.
    """
    gray = to_gray(gray)
    scale = 1000.0 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    mask = _ink_mask(small)
    if mask.mean() < 0.5:
        return 0.0
    h, w = mask.shape
    centre = (w / 2, h / 2)

    def sharpness(angle: float) -> float:
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rot = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rot.sum(axis=1).astype(np.float64)
        return float(np.var(np.diff(profile)))

    best = max(np.arange(-max_angle, max_angle + 1e-9, 0.5), key=sharpness)
    fine = max(np.arange(best - 0.5, best + 0.5 + 1e-9, 0.05), key=sharpness)
    # Rotating by +fine levels the text, so the text was turned by -fine.
    return round(-float(fine), 2)


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate counter-clockwise by ``angle`` degrees on a white canvas large enough to keep everything."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos + 0.5), int(h * cos + w * sin + 0.5)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    border = 255 if img.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=border)


def deskew(img: np.ndarray, max_angle: float = 5.0) -> tuple[np.ndarray, float]:
    """Correct small scan tilt; never turns a page by 90°. Returns (image, skew found)."""
    angle = estimate_skew(to_gray(img), max_angle)
    if abs(angle) < 0.15:
        return img, 0.0
    return rotate(img, -angle), angle


def content_box(gray: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of all ink inside the scanner border; the whole image if blank."""
    gray = to_gray(gray)
    bx0, by0, bx1, by1 = _strip_dark_edges(gray)
    inner = gray[by0:by1, bx0:bx1]
    mask = _ink_mask(inner)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]
    if not keep:
        return 0, 0, gray.shape[1], gray.shape[0]
    xs0 = min(stats[i, cv2.CC_STAT_LEFT] for i in keep)
    ys0 = min(stats[i, cv2.CC_STAT_TOP] for i in keep)
    xs1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in keep)
    ys1 = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in keep)
    return int(bx0 + xs0), int(by0 + ys0), int(bx0 + xs1), int(by0 + ys1)


def crop_borders(img: np.ndarray, pad: int = 20) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Remove scanner borders and empty margins, keeping ``pad`` pixels around all ink."""
    gray = to_gray(img)
    h, w = gray.shape
    x0, y0, x1, y1 = content_box(gray)
    box = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))
    if box[2] - box[0] < w * 0.2 or box[3] - box[1] < h * 0.2:
        return img, (0, 0, w, h)  # too little content to trust the box
    return img[box[1]:box[3], box[0]:box[2]], box


def find_spread_split(gray: np.ndarray) -> int | None:
    """x of the gutter of a two-page spread, or None.

    Only landscape images qualify; the gutter is the lowest-ink column band in the
    middle 30% of the width, with substantial ink on both sides.
    """
    gray = to_gray(gray)
    h, w = gray.shape
    if w < 1.15 * h:
        return None
    mask = _ink_mask(gray)
    cols = mask.mean(axis=0) / 255.0
    kernel = max(3, w // 100)
    smooth = np.convolve(cols, np.ones(kernel) / kernel, mode="same")
    lo, hi = int(w * 0.35), int(w * 0.65)
    x = lo + int(np.argmin(smooth[lo:hi]))
    # Centre of the low-ink band around the minimum (the gutter, not the edge of the text).
    floor = smooth[x] + 0.02 * max(smooth.max(), 1e-6)
    a_, b_ = x, x
    while a_ > lo and smooth[a_ - 1] <= floor:
        a_ -= 1
    while b_ < hi - 1 and smooth[b_ + 1] <= floor:
        b_ += 1
    x = (a_ + b_) // 2
    left, right = cols[: x].mean(), cols[x:].mean()
    if smooth[x] > 0.2 * min(left, right) or min(left, right) < 0.01:
        return None
    return x


def split_spread(img: np.ndarray, x: int) -> tuple[np.ndarray, np.ndarray]:
    """(right page, left page): in a right-to-left book the right half comes first."""
    return img[:, x:], img[:, :x]


def is_color(img: np.ndarray, tolerance: int = 12) -> bool:
    if img.ndim == 2 or img.shape[2] < 3:
        return False
    small = cv2.resize(img[:, :, :3], (200, 200), interpolation=cv2.INTER_AREA).astype(np.int16)
    spread = small.max(axis=2) - small.min(axis=2)
    return float((spread > tolerance).mean()) > 0.01
