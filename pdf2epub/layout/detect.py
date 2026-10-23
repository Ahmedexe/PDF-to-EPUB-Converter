"""Region detection (spec §6.3 step 1).

Margin and framed حاشية break the simple "one page, one model call" approach:
margin text is often rotated and sits level with main-text lines. Pages that
need it get a layout pass first: the VLM returns region boxes, or a classical
CV estimate is used when its answer is unusable. Pages with notes only at the
foot are read in one call (step 3).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import cv2
import numpy as np

from ..config import BookConfig
from ..model import LayoutRegion, PageEntry, PageLayout, save_model
from ..ocr.prompts import layout_prompts
from ..ocr.types import Reader
from ..workdir import WorkDir

log = logging.getLogger(__name__)
KIND_ALIASES = {
    "main": "main", "text": "main", "body": "main", "matn": "main", "sharh": "main",
    "hashiya": "hashiya", "notes": "hashiya", "note": "hashiya", "footnotes": "hashiya", "footnote": "hashiya",
    "margin": "hashiya", "marginalia": "hashiya",
    "header": "header", "running_header": "header", "footer": "header",
    "page_number": "page_number", "pagenum": "page_number", "page-number": "page_number", "folio": "page_number",
    "figure": "figure", "image": "figure", "picture": "figure", "table": "figure",
    "frame": "frame",
}
POS_ALIASES = {"bottom": "bottom", "foot": "bottom", "footer": "bottom", "right": "margin-right",
               "margin-right": "margin-right", "left": "margin-left", "margin-left": "margin-left",
               "top": "top", "frame": "frame"}


# --------------------------------------------------------------------------- CV


def _blocks(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Text blocks: ink dilated so lines and words merge, as (x0, y0, x1, y1)."""
    _t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = mask.shape
    k = max(3, w // 60)
    merged = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, max(3, h // 120))))
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw * bh < (w * h) * 0.002:
            continue
        out.append((int(x), int(y), int(x + bw), int(y + bh)))
    return _merge_columns(out, gap=max(10, h // 25))


def _merge_columns(blocks: list[tuple[int, int, int, int]], gap: int) -> list[tuple[int, int, int, int]]:
    """Merge blocks stacked in the same column (x-overlap over half the narrower one, small vertical gap)."""
    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
    changed = True
    while changed:
        changed = False
        out: list[tuple[int, int, int, int]] = []
        for b in blocks:
            for k, a in enumerate(out):
                overlap = min(a[2], b[2]) - max(a[0], b[0])
                narrow = min(a[2] - a[0], b[2] - b[0])
                vgap = max(a[1], b[1]) - min(a[3], b[3])
                if overlap > 0.5 * narrow and vgap < gap:
                    out[k] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    changed = True
                    break
            else:
                out.append(b)
        blocks = out
    return blocks


def _long_lines(gray: np.ndarray) -> tuple[int, int]:
    """(long horizontal lines, long vertical lines) — ruled frames."""
    _t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = mask.shape
    hor = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.5), 1)))
    ver = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.5))))
    nh = cv2.connectedComponents(hor)[0] - 1
    nv = cv2.connectedComponents(ver)[0] - 1
    return nh, nv


def _main_block(blocks: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not blocks:
        return None
    return max(blocks, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def _line_extents(gray: np.ndarray) -> list[tuple[int, int, int]]:
    """(left, right, height) of each text line band, from the horizontal ink projection."""
    _t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    h, w = mask.shape
    rows = (mask > 0).sum(axis=1) > max(2, int(0.002 * w))
    out = []
    y = 0
    while y < h:
        if not rows[y]:
            y += 1
            continue
        y0 = y
        while y < h and rows[y]:
            y += 1
        band = mask[y0:y]
        cols = np.where((band > 0).sum(axis=0) > 0)[0]
        if len(cols) and (y - y0) >= 4:
            out.append((int(cols[0]), int(cols[-1]) + 1, y - y0))
    return out


def _text_area(gray: np.ndarray) -> tuple[int, int] | None:
    """Horizontal extent of the main text: the median left and right edges of its wider lines.

    The ends of poem lines and short last lines can fall outside it; ``needs_layout`` therefore
    also requires margin blocks to be several lines tall. Books whose margins are full of notes
    should use ``--regions always``.
    """
    w = gray.shape[1]
    lines = [ln for ln in _line_extents(gray) if (ln[1] - ln[0]) >= 0.3 * w]
    if not lines:
        return None
    lefts = sorted(ln[0] for ln in lines)
    rights = sorted(ln[1] for ln in lines)
    return lefts[len(lefts) // 2], rights[len(rights) // 2]


def _margin_blocks(blocks: list[tuple[int, int, int, int]], width: int,
                   area: tuple[int, int] | None) -> list[tuple[str, tuple[int, int, int, int]]]:
    if area is None:
        return []
    tx0, tx1 = area
    tol = 0.015 * width
    out = []
    for b in blocks:
        x0, _y0, x1, _y1 = b
        if (x1 - x0) >= 0.25 * width:
            continue
        if x1 <= tx0 - tol:
            out.append(("margin-left", b))
        elif x0 >= tx1 + tol:
            out.append(("margin-right", b))
    return out


def needs_layout(gray: np.ndarray) -> bool:
    """True for pages whose notes are not simply at the foot: substantial text beside the main
    text area (not just reference numbers in the margin), or ruled frames. Conservative."""
    gray = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    nh, nv = _long_lines(gray)
    if nv >= 2:
        return True
    h, w = gray.shape
    lines = _line_extents(gray)
    if not lines:
        return False
    line_h = sorted(ln[2] for ln in lines)[len(lines) // 2]
    margins = [(pos, b) for pos, b in _margin_blocks(_blocks(gray), w, _text_area(gray))
               if (b[3] - b[1]) >= max(3 * line_h, 0.05 * h)]  # a block of several lines, not a stray word
    ink = sum((b[2] - b[0]) * (b[3] - b[1]) for _pos, b in margins)
    return ink >= 0.012 * h * w


def _rule_y(gray: np.ndarray, main: tuple[int, int, int, int], *, first: bool = False) -> int | None:
    """y of a horizontal rule in the lower part of the page (the foot-note separator): the lowest
    one, or with ``first`` the topmost (where the main text ends when notes come in two sets)."""
    _t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = mask.shape
    # Thicken by a pixel or two first: a thin rule on a slightly tilted scan steps between rows.
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 8), 1)))
    n, _l, stats, _ = cv2.connectedComponentsWithStats(lines)
    best = None
    for i in range(1, n):
        x, y, bw, bh, _a = stats[i]
        # short (classic) or full-width rules; not a frame line at the very bottom edge
        if h * 0.4 < y < h * 0.96 and bh <= max(4, h // 300) and bw < w * 0.95:
            best = y if best is None else (min(best, y) if first else max(best, y))
    return best if best is not None else _dotted_rule_y(mask)


def _dotted_rule_y(mask: np.ndarray) -> int | None:
    """y of a dotted rule (a row of small, evenly spaced dots) in the lower part of the page."""
    h, w = mask.shape
    n, _l, stats, cents = cv2.connectedComponentsWithStats(mask)
    size = max(3, h // 250)
    dots = [(float(cents[i][1]), float(cents[i][0])) for i in range(1, n)
            if stats[i][2] <= size and stats[i][3] <= size and stats[i][4] >= 2 and h * 0.4 < cents[i][1] < h * 0.96]
    dots.sort()
    rows = []
    k = 0
    while k < len(dots):
        row = [dots[k]]
        while k + len(row) < len(dots) and dots[k + len(row)][0] - row[0][0] <= size:
            row.append(dots[k + len(row)])
        xs = sorted(x for _y, x in row)
        if len(xs) >= 15 and xs[-1] - xs[0] >= w / 8:
            gaps = np.diff(xs)
            # evenly spaced: most gaps close to the median gap (diacritics under a text line are not)
            if np.mean(np.abs(gaps - np.median(gaps)) <= max(2.0, 0.35 * np.median(gaps))) >= 0.7:
                rows.append(int(np.median([y for y, _x in row])))
        k += len(row)
    # A notes separator is one dotted row; several are dot leaders (a table of contents) or a
    # halftone pattern.
    return rows[0] if len(rows) == 1 else None


def cv_regions(gray: np.ndarray) -> list[LayoutRegion]:
    """Best-effort regions from classical CV: main text area, foot notes below a rule, margin blocks."""
    gray = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blocks = _blocks(gray)
    main = _main_block(blocks)
    if main is None:
        return []
    area = _text_area(gray)
    margins = [(pos, b) for pos, b in _margin_blocks(blocks, w, area) if (b[2] - b[0]) * (b[3] - b[1]) >= 0.004 * h * w]
    small = {b for _pos, b in _margin_blocks(blocks, w, area)}
    body = [b for b in blocks if b not in small] or [main]
    bx0, by0 = min(b[0] for b in body), min(b[1] for b in body)
    bx1, by1 = max(b[2] for b in body), max(b[3] for b in body)
    regions: list[LayoutRegion] = []
    rule = _rule_y(gray, main)
    if rule is not None and by0 < rule < by1:
        regions.append(LayoutRegion(id="main", kind="main", bbox=[bx0, by0, bx1, rule - 2]))
        regions.append(LayoutRegion(id="h1", kind="hashiya", pos="bottom", bbox=[bx0, rule + 2, bx1, by1]))
    else:
        regions.append(LayoutRegion(id="main", kind="main", bbox=[bx0, by0, bx1, by1]))
    n = sum(1 for r in regions if r.kind == "hashiya")
    for pos, (x0, y0, x1, y1) in margins:
        n += 1
        rotation = 90.0 if (y1 - y0) > 2.5 * (x1 - x0) else 0.0
        regions.append(LayoutRegion(id=f"h{n}", kind="hashiya", pos=pos, bbox=[x0, y0, x1, y1], rotation=rotation))
    return regions


def cv_bands(gray: np.ndarray, n: int = 2) -> list[LayoutRegion]:
    """The page cut into ``n`` main-text bands at white gaps between lines, plus the foot notes
    below the rule. Cuts fall in the widest gaps near equal heights, so no line is split."""
    gray = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    rows = (mask > 0).sum(axis=1)
    ink_rows = np.where(rows > max(2, int(0.002 * w)))[0]
    if len(ink_rows) == 0:
        return []
    top, bottom = int(ink_rows[0]), int(ink_rows[-1]) + 1
    blocks = _blocks(gray)
    main = _main_block(blocks)
    rule = _rule_y(gray, main) if main else None
    body_end = rule - 2 if rule is not None and top < rule < bottom else bottom
    cols = np.where(mask[top:bottom].sum(axis=0) > 0)[0]
    x0, x1 = (int(cols[0]), int(cols[-1]) + 1) if len(cols) else (0, w)
    empty = rows <= max(2, int(0.002 * w))
    ink_above = np.where(~empty[top:body_end])[0]
    body_end = top + int(ink_above[-1]) + 1 if len(ink_above) else body_end
    # White gaps between lines: (start, end) row ranges inside the body.
    gaps, y = [], top
    while y < body_end:
        if empty[y]:
            start = y
            while y < body_end and empty[y]:
                y += 1
            gaps.append((start, y))
        y += 1
    real = [g for g in gaps if g[1] - g[0] >= 3]
    typical = sorted(g[1] - g[0] for g in real)[len(real) // 2] if real else 0
    line_gaps = [g for g in real if g[1] - g[0] >= 0.5 * typical]
    cuts = [top]
    for k in range(1, max(1, n)):
        target = top + (body_end - top) * k // n
        if line_gaps:  # the line gap nearest the target height
            g = min(line_gaps, key=lambda g: abs((g[0] + g[1]) // 2 - target))
            cut = (g[0] + g[1]) // 2
            if cuts[-1] + 40 < cut < body_end - 40:
                cuts.append(cut)
    cuts.append(body_end)
    regions = [LayoutRegion(id=f"main{k + 1}" if k else "main", kind="main", bbox=[x0, a, x1, b])
               for k, (a, b) in enumerate(zip(cuts, cuts[1:])) if b - a > 20]
    if rule is not None and top < rule < bottom:
        regions.append(LayoutRegion(id="h1", kind="hashiya", pos="bottom", bbox=[x0, rule + 2, x1, bottom]))
    return regions


# --------------------------------------------------------------------------- VLM answer


def _extract_json(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", "", text.strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                continue
    raise ValueError("no JSON in the layout answer")


def _snap(angle: float) -> float:
    angle = angle % 360
    for target in (0, 90, 180, 270, 360):
        if abs(angle - target) <= 8:
            return float(target % 360)
    return round(angle, 1)


def parse_layout_response(text: str, width: int, height: int, bbox_format: str = "rel1000") -> list[LayoutRegion]:
    """Regions from the model's JSON answer (tolerant of fences, prose and key variants)."""
    data = _extract_json(text)
    items = data.get("regions", data.get("items", [])) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("layout answer has no region list")
    out: list[LayoutRegion] = []
    counters: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = KIND_ALIASES.get(str(item.get("type") or item.get("kind") or item.get("label") or "").lower().strip())
        box = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        if kind is None or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        if bbox_format == "rel1000":
            x0, x1 = x0 * width / 1000, x1 * width / 1000
            y0, y1 = y0 * height / 1000, y1 * height / 1000
        x0, x1 = sorted((max(0.0, min(width, x0)), max(0.0, min(width, x1))))
        y0, y1 = sorted((max(0.0, min(height, y0)), max(0.0, min(height, y1))))
        if (x1 - x0) < 8 or (y1 - y0) < 8:
            continue
        pos = None
        if kind == "hashiya":
            pos = POS_ALIASES.get(str(item.get("pos") or item.get("position") or "bottom").lower(), "bottom")
        try:
            rotation = _snap(float(item.get("rotation") or item.get("angle") or 0))
        except (TypeError, ValueError):
            rotation = 0.0
        prefix = {"main": "main", "hashiya": "h", "header": "header", "page_number": "pagenum",
                  "figure": "fig", "frame": "frame"}[kind]
        counters[prefix] = counters.get(prefix, 0) + 1
        rid = prefix if (prefix in ("main", "header", "pagenum") and counters[prefix] == 1) else f"{prefix}{counters[prefix]}"
        text_value = item.get("text")
        out.append(LayoutRegion(id=rid, kind=kind, pos=pos, bbox=[x0, y0, x1, y1], rotation=rotation,
                                layer=int(item.get("layer") or 1), independent=kind == "frame",
                                text=str(text_value).strip() if isinstance(text_value, (str, int)) else None))
    return out


def _needs_split(regions: list[LayoutRegion]) -> bool:
    for r in regions:
        if r.kind == "frame":
            return True
        if r.kind == "hashiya" and (r.pos != "bottom" or r.rotation):
            return True
    return False


async def detect_layout(entry: PageEntry, wd: WorkDir, cfg: BookConfig, client: Reader, *,
                        refresh: bool = False) -> PageLayout:
    """Decide how the page is read and save ``pages/NNNN.regions.json``; never raises for a bad answer."""
    layout = PageLayout(index=entry.index, width=entry.width, height=entry.height)
    mode = cfg.layout.regions
    if mode == "never" or entry.blank:
        return layout
    path = wd.abs(entry.image)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        layout.error = f"cannot read {entry.image}"
        return layout
    if mode == "bands":
        layout.regions, layout.source = cv_bands(gray, cfg.layout.bands), "cv"
        layout.split = len(layout.regions) > 1
        save_model(layout, wd.page_regions(entry.index))
        return layout
    if mode == "auto" and not needs_layout(gray):
        return layout
    system, user = layout_prompts(cfg)
    try:
        completion, _cached = await client.read(path, system, user, request_id=f"{wd.stem(entry.index)}.layout",
                                                max_side=cfg.ocr.max_side, refresh=refresh)
        regions = parse_layout_response(completion.text, entry.width, entry.height, cfg.layout.bbox_format)
        if not any(r.kind == "main" for r in regions):
            raise ValueError("no main-text region in the answer")
        layout.regions, layout.source = regions, "vlm"
    except Exception as exc:  # unusable answer or failed call: fall back to CV
        log.info("layout for page %d falls back to CV: %s", entry.index, exc)
        layout.regions, layout.source, layout.error = cv_regions(gray), "cv", str(exc)[:300]
    layout.split = _needs_split(layout.regions)
    save_model(layout, wd.page_regions(entry.index))
    return layout
