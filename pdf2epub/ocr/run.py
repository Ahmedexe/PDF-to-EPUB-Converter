"""Read pages: layout, one call or one call per region, retry, merge, cache (spec §4, §6.3).

``pages/NNNN.tags`` holds the accepted model output and ``NNNN.ocr.json`` its
``OcrInfo`` plus every attempt's text. A page whose calls fail writes only the
``.ocr.json`` (with the error) and is retried by the next run; ``force`` re-reads
a page and bypasses the response cache.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from ..config import BookConfig
from ..model import LayoutRegion, OcrInfo, PageEntry, PageLayout
from ..parse import tags_parse_ok
from ..workdir import WorkDir
from .client import OcrClient
from .prompts import page_prompts, prompt_version, region_prompts
from .types import Completion, OcrError

log = logging.getLogger(__name__)
_FIG_BOX_RE = re.compile(r'(<figure\b[^>]*\bbox=")([^"]*)(")')
_HASHIYA_WRAP_RE = re.compile(r"^\s*<hashiya\b[^>]*>(.*)</hashiya>\s*$", re.S)


def plain_to_tags(text: str) -> str:
    """Plain-text model output -> one <p> per paragraph (blank-line separated)."""
    import html as _html

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", "", text.strip())
    # Some OCR models answer in light HTML (<b>, <i>, <h1>, <br>): keep the text, not the markup.
    text = re.sub(r"<\s*(br|/p|/h[1-6]|/div|/li|/tr)\s*/?>", "\n\n", text, flags=re.I)
    text = re.sub(r"</?[A-Za-z][^<>]*>", "", text)
    text = _html.unescape(text)
    paras = [" ".join(chunk.split()) for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    return "\n".join(f"<p>{escape(p)}</p>" for p in paras) + ("\n" if paras else "")


@dataclass
class PageOcr:
    tags: str
    info: OcrInfo
    layout: PageLayout | None = None

    @property
    def failed(self) -> bool:
        return "error" in self.info.finish_reasons


def _load_done(wd: WorkDir, index: int) -> PageOcr | None:
    tags, meta = wd.page_tags(index), wd.page_ocr_info(index)
    if not tags.exists() or not meta.exists():
        return None
    try:
        info = OcrInfo.model_validate(json.loads(meta.read_text(encoding="utf-8"))["info"])
    except (ValueError, KeyError):
        return None
    if "error" in info.finish_reasons:
        return None
    layout = None
    if wd.page_regions(index).exists():
        layout = PageLayout.model_validate_json(wd.page_regions(index).read_text(encoding="utf-8"))
    info.cached = True
    return PageOcr(tags.read_text(encoding="utf-8"), info, layout)


def _save(wd: WorkDir, index: int, result: PageOcr, attempts: list[dict[str, object]]) -> None:
    record = {"info": json.loads(result.info.model_dump_json()), "attempts": attempts}
    if not result.failed:
        wd.write_text(wd.page_tags(index), result.tags)
    else:
        wd.page_tags(index).unlink(missing_ok=True)
    wd.write_json(wd.page_ocr_info(index), record)


async def _read_with_retry(client: OcrClient, image: Path, system: str, user: str, request_id: str,
                           cfg: BookConfig, info: OcrInfo, attempts: list[dict[str, object]], *,
                           refresh: bool) -> Completion:
    """One read, plus one higher-resolution retry when the output does not parse or is cut off."""
    from ..checks.flags import find_repetition

    sides = [cfg.ocr.max_side, cfg.ocr.retry_max_side]
    best: Completion | None = None
    for n, side in enumerate(sides):
        completion, cached = await client.read(image, system, user, request_id=request_id, max_side=side,
                                               refresh=refresh)
        if completion.finish_reason == "length" or find_repetition(completion.text, 4):
            # The model looped (or ran out of tokens, usually by looping): read again with a
            # repetition penalty (an extra request parameter).
            attempts.append({"request": request_id, "max_side": side, "finish_reason": "repetition",
                             "cached": cached, "text": completion.text})
            completion, cached = await client.read(image, system, user, request_id=request_id, max_side=side,
                                                   refresh=refresh, extra={"repetition_penalty": 1.08})
        info.calls += 0 if cached else 1
        info.cached = info.cached or cached
        attempts.append({"request": request_id, "max_side": side, "finish_reason": completion.finish_reason,
                         "cached": cached, "text": completion.text})
        info.attempts = max(info.attempts, n + 1)
        if completion.usage.get("image_w"):
            info.image_px = [completion.usage["image_w"], completion.usage["image_h"]]
        info.elapsed_s = round((info.elapsed_s or 0.0) + completion.elapsed_s, 3)
        if info.model in ("", None):
            info.model = completion.model
        plain = cfg.ocr.prompt_preset == "plain"
        ok = completion.finish_reason != "length" and (plain or tags_parse_ok(completion.text))
        best = completion
        if ok:
            break
        if n + 1 < len(sides):
            log.info("%s: output %s; retrying at a higher resolution", request_id,
                     "cut off" if completion.finish_reason == "length" else "does not parse")
    assert best is not None
    info.finish_reasons.append(best.finish_reason)
    return best


def _page_units(box: str, region: LayoutRegion, pad: int, width: int, height: int) -> str:
    """A figure box given in 0–1000 units of an (unrotated) region crop -> 0–1000 page units."""
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", box)]
    if len(nums) != 4 or region.rotation:
        return box
    cx0, cy0 = max(0.0, region.bbox[0] - pad), max(0.0, region.bbox[1] - pad)
    cw = min(width, region.bbox[2] + pad) - cx0
    ch = min(height, region.bbox[3] + pad) - cy0
    x0, y0, x1, y1 = nums
    out = [(cx0 + x0 / 1000 * cw) * 1000 / width, (cy0 + y0 / 1000 * ch) * 1000 / height,
           (cx0 + x1 / 1000 * cw) * 1000 / width, (cy0 + y1 / 1000 * ch) * 1000 / height]
    return " ".join(str(round(v)) for v in out)


def merge_regions(parts: list[tuple[LayoutRegion, str]], layout: PageLayout, pad: int) -> str:
    """One tags document for a page read region by region (main text first, then notes)."""
    out: list[str] = []
    for region in layout.regions:
        if region.kind == "header" and region.text:
            out.append(f"<header>{escape(region.text)}</header>")
    previous_main = False
    for region, text in parts:
        text = text.strip()
        if region.kind == "main":
            text = _FIG_BOX_RE.sub(lambda m: m.group(1) + _page_units(m.group(2), region, pad, layout.width,
                                                                       layout.height) + m.group(3), text)
            if previous_main and out:
                # A paragraph cut by a band boundary: the band ends mid-sentence and the next starts a <p>.
                prev = out[-1].rstrip()
                if prev.endswith("</p>") and text.startswith("<p>") and \
                        not re.search(r"[.:؟!»)\]]\s*</p>$", prev):
                    out[-1] = prev[: -len("</p>")] + " " + text[len("<p>"):]
                    continue
            out.append(text)
            previous_main = True
            continue
        previous_main = False
        inner = _HASHIYA_WRAP_RE.match(text)
        if inner:
            text = inner.group(1).strip()
        bbox = ",".join(f"{v:.1f}" for v in region.bbox)
        attrs = f'pos="{region.pos or "frame"}" bbox="{bbox}"'
        if region.rotation:
            attrs += f' rotation="{region.rotation:g}"'
        if region.layer != 1:
            attrs += f' layer="{region.layer}"'
        if region.kind == "frame":
            attrs = f'pos="frame" independent="1" bbox="{bbox}"'
        out.append(f"<hashiya {attrs}>\n{text}\n</hashiya>")
    for region in layout.regions:
        if region.kind == "figure":
            b = region.bbox
            box = " ".join(str(round(v)) for v in (b[0] * 1000 / layout.width, b[1] * 1000 / layout.height,
                                                     b[2] * 1000 / layout.width, b[3] * 1000 / layout.height))
            out.append(f'<figure box="{box}"/>')
    for region in layout.regions:
        if region.kind == "page_number" and region.text:
            out.append(f"<pagenum>{escape(region.text)}</pagenum>")
    return "\n".join(out) + "\n"


async def _detect_layout(entry: PageEntry, wd: WorkDir, cfg: BookConfig, client: OcrClient, refresh: bool) -> PageLayout:
    from ..layout import detect

    return await detect.detect_layout(entry, wd, cfg, client, refresh=refresh)


def _write_crops(entry: PageEntry, layout: PageLayout, wd: WorkDir, pad: int) -> PageLayout:
    from ..layout import crop

    return crop.write_region_crops(entry, layout, wd, pad)


def _reading_order(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    from ..layout import crop

    return crop.reading_order(regions)


async def ocr_page(entry: PageEntry, wd: WorkDir, cfg: BookConfig, client: OcrClient, *,
                   force: bool = False) -> PageOcr:
    """Read one page (see the module docstring)."""
    if not force:
        done = _load_done(wd, entry.index)
        if done is not None:
            return done
    info = OcrInfo(backend=client.backend.name, model=client.model, prompt_version=prompt_version(cfg))
    attempts: list[dict[str, object]] = []
    stem = wd.stem(entry.index)
    if entry.blank:
        info.finish_reasons = ["blank"]
        result = PageOcr("", info, None)
        _save(wd, entry.index, result, attempts)
        return result
    start = time.monotonic()
    layout: PageLayout | None = None
    try:
        layout = await _detect_layout(entry, wd, cfg, client, force)
        info.layout_source = layout.source if layout.source != "none" else None
        info.layout_error = layout.error
        if layout.split:
            layout = _write_crops(entry, layout, wd, cfg.layout.pad_px)
            parts: list[tuple[LayoutRegion, str]] = []
            for region in _reading_order(layout.regions):
                if region.kind not in ("main", "hashiya", "frame") or not region.image:
                    continue
                system, user = region_prompts(cfg, region)
                completion = await _read_with_retry(client, wd.abs(region.image), system, user,
                                                    f"{stem}.{region.id}", cfg, info, attempts, refresh=force)
                parts.append((region, completion.text))
            tags = merge_regions(parts, layout, cfg.layout.pad_px)
        else:
            system, user = page_prompts(cfg)
            completion = await _read_with_retry(client, wd.abs(entry.image), system, user, stem, cfg, info,
                                                attempts, refresh=force)
            tags = plain_to_tags(completion.text) if cfg.ocr.prompt_preset == "plain" else completion.text
            tokens_file = wd.page_tags(entry.index).with_name(f"{stem}.tokens.json")
            if completion.tokens and "".join(t["t"] for t in completion.tokens) == tags:
                # compact: a token the model was 99% sure of keeps no alternatives (they never win)
                slim = [{"t": t["t"], "lp": round(t["lp"], 3), "alt": [] if t["lp"] > -0.01 else
                         [[a, round(lp, 3)] for a, lp in t["alt"]]} for t in completion.tokens]
                wd.write_text(tokens_file, json.dumps(slim, ensure_ascii=False, separators=(",", ":")))
            else:
                tokens_file.unlink(missing_ok=True)
    except (OcrError, FileNotFoundError, OSError) as exc:
        log.warning("page %d: OCR failed: %s", entry.index, exc)
        info.finish_reasons.append("error")
        attempts.append({"error": str(exc)})
        tags = ""
    info.elapsed_s = round(time.monotonic() - start, 3)
    result = PageOcr(tags, info, layout)
    _save(wd, entry.index, result, attempts)
    return result


async def ocr_book(entries: list[PageEntry], wd: WorkDir, cfg: BookConfig, *, force: bool = False,
                   client: OcrClient | None = None,
                   progress: Callable[[int, int, PageOcr | None], None] | None = None) -> dict[int, PageOcr]:
    """Read many pages concurrently (bounded by the client); one failed page never stops the book."""
    own = client is None
    client = client or OcrClient(cfg, wd)
    results: dict[int, PageOcr] = {}
    done = 0
    lock = asyncio.Lock()
    # Pages enter a bounded pipeline: started all at once, every page's local work (layout
    # detection, image encoding) would run before the first request goes out.
    pages = asyncio.Semaphore(max(1, cfg.ocr.concurrency) * 2)

    async def one(entry: PageEntry) -> None:
        nonlocal done
        try:
            async with pages:
                res = await ocr_page(entry, wd, cfg, client, force=force)
        except Exception as exc:  # never abort the book
            log.exception("page %d failed", entry.index)
            res = PageOcr("", OcrInfo(finish_reasons=["error"], backend=client.backend.name), None)
            _save(wd, entry.index, res, [{"error": repr(exc)}])
        async with lock:
            results[entry.index] = res
            done += 1
            if progress:
                progress(done, len(entries), res)

    try:
        await asyncio.gather(*(one(e) for e in entries))
    finally:
        if own:
            await client.aclose()
    return results
