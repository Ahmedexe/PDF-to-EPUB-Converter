"""Catching bad pages automatically (spec §4).

VLMs can produce fluent text that is not on the page, or loop and repeat a
line. A page is flagged when its output is empty although the page has ink,
was cut off, loops, contains unclear words, or its word count differs sharply
from Tesseract's count for the same image. Parse, Quran and note problems are
flagged by their own stages; :func:`finalize_status` turns all flags into the
page status.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections import Counter
from pathlib import Path

from ..config import BookConfig, default_tessdata_dir
from ..model import FlagCode, Page, PageEntry, Quran, Unclear, block_containers, iter_blocks, walk_inline_slots
from ..workdir import WorkDir

log = logging.getLogger(__name__)
STAGE = "health"
_WORD_RE = re.compile(r"\w", re.UNICODE)
_TAG_RE = re.compile(r"<[^<>]*>")


def _count_words(text: str) -> int:
    """Whitespace tokens with at least one letter or digit (same rule on both sides)."""
    return sum(1 for tok in text.split() if _WORD_RE.search(tok))


def _page_texts(page: Page, *, quran: bool = True) -> list[str]:
    """Every text line the model produced for the page, in reading order."""
    out: list[str] = []
    for block, _owner in iter_blocks(page):
        for container in block_containers(block):
            parts: list[str] = []
            for _c, _i, node in walk_inline_slots(container):
                if isinstance(node, str):
                    parts.append(node)
                elif isinstance(node, Unclear):
                    parts.append(node.text)
                elif isinstance(node, Quran):
                    parts.append(node.text if quran else " ")
            out.append("".join(parts))
    return out


def model_word_count(page: Page) -> int:
    """Words the model read on the page: main text, notes, independent regions,
    running header, printed page number and unclear spans (spec rule 7 of §5)."""
    words = sum(_count_words(t) for t in _page_texts(page))
    words += _count_words(page.running_header or "") + _count_words(page.printed_page or "")
    words += sum(_count_words(n.marker or "") for n in page.notes())
    return words


def find_repetition(text: str, min_repeats: int = 3) -> str | None:
    """A line or phrase that loops: the same line ≥ ``min_repeats`` times, or a word
    n-gram (4–60 words) repeated back to back ≥ ``min_repeats`` times."""
    lines = [ln.strip() for ln in text.split("\n") if len(ln.split()) >= 4]
    counts = Counter(lines)
    for line, n in counts.most_common(1):
        if n >= min_repeats:
            return line
    words = text.split()
    for n in range(4, 61):
        if len(words) < n * min_repeats:
            break
        for i in range(0, len(words) - n * min_repeats + 1):
            gram = words[i:i + n]
            if all(words[i + k * n:i + (k + 1) * n] == gram for k in range(1, min_repeats)):
                return " ".join(gram)
    return None


_MARKER_RE = re.compile(r"[(\[][\d٠-٩۰-۹]{1,2}[)\]]")


def _has_foot_rule(path: Path) -> bool:
    """True when the page image has a short horizontal rule in its lower part (the foot-note separator)."""
    try:
        import cv2

        from ..layout.detect import _blocks, _main_block, _rule_y

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return False
        main = _main_block(_blocks(gray))
        return main is not None and _rule_y(gray, main) is not None
    except Exception:  # never let a heuristic break the checks
        log.debug("foot-rule detection failed", exc_info=True)
        return False


def _has_ink(path: Path) -> bool | None:
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    except Exception:  # pragma: no cover - OpenCV missing
        return None
    if img is None:
        return None
    from ..ingest.cleanup import is_blank

    return not is_blank(img)


def tesseract_text(wd: WorkDir | None, entry_or_page: PageEntry | Page, cfg: BookConfig) -> str | None:
    """Tesseract's reading of the page image, cached in ``pages/NNNN.tess.txt``.

    Returns None (and raises no flag) when Tesseract or the Arabic data is missing.
    """
    if wd is None or not entry_or_page.image:
        return None
    cache = wd.page_tesseract(entry_or_page.index)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    image = wd.abs(entry_or_page.image)
    if not image.exists():
        return None
    # A shared cache keyed by the image bytes: the same page is never read twice, across books and runs.
    import hashlib
    import os

    digest = hashlib.sha256(image.read_bytes() + cfg.checks.tesseract_lang.encode()).hexdigest()[:32]
    shared = Path(os.environ.get("PDF2EPUB_CACHE") or Path.home() / ".cache" / "pdf2epub") / "tesseract" / f"{digest}.txt"
    if shared.exists():
        text = shared.read_text(encoding="utf-8")
        wd.write_text(cache, text)
        return text
    if shutil.which("tesseract") is None:
        return None
    tessdata = cfg.checks.tessdata_dir or default_tessdata_dir()
    if tessdata is None or not (Path(tessdata) / f"{cfg.checks.tesseract_lang.split('+')[0]}.traineddata").exists():
        return None
    try:
        import pytesseract

        text = pytesseract.image_to_string(str(image), lang=cfg.checks.tesseract_lang,
                                           config=f'--tessdata-dir "{tessdata}" --psm 6')
    except Exception:
        log.warning("Tesseract failed on %s", image, exc_info=True)
        return None
    wd.write_text(cache, text)
    try:
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_text(text, encoding="utf-8")
    except OSError:
        pass
    return text


def check_health(page: Page, cfg: BookConfig, *, tags: str | None = None, tess_text: str | None = None,
                 wd: WorkDir | None = None) -> None:
    """Add the stage-``health`` flags for one page (idempotent)."""
    page.clear_flags(STAGE)
    flag = lambda code, msg, sev="warning", target=None: page.add_flag(  # noqa: E731
        code, msg, severity=sev, target=target, stage=STAGE)
    reasons = page.ocr.finish_reasons if page.ocr else []
    if "error" in reasons:
        flag(FlagCode.OCR_FAILED, "the model call failed for this page", "error")
    if "length" in reasons:
        flag(FlagCode.TRUNCATED, "the model output was cut off (token limit)", "error")
    if page.ocr and page.ocr.layout_error:
        flag(FlagCode.LAYOUT_FAILED, f"layout step failed: {page.ocr.layout_error}")
    words = model_word_count(page)
    if words == 0 and not page.blank and "blank" not in reasons:
        ink = _has_ink(wd.abs(page.image)) if (wd is not None and page.image) else None
        if ink is not False:
            flag(FlagCode.EMPTY_OUTPUT, "no text was read, but the page is not blank", "error")
    if wd is not None and page.image and not page.notes() and words:
        if _has_foot_rule(wd.abs(page.image)):
            flag(FlagCode.MISSING_NOTES, "the page shows a foot-note rule, but no notes were read")
    if not page.notes():
        main_text = "\n".join(_page_texts(page))
        found = _MARKER_RE.findall(main_text)
        if found:
            flag(FlagCode.MARKER_IN_TEXT, f"note markers left in the text with no notes: {', '.join(found[:5])}")
    text = "\n".join(_page_texts(page, quran=False))
    loop = find_repetition(text, cfg.checks.repetition_min_repeats)
    if loop is None and tags:
        raw = _TAG_RE.sub("\n", tags)
        loop = find_repetition(raw, cfg.checks.repetition_min_repeats + 1)
    if loop:
        flag(FlagCode.REPETITION, f"repeated text: «{loop[:80]}»", "error")
    rescore_record = wd.page_tags(page.index).with_name(f"{wd.stem(page.index)}.rescore.json") if wd is not None else None
    if rescore_record is not None and rescore_record.exists():  # language-model rescoring (pdf2epub/rescore.py)
        import json

        changes = json.loads(rescore_record.read_text(encoding="utf-8")).get("changes") or []
        if changes:
            flag(FlagCode.WORDS_CORRECTED, f"{len(changes)} word(s) corrected by the language model: "
                 + ", ".join(f"{c['from']} → {c['to']}" for c in changes[:8]), "info")
    lines_record = wd.page_tags(page.index).with_name(f"{wd.stem(page.index)}.lines.json") if wd is not None else None
    if lines_record is not None and lines_record.exists():  # the completeness check (checks/completeness.py)
        import json

        record = json.loads(lines_record.read_text(encoding="utf-8"))
        if record.get("inserted"):
            flag(FlagCode.LINES_INSERTED, f"{len(record['inserted'])} skipped line(s) read on their own and added: "
                 + " / ".join(f"«{x['text'][:40]}»" for x in record["inserted"][:3]), "info")
        for x in record.get("skipped_missing", [])[:3]:
            flag(FlagCode.TEXT_LOST, f"a printed line seems missing: «{x['text'][:60]}»")
        for x in record.get("extra", [])[:3]:
            flag(FlagCode.EXTRA_TEXT, f"text no printed line matches: «{x['words'][:60]}»")
    for block, _owner in iter_blocks(page):
        for container in block_containers(block):
            for _c, _i, node in walk_inline_slots(container):
                if isinstance(node, Unclear):
                    flag(FlagCode.UNCLEAR_TEXT, f"unclear reading: «{node.text}»", target=block.id)
    if tess_text is not None:
        tw = _count_words(tess_text)
        if max(tw, words) >= 20:
            diff = abs(tw - words) / max(tw, words)
            if diff > cfg.checks.wordcount_tolerance:
                flag(FlagCode.WORDCOUNT_MISMATCH,
                     f"the model read {words} words, Tesseract {tw} ({diff:.0%} apart)")


def finalize_status(page: Page, cfg: BookConfig) -> None:
    """``flagged`` when any flag's severity blocks (``checks.block_on``), else ``clean``.
    Reviewed pages stay reviewed."""
    if page.status == "reviewed":
        return
    blocking = set(cfg.checks.block_on)
    page.status = "flagged" if any(f.severity in blocking for f in page.flags) else "clean"
