"""Book-level checks: Quran, poetry, hashiya links, bad pages (spec §4, §5, §6).

``run_checks`` is the one entry point the pipeline and the review screen call.
Checks run over the whole book in page order so each can look at neighbouring
pages. Reviewed pages, and pages outside ``only``, are read-only context (the
linker may still add or move generated refs in them for notes of other pages).
"""

from __future__ import annotations

import logging

from ..config import BookConfig
from ..model import Page
from ..workdir import WorkDir

log = logging.getLogger(__name__)


def run_checks(
    pages: list[Page],
    wd: WorkDir | None,
    cfg: BookConfig,
    *,
    tags: dict[int, str] | None = None,
    tesseract: bool | None = None,
    only: set[int] | None = None,
) -> list[Page]:
    """Run every check over ``pages`` (in book order), mutating and returning them.

    ``tags`` maps page index to the raw model output (read from the work folder
    when omitted). ``tesseract`` overrides ``cfg.checks.tesseract``. When the
    Quran style is ``auto`` and not yet resolved, it is resolved from these pages
    and saved in the book's settings (whole-book runs only).
    """
    from . import flags, hashiya, poetry, quran

    pages = sorted(pages, key=lambda p: p.index)
    use_tess = cfg.checks.tesseract if tesseract is None else tesseract
    editable = [p for p in pages if p.status != "reviewed" and (only is None or p.index in only)]
    for page in editable:
        raw = (tags or {}).get(page.index)
        if raw is None and wd is not None and wd.page_tags(page.index).exists():
            raw = wd.page_tags(page.index).read_text(encoding="utf-8")
        tess = flags.tesseract_text(wd, page, cfg) if (use_tess and wd is not None) else None
        flags.check_health(page, cfg, tags=raw, tess_text=tess, wd=wd)
    if cfg.text.quran_style == "auto" and not cfg.text.quran_style_detected and only is None:
        cfg.text.quran_style_detected = quran.resolve_style(pages, cfg)
        if wd is not None and wd.config_path.exists():
            wd.save_config(cfg)
    quran.check_quran(pages, cfg, log_path=wd.quran_log if wd is not None else None, only=only)
    poetry.check_poetry(pages, cfg, only=only)
    hashiya.link_notes(pages, cfg, only=only)
    for page in editable:
        flags.finalize_status(page, cfg)
    return pages
