"""Local review screen (spec §2 step 6, §6.5, §9).

The page image beside the recognised text, with flagged words, verses and note
links highlighted. A page is fixed by editing it as tags, a wrong note link by
selecting the right words; every save is also written as training data, and
approving a page makes it ``reviewed`` (read-only for later checks).

Runs on 127.0.0.1 only by default; it serves files from the book's work
folder and nothing else.
"""

from __future__ import annotations

import logging
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..arabic import normalize_for_match, normalize_with_map, source_span
from ..config import BookConfig
from ..model import (
    BlockText,
    Figure,
    Heading,
    Matn,
    Note,
    NoteLink,
    Page,
    PageBreak,
    PageLayout,
    Poem,
    Quote,
    Quran,
    Ref,
    Separator,
    Unclear,
    block_containers,
    iter_blocks,
    load_page,
    next_ref_id,
    normalize_inlines,
    walk_inline_slots,
)
from ..parse import parse_tags, serialize_page
from ..workdir import WorkDir
from .training import save_training_record

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"
Recheck = Callable[[list[Page], WorkDir, BookConfig, set[int]], list[Page]]


def _default_recheck(pages: list[Page], wd: WorkDir, cfg: BookConfig, only: set[int]) -> list[Page]:
    from ..checks import run_checks

    return run_checks(pages, wd, cfg, only=only)


# --------------------------------------------------------------------------- rendering


def _render_inlines(nodes: list[Any], start: int, flags_for: set[str]) -> tuple[str, int]:
    """HTML for an inline list; text spans carry their BlockText offsets (data-off)."""
    out: list[str] = []
    pos = start
    for node in nodes:
        if isinstance(node, str):
            if node:
                out.append(f'<span data-off="{pos}">{escape(node)}</span>')
                pos += len(node)
        elif isinstance(node, Ref):
            cls = "ref generated" if node.generated else "ref"
            if node.note is None:
                cls += " unpaired"
            label = escape(node.n or "◆")
            out.append(f'<sup class="{cls}" data-ref={quoteattr(node.id)} data-note={quoteattr(node.note or "")}>'
                       f"{label}</sup>")
        elif isinstance(node, Quran):
            title = f"{node.status} {node.ref or ''} {node.score or ''}".strip()
            if node.ocr_text and node.ocr_text != node.text:
                title += f" — read: {node.ocr_text}"
            out.append(f'<span class="quran q-{node.status}" title={quoteattr(title)} data-off="{pos}">'
                       f"﴿{escape(node.text)}﴾</span>")
            pos += len(node.text)
        elif isinstance(node, Unclear):
            out.append(f'<span class="unclear" data-off="{pos}" title="unclear reading">{escape(node.text)}</span>')
            pos += len(node.text)
        elif isinstance(node, (Quote, Matn)):
            inner, pos = _render_inlines(node.c, pos, flags_for)
            cls = "quote" if isinstance(node, Quote) else "matn"
            out.append(f'<span class="{cls}">{inner}</span>')
        elif isinstance(node, PageBreak):
            continue
    return "".join(out), pos


def render_block(block: Any, flagged: set[str]) -> str:
    bid = getattr(block, "id", "")
    cls = "blk " + block.type + (" flagged" if bid in flagged else "")
    head = f'<div class={quoteattr(cls)} data-block={quoteattr(bid)}>'
    containers = block_containers(block)
    if isinstance(block, Poem):
        rows, pos = [], 0
        for verse in block.verses:
            sadr, pos = _render_inlines(verse.sadr, pos, flagged)
            ajuz = ""
            if verse.ajuz is not None:
                pos += 1
                ajuz, pos = _render_inlines(verse.ajuz, pos, flagged)
            pos += 1
            rows.append(f'<div class="verse"><span class="sadr">{sadr}</span><span class="ajuz">{ajuz or "—"}</span></div>')
        return head + "".join(rows) + "</div>"
    if isinstance(block, Separator):
        return head + escape(block.text or "* * *") + "</div>"
    if isinstance(block, Figure):
        cap, _ = _render_inlines(block.caption, 0, flagged)
        return head + f"[صورة] {cap}</div>"
    tag = f"h{min(block.level, 4)}" if isinstance(block, Heading) else "p"
    inner, _ = _render_inlines(containers[0] if containers else [], 0, flagged)
    return head + f"<{tag}>{inner}</{tag}></div>"


def render_note(note: Note, flagged: set[str]) -> str:
    link = note.link
    cls = f"note link-{link.status}" + (" flagged" if note.id in flagged else "")
    head = []
    if note.marker:
        head.append(f'<span class="marker">({escape(note.marker)})</span>')
    if note.lemma:
        head.append(f'<b class="lemma">قوله: ({escape(note.lemma)})</b>')
    if note.cont:
        head.append('<span class="cont" title="continues the previous page">=</span>')
    body = "".join(render_block(b, flagged) for b in note.blocks)
    meta = escape(f"{note.anchor_type} · {link.status}{(' · ' + link.method) if link.method else ''}"
                  f"{(' · ' + note.kind) if note.kind != 'author' else ''}")
    return (f'<div class={quoteattr(cls)} data-note={quoteattr(note.id)} data-ref={quoteattr(link.ref or "")} '
            f'data-page="{link.page or ""}"><div class="note-head">{" ".join(head)}'
            f'<span class="meta">{meta}</span><button class="linkbtn" data-note={quoteattr(note.id)}>ربط</button>'
            f"</div>{body}</div>")


def render_page(page: Page) -> dict[str, str]:
    flagged = {f.target for f in page.flags if f.target and f.severity != "info"}
    main = "".join(render_block(b, flagged) for b in page.blocks)
    notes = "".join(f'<div class="region" data-region="{escape(r.region)}"><div class="region-name">{escape(r.region)}'
                    f'{" · مستقل" if r.independent else ""}</div>'
                    + "".join(render_block(b, flagged) for b in r.blocks)
                    + "".join(render_note(n, flagged) for n in r.notes) + "</div>"
                    for r in page.hashiya)
    return {"main": main, "notes": notes}


# --------------------------------------------------------------------------- app


def create_app(wd: WorkDir, *, recheck: Recheck | None = None) -> FastAPI:
    app = FastAPI(title="pdf2epub review", docs_url=None, redoc_url=None, openapi_url=None)
    recheck = recheck or _default_recheck
    started: dict[int, float] = {}

    def cfg() -> BookConfig:
        return wd.load_config()

    def manifest_entry(i: int):
        entry = next((e for e in wd.load_manifest().pages if e.index == i), None)
        if entry is None:
            raise HTTPException(404, f"no page {i}")
        return entry

    def best(i: int) -> Page:
        page = wd.best_page(i)
        if page is None:
            raise HTTPException(404, f"page {i} has not been read yet")
        return page

    def log_action(i: int, action: str, **extra: Any) -> None:
        seconds = round(time.time() - started.get(i, time.time()), 1)
        wd.append_jsonl(wd.review_log, {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "page": i, "action": action,
                                        "seconds": seconds, **extra})

    def view(page: Page) -> dict[str, Any]:
        layout = None
        if wd.page_regions(page.index).exists():
            layout = PageLayout.model_validate_json(wd.page_regions(page.index).read_text(encoding="utf-8"))
        regions = [{"id": r.id, "kind": r.kind, "pos": r.pos, "bbox": r.bbox} for r in (layout.regions if layout else [])]
        regions += [{"id": h.id, "kind": "hashiya", "pos": h.region, "bbox": h.bbox} for h in page.hashiya
                    if h.bbox and not any(r["bbox"] == h.bbox for r in regions)]
        return {
            "index": page.index, "status": page.status, "printed_page": page.printed_page,
            "width": page.width, "height": page.height, "image": f"/image/{page.index}",
            "flags": [f.model_dump() for f in page.flags],
            "tags": serialize_page(page, full=True), "html": render_page(page), "regions": regions,
            "reviewed_file": wd.page_reviewed(page.index).exists(),
        }

    def window(i: int, edited: Page) -> list[Page]:
        """The edited page with its neighbours (read-only context for the checks)."""
        c = cfg()
        lo = i - 1 - c.checks.lemma_lookback_pages
        pages = []
        for j in range(lo, i + 2):
            if j == i:
                pages.append(edited)
                continue
            if wd.page_reviewed(j).exists():
                pages.append(load_page(wd.page_reviewed(j)))
            elif wd.page_checked(j).exists():
                pages.append(load_page(wd.page_checked(j)))
            elif wd.page_parsed(j).exists():
                pages.append(load_page(wd.page_parsed(j)))
        return pages

    def save_window(pages: list[Page], i: int) -> None:
        for p in pages:
            if p.index == i or wd.page_reviewed(p.index).exists():
                wd.save_page(p, "reviewed")
            else:
                wd.save_page(p, "checked")

    def recheck_and_save(i: int, page: Page, original: Page | None) -> Page:
        c = cfg()
        pages = window(i, page)
        recheck(pages, wd, c, {i})
        page = next(p for p in pages if p.index == i)
        save_window(pages, i)
        save_training_record(wd, page, original)
        return page

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/book")
    def book() -> dict[str, Any]:
        c = cfg()
        pages = []
        for i in wd.page_indices():
            page = wd.best_page(i)
            if page is None:
                pages.append({"index": i, "status": "unread", "printed_page": None, "counts": {}, "codes": []})
                continue
            counts: dict[str, int] = {}
            for f in page.flags:
                counts[f.severity] = counts.get(f.severity, 0) + 1
            pages.append({"index": i, "status": page.status, "printed_page": page.printed_page, "counts": counts,
                          "codes": sorted({f.code for f in page.flags if f.severity != "info"})})
        return {"slug": c.slug, "title": c.title or c.slug, "pages": pages}

    @app.get("/api/page/{i}")
    def get_page(i: int) -> dict[str, Any]:
        started[i] = time.time()
        return view(best(i))

    @app.get("/image/{i}")
    def image(i: int) -> FileResponse:
        entry = manifest_entry(i)
        path = wd.abs(entry.image)
        if not path.exists():
            raise HTTPException(404, "image missing")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/page/{i}/tags")
    def save_tags(i: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        tags = payload.get("tags")
        if not isinstance(tags, str):
            raise HTTPException(422, "tags must be a string")
        original = best(i)
        base = original.model_copy(update={"blocks": [], "hashiya": [], "flags": [], "status": "new"})
        page = parse_tags(tags, base, commentary=cfg().text.commentary).page
        # Manual links keep their anchors through the edit (re-placed by the linker if needed).
        manual = {n.id: n.link for n in original.notes() if n.link.method == "manual"}
        for note in page.notes():
            if note.id in manual and note.link.method != "manual":
                note.link = manual[note.id].model_copy()
        page = recheck_and_save(i, page, original)
        log_action(i, "edit")
        return view(page)

    @app.post("/api/page/{i}/link")
    def set_link(i: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        note_id, block_id = payload.get("note"), payload.get("block")
        try:
            start, end = int(payload.get("start")), int(payload.get("end"))
        except (TypeError, ValueError):
            raise HTTPException(422, "start and end must be integers")
        page = best(i)
        original = page.model_copy(deep=True)
        note = next((n for n in page.notes() if n.id == note_id), None)
        block = next((b for b in page.blocks if b.id == block_id), None)
        if note is None or block is None:
            raise HTTPException(404, "note or block not on this page")
        bt = BlockText(block)
        if not (0 <= start < end <= len(bt.text)):
            raise HTTPException(422, "selection outside the block")
        # Remove the note's current generated anchor on this page.
        for b, _o in iter_blocks(page):
            for container in block_containers(b):
                for c, k, node in list(walk_inline_slots(container)):
                    if isinstance(node, Ref) and node.generated and node.note == note.id:
                        c.remove(node)
                        normalize_inlines(c)
        bt = BlockText(block)
        words = bt.text[start:end].strip()
        norm, idx = normalize_with_map(bt.text)
        anchor = normalize_for_match(words)
        # Which occurrence of these words (in the page's main text, in order) was selected?
        occurrence, count = 1, 0
        for b in page.blocks:
            other = BlockText(b)
            onorm, oidx = normalize_with_map(other.text)
            pos = onorm.find(anchor)
            while pos >= 0 and anchor:
                count += 1
                a, z = source_span(other.text, oidx, pos, pos + len(anchor))
                if b is block and a >= start - 1 and z <= end + 1:
                    occurrence = count
                pos = onorm.find(anchor, pos + 1)
        end_off = end
        while end_off > start and bt.text[end_off - 1].isspace():
            end_off -= 1
        ref = Ref(id=next_ref_id(page), generated=True, note=note.id)
        bt.insert_at(end_off, ref)
        note.link = NoteLink(status="linked", method="manual", page=i, block=block.id, ref=ref.id,
                             offset=end_off, anchor=words, occurrence=occurrence)
        page = recheck_and_save(i, page, original)
        log_action(i, "link", note=note.id)
        return view(page)

    @app.post("/api/page/{i}/approve")
    def approve(i: int) -> dict[str, Any]:
        page = best(i)
        original = page.model_copy(deep=True)
        for block, _o in iter_blocks(page):
            for container in block_containers(block):
                for _c, _k, node in walk_inline_slots(container):
                    if isinstance(node, Quran) and node.status in ("flagged", "unchecked"):
                        node.status = "accepted"
        page.status = "reviewed"
        wd.save_page(page, "reviewed")
        save_training_record(wd, page, original)
        log_action(i, "approve")
        return view(page)

    @app.post("/api/page/{i}/revert")
    def revert(i: int) -> dict[str, Any]:
        path = wd.page_reviewed(i)
        if path.exists():
            path.unlink()
        log_action(i, "revert")
        return view(best(i))

    @app.exception_handler(ValueError)
    def value_error(_request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    return app


def serve(wd: WorkDir, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn

    app = create_app(wd)
    url = f"http://{host}:{port}/"
    if open_browser:
        webbrowser.open(url)
    print(f"Review screen: {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
