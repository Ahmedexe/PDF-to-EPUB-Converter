"""Notes across pages and chapters (spec §6.4 step 4, §6.6).

Anchors are located only through ``Ref.note == note.id``; ``NoteLink.block``
and ``offset`` are informational and not maintained after pages are joined.
"""

from __future__ import annotations

from ..model import Chapter, Note, Page, Paragraph, Ref, block_containers, normalize_inlines, walk_inlines


def join_continued_notes(pages: list[Page]) -> tuple[dict[str, Note], dict[str, int], dict[str, str]]:
    """Merge notes continued across pages into their first part.

    Returns ``(notes, home, alias)``: merged notes by id (continuation parts
    removed), the page index each note was printed on, and continuation-part id ->
    id of the note it was merged into.
    """
    originals: dict[str, Note] = {}
    home: dict[str, int] = {}
    for page in sorted(pages, key=lambda p: p.index):
        for note in page.notes():
            originals[note.id] = note
            home[note.id] = page.index
    merged: dict[str, Note] = {}
    alias: dict[str, str] = {}
    for note_id, note in originals.items():
        if note.continues_from and note.continues_from in originals:
            continue  # merged into its first part below
        first = note.model_copy(deep=True)
        part = note
        seen = {note_id}
        while part.continues_to and part.continues_to in originals and part.continues_to not in seen:
            nxt = originals[part.continues_to]
            seen.add(nxt.id)
            alias[nxt.id] = first.id
            blocks = [b.model_copy(deep=True) for b in nxt.blocks]
            if first.blocks and blocks and isinstance(first.blocks[-1], Paragraph) and isinstance(blocks[0], Paragraph):
                last = first.blocks[-1]
                last.c.extend([" ", *blocks[0].c])
                normalize_inlines(last.c)
                blocks = blocks[1:]
            first.blocks.extend(blocks)
            part = nxt
        first.cont_next = False
        first.continues_to = None
        merged[first.id] = first
    # Continuation parts that never found their first part are kept as notes of their own.
    for note_id, note in originals.items():
        if note_id not in merged and note_id not in alias:
            merged[note_id] = note.model_copy(deep=True)
    return merged, home, alias


def _ref_order(blocks: list[object]) -> list[str]:
    out = []
    for block in blocks:
        for container in block_containers(block):
            for node in walk_inlines(container):
                if isinstance(node, Ref) and node.note:
                    out.append(node.note)
    return out


def chapter_notes(chapter: Chapter, notes: dict[str, Note], claimed: set[str]) -> list[Note]:
    """Notes anchored in ``chapter`` in anchor order, each followed by its own layer-2/3 notes.

    ``claimed`` collects the ids already placed (shared across chapters).
    """
    out: list[Note] = []

    def add(note_id: str) -> None:
        if note_id in claimed or note_id not in notes:
            return
        claimed.add(note_id)
        note = notes[note_id]
        out.append(note)
        for sub in _ref_order(note.blocks):
            add(sub)

    for note_id in _ref_order(chapter.blocks):
        add(note_id)
    return out


def attach_orphans(chapters: list[Chapter], notes: dict[str, Note], home: dict[str, int], claimed: set[str]) -> None:
    """Every note not reached through an anchor goes to the chapter holding its page, so no
    note text is ever lost."""
    for note_id, note in notes.items():
        if note_id in claimed:
            continue
        page = home.get(note_id)
        target = next((c for c in chapters if page in c.pages), chapters[-1] if chapters else None)
        if target is None:
            continue
        claimed.add(note_id)
        target.notes.append(note)
