"""Corrections saved as training data (spec §2 step 6, §8 "Keep improving").

Each review save writes ``training/NNNN.json``: the page image, the corrected
page as tags in exactly the format the model is taught to emit, the note
links, and the model's original output, so fine-tuning (phase 3) can use every
fix made in the review screen.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..model import BlockText, Page, Quran, block_containers, iter_blocks, walk_inlines
from ..parse import serialize_page
from ..workdir import WorkDir


def _printed_tags(page: Page) -> str:
    """Tags of the page as printed: Quran spans carry what the page shows (the OCR
    reading when the reviewer accepted it, else the verified text)."""
    copy = page.model_copy(deep=True)
    for block, _o in iter_blocks(copy):
        for container in block_containers(block):
            for node in walk_inlines(container):
                if isinstance(node, Quran) and node.status == "accepted" and node.ocr_text:
                    node.text = node.ocr_text
    return serialize_page(copy)


def save_training_record(wd: WorkDir, page: Page, original: Page | None) -> Path:
    links = []
    by_id = {}
    for block, _o in iter_blocks(page, notes=False):
        bt = BlockText(block)
        for ref_id, off in bt.ref_offsets.items():
            by_id[ref_id] = (block.id, off, bt.text[max(0, off - 40):off])
    for note in page.notes():
        if note.link.ref in by_id:
            block_id, off, before = by_id[note.link.ref]
            links.append({"note": note.id, "method": note.link.method, "block": block_id, "offset": off,
                          "words_before": before})
    original_tags = wd.page_tags(page.index).read_text(encoding="utf-8") if wd.page_tags(page.index).exists() else None
    record = {
        "index": page.index,
        "image": page.image,
        "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": page.status,
        "tags": _printed_tags(page),
        "links": links,
        "model_tags": original_tags,
        "previous_tags": serialize_page(original) if original is not None else None,
    }
    path = wd.training_record(page.index)
    wd.write_json(path, record)
    return path


def export_training_set(wd: WorkDir, out: Path) -> int:
    """Copy reviewed pages' records to a JSONL file (image path, target tags)."""
    count = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for path in sorted(wd.training_dir.glob("*.json")):
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("status") != "reviewed":
                continue
            fh.write(json.dumps({"image": str(wd.abs(rec["image"])), "target": rec["tags"]}, ensure_ascii=False) + "\n")
            count += 1
    return count
