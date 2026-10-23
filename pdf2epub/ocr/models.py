"""Candidate OCR models for the phase-1 comparison (spec §4, roadmap phase 1).

Each entry says how the model is served (Hugging Face id, GPU) and prompted:
``tags`` models get our tag-format prompt; ``plain`` models (OCR specialists
trained on their own simple prompt) get a plain-text prompt, which loses
structure but keeps the text comparison fair. Checked on Hugging Face on
2026-10-19 (all open weights, none gated).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Candidate:
    key: str
    hf_id: str
    params_b: float
    preset: str = "tags"  # "tags" | "plain"
    gpu: str = "L40S"
    extra_body: dict[str, Any] = field(default_factory=dict)
    max_model_len: int = 16384
    note: str = ""


CANDIDATES: dict[str, Candidate] = {c.key: c for c in [
    Candidate("qwen3-vl-8b", "Qwen/Qwen3-VL-8B-Instruct", 8.8,
              note="the spec's main candidate and the base to fine-tune"),
    Candidate("qwen3.5-9b", "Qwen/Qwen3.5-9B", 9.7,
              extra_body={"chat_template_kwargs": {"enable_thinking": False}},
              note="newer natively multimodal Qwen (2026); thinking switched off"),
    Candidate("ain-7b", "MBZUAI/AIN", 8.3, preset="plain",
              note="Arabic-English VLM; best open model on KITAB-Bench in the spec. Ignores the tag format "
                   "(answers in plain text, often only the page number), so it gets the plain prompt"),
    Candidate("qari-ocr", "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct", 2.2, preset="plain", gpu="L4",
              note="Arabic OCR specialist with diacritics; plain-text output"),
]}


DEFAULT_KEY = "ft-ft2"  # the model `pdf2epub convert` uses unless told otherwise
DEFAULT_MODEL = "/models/ft2/merged"


def key_for(model: str) -> str | None:
    """The candidate key serving a model id: ``ft-<run>`` for ``/models/<run>/merged``."""
    m = re.fullmatch(r"/models/([^/]+)/merged", model)
    if m:
        return f"ft-{m.group(1)}"
    return next((c.key for c in CANDIDATES.values() if c.hf_id == model), None)


def recorded_endpoint(model: str) -> str | None:
    """The endpoint ``pdf2epub bench deploy`` recorded for this model (~/.config/pdf2epub/endpoints.json)."""
    key = key_for(model)
    path = Path(os.environ.get("PDF2EPUB_CONFIG", Path.home() / ".config" / "pdf2epub")) / "endpoints.json"
    if key is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get(key)


def get(key: str) -> Candidate:
    """A candidate by key; ``ft-<run>`` is a fine-tuned model in the ``pdf2epub-models`` volume
    (``train/finetune_modal.py --run <run>``), served from ``/models/<run>/merged``."""
    if key.startswith("ft-") and key not in CANDIDATES:
        return Candidate(key, f"/models/{key[3:]}/merged", 8.8, note="fine-tuned Qwen3-VL-8B")
    if key not in CANDIDATES:
        raise KeyError(f"unknown model '{key}'; choose from {', '.join(CANDIDATES)}")
    return CANDIDATES[key]
