"""Per-book settings (spec §10 CLI options, §6.6 display modes, §5.3 commentary style).

A book's settings live in ``<work>/<slug>/book.json``. The first ``convert``
writes them from the command-line flags; later commands load them and apply
only the flags given explicitly (``apply_overrides``).

Endpoint secrets are never stored: ``OcrSettings.api_key_env`` names the
environment variable that holds the key.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

QuranStyle = Literal["auto", "uthmani", "uthmani-plain", "uthmani-min", "simple", "simple-plain", "simple-clean"]


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class IngestSettings(_Settings):
    dpi: int = 300  # render resolution when a page is not a single embedded scan
    extract: bool = True  # take the original scan untouched when a page is one image
    spreads: Literal["auto", "yes", "no"] = "auto"  # split two-page spreads
    deskew: bool = True
    crop: bool = True  # remove scanner borders and empty margins (never printed text)
    grayscale: bool = True  # OCR image in grayscale; figures are cropped from the colour source


class LayoutSettings(_Settings):
    # never: one model call per page; always: layout step on every page;
    # auto: layout step only where a CV check finds margin text or frames (spec §6.3 step 3).
    # bands: cut every page at white gaps into ``bands`` main-text bands plus the foot notes, and read
    # each separately (fewer skipped lines on dense pages).
    regions: Literal["auto", "always", "never", "bands"] = "auto"
    bands: int = 2
    bbox_format: Literal["rel1000", "abs"] = "rel1000"  # Qwen3-VL returns 0..1000 relative boxes
    pad_px: int = 12


class OcrSettings(_Settings):
    backend: Literal["openai", "fixture", "tesseract"] = "openai"
    # OpenAI-compatible base URL, e.g. https://<app>.modal.run/v1; when unset, the endpoint
    # `pdf2epub bench deploy` recorded for the model is used.
    endpoint: str | None = None
    # The fine-tuned Qwen3-VL-8B (train/finetune_modal.py run ft2), served from the Modal volume;
    # the base model is "Qwen/Qwen3-VL-8B-Instruct".
    model: str = "/models/ft2/merged"
    api_key_env: str = "PDF2EPUB_API_KEY"
    fixtures: str | None = None  # fixture backend: folder of NNNN.tags files
    concurrency: int = 8
    timeout_s: float = 600.0  # serverless cold starts can take minutes
    retries: int = 3  # network / 5xx / 429 retries per call, with backoff
    max_tokens: int = 8192
    temperature: float = 0.0  # spec §4 rule 6
    max_side: int = 2400  # longest image side sent to the model
    retry_max_side: int = 3400  # higher-resolution retry when the output does not parse
    image_format: Literal["png", "jpeg"] = "png"
    extra_body: dict[str, Any] = Field(default_factory=dict)  # passed through to the endpoint
    # tags: our tag format (structure, notes, verses); plain: plain text, for OCR models trained on
    # their own simple prompt (structure is then lost, but text accuracy can still be compared).
    prompt_preset: Literal["tags", "plain"] = "tags"
    prompt_dir: str | None = None  # folder of prompt template overrides (same file names as ocr/prompts/)
    cache_salt: str = ""  # change to invalidate cached responses (e.g. after redeploying a model)

    def api_key(self) -> str | None:
        """From the environment, else the key ``pdf2epub bench deploy`` saved in ~/.config/pdf2epub/api-key."""
        if os.environ.get(self.api_key_env):
            return os.environ[self.api_key_env]
        path = Path(os.environ.get("PDF2EPUB_CONFIG", Path.home() / ".config" / "pdf2epub")) / "api-key"
        return path.read_text(encoding="utf-8").strip() or None if path.exists() else None


class TextSettings(_Settings):
    """How this book prints things; tells the prompts and checks what to expect."""

    commentary: Literal["none", "mixed"] = "none"  # mixed = شرح ممزوج: matn words in ( ) or bold
    quran_style: QuranStyle = "auto"  # Tanzil text used for replacement (spec §5.1 step 5)
    quran_style_detected: str | None = None  # resolved once per book when quran_style == "auto"
    quran_brackets: list[str] = Field(default_factory=lambda: ["﴿﴾", "﴾﴿"])  # untagged-verse regex pass
    hashiya_forms: list[Literal["foot", "lemma", "margin", "frame", "layered", "editor"]] = Field(
        default_factory=lambda: ["foot", "lemma"]
    )
    note_kind_default: Literal["author", "editor", "unknown"] = "author"
    editor_markers: Literal["none", "letters", "numbers", "symbols"] = "none"  # marker style of editor notes
    lemma_template: str = "قوله: ({lemma})"  # how a lemma note opens in the EPUB, in bold
    digits: Literal["auto", "arabic-indic", "western"] = "auto"
    poem_meter: bool = False  # run the optional meter classifier when installed
    # auto: a poem whose verses all lack a left half-line is single-column (info);
    # two: every verse must have both halves; single: modern / مشطور poetry book.
    poem_columns: Literal["auto", "two", "single"] = "auto"


class CheckSettings(_Settings):
    quran_threshold: float = 0.90  # >= : corrected; below: flagged (spec §5.1 step 4)
    quran_min_words: int = 3  # shorter spans are only verified on an exact, unambiguous match
    lemma_threshold: float = 0.85  # spec §6.5
    lemma_lookback_pages: int = 1  # glosses may lag a page behind (spec §6.4 step 2)
    poem_half_ratio: float = 2.2  # max length ratio between the two halves of a verse
    tesseract: bool = True  # word-count cross-check (spec §4)
    tesseract_lang: str = "ara"
    tessdata_dir: str | None = None  # defaults to <repo>/vendor/tessdata when present
    wordcount_tolerance: float = 0.40  # flag when |model - tesseract| / max(model, tesseract) exceeds it
    repetition_min_repeats: int = 3
    block_on: list[Literal["info", "warning", "error"]] = Field(default_factory=lambda: ["warning", "error"])


class BuildSettings(_Settings):
    poem_layout: Literal["two-column", "stacked"] = "two-column"
    notes: Literal["popup", "inline", "chapter-end"] = "popup"
    quran_refs: Literal["label", "footnote", "none"] = "none"
    author_notes: Literal["show", "hide"] = "show"
    editor_notes: Literal["show", "hide"] = "show"
    note_markers: Literal["renumber", "printed"] = "renumber"  # renumber per chapter, or keep printed signs
    body_font: Literal["amiri", "noto-naskh"] = "amiri"
    embed_fonts: bool = True
    split_level: int | None = None  # heading level that starts a new XHTML file; None = top level used
    max_chapter_pages: int = 40  # split long runs without headings so e-ink readers open them fast
    frames: Literal["section", "separate-epub"] = "section"  # independent works printed بهامشه
    cover: Literal["first-page", "none"] = "first-page"
    allow_flagged: bool = False  # build even if some pages still need review (draft)
    epubcheck: bool = True


class BookConfig(_Settings):
    slug: str
    title: str | None = None
    author: str | None = None
    source_pdf: str | None = None
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    layout: LayoutSettings = Field(default_factory=LayoutSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    text: TextSettings = Field(default_factory=TextSettings)
    checks: CheckSettings = Field(default_factory=CheckSettings)
    build: BuildSettings = Field(default_factory=BuildSettings)


def slugify(name: str) -> str:
    """Work-folder name for a book: ``sharh.pdf`` -> ``sharh``. Arabic letters are kept."""
    stem = Path(name).stem
    stem = unicodedata.normalize("NFC", stem).strip().lower()
    stem = re.sub(r"[^\w\-]+", "-", stem, flags=re.UNICODE).strip("-_")
    return stem or "book"


def apply_overrides(cfg: BookConfig, overrides: dict[str, Any]) -> BookConfig:
    """Return a copy of ``cfg`` with dotted-key overrides applied, e.g.
    ``{"build.notes": "inline", "title": "..."}``. ``None`` values are ignored,
    so unset CLI flags never clobber saved settings."""
    data = cfg.model_dump()
    for key, value in overrides.items():
        if value is None:
            continue
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"unknown setting: {key}")
        node[parts[-1]] = value
    return BookConfig.model_validate(data)


def repo_root() -> Path:
    """The source checkout root (holds ``vendor/``), when running from a checkout."""
    return Path(__file__).resolve().parent.parent


def default_tessdata_dir() -> str | None:
    env = os.environ.get("PDF2EPUB_TESSDATA") or os.environ.get("TESSDATA_PREFIX")
    if env:
        return env
    candidate = repo_root() / "vendor" / "tessdata"
    return str(candidate) if (candidate / "ara.traineddata").exists() else None


def default_epubcheck_jar() -> str | None:
    env = os.environ.get("PDF2EPUB_EPUBCHECK")
    if env:
        return env
    candidate = repo_root() / "vendor" / "epubcheck" / "epubcheck.jar"
    return str(candidate) if candidate.exists() else None
