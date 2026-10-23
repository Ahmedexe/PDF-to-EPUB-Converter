"""Prompt templates for page, region and layout calls (spec §4 "Prompting rules").

Templates live in ``ocr/prompts/*.txt`` and use ``string.Template`` placeholders:
``$hashiya_rules``, ``$commentary_rules`` and, for regions, ``$region_name``.
``PROMPT_VERSION`` changes whenever any template changes, which invalidates the
OCR cache for that book.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from string import Template

from ..config import BookConfig
from ..model import LayoutRegion

PROMPT_DIR = Path(__file__).parent / "prompts"
_TEMPLATE_FILES = ("page.txt", "page_user.txt", "region_main.txt", "region_hashiya.txt", "layout.txt",
                   "layout_user.txt", "plain.txt")

REGION_NAMES = {
    "bottom": "the foot of a page",
    "margin-right": "the right-hand margin of a page",
    "margin-left": "the left-hand margin of a page",
    "top": "the top margin of a page",
    "frame": "a frame on a page",
}

COMMENTARY_RULES = {
    "none": "",
    "mixed": (
        "   This book is a commentary that weaves the main text (متن) into its explanation "
        "(شرح ممزوج). Main-text words are printed in parentheses or in bold: wrap each such passage "
        "in <matn>…</matn>, keeping the printed parentheses inside. They are not quotations and not notes."
    ),
}

HASHIYA_RULES = {
    "margin": (
        "   - Notes printed in a side margin go in <hashiya pos=\"margin-right\"> or "
        "<hashiya pos=\"margin-left\">, notes in the top margin in <hashiya pos=\"top\">. Never mix "
        "margin text into the lines of the main text."
    ),
    "frame": (
        "   - Notes printed in their own frame go in <hashiya pos=\"frame\">. A separate work printed in a "
        "frame around the main text goes in <hashiya pos=\"frame\" independent=\"1\"> with block tags "
        "instead of notes."
    ),
    "layered": (
        "   - Notes on the commentary rather than on the main text go in a region with layer=\"2\" "
        "(layer=\"3\" for glosses on those notes); markers of such notes inside a note are <ref n=\"…\"/>."
    ),
    "editor": (
        "   - Notes added by the editor of this edition (حواشي المحقق) get kind=\"editor\"; the author's "
        "own notes get no kind."
    ),
}


_override_dir: Path | None = None


def use_prompt_dir(folder: str | Path | None) -> None:
    """Read templates from ``folder`` first (``ocr.prompt_dir``), falling back to the bundled ones."""
    global _override_dir, PROMPT_VERSION
    new = Path(folder) if folder else None
    if new != _override_dir:
        _override_dir = new
        _template.cache_clear()
        PROMPT_VERSION = _version()


LEMMA_RULE = (
    "   - A note that opens with قوله followed by quoted words of the main text: "
    "<note lemma=\"the quoted words\">the explanation</note>. Do not repeat \"قوله: (…)\" inside the note text."
)


@lru_cache(maxsize=None)
def _template(name: str) -> str:
    if _override_dir is not None and (_override_dir / name).exists():
        return (_override_dir / name).read_text(encoding="utf-8")
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _version() -> str:
    h = hashlib.sha256()
    for name in _TEMPLATE_FILES:
        h.update(name.encode())
        h.update(_template(name).encode("utf-8"))
    h.update(repr(sorted(COMMENTARY_RULES.items())).encode("utf-8"))
    h.update(repr(sorted(HASHIYA_RULES.items())).encode("utf-8"))
    h.update(LEMMA_RULE.encode("utf-8"))
    return h.hexdigest()[:12]


PROMPT_VERSION = _version()


def _fill(name: str, **values: str) -> str:
    out = []
    for line in _template(name).split("\n"):
        key = line.strip()
        if key.startswith("$") and key[1:] in values and not values[key[1:]]:
            continue  # a rule that is switched off leaves no blank line behind
        out.append(Template(line).safe_substitute(**values))
    return "\n".join(out).strip() + "\n"


def _hashiya_rules(cfg: BookConfig) -> str:
    return "\n".join(HASHIYA_RULES[form] for form in ("margin", "frame", "layered", "editor")
                     if form in cfg.text.hashiya_forms)


def _commentary_rules(cfg: BookConfig) -> str:
    return COMMENTARY_RULES.get(cfg.text.commentary, "")


def prompt_version(cfg: BookConfig) -> str:
    use_prompt_dir(cfg.ocr.prompt_dir)
    return PROMPT_VERSION + ("-plain" if cfg.ocr.prompt_preset == "plain" else "")


def page_prompts(cfg: BookConfig) -> tuple[str, str]:
    """(system, user) prompts for reading a whole page in one call."""
    use_prompt_dir(cfg.ocr.prompt_dir)
    if cfg.ocr.prompt_preset == "plain":
        return _template("plain.txt").strip() + "\n", _template("page_user.txt").strip()
    system = _fill("page.txt", hashiya_rules=_hashiya_rules(cfg), commentary_rules=_commentary_rules(cfg),
                   lemma_rule=LEMMA_RULE if "lemma" in cfg.text.hashiya_forms else "")
    return system, _template("page_user.txt").strip()


def region_prompts(cfg: BookConfig, region: LayoutRegion) -> tuple[str, str]:
    """(system, user) prompts for one region cut from a page (main text, hashiya or frame)."""
    use_prompt_dir(cfg.ocr.prompt_dir)
    if region.kind == "hashiya":
        name = REGION_NAMES.get(region.pos or "bottom", "the page")
        system = _fill("region_hashiya.txt", region_name=name, hashiya_rules=_hashiya_rules(cfg))
        return system, "Transcribe the notes in this region."
    system = _fill("region_main.txt", commentary_rules=_commentary_rules(cfg))
    return system, "Transcribe this region."


def layout_prompts(cfg: BookConfig) -> tuple[str, str]:
    """(system, user) prompts for the layout step (region boxes and rotation)."""
    return _template("layout.txt").strip() + "\n", _template("layout_user.txt").strip()
