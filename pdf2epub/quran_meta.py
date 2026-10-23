"""Sura metadata from Tanzil's quran-data.xml, and Quran reference strings.

A reference string is "2:155" (one aya), "2:155-157" (a range within a sura)
or "2:286-3:1" (across suras). Used by the Quran check and the EPUB builder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from lxml import etree

from .arabic import format_number

DATA_DIR = Path(__file__).parent / "data" / "tanzil"
_REF_RE = re.compile(r"^\s*(\d+):(\d+)(?:-(?:(\d+):)?(\d+))?\s*$")


@dataclass(frozen=True)
class Sura:
    index: int
    name: str  # Arabic name, e.g. "البقرة"
    tname: str  # transliterated name
    ayas: int
    type: str  # Meccan / Medinan


@lru_cache(maxsize=None)
def suras(data_dir: str | None = None) -> tuple[Sura, ...]:
    path = Path(data_dir) / "quran-data.xml" if data_dir else DATA_DIR / "quran-data.xml"
    root = etree.parse(str(path)).getroot()
    out = []
    for el in root.iter("sura"):
        out.append(Sura(int(el.get("index")), el.get("name"), el.get("tname"), int(el.get("ayas")),
                        el.get("type")))
    return tuple(out)


def sura(index: int) -> Sura:
    return suras()[index - 1]


def sura_name(index: int) -> str:
    return sura(index).name


@dataclass(frozen=True)
class QuranRef:
    sura: int
    aya: int
    end_sura: int
    end_aya: int

    def __str__(self) -> str:
        if (self.sura, self.aya) == (self.end_sura, self.end_aya):
            return f"{self.sura}:{self.aya}"
        if self.sura == self.end_sura:
            return f"{self.sura}:{self.aya}-{self.end_aya}"
        return f"{self.sura}:{self.aya}-{self.end_sura}:{self.end_aya}"


def parse_ref(ref: str) -> QuranRef | None:
    m = _REF_RE.match(ref or "")
    if not m:
        return None
    s, a = int(m.group(1)), int(m.group(2))
    if m.group(4) is None:
        return QuranRef(s, a, s, a)
    return QuranRef(s, a, int(m.group(3)) if m.group(3) else s, int(m.group(4)))


def format_ref(sura: int, aya: int, end_sura: int | None = None, end_aya: int | None = None) -> str:
    return str(QuranRef(sura, aya, end_sura or sura, end_aya or aya))


def ref_label(ref: str, digits: str = "arabic-indic") -> str:
    """Printed-style label for a reference: "[البقرة: ١٥٥]", "[البقرة: ١٥٥-١٥٦]",
    "[البقرة: ٢٨٦ - آل عمران: ١]". Returns "" for an unparseable reference."""
    r = parse_ref(ref)
    if r is None:
        return ""
    num = lambda n: format_number(n, digits)  # noqa: E731
    if r.sura == r.end_sura:
        ayas = num(r.aya) if r.aya == r.end_aya else f"{num(r.aya)}-{num(r.end_aya)}"
        return f"[{sura_name(r.sura)}: {ayas}]"
    return f"[{sura_name(r.sura)}: {num(r.aya)} - {sura_name(r.end_sura)}: {num(r.end_aya)}]"
