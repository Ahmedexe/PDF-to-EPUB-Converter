"""Shared test helpers.

Pages are built from the tagged fixtures in tests/fixtures/tags/ with the real parser:

    page = fixture_page("basic", index=37)
    pages = fixture_sequence()      # seq_0001..0003 as pages 1..3
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pdf2epub.config import BookConfig, default_epubcheck_jar, default_tessdata_dir
from pdf2epub.model import Page
from pdf2epub.parse import parse_tags
from pdf2epub.workdir import WorkDir

FIXTURES = Path(__file__).parent / "fixtures"
TAGS = FIXTURES / "tags"
REPO = Path(__file__).resolve().parent.parent


def fixture_tags(name: str) -> str:
    return (TAGS / f"{name}.tags").read_text(encoding="utf-8")


def fixture_page(name: str, index: int = 1, *, commentary: str = "none", **base) -> Page:
    """Parse a fixture into a Page with the given index (and optional base fields)."""
    return parse_tags(fixture_tags(name), Page(index=index, **base), commentary=commentary).page


def fixture_sequence() -> list[Page]:
    """The three consecutive pages seq_0001..seq_0003 as book pages 1..3."""
    return [fixture_page(f"seq_{i:04d}", index=i) for i in (1, 2, 3)]


@pytest.fixture
def cfg() -> BookConfig:
    return BookConfig(slug="test")


@pytest.fixture
def workdir(tmp_path: Path) -> WorkDir:
    return WorkDir(tmp_path / "work" / "test").ensure()


def have_java_epubcheck() -> bool:
    return shutil.which("java") is not None and default_epubcheck_jar() is not None


def have_tesseract_ara() -> bool:
    return shutil.which("tesseract") is not None and default_tessdata_dir() is not None


requires_epubcheck = pytest.mark.skipif(not have_java_epubcheck(), reason="Java or EPUBCheck missing")
requires_tesseract = pytest.mark.skipif(not have_tesseract_ara(), reason="Tesseract or Arabic data missing")
