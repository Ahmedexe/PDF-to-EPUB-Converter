"""End to end: sample book -> EPUB, review screen, CLI (spec §1 criteria 5-6, §10)."""

from __future__ import annotations

import json
import zipfile

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from conftest import requires_epubcheck
from pdf2epub import pipeline
from pdf2epub.cli import main
from pdf2epub.devtools.sample import make_sample_book
from pdf2epub.model import Quran, block_containers, iter_blocks, load_page, walk_inlines
from pdf2epub.review.app import create_app

STAMP = "2026-10-19T00:00:00Z"


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return make_sample_book(tmp_path_factory.mktemp("sample"))


def overrides(sample, **extra):
    base = {"ocr.backend": "fixture", "ocr.fixtures": str(sample.fixtures), "title": "تحفة الصابرين",
            "checks.tesseract": False}
    base.update(extra)
    return base


def test_convert_sample_book(sample, tmp_path):
    report = pipeline.convert(sample.pdf, tmp_path / "work", overrides(sample), modified=STAMP)
    assert not report.blocked and len(report.epubs) == 1
    wd, cfg = pipeline.open_book(tmp_path / "work", "sample")
    assert cfg.text.quran_style_detected == "simple-plain"
    epub = report.epubs[0]
    with zipfile.ZipFile(epub) as zf:
        names = zf.namelist()
        text = "".join(zf.read(n).decode() for n in names if n.startswith("OEBPS/text/ch"))
        nav = zf.read("OEBPS/nav.xhtml").decode()
    assert "OEBPS/images/cover.jpg" in names
    assert "تنال بالصبر واليقين، وهذا" in text and "معنى قول بعض السلف" in text  # paragraph joined
    for label in "١٢٣٤٥":
        assert f">{label}</a>" in nav  # page list from printed page numbers
    # Every Quran span on every page was verified against Tanzil.
    for i in wd.page_indices():
        page = load_page(wd.page_checked(i))
        spans = [n for b, _o in iter_blocks(page) for c in block_containers(b) for n in walk_inlines(c)
                 if isinstance(n, Quran)]
        assert all(s.status in ("verified", "corrected") for s in spans)
    # Re-running does not call the model again and gives the same book.
    again = pipeline.convert(sample.pdf, tmp_path / "work", overrides(sample), modified=STAMP)
    assert again.epubs[0].read_bytes() == epub.read_bytes()


@requires_epubcheck
@pytest.mark.slow
def test_sample_epub_passes_epubcheck(sample, tmp_path):
    report = pipeline.convert(sample.pdf, tmp_path / "work", overrides(sample, **{"build.epubcheck": True}),
                              modified=STAMP)
    assert report.epubcheck and report.epubcheck[0]["ok"], report.epubcheck
    assert not report.epubcheck[0]["warnings"]


def flawed(sample, tmp_path):
    """The sample with a model error on page 2: a Quran verse misread beyond correction."""
    bad = tmp_path / "fixtures"
    bad.mkdir()
    for path in sample.fixtures.glob("*.tags"):
        text = path.read_text(encoding="utf-8")
        if path.stem == "0002":
            text = text.replace("<quran>وَلَنَبْلُوَنَّكُم", "<quran>هذا نص مختلف تماما وَلَنَبْلُوَنَّكُم")
            text = text.replace("وَالثَّمَرَاتِ وَبَشِّرِ الصَّابِرِينَ</quran>", "</quran>")
        (bad / path.name).write_text(text, encoding="utf-8")
    return bad


def test_unattended_build_is_final_with_quality_report(sample, tmp_path):
    report = pipeline.convert(sample.pdf, tmp_path / "work", overrides(sample, **{"ocr.fixtures": str(flawed(sample, tmp_path))}),
                              modified=STAMP)
    wd, _cfg = pipeline.open_book(tmp_path / "work", "sample")
    assert report.blocked == [2]                                  # the re-reads could not fix it
    assert report.epubs and report.epubs[0].name == "sample.epub"  # final, not a draft
    text = report.quality.read_text(encoding="utf-8")
    assert "| 2 |" in text and "quran" in text.lower()
    assert wd.page_tags(2).exists() and "quran" in wd.page_tags(2).read_text(encoding="utf-8")  # first reading kept


def test_flagged_page_blocks_build_until_reviewed(sample, tmp_path):
    report = pipeline.convert(sample.pdf, tmp_path / "work",
                              overrides(sample, **{"ocr.fixtures": str(flawed(sample, tmp_path)), "build.review": "required"}),
                              modified=STAMP)
    assert report.blocked == [2] and not report.epubs
    wd, cfg = pipeline.open_book(tmp_path / "work", "sample")
    draft = pipeline.build(wd, cfg, draft=True, modified=STAMP)
    assert draft.epubs and draft.epubs[0].name == "sample-draft.epub"
    client = TestClient(create_app(wd))
    book = client.get("/api/book").json()
    assert [p["index"] for p in book["pages"] if p["status"] == "flagged"] == [2]
    page = client.get("/api/page/2").json()
    assert "quran_unmatched" in {f["code"] for f in page["flags"]}
    assert client.get("/image/2").status_code == 200
    # Approving accepts the verse as printed; the book now builds.
    approved = client.post("/api/page/2/approve").json()
    assert approved["status"] == "reviewed" and wd.page_reviewed(2).exists()
    reviewed = load_page(wd.page_reviewed(2))
    statuses = {n.status for b, _o in iter_blocks(reviewed) for c in block_containers(b) for n in walk_inlines(c)
                if isinstance(n, Quran)}
    assert "accepted" in statuses
    assert json.loads(wd.training_record(2).read_text())["status"] == "reviewed"
    assert wd.review_log.exists()
    final = pipeline.build(wd, cfg, modified=STAMP)
    assert final.epubs and not final.blocked
    # Re-checking the book never touches the reviewed page.
    pipeline.check_book(wd, cfg)
    assert load_page(wd.page_reviewed(2)).status == "reviewed"


def test_review_edit_tags_and_manual_link(sample, tmp_path):
    pipeline.convert(sample.pdf, tmp_path / "work", overrides(sample), modified=STAMP)
    wd, cfg = pipeline.open_book(tmp_path / "work", "sample")
    client = TestClient(create_app(wd))
    page = client.get("/api/page/4").json()
    edited = page["tags"].replace("وأعلاها الصبر على الطاعة", "وأعلاها الصبر على طاعة الله")
    out = client.post("/api/page/4/tags", json={"tags": edited}).json()
    assert "على طاعة الله" in out["html"]["main"] and wd.page_reviewed(4).exists()
    assert out["status"] in ("clean", "flagged")
    # Move the lemma note's anchor to other words by selecting them.
    reviewed = load_page(wd.page_reviewed(4))
    lemma_note = next(n for n in reviewed.notes() if n.lemma)
    block = next(b for b in reviewed.blocks if b.type == "paragraph" and "ثلاثة أقسام" in
                 "".join(x for x in b.c if isinstance(x, str)))
    from pdf2epub.model import BlockText

    text = BlockText(block).text
    start = text.index("ثلاثة أقسام")
    res = client.post("/api/page/4/link", json={"note": lemma_note.id, "block": block.id, "start": start,
                                                "end": start + len("ثلاثة أقسام")})
    assert res.status_code == 200
    page = load_page(wd.page_reviewed(4))
    note = next(n for n in page.notes() if n.id == lemma_note.id)
    assert note.link.method == "manual" and note.link.anchor == "ثلاثة أقسام"
    bt = BlockText(next(b for b in page.blocks if b.id == block.id))
    assert bt.text[: bt.ref_offsets[note.link.ref]].endswith("ثلاثة أقسام")
    # The manual link survives a full re-check of the book.
    pipeline.check_book(wd, cfg)
    page = load_page(wd.page_reviewed(4))
    note = next(n for n in page.notes() if n.id == lemma_note.id)
    bt = BlockText(next(b for b in page.blocks if b.id == block.id))
    assert note.link.method == "manual" and bt.text[: bt.ref_offsets[note.link.ref]].endswith("ثلاثة أقسام")
    # Revert drops the review file.
    client.post("/api/page/4/revert")
    assert not wd.page_reviewed(4).exists()


def test_review_rejects_bad_input(sample, tmp_path):
    pipeline.convert(sample.pdf, tmp_path / "work", overrides(sample), modified=STAMP)
    wd, _cfg = pipeline.open_book(tmp_path / "work", "sample")
    client = TestClient(create_app(wd))
    assert client.get("/api/page/abc").status_code == 422
    assert client.get("/api/page/99").status_code == 404
    assert client.get("/image/99").status_code == 404
    assert client.post("/api/page/1/tags", json={"tags": 5}).status_code == 422
    assert client.post("/api/page/1/link", json={"note": "x", "block": "y", "start": 0, "end": 1}).status_code == 404
    assert client.get("/").status_code == 200 and "app.js" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_redo_rereads_pages(sample, tmp_path):
    pipeline.convert(sample.pdf, tmp_path / "work", overrides(sample), modified=STAMP)
    wd, cfg = pipeline.open_book(tmp_path / "work", "sample")
    done = pipeline.redo(wd, cfg, [3])
    assert done == [3]
    page = load_page(wd.page_checked(3))
    assert page.status == "clean"


def test_cli(sample, tmp_path):
    runner = CliRunner()
    work = str(tmp_path / "work")
    res = runner.invoke(main, ["--work", work, "convert", str(sample.pdf), "--fixtures", str(sample.fixtures),
                               "--no-tesseract", "--no-epubcheck", "--notes", "inline", "--layout", "stacked"])
    assert res.exit_code == 0, res.output
    assert "Wrote" in res.output
    res = runner.invoke(main, ["--work", work, "status", "sample"])
    assert json.loads(res.output)["status"] == {"clean": 5}
    res = runner.invoke(main, ["--work", work, "build", "sample", "--notes", "chapter-end", "--no-epubcheck"])
    assert res.exit_code == 0, res.output
    _wd, cfg = pipeline.open_book(tmp_path / "work", "sample")
    assert cfg.build.notes == "chapter-end" and cfg.build.poem_layout == "stacked"
    res = runner.invoke(main, ["--work", work, "check", "sample", "--reparse"])
    assert res.exit_code == 0 and "5 ready" in res.output
