"""OCR client, backends, page reading, and the layout step (spec §4, §6.3)."""

from __future__ import annotations

import asyncio
import json

import cv2
import httpx
import numpy as np
import pytest

from pdf2epub.config import BookConfig
from pdf2epub.devtools.sample import make_sample_book
from pdf2epub.ingest import render
from pdf2epub.layout.crop import crop_region, reading_order
from pdf2epub.layout.detect import cv_regions, needs_layout, parse_layout_response
from pdf2epub.model import LayoutRegion, PageEntry
from pdf2epub.ocr.backends import FixtureBackend, OpenAIBackend
from pdf2epub.ocr.client import OcrClient, encode_image
from pdf2epub.ocr.prompts import page_prompts
from pdf2epub.ocr.run import ocr_book, ocr_page
from pdf2epub.ocr.types import Completion, OcrError
from pdf2epub.parse import parse_tags
from pdf2epub.model import Page
from pdf2epub.workdir import WorkDir

run = asyncio.run


def ok_response(text="<p>نص</p>", finish="stop"):
    return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": text}, "finish_reason": finish}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def backend(handler, retries=3, **kw):
    cfg = BookConfig(slug="o")
    cfg.ocr.endpoint, cfg.ocr.retries = "http://gpu.test/v1", retries
    for k, v in kw.items():
        setattr(cfg.ocr, k, v)
    slept = []

    async def sleep(d):
        slept.append(d)

    return OpenAIBackend(cfg.ocr, transport=httpx.MockTransport(handler), sleep=sleep), slept


def test_openai_backend_payload_and_auth(monkeypatch):
    monkeypatch.setenv("PDF2EPUB_API_KEY", "secret")
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return ok_response("<p>أ</p>")

    b, _ = backend(handler, extra_body={"top_k": 1})
    c = run(b.complete(b"\x89PNGxx", "sys", "user", request_id="0001", max_tokens=100, temperature=0.0, extra={"top_k": 1}))
    assert c.text == "<p>أ</p>" and c.finish_reason == "stop"
    assert seen["url"] == "http://gpu.test/v1/chat/completions" and seen["auth"] == "Bearer secret"
    content = seen["body"]["messages"][1]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert seen["body"]["temperature"] == 0.0 and seen["body"]["top_k"] == 1


def test_openai_backend_retries():
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, text="slow down")
        if calls["n"] == 2:
            raise httpx.ConnectError("boom")
        return ok_response()

    b, slept = backend(flaky)
    c = run(b.complete(b"x", "s", "u", request_id="1", max_tokens=1, temperature=0, extra={}))
    assert c.text and calls["n"] == 3 and slept[0] >= 7
    b, _ = backend(lambda r: httpx.Response(500, text="err"), retries=2)
    with pytest.raises(OcrError):
        run(b.complete(b"x", "s", "u", request_id="1", max_tokens=1, temperature=0, extra={}))
    count = {"n": 0}

    def bad_request(r):
        count["n"] += 1
        return httpx.Response(400, text="bad")

    b, _ = backend(bad_request)
    with pytest.raises(OcrError):
        run(b.complete(b"x", "s", "u", request_id="1", max_tokens=1, temperature=0, extra={}))
    assert count["n"] == 1  # no retry on other 4xx


def test_encode_image_downscales_only():
    img = np.full((3000, 2000), 255, np.uint8)
    data, size = encode_image(img, 2400)
    assert size == (1600, 2400) and data[:4] == b"\x89PNG"
    assert encode_image(img, 4000)[1] == (2000, 3000)


class FakeBackend:
    name = "fake"

    def __init__(self, answers):
        self.answers = answers  # request_id -> list of texts (consumed in order) or a text
        self.requests = []

    async def complete(self, image_png, system, user, *, request_id, max_tokens, temperature, extra):
        self.requests.append((request_id, len(image_png)))
        ans = self.answers.get(request_id)
        if isinstance(ans, list):
            ans = ans.pop(0) if len(ans) > 1 else ans[0]
        if isinstance(ans, Exception):
            raise ans
        if ans is None:
            raise FileNotFoundError(request_id)
        return Completion(text=ans, model="fake")

    async def aclose(self):
        pass


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return make_sample_book(tmp_path_factory.mktemp("sample"))


def ingested(sample, tmp_path, **ocr):
    cfg = BookConfig(slug="s")
    cfg.layout.regions = "never"
    for k, v in ocr.items():
        setattr(cfg.ocr, k, v)
    wd = WorkDir(tmp_path / "w")
    manifest = render.ingest_pdf(sample.pdf, wd, cfg)
    return cfg, wd, manifest


def test_ocr_book_with_fixtures_and_cache(sample, tmp_path):
    cfg, wd, manifest = ingested(sample, tmp_path, backend="fixture", fixtures=str(sample.fixtures))
    results = run(ocr_book(manifest.pages, wd, cfg))
    assert sorted(results) == [1, 2, 3, 4, 5]
    assert all(wd.page_tags(i).exists() for i in results)
    assert results[2].tags == (sample.fixtures / "0002.tags").read_text()
    # Second run: nothing is re-read.
    client = OcrClient(cfg, wd, FakeBackend({}))
    again = run(ocr_book(manifest.pages, wd, cfg, client=client))
    assert client.calls == 0 and again[2].info.cached


def test_retry_at_higher_resolution(sample, tmp_path):
    cfg, wd, manifest = ingested(sample, tmp_path, max_side=800, retry_max_side=1200)
    fake = FakeBackend({"0001": ["<p>unclosed <quran>", "<p>نص سليم</p>"]})
    res = run(ocr_page(manifest.pages[0], wd, cfg, OcrClient(cfg, wd, fake)))
    assert res.tags == "<p>نص سليم</p>" and res.info.attempts == 2
    assert fake.requests[1][1] > fake.requests[0][1]  # the retry sent a bigger image


def test_failed_pages_are_not_cached(sample, tmp_path):
    cfg, wd, manifest = ingested(sample, tmp_path)
    fake = FakeBackend({"0001": OcrError("endpoint down")})
    res = run(ocr_page(manifest.pages[0], wd, cfg, OcrClient(cfg, wd, fake)))
    assert res.failed and not wd.page_tags(1).exists() and wd.page_ocr_info(1).exists()
    fake = FakeBackend({"0001": "<p>نص</p>"})
    res = run(ocr_page(manifest.pages[0], wd, cfg, OcrClient(cfg, wd, fake)))
    assert not res.failed and wd.page_tags(1).exists()


def test_force_bypasses_response_cache(sample, tmp_path):
    cfg, wd, manifest = ingested(sample, tmp_path)
    run(ocr_page(manifest.pages[0], wd, cfg, OcrClient(cfg, wd, FakeBackend({"0001": "<p>أول</p>"}))))
    fake = FakeBackend({"0001": "<p>ثان</p>"})
    res = run(ocr_page(manifest.pages[0], wd, cfg, OcrClient(cfg, wd, fake), force=True))
    assert res.tags == "<p>ثان</p>" and fake.requests


def margin_page() -> np.ndarray:
    img = np.full((1400, 1000), 255, np.uint8)
    rng = np.random.default_rng(3)
    for k in range(20):  # main column
        y = 120 + k * 50
        x = 260
        while x < 940:
            w = int(rng.integers(30, 80))
            cv2.rectangle(img, (x, y), (min(x + w, 940), y + 16), 0, -1)
            x += w + 15
    for k in range(10):  # a margin column on the left, beside the main text
        y = 200 + k * 40
        cv2.rectangle(img, (40, y), (200, y + 14), 0, -1)
    return img


def test_needs_layout_and_cv_regions(sample, tmp_path):
    assert needs_layout(margin_page())
    cfg, wd, manifest = ingested(sample, tmp_path)
    plain = cv2.imread(str(wd.page_image(2)), cv2.IMREAD_GRAYSCALE)
    assert not needs_layout(plain)  # foot notes under a rule only
    regions = cv_regions(margin_page())
    kinds = {(r.kind, r.pos) for r in regions}
    assert ("main", None) in kinds and ("hashiya", "margin-left") in kinds
    foot = cv_regions(plain)
    assert any(r.kind == "hashiya" and r.pos == "bottom" for r in foot)


def test_parse_layout_response():
    text = 'Here you go:\n```json\n{"regions": [{"type": "main", "bbox_2d": [250, 80, 950, 900]},' \
           ' {"type": "hashiya", "pos": "right", "bbox_2d": [960, 100, 1000, 800], "rotation": 88},' \
           ' {"type": "page_number", "bbox_2d": [480, 950, 520, 980], "text": "٣٧"},' \
           ' {"type": "junk", "bbox_2d": [1, 2, 3, 4]}, {"type": "figure", "bbox": [0, 0, 1, 1]}]}\n```'
    regions = parse_layout_response(text, 1000, 1400)
    assert [r.id for r in regions] == ["main", "h1", "pagenum"]
    assert regions[1].pos == "margin-right" and regions[1].rotation == 90.0
    assert regions[0].bbox == [250.0, 112.0, 950.0, 1260.0]
    assert regions[2].text == "٣٧"
    with pytest.raises(ValueError):
        parse_layout_response("no json here", 10, 10)


def test_crop_rotation_makes_text_upright():
    img = np.full((200, 100), 255, np.uint8)
    cv2.rectangle(img, (10, 10), (30, 190), 0, -1)  # a tall bar on the left
    region = LayoutRegion(id="h1", kind="hashiya", pos="margin-left", bbox=[0, 0, 100, 200], rotation=90)
    out = crop_region(img, region, pad=0)
    assert out.shape == (100, 200)  # turned clockwise: now wide
    assert out[:40, :].mean() < out[60:, :].mean()  # the bar is now at the top
    slanted = crop_region(img, region.model_copy(update={"rotation": 30.0}), pad=0)
    assert slanted.shape[0] > 100 and slanted.shape[1] > 100


def test_reading_order():
    regions = [LayoutRegion(id="h2", kind="hashiya", pos="top", bbox=[0, 0, 1, 1]),
               LayoutRegion(id="h1", kind="hashiya", pos="margin-right", bbox=[0, 0, 1, 1]),
               LayoutRegion(id="main", kind="main", bbox=[0, 0, 1, 1]),
               LayoutRegion(id="h3", kind="hashiya", pos="bottom", bbox=[0, 0, 1, 1]),
               LayoutRegion(id="pagenum", kind="page_number", bbox=[0, 0, 1, 1])]
    assert [r.id for r in reading_order(regions)] == ["main", "h3", "h1", "h2"]


def test_split_page_end_to_end(tmp_path):
    wd = WorkDir(tmp_path / "w").ensure()
    cv2.imwrite(str(wd.page_image(1)), margin_page())
    entry = PageEntry(index=1, pdf_page=1, image="pages/0001.png", width=1000, height=1400, sha256="x")
    cfg = BookConfig(slug="m")
    layout = ('{"regions": [{"type": "main", "bbox_2d": [250, 70, 950, 860]},'
              ' {"type": "hashiya", "pos": "margin-left", "bbox_2d": [30, 130, 210, 500]},'
              ' {"type": "header", "bbox_2d": [300, 10, 700, 40], "text": "باب الطهارة"},'
              ' {"type": "page_number", "bbox_2d": [480, 950, 520, 980], "text": "١٢"}]}')
    fake = FakeBackend({"0001.layout": layout, "0001.main": "<p>قال المصنف في الطهارة</p>",
                        "0001.h1": "<note>حاشية في الهامش.</note>"})
    res = run(ocr_page(entry, wd, cfg, OcrClient(cfg, wd, fake)))
    assert res.layout.split and res.info.layout_source == "vlm"
    page = parse_tags(res.tags, Page(index=1, width=1000, height=1400)).page
    assert page.running_header == "باب الطهارة" and page.printed_page == "١٢"
    region = page.hashiya[0]
    assert region.region == "margin-left" and region.bbox and page.notes()[0].blocks
    assert wd.region_image(1, "h1").exists()
    # A useless layout answer falls back to CV and records why.
    fake = FakeBackend({"0001.layout": "I cannot do that", "0001.main": "<p>نص</p>",
                        "0001.h1": "<note>ح</note>", "0001.h2": "<note>ح</note>"})
    res = run(ocr_page(entry, wd, cfg, OcrClient(cfg, wd, fake), force=True))
    assert res.info.layout_source == "cv" and res.info.layout_error


def test_prompts_mention_book_conventions():
    cfg = BookConfig(slug="x")
    system, _ = page_prompts(cfg)
    assert "<matn>" not in system.split("Example")[0] or "commentary" not in system
    cfg.text.commentary = "mixed"
    assert "شرح ممزوج" in page_prompts(cfg)[0]


def test_fixture_backend_missing():
    with pytest.raises(OcrError):
        FixtureBackend("/nonexistent/folder")


def test_serve_modal_imports_without_modal():
    from pdf2epub.ocr import serve_modal

    cmd = serve_modal.vllm_command("k")
    assert "--api-key" in cmd and "bfloat16" in cmd
