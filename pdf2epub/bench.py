"""Phase-1 model comparison on Modal (roadmap phase 1: "test 4 models").

    pdf2epub bench deploy qwen3-vl-8b          # serve a candidate on Modal (scales to zero)
    pdf2epub bench run qwen3-vl-8b             # read the test PDF with it (work/bench-qwen3-vl-8b)
    pdf2epub bench run tesseract               # free CPU baseline, no GPU
    pdf2epub bench report                      # compare every run
    pdf2epub bench stop qwen3-vl-8b            # stop the app now instead of waiting for idle

Without hand-checked truth, the report compares what can be measured without
labels: pages that pass every check, Quran verses resolved against Tanzil,
notes linked, poems with both half-lines, agreement with Tesseract and between
models, time and GPU cost. With a truth folder (``--truth``, pages reviewed in
the review screen and exported with ``pdf2epub bench export-truth``) it adds
CER and the other spec §9 metrics.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import time
from html import escape
from pathlib import Path
from typing import Any

import httpx

from . import pipeline
from .arabic import normalize_for_score
from .eval.metrics import cer, page_text, score_page
from .model import Page, Poem, Quran, block_containers, iter_blocks, load_page, walk_inlines
from .ocr.models import get
from .workdir import WorkDir, book_workdir

CONFIG_DIR = Path(os.environ.get("PDF2EPUB_CONFIG", Path.home() / ".config" / "pdf2epub"))
GPU_PRICE_PER_S = {"L40S": 0.000542, "L4": 0.000222, "A10": 0.000306, "A100-40GB": 0.000583,
                   "A100-80GB": 0.000694, "H100": 0.001097}  # Modal list prices, checked 2026-10-19
SECRET_NAME = "pdf2epub-api-key"
REPO = Path(__file__).resolve().parent.parent
DEFAULT_TEST = REPO / "sample" / "test" / "part.pdf"


class BenchError(RuntimeError):
    pass


# --------------------------------------------------------------------------- credentials and endpoints


def api_key(create: bool = False) -> str | None:
    """The key the model servers expect: env PDF2EPUB_API_KEY, else ~/.config/pdf2epub/api-key."""
    if os.environ.get("PDF2EPUB_API_KEY"):
        return os.environ["PDF2EPUB_API_KEY"]
    path = CONFIG_DIR / "api-key"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    if not create:
        return None
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(32)
    path.write_text(key + "\n", encoding="utf-8")
    path.chmod(0o600)
    return key


def endpoints() -> dict[str, str]:
    path = CONFIG_DIR / "endpoints.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_endpoint(key: str, url: str) -> None:
    data = endpoints()
    data[key] = url
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "endpoints.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def modal_bin() -> str:
    for cand in (Path(sys.executable).parent / "modal", shutil.which("modal")):
        if cand and Path(cand).exists():
            return str(cand)
    raise BenchError("Modal is not installed: run  .venv/bin/pip install modal  then  .venv/bin/modal setup")


def _modal(args: list[str], env: dict[str, str] | None = None, echo=print) -> str:
    proc = subprocess.run([modal_bin(), *args], capture_output=True, text=True,
                          env={**os.environ, "COLUMNS": "500", "TERM": "dumb", **(env or {})}, cwd=str(REPO))
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        low = out.lower()
        if "payment method" in low:
            raise BenchError("Modal needs a payment method before it runs GPUs (the $30/month included "
                             "compute is still used first): add one at https://modal.com/settings/plans")
        if "token missing" in low or "not authenticated" in low or "no token" in low:
            raise BenchError("Modal is not logged in: run  .venv/bin/modal setup  (opens your browser)")
        raise BenchError(f"modal {' '.join(args[:2])} failed:\n{out[-2000:]}")
    return out


# --------------------------------------------------------------------------- deploy / stop


def deploy(key: str, echo=print) -> str:
    """Deploy a candidate as its own Modal app and return its /v1 endpoint."""
    from .ocr.serve_modal import app_name

    cand = get(key)
    secret = api_key(create=True)
    echo(f"Creating the Modal secret {SECRET_NAME} (the endpoint's API key)…")
    _modal(["secret", "create", SECRET_NAME, f"PDF2EPUB_API_KEY={secret}", "--force"], echo=echo)
    env = {"PDF2EPUB_MODEL": cand.hf_id, "PDF2EPUB_GPU": cand.gpu, "PDF2EPUB_MAX_MODEL_LEN": str(cand.max_model_len)}
    echo(f"Deploying {cand.hf_id} on a {cand.gpu} as app {app_name(cand.hf_id)}…")
    out = _modal(["deploy", "pdf2epub/ocr/serve_modal.py"], env=env, echo=echo)
    urls = re.findall(r"https://[\w.-]+\.modal\.run", out)
    if not urls:
        raise BenchError(f"deployed, but no URL found in Modal's output:\n{out[-1500:]}")
    url = urls[-1].rstrip("/") + "/v1"
    _save_endpoint(key, url)
    echo(f"Endpoint: {url}")
    return url


def stop(key: str, echo=print) -> None:
    from .ocr.serve_modal import app_name

    _modal(["app", "stop", "--yes", app_name(get(key).hf_id)], echo=echo)
    echo(f"Stopped {app_name(get(key).hf_id)}")


def wait_ready(url: str, timeout_s: float = 25 * 60, echo=print) -> float:
    """Poll ``/models`` until the server answers (the first call triggers a cold start)."""
    headers = {"Authorization": f"Bearer {api_key()}"} if api_key() else {}
    start = time.monotonic()
    last = ""
    while time.monotonic() - start < timeout_s:
        try:
            r = httpx.get(url.rstrip("/") + "/models", headers=headers, timeout=120)
            if r.status_code == 200:
                waited = time.monotonic() - start
                echo(f"Server ready after {waited:.0f} s")
                return waited
            last = f"HTTP {r.status_code}"
        except httpx.HTTPError as exc:
            last = type(exc).__name__
        echo(f"  waiting for the model server to start ({last}, {time.monotonic() - start:.0f} s)…")
        time.sleep(15)
    raise BenchError(f"the server at {url} did not start within {timeout_s / 60:.0f} minutes ({last})")


# --------------------------------------------------------------------------- run


def run(key: str, *, test_pdf: Path = DEFAULT_TEST, work_root: Path = Path("work"), endpoint: str | None = None,
        concurrency: int = 8, force: bool = False, max_side: int | None = None, tag: str = "",
        bands: int | None = None, echo=print) -> WorkDir:
    """Read the test PDF with one model into ``work/bench-<key><tag>`` (OCR, parse, checks; no EPUB)."""
    slug = f"bench-{key}{tag}"
    overrides: dict[str, Any] = {"checks.tesseract": True, "build.epubcheck": False, "title": f"bench {key}"}
    gpu = None
    if key == "tesseract":
        overrides["ocr.backend"] = "tesseract"
    else:
        cand = get(key)
        endpoint = endpoint or endpoints().get(key)
        if not endpoint:
            raise BenchError(f"no endpoint for {key}: run  pdf2epub bench deploy {key}  first")
        if api_key():
            os.environ.setdefault("PDF2EPUB_API_KEY", api_key() or "")
        overrides.update({"ocr.backend": "openai", "ocr.endpoint": endpoint, "ocr.model": cand.hf_id,
                          "ocr.prompt_preset": cand.preset, "ocr.extra_body": dict(cand.extra_body),
                          "ocr.concurrency": concurrency})
        if max_side:
            overrides.update({"ocr.max_side": max_side, "ocr.retry_max_side": max(max_side, 3400)})
        if bands:
            overrides.update({"layout.regions": "bands", "layout.bands": bands})
        gpu = cand.gpu
    wd, cfg = pipeline.setup_book(Path(test_pdf), work_root, overrides, slug)
    started = time.time()
    if gpu:
        wait_ready(endpoint, echo=echo)
    manifest = pipeline.ingest_pdf(Path(test_pdf), wd, cfg)
    _reuse_tesseract(wd, Path(work_root) / "bench-tesseract", manifest)
    echo(f"Reading {len(manifest.pages)} pages with {key}…")
    results = pipeline.ocr_pages(wd, cfg, manifest.pages, force=force, echo=echo)
    for entry in manifest.pages:
        if entry.index in results:
            pipeline.parse_page(entry, results[entry.index], wd, cfg)
    pages = pipeline.check_book(wd, cfg)
    if key != "tesseract" and pipeline.rescue_short_pages(wd, cfg, echo=echo):
        pages = pipeline.check_book(wd, cfg)
    seconds = time.time() - started
    failed = sum(1 for r in results.values() if r.failed)
    record = {"model": key + tag, "hf_id": get(key).hf_id if key != "tesseract" else key, "gpu": gpu,
              "pages": len(manifest.pages), "failed": failed, "seconds": round(seconds, 1),
              "cost_usd": round(seconds * GPU_PRICE_PER_S.get(gpu or "", 0.0), 3),
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "test_pdf": str(Path(test_pdf).resolve())}
    wd.write_json(wd.root / "bench.json", record)
    pipeline.report_flags(pages, echo)
    echo(f"Done in {seconds / 60:.1f} min" + (f", about ${record['cost_usd']:.2f} of GPU time" if gpu else ""))
    return wd


def rebuild(key: str, work_root: Path = Path("work"), echo=print) -> int:
    """Rebuild a run's pages from the replies already saved (``NNNN.ocr.json``): no model calls.

    Used after a change to how replies are converted (for example plain-text clean-up).
    """
    from .ocr.run import plain_to_tags

    wd = book_workdir(work_root, f"bench-{key}")
    cfg = wd.load_config()
    plain = cfg.ocr.prompt_preset == "plain"
    count = 0
    for entry in wd.load_manifest().pages:
        meta = wd.page_ocr_info(entry.index)
        if not meta.exists():
            continue
        record = json.loads(meta.read_text(encoding="utf-8"))
        texts = [a["text"] for a in record.get("attempts", []) if isinstance(a, dict) and a.get("text")]
        if not texts or "error" in record["info"].get("finish_reasons", []):
            continue
        wd.write_text(wd.page_tags(entry.index), plain_to_tags(texts[-1]) if plain else texts[-1])
        count += 1
    pipeline.reparse(wd, cfg)
    pages = pipeline.check_book(wd, cfg)
    pipeline.report_flags(pages, echo)
    return count


def _reuse_tesseract(wd: WorkDir, baseline: Path, manifest: Any) -> None:
    """Copy the baseline run's Tesseract readings for identical page images (saves minutes per run)."""
    base = WorkDir(baseline)
    if not base.manifest_path.exists() or base.root == wd.root:
        return
    theirs = {e.index: e.sha256 for e in base.load_manifest().pages}
    for entry in manifest.pages:
        src = base.page_tesseract(entry.index)
        if theirs.get(entry.index) == entry.sha256 and src.exists() and not wd.page_tesseract(entry.index).exists():
            shutil.copyfile(src, wd.page_tesseract(entry.index))


# --------------------------------------------------------------------------- report


def _pages(wd: WorkDir) -> dict[int, Page]:
    out = {}
    for i in wd.page_indices():
        page = wd.best_page(i)
        if page is not None:
            out[i] = page
    return out


def _run_stats(wd: WorkDir, pages: dict[int, Page]) -> dict[str, Any]:
    from .checks.flags import _count_words, model_word_count

    meta = json.loads((wd.root / "bench.json").read_text(encoding="utf-8")) if (wd.root / "bench.json").exists() else {}
    n = len(pages) or 1
    quran = {"verified": 0, "corrected": 0, "flagged": 0, "accepted": 0, "unchecked": 0}
    notes_total = notes_linked = notes_guessed = 0
    verses = verses_full = 0
    codes: dict[str, int] = {}
    agree = []
    for page in pages.values():
        for f in page.flags:
            if f.severity != "info":
                codes[f.code] = codes.get(f.code, 0) + 1
        for b, _o in iter_blocks(page):
            for c in block_containers(b):
                for node in walk_inlines(c):
                    if isinstance(node, Quran):
                        quran[node.status] = quran.get(node.status, 0) + 1
            if isinstance(b, Poem) and not b.single:
                verses += len(b.verses)
                verses_full += sum(1 for v in b.verses if v.sadr and v.ajuz)
        for note in page.notes():
            notes_total += 1
            notes_linked += note.link.status == "linked"
            notes_guessed += note.link.status == "guessed"
        tess = wd.page_tesseract(page.index)
        if tess.exists():
            tw, mw = _count_words(tess.read_text(encoding="utf-8")), model_word_count(page)
            if max(tw, mw) >= 20:
                agree.append(1 - abs(tw - mw) / max(tw, mw))
    spans = sum(quran.values())
    return {
        "model": meta.get("model", wd.root.name), "hf_id": meta.get("hf_id", ""), "pages": len(pages),
        "failed": meta.get("failed", 0),
        "clean": sum(1 for p in pages.values() if p.status in ("clean", "reviewed")) / n,
        "quran_spans": spans, "quran_resolved": (quran["verified"] + quran["corrected"]) / spans if spans else None,
        "notes": notes_total, "notes_linked": (notes_linked + notes_guessed) / notes_total if notes_total else None,
        "poem_verses": verses, "poem_both_halves": verses_full / verses if verses else None,
        "tesseract_agreement": statistics.median(agree) if agree else None,
        "minutes": (meta.get("seconds") or 0) / 60, "cost_usd": meta.get("cost_usd"),
        "top_flags": sorted(codes.items(), key=lambda kv: -kv[1])[:6],
    }


def _text(page: Page) -> str:
    return normalize_for_score(page_text(page, notes=True), keep_diacritics=False)


def _lenient(text: str) -> str:
    """Letters and digits only: no diacritics, punctuation or note markers; ى/ي and alef forms unified."""
    from .arabic import normalize_for_match

    text = re.sub(r"[(\[][٠-٩0-9]{1,2}[)\]]", " ", text)
    return " ".join(normalize_for_match(text).split())


def reference_scores(pages: dict[int, Page], reference: Path) -> dict[str, float | int | None]:
    """CER/WER of a run against plain reference texts (``NNNN.txt``: main, ``---``, notes)."""
    from rapidfuzz.distance import Levenshtein

    tot = {"main": [0, 0, 0, 0], "notes": [0, 0, 0, 0]}  # char edits, chars, word edits, words
    scored = 0
    for f in sorted(Path(reference).glob("*.txt")):
        i = int(f.stem)
        if i not in pages:
            continue
        ref_main, _, ref_notes = f.read_text(encoding="utf-8").partition("\n---\n")
        page = pages[i]
        scored += 1
        for key, ref, hyp in (("main", ref_main, page_text(page)), ("notes", ref_notes, page_text(page, notes=True, main=False))):
            r, h = _lenient(ref), _lenient(hyp)
            t = tot[key]
            t[0] += Levenshtein.distance(r, h)
            t[1] += len(r)
            t[2] += Levenshtein.distance(r.split(), h.split())
            t[3] += len(r.split())
    rate = lambda a, b: a / b if b else None  # noqa: E731
    m, n = tot["main"], tot["notes"]
    return {"pages": scored, "cer_main": rate(m[0], m[1]), "wer_main": rate(m[2], m[3]),
            "cer_notes": rate(n[0], n[1]), "wer_notes": rate(n[2], n[3]),
            "cer_all": rate(m[0] + n[0], m[1] + n[1]), "wer_all": rate(m[2] + n[2], m[3] + n[3])}


def report(work_root: Path = Path("work"), *, truth: Path | None = None, out: Path | None = None,
           reference: Path | None = None) -> str:
    runs = sorted(p for p in Path(work_root).glob("bench-*") if (p / "book.json").exists())
    if not runs:
        raise BenchError(f"no runs in {work_root} (run  pdf2epub bench run <model>  first)")
    data = {}
    for root in runs:
        wd = WorkDir(root)
        pages = _pages(wd)
        data[root.name.removeprefix("bench-")] = (wd, pages, _run_stats(wd, pages))
    names = list(data)
    pct = lambda v: "—" if v is None else (f"{v * 100:.1f}%" if v < 0.1 else f"{v * 100:.0f}%")  # noqa: E731
    lines = ["# Model comparison on the test set", "",
             "Measured without hand-checked truth unless a truth folder was given. Higher is better except time and cost.", "",
             "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    rows = [
        ("Model", lambda s: s["hf_id"] or s["model"]),
        ("Pages read (failed)", lambda s: f"{s['pages']} ({s['failed']})"),
        ("Pages passing every check", lambda s: pct(s["clean"])),
        ("Quran verses resolved against Tanzil", lambda s: f"{pct(s['quran_resolved'])} of {s['quran_spans']}"),
        ("Notes found / linked", lambda s: f"{s['notes']} / {pct(s['notes_linked'])}"),
        ("Poem verses with both halves", lambda s: f"{pct(s['poem_both_halves'])} of {s['poem_verses']}"),
        ("Word-count agreement with Tesseract", lambda s: pct(s["tesseract_agreement"])),
        ("Time (min)", lambda s: f"{s['minutes']:.1f}"),
        ("GPU cost", lambda s: "—" if not s["cost_usd"] else f"${s['cost_usd']:.2f}"),
        ("Most frequent flags", lambda s: ", ".join(f"{c} {n}" for c, n in s["top_flags"][:4]) or "none"),
    ]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(str(fn(data[k][2])) for k in names) + " |")
    # Agreement between models: mean whole-page CER of one against the other.
    if len(names) > 1:
        lines += ["", "## Agreement between models", "",
                  "Mean character difference between two models' text on the same pages (lower = they read alike;"
                  " models that agree with each other are usually right).", "",
                  "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for a in names:
            cells = []
            for b in names:
                if a == b:
                    cells.append("—")
                    continue
                pa, pb = data[a][1], data[b][1]
                common = [i for i in pa if i in pb]
                vals = [cer(_text(pa[i]), _text(pb[i])) for i in common if _text(pa[i])]
                cells.append(f"{statistics.mean(vals) * 100:.1f}%" if vals else "—")
            lines.append(f"| {a} | " + " | ".join(cells) + " |")
    if reference is not None:
        lines += ["", "## Against a reference text", "",
                  "Letters and words only (diacritics, punctuation and note markers ignored; ى/ي and alef forms "
                  "unified), against a digital text of the same edition. Lower is better.", "",
                  "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        ref = {k: reference_scores(data[k][1], reference) for k in names}
        for label, key in [("Word error rate, main text", "wer_main"), ("Character error rate, main text", "cer_main"),
                           ("Word error rate, notes", "wer_notes"), ("Character error rate, notes", "cer_notes"),
                           ("Word error rate, whole page", "wer_all"), ("Character error rate, whole page", "cer_all")]:
            lines.append(f"| {label} | " + " | ".join(pct(ref[k][key]) if ref[k][key] is not None else "—" for k in names) + " |")
        lines.append("| Pages scored | " + " | ".join(str(ref[k]["pages"]) for k in names) + " |")
    if truth is not None:
        lines += ["", "## Against hand-checked truth", ""]
        truth_pages = {int(p.stem): load_page(p) for p in Path(truth).glob("*.json")}
        lines += ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        scores = {k: [score_page(t, data[k][1][i]) for i, t in truth_pages.items() if i in data[k][1]] for k in names}
        def agg(k: str, num: str, den: str) -> str:
            d = sum(getattr(s, den) for s in scores[k])
            return "—" if not d else f"{sum(getattr(s, num) for s in scores[k]) / d * 100:.1f}%"
        for label, num, den in [("CER, diacritics removed", "edits_nodia", "chars"),
                                ("CER, diacritics kept", "edits_dia", "chars_dia"),
                                ("Whole-page CER (with notes)", "whole_edits", "whole_chars"),
                                ("WER", "word_edits", "words"), ("Note recall", "notes_found", "notes_truth"),
                                ("Note link accuracy", "links_correct", "links_total"),
                                ("Poem verse accuracy", "poem_correct", "poem_verses")]:
            lines.append(f"| {label} | " + " | ".join(agg(k, num, den) for k in names) + " |")
        lines.append(f"\nTruth pages: {len(truth_pages)}")
    text = "\n".join(lines) + "\n"
    out = out or Path(work_root) / "bench-report.md"
    out.write_text(text, encoding="utf-8")
    _side_by_side(data, Path(work_root) / "bench-compare.html")
    return text


def _side_by_side(data: dict[str, tuple[WorkDir, dict[int, Page], dict[str, Any]]], path: Path) -> None:
    """One section per page: the image, then each model's text (notes after a rule)."""
    names = list(data)
    first_wd = data[names[0]][0]
    indices = sorted(set().union(*(set(d[1]) for d in data.values())))
    parts = ['<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><title>مقارنة النماذج</title>',
             "<style>body{font-family:Amiri,'Noto Naskh Arabic',serif;margin:12px}section{display:grid;"
             f"grid-template-columns:420px repeat({len(names)},1fr);gap:10px;border-top:2px solid #999;padding:8px 0}}"
             "img{width:420px}pre{white-space:pre-wrap;font-family:inherit;font-size:17px;line-height:1.8;margin:0}"
             "h3{margin:0 0 4px;font-size:14px;font-family:sans-serif;direction:ltr;text-align:left}"
             ".flag{color:#b00;font-size:12px;font-family:sans-serif;direction:ltr}</style></head><body>",
             "<h1>Model comparison</h1>"]
    for i in indices:
        rel = os.path.relpath(first_wd.page_image(i), path.parent)
        parts.append(f'<section id="p{i}"><div><h3>page {i}</h3><img src="{escape(rel)}" loading="lazy"></div>')
        for k in names:
            page = data[k][1].get(i)
            if page is None:
                parts.append(f"<div><h3>{escape(k)}</h3>—</div>")
                continue
            main = page_text(page)
            notes = page_text(page, notes=True, main=False)
            flags = ", ".join(sorted({f.code for f in page.flags if f.severity != "info"}))
            parts.append(f"<div><h3>{escape(k)} · {escape(page.status)}</h3>"
                         f'<div class="flag">{escape(flags)}</div><pre>{escape(main)}</pre>'
                         + (f"<hr><pre>{escape(notes)}</pre>" if notes else "") + "</div>")
        parts.append("</section>")
    parts.append("</body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def export_truth(slug: str, work_root: Path, out: Path) -> int:
    """Reviewed pages of a bench run -> a truth folder (``NNNN.json``) for ``bench report --truth``."""
    wd = book_workdir(work_root, slug)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for i in wd.page_indices():
        if wd.page_reviewed(i).exists():
            page = load_page(wd.page_reviewed(i))
            if page.status == "reviewed":
                shutil.copyfile(wd.page_reviewed(i), out / f"{i:04d}.json")
                count += 1
    return count
