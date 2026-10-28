"""How much could rescoring with a language model fix? An upper bound from the model's own alternatives.

    python tools/oracle_alternatives.py MODEL_KEY [--test sample/test/part.pdf] [--ref sample/test/reference] [--out DIR]

Re-reads the test pages with the same prompt and settings as ``pdf2epub bench run``, asking the
server for each token's top alternatives (``logprobs``), and saves them in
``work/oracle-<key>/NNNN.json``. Then aligns each page's main text with the reference and sorts
every error:

* substitution fixable from the model's alternatives: the reference word is one token swap away
  from what the model wrote, using a token the model itself considered;
* substitution by dots only: same letter skeleton (rasm), different dots; a language model
  choosing among dot variants could fix it;
* other substitutions, missing words, extra words.

A language model rescoring the model's candidates can fix at most the first two groups.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from pdf2epub import pipeline  # noqa: E402
from pdf2epub.bench import _lenient, api_key, endpoints, wait_ready  # noqa: E402
from pdf2epub.ocr.client import encode_image  # noqa: E402
from pdf2epub.ocr.models import get  # noqa: E402
from pdf2epub.ocr.prompts import page_prompts  # noqa: E402

RASM = str.maketrans({"ب": "ٮ", "ت": "ٮ", "ث": "ٮ", "ن": "ٮ", "ي": "ى", "ج": "ح", "خ": "ح", "ذ": "د", "ز": "ر",
                      "ش": "س", "ض": "ص", "ظ": "ط", "غ": "ع", "ف": "ڡ", "ق": "ڡ", "ة": "ه"})
TOP = 5


async def read_pages(key: str, test_pdf: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cand = get(key)
    wd, cfg = pipeline.setup_book(test_pdf, Path("work"), {"ocr.model": cand.hf_id, "checks.tesseract": False},
                                  f"{out.name}-book")
    manifest = pipeline.ingest_pdf(test_pdf, wd, cfg)
    system, user = page_prompts(cfg)
    url = endpoints()[key].rstrip("/") + "/chat/completions"
    wait_ready(endpoints()[key])
    headers = {"Authorization": f"Bearer {api_key()}"}
    sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient(timeout=900, follow_redirects=True, headers=headers) as client:
        async def one(entry) -> None:
            dest = out / f"{entry.index:04d}.json"
            if dest.exists() or entry.blank:
                return
            data, _ = encode_image(wd.abs(entry.image), cfg.ocr.max_side)
            body = {"model": cand.hf_id, "temperature": 0, "max_tokens": cfg.ocr.max_tokens,
                    "logprobs": True, "top_logprobs": TOP, **cand.extra_body,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}},
                        {"type": "text", "text": user}]}]}
            async with sem:
                for attempt in range(4):
                    r = await client.post(url, json=body)
                    if r.status_code == 200:
                        break
                    await asyncio.sleep(5 + 10 * attempt)
            choice = r.json()["choices"][0]
            toks = [{"t": t["token"], "lp": t["logprob"], "alt": [[a["token"], a["logprob"]] for a in t["top_logprobs"]]}
                    for t in choice["logprobs"]["content"]]
            dest.write_text(json.dumps({"text": choice["message"]["content"], "tokens": toks}, ensure_ascii=False))
        await asyncio.gather(*(one(e) for e in manifest.pages))


def words_with_candidates(tokens: list[dict]) -> list[tuple[str, set[str]]]:
    """Main-text words in order (no header, page number or notes), each with the words that one
    swap of a token for one of its alternatives would have produced."""
    text = "".join(t["t"] for t in tokens)
    starts, pos = [], 0
    for t in tokens:
        starts.append(pos)
        pos += len(t["t"])
    cut = text.find("<hashiya")
    end = cut if cut >= 0 else len(text)
    skip = [(m.start(), m.end()) for m in re.finditer(r"<header>.*?</header>|<pagenum>.*?</pagenum>|<[^>]+>", text[:end], re.S)]
    out = []
    for m in re.finditer(r"[^\s<>]+", text[:end]):
        a, b = m.span()
        if any(s <= a < e for s, e in skip):
            continue
        word = m.group(0)
        cands = set()
        for k, t in enumerate(tokens):
            ta, tb = starts[k], starts[k] + len(t["t"])
            if tb <= a or ta >= b:
                continue
            for alt, _lp in t["alt"]:
                if alt == t["t"]:
                    continue
                new = text[a:ta] + alt + text[tb:b] if ta >= a else (alt + text[tb:b])[max(0, a - ta):]
                cands.add(_lenient(new).replace(" ", ""))
        out.append((_lenient(word), cands))
    return [(w, c) for w, c in out if w]


def analyse(out: Path, ref_dir: Path) -> None:
    kinds: Counter = Counter()
    total = 0
    examples: dict[str, list] = {}
    for f in sorted(ref_dir.glob("*.txt")):
        page = out / f"{int(f.stem):04d}.json"
        if not page.exists():
            continue
        ref = _lenient(f.read_text(encoding="utf-8").partition("\n---\n")[0]).split()
        hyp = words_with_candidates(json.loads(page.read_text(encoding="utf-8"))["tokens"])
        hw = [w for w, _ in hyp]
        total += len(ref)
        for op in Levenshtein.opcodes(ref, hw):
            if op.tag == "equal":
                continue
            r, h = ref[op.src_start:op.src_end], hyp[op.dest_start:op.dest_end]
            if op.tag == "delete":
                kind = "missing words (not fixable by rescoring)"
                n = len(r)
            elif op.tag == "insert":
                kind = "extra words (not fixable by rescoring)"
                n = len(h)
            elif len(r) == len(h):
                for rw, (hw1, cands) in zip(r, h):
                    if rw in cands:
                        k = "wrong word, right one among the model's alternatives"
                    elif rw.translate(RASM) == hw1.translate(RASM):
                        k = "wrong dots only (a dot-variant chooser could fix)"
                    else:
                        k = "wrong word, right one not among the alternatives"
                    kinds[k] += 1
                    examples.setdefault(k, []).append((rw, hw1))
                continue
            else:
                kind = "several words garbled"
                n = max(len(r), len(h))
            kinds[kind] += n
            examples.setdefault(kind, []).append((" ".join(r)[:40], " ".join(w for w, _ in h)[:40]))
    err = sum(kinds.values())
    print(f"reference words {total}; error words {err} ({err / max(1, total):.1%})")
    for k, n in kinds.most_common():
        print(f"  {n:5} {n / max(1, err):6.1%}  {k}   e.g. {examples[k][:3]}")
    fixable = kinds["wrong word, right one among the model's alternatives"] + kinds["wrong dots only (a dot-variant chooser could fix)"]
    print(f"upper bound for rescoring: {fixable} of {err} error words ({fixable / max(1, err):.0%}), "
          f"word error rate {err / total:.1%} -> {(err - fixable) / total:.1%}")


def main() -> None:
    key = sys.argv[1]
    test = Path(sys.argv[sys.argv.index("--test") + 1]) if "--test" in sys.argv else Path("sample/test/part.pdf")
    ref = Path(sys.argv[sys.argv.index("--ref") + 1]) if "--ref" in sys.argv else Path("sample/test/reference")
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path("work") / f"oracle-{key}"
    os.environ.setdefault("PDF2EPUB_API_KEY", api_key() or "")
    asyncio.run(read_pages(key, test, out))
    analyse(out, ref)


if __name__ == "__main__":
    main()
