"""Diacritic accuracy on held-out pages with checked vowelling (sample/test4/heldout.jsonl).

    python tools/score_diacritics.py MODEL_KEY [MODEL_KEY ...] [--test sample/test4/heldout.jsonl]

Each model (a key from ``pdf2epub bench list``, e.g. ``qwen3-vl-8b`` or ``ft-ft2``) reads every
held-out image with the prompt it would get in use (whole page, or the main-text region). Words
are aligned with the reference by their letters; for words read with the right letters, every
letter's marks are compared. Reports, per model:

* diacritic error rate (DER): letters whose marks differ from the reference, over all letters
  of correctly read words (a mark printed but missing counts, and so does one added);
* the same counting only letters the reference marks (missed or wrong marks);
* diacritics per letter, in the reference and in the output.

Readings are cached in ``work/diacritics-<key>/``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from pdf2epub.arabic import DIACRITICS, normalize_for_match  # noqa: E402
from pdf2epub.bench import api_key, endpoints, wait_ready  # noqa: E402
from pdf2epub.config import OcrSettings  # noqa: E402
from pdf2epub.ocr.backends import OpenAIBackend  # noqa: E402
from pdf2epub.ocr.client import encode_image  # noqa: E402
from pdf2epub.ocr.models import get  # noqa: E402
from train.bundle import prompts  # noqa: E402

TAG = re.compile(r"<[^>]+>")


def words(tags: str) -> list[str]:
    return [w for w in TAG.sub(" ", tags).split() if normalize_for_match(w)]


def marks(word: str) -> list[tuple[str, str]]:
    """(letter, its marks) for each letter; marks sorted so order does not matter."""
    out: list[tuple[str, str]] = []
    for ch in word:
        if ch in DIACRITICS:
            if out:
                out[-1] = (out[-1][0], "".join(sorted(out[-1][1] + ch)))
        elif "ء" <= ch <= "ي" or ch in "ٱآأإ":
            out.append((ch, ""))
    return out


async def read_all(key: str, items: list[dict], cache: Path) -> dict[int, str]:
    cache.mkdir(parents=True, exist_ok=True)
    name = {k: Path(item["image"]).stem for k, item in enumerate(items)}  # cache by page, not position
    todo = [k for k in range(len(items)) if not (cache / f"{name[k]}.tags").exists()]
    if todo:
        url = endpoints().get(key)
        if not url:
            raise SystemExit(f"no endpoint for {key}: run  pdf2epub bench deploy {key}")
        os.environ.setdefault("PDF2EPUB_API_KEY", api_key() or "")
        cand = get(key)
        backend = OpenAIBackend(OcrSettings(endpoint=url, model=cand.hf_id, timeout_s=600))
        wait_ready(url)
        sem = asyncio.Semaphore(8)

        async def one(k: int) -> None:
            item = items[k]
            system, user = prompts(item.get("instruction", "keep_diacritics"))
            data, _ = encode_image(Path(item["image"]), 2400)
            async with sem:
                c = await backend.complete(data, system, user, request_id=f"{k:03d}", max_tokens=8192,
                                           temperature=0.0, extra=dict(cand.extra_body))
            (cache / f"{name[k]}.tags").write_text(c.text, encoding="utf-8")

        await asyncio.gather(*(one(k) for k in todo))
        await backend.aclose()
    return {k: (cache / f"{name[k]}.tags").read_text(encoding="utf-8") for k in range(len(items))}


def score(items: list[dict], readings: dict[int, str]) -> dict[str, float]:
    letters = wrong = marked = wrong_marked = ref_d = out_d = ref_l = out_l = 0
    for k, item in enumerate(items):
        ref, hyp = words(item["target"]), words(readings[k])
        for w in ref:
            m = marks(w); ref_l += len(m); ref_d += sum(len(x) for _, x in m)
        for w in hyp:
            m = marks(w); out_l += len(m); out_d += sum(len(x) for _, x in m)
        rn, hn = [normalize_for_match(w) for w in ref], [normalize_for_match(w) for w in hyp]
        for op in Levenshtein.opcodes(rn, hn):
            if op.tag != "equal":
                continue
            for a, b in zip(range(op.src_start, op.src_end), range(op.dest_start, op.dest_end)):
                mr, mh = marks(ref[a]), marks(hyp[b])
                if len(mr) != len(mh):
                    continue
                for (_l1, d1), (_l2, d2) in zip(mr, mh):
                    letters += 1
                    wrong += d1 != d2
                    if d1:
                        marked += 1
                        wrong_marked += d1 != d2
    return {"der": wrong / max(1, letters), "der_marked": wrong_marked / max(1, marked),
            "ref_density": ref_d / max(1, ref_l), "out_density": out_d / max(1, out_l), "letters": letters}


def main() -> None:
    args = [a for a in sys.argv[1:]]
    test = Path(args[args.index("--test") + 1]) if "--test" in args else Path("sample/test4/heldout.jsonl")
    keys = [a for a in args if not a.startswith("--") and a != str(test)]
    items = [json.loads(l) for l in test.read_text(encoding="utf-8").splitlines() if l.strip()]
    for key in keys:
        readings = asyncio.run(read_all(key, items, Path("work") / f"diacritics-{key}"))
        s = score(items, readings)
        print(f"{key:14} DER {s['der']:6.1%} | on marked letters {s['der_marked']:6.1%} | "
              f"diacritics per letter: reference {s['ref_density']:.2f}, output {s['out_density']:.2f} "
              f"({s['letters']} letters compared)")


if __name__ == "__main__":
    main()
