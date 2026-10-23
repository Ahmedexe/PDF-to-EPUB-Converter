"""Gather training pairs into one bundle for fine-tuning (spec §8 "Training data").

    python -m train.bundle OUT_DIR PAIRS.jsonl [PAIRS.jsonl ...] [--real-repeat 2] [--max-side 2400] [--val 30]

Each pairs file holds ``{"image", "target", "instruction"}`` lines (from
tools/build_training_pairs.py or tools/synth_pages.py). Images are resized so the
longest side is ``--max-side`` and saved as JPEG (quality 95); the instruction becomes the system
and user prompts the model will see at inference:

* ``keep_diacritics``: the normal page prompt (diacritics exactly as printed);
* ``no_diacritics``: the page prompt plus "write the text without diacritics";
* ``main_region``: the main-region prompt (the region above the foot-note rule);
* ``no_diacritics_main_region``: the main-region prompt plus "without diacritics";
* ``text_only_main_region``: the main-region prompt, without markers, header or page number
  (pages of another edition of the text, whose markers fall elsewhere).

Real pages (everything but the synthetic pages, source train-synth) are repeated
``--real-repeat`` times. ``--val`` real pages are held out. Writes ``images/``,
``train.jsonl`` and ``val.jsonl`` with chat messages ready for the trainer.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

from pdf2epub.config import BookConfig
from pdf2epub.model import LayoutRegion
from pdf2epub.ocr.prompts import page_prompts, region_prompts

NO_DIACRITICS = ("\nThis page must be written WITHOUT any diacritics: no harakat, tanwin, shadda or sukun, "
                 "only the letters, exactly as they are printed otherwise.")


TEXT_ONLY = ("\nFor this image write the words only: no note markers, no running header, no page number "
             "and no line numbers from the margin.")


def prompts(instruction: str) -> tuple[str, str]:
    cfg = BookConfig(slug="train")
    if instruction in ("main_region", "no_diacritics_main_region", "text_only_main_region"):
        system, user = region_prompts(cfg, LayoutRegion(id="main", kind="main", bbox=[0, 0, 1, 1]))
        if instruction == "no_diacritics_main_region":
            system += NO_DIACRITICS
        return (system + TEXT_ONLY if instruction == "text_only_main_region" else system), user
    system, user = page_prompts(cfg)
    if instruction == "no_diacritics":
        return system + NO_DIACRITICS, user
    return system, user


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("pairs", type=Path, nargs="+")
    ap.add_argument("--real-repeat", type=int, default=2)
    ap.add_argument("--repeat", action="append", default=[], metavar="SOURCE=N",
                    help="repeat this source's pages N times instead (source = the pairs file's folder name)")
    ap.add_argument("--limit", action="append", default=[], metavar="SOURCE=N",
                    help="use at most N pages of this source")
    ap.add_argument("--max-side", type=int, default=2400)  # same as OcrSettings.max_side
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    records = []
    repeats = {k: int(v) for k, v in (r.split("=") for r in args.repeat)}
    limits = {k: int(v) for k, v in (r.split("=") for r in args.limit)}
    for pf in args.pairs:
        recs = []
        for line in pf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["source"] = pf.parent.name
                recs.append(rec)
        if pf.parent.name in limits:
            rng.shuffle(recs)
            recs = recs[: limits[pf.parent.name]]
        records.extend(recs)
    real = [r for r in records if r.get("source") != "train-synth"]
    rng.shuffle(real)
    val_ids = {id(r) for r in real[: args.val]}
    train_lines, val_lines = [], []
    for n, rec in enumerate(records):
        img = cv2.imread(rec["image"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        scale = min(1.0, args.max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        name = f"images/{n:06d}.jpg"  # high-quality JPEG: a quarter of the PNG size to upload
        cv2.imwrite(str(args.out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        system, user = prompts(rec.get("instruction", "keep_diacritics"))
        item = {"image": name, "system": system, "user": user, "target": rec["target"],
                "instruction": rec.get("instruction"), "source": rec["source"]}
        line = json.dumps(item, ensure_ascii=False)
        if id(rec) in val_ids:
            val_lines.append(line)
        else:
            repeat = repeats.get(rec["source"], args.real_repeat if rec["source"] != "train-synth" else 1)
            train_lines.extend([line] * repeat)
    rng.shuffle(train_lines)
    (args.out / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (args.out / "val.jsonl").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    print(f"train {len(train_lines)} examples, val {len(val_lines)}, images {len(records)}")


if __name__ == "__main__":
    main()
