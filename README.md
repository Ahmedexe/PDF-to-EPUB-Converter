# pdf2epub

Convert scanned Arabic PDF books into right-to-left, reflowable EPUB3 for Kindle and Boox.

An open vision-language model reads each page image and returns lightly tagged text. The default
is `ft4`, Qwen3-VL-8B fine-tuned on Arabic book pages in many typefaces (see "Fine-tuning" below). The model runs
on a rented cloud GPU or any OpenAI-compatible endpoint. The tool then checks that text and
rebuilds the book. Along the way it:

- **matches every Quran verse** against the verified Tanzil text: it replaces the verse with the
  book's spelling (Uthmani or standard) or flags it for review;
- **keeps poems whole**: both half-lines of each verse, in order, checked for rhyme;
- **links every حاشية note** to the words it explains: by printed marker, by قوله lemma
  (including on the previous page), by position for margin notes, or as a continuation;
- **flags suspicious pages** for a local review screen: looping text, word counts far from
  Tesseract's, unclear words, missing or unpaired notes, failed or cut-off model calls;
- **builds a clean EPUB3**: RTL page turns, embedded Amiri fonts, ﴿ ﴾ in the right order, a page
  list matching the printed edition, pop-up, inline or chapter-end notes, and a Tanzil credits
  page. Every book is validated with EPUBCheck.

The design is in [arabic-pdf-to-epub-spec.md](arabic-pdf-to-epub-spec.md). Module contracts and
the rules they follow are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Install

```sh
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Optional tools, both vendored:

- **Java**, to run EPUBCheck 5.4.0 from `vendor/epubcheck/`.
- **Tesseract**, for the word-count cross-check, using the Arabic `tessdata_best` model in
  `vendor/tessdata/`.

## Try it without a GPU

```sh
pdf2epub sample demo/                                        # a 5-page synthetic scanned book
pdf2epub convert demo/sample.pdf --fixtures demo/fixtures    # replays recorded model output
```

This writes `work/sample/out/sample.epub`, which passes EPUBCheck. `--backend tesseract` is an
offline dry run with real but weak OCR. The checks flag every page it gets wrong, which shows
the review step at work.

## Converting a real book

1. **Run the model somewhere.** `pdf2epub bench deploy ft-ft4` serves the default model with
   vLLM on a scale-to-zero L40S on Modal (no cost while idle) and records its endpoint and API key
   in `~/.config/pdf2epub/`. Any OpenAI-compatible endpoint works too (`--endpoint`, `--model`),
   including a vLLM server on a rented pod; the base model is `Qwen/Qwen3-VL-8B-Instruct`.
2. **Convert:**

   ```sh
   pdf2epub convert sharh.pdf --layout two-column --notes popup --commentary mixed \
       --title "..." --author "..."
   ```

   Conversion is unattended. Pages the model ends too early (far fewer words than Tesseract
   finds) are read again in horizontal bands. Pages the checks still flag are read once more at
   higher resolution and then in bands, and a new reading is kept only when the checks find fewer
   problems in it. The final EPUB is then built, with `out/<book>-quality.md` listing what the
   checks still flag, page by page, for spot checks. Nothing waits for a person.

   Try a few pages first with `--pdf-pages 1-20`. Books with margin or framed notes:
   `--hashiya foot,lemma,margin --regions auto`.
3. **Review (optional):** `pdf2epub review sharh` opens a local page in your browser. It shows
   the page image beside the text, with flags highlighted and lines from each note to its anchor.
   Edit a page as tags, fix a note link by selecting the right words, then approve. Every fix is
   saved under `work/<book>/training/` for fine-tuning later. With `--review required` the EPUB
   is not built until every flagged page is fixed or approved here.
4. **Rebuild or re-read:**

   ```sh
   pdf2epub build sharh --notes inline    # display options only; no model calls
   pdf2epub redo sharh --pages 37,112     # re-read pages with the model
   pdf2epub check sharh --reparse         # re-run parser and checks from cached output
   pdf2epub status sharh
   ```

Every stage caches its output in `work/<book>/`. A crash or a re-run picks up where it stopped,
and pages whose model call failed are retried. With `--review required`, `build` refuses while
pages still need review, unless you pass `--draft`.

## Comparing models (phase 1)

Put about 100 hand-verified pages in a test-set folder. Each page needs `pages/<id>.png` and a
`truth/<id>.json` (a reviewed page), `.tags` or `.txt`, plus an optional `meta.json` for per-book
settings. Then run:

```sh
pdf2epub eval testset/ --endpoint URL --model Qwen/Qwen3-VL-8B-Instruct
pdf2epub eval testset/ --endpoint URL2 --model other/model --compare testset/results/Qwen_Qwen3-VL-8B-Instruct/scores.json
```

The report covers every spec §9 metric against its target, ranking models by whole-page CER:
CER with and without diacritics, WER, Quran wrong-verse and auto-resolved rates, poem verse
accuracy, block types, note recall and link accuracy, region detection and missed errors.

## Project layout

```
pdf2epub/
  cli.py  pipeline.py          commands and stage orchestration
  model.py  arabic.py          page model; Arabic normalisation
  config.py  workdir.py        per-book settings; work-folder layout and caching
  ingest/                      PDF -> page images (extract scans, deskew, crop, split spreads)
  layout/                      region detection (VLM boxes, CV fallback), upright crops
  ocr/                         backends, client and cache, prompts, page reading, Modal server
  parse/                       tagged text <-> page model
  checks/                      quran.py, poetry.py, hashiya.py, flags.py
  assemble/                    join pages, continued notes, chapters
  epub/                        XHTML rendering, packaging, EPUBCheck
  review/                      local review screen and training-data export
  eval/                        test-set scoring and model comparison
  devtools/sample.py           synthetic scanned book with ground truth
  data/tanzil/                 Quran text, verbatim, with its licence
tests/                         pytest suite (`.venv/bin/python -m pytest`)
```

## Fine-tuning

Training pairs come from scans whose digital edition (al-Maktaba al-Shamela, turath.io) is the same
print: the base model reads each page, the page is located in the edition, and the target takes the
edition's letters and structure (`tools/build_training_pairs.py`). Synthetic vowelled pages come from
`tools/synth_pages.py`. `python -m train.bundle` packs the pairs, `python -m train.launch` trains a
LoRA on a Modal H100 and merges it into `/models/<run>/merged`, and `pdf2epub bench run ft-<run>`
scores it on the test sets (`sample/test`, and `sample/test2`, a book never used for training).

The web app (phase 4) is not built yet.

## Licences

- Quran text: Tanzil Project, CC BY 3.0, used verbatim
  ([pdf2epub/data/tanzil](pdf2epub/data/tanzil/README.md)).
- Fonts: Amiri, Amiri Quran and Noto Naskh Arabic, SIL Open Font License.
- EPUBCheck: BSD-3-Clause. Tesseract data: Apache-2.0.
