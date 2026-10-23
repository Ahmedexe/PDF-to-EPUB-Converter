# Architecture and module contracts

This file is the contract between modules. The design itself is in
`arabic-pdf-to-epub-spec.md` (cited below as "spec §N"). Where this file and
the spec disagree on an interface, this file wins; where they disagree on
behaviour, the spec wins and this file should be fixed.

## 0. Status (19 October 2026)

All modules below are implemented and tested (165 tests, `.venv/bin/python -m pytest`). The
synthetic sample book converts end to end (fixture backend) into an EPUB that passes EPUBCheck
with no warnings in all three note modes. Not yet built (gated by the spec's roadmap): the
phase-3 synthetic training-data renderer, fine-tuning scripts and layout model, and the phase-4
web app. Section 5 records the rules settled by the contract review; where the signatures in
section 3 changed during implementation (for example `only=` on the checks, `refresh=` on the
OCR client, `build_epub` returning a list), the code is the reference.

## 1. Data flow

```
PDF ──ingest──▶ pages/NNNN.png ──layout──▶ NNNN.regions.json ──ocr──▶ NNNN.tags
    ──parse──▶ NNNN.parsed.json ──checks (book-level)──▶ NNNN.json ──review──▶ NNNN.reviewed.json
    ──assemble──▶ out/book.json (Book) ──epub──▶ out/<slug>.epub ──epubcheck──▶ out/epubcheck.json
```

* Only layout detection and OCR call the GPU endpoint. Everything else runs locally.
* Every stage caches its output in the work folder (`pdf2epub/workdir.py` lists every file).
* Checks run over the whole book in page order, so a check can look at neighbouring pages
  (poems and notes continued across a page break, lemmas that match the previous page).
* A page reaches the EPUB only when its status is `clean` or `reviewed`
  (spec §2), unless the build is a draft (`build.allow_flagged`).

## 2. Shared core (frozen)

These are written and tested. Module work must not edit them; if a change is needed,
report it as a contract change request instead.

| File | What it gives you |
| --- | --- |
| `pdf2epub/model.py` | The page model (pydantic): inline nodes (`str`, `Ref`, `Quran`, `Quote`, `Matn`, `Unclear`, `PageBreak`), blocks (`Heading`, `Paragraph`, `QuranBlock`, `Poem`/`Verse`, `Figure`, `Separator`), `Note`, `NoteLink`, `HashiyaRegion`, `Flag`/`FlagCode`, `OcrInfo`, `Page`, `LayoutRegion`, `PageLayout`, `PageEntry`, `Manifest`, `Chapter`, `Book`; traversal (`iter_blocks`, `block_containers`, `walk_inlines`, `walk_inline_slots`, `iter_quran`, `iter_refs`, `inline_text`, `block_text`); `BlockText` (offsets ↔ nodes, `insert_at`, `wrap_range`); ids (`assign_ids`, `next_ref_id`, `page_prefix`); JSON io (`save_model`, `load_page`, `load_book`) |
| `pdf2epub/arabic.py` | `clean_text` (parser clean-up), `normalize_with_map` / `normalize_for_match` / `match_words` (matching), `normalize_for_score` (eval), `strip_diacritics`, `unify_letters`, `to_rasm`, feature tests (`has_diacritics`, `diacritic_ratio`, `has_pause_marks`, `has_uthmani_features`), digits (`parse_number`, `format_number`, `detect_digit_style`), `normalize_marker`, `ORNATE_OPEN` (U+FD3F, before a verse) and `ORNATE_CLOSE` (U+FD3E, after) |
| `pdf2epub/config.py` | `BookConfig` with groups `ingest`, `layout`, `ocr`, `text`, `checks`, `build`; `apply_overrides`, `slugify`, `default_tessdata_dir`, `default_epubcheck_jar` |
| `pdf2epub/workdir.py` | `WorkDir`: every path in a book's work folder, manifest/config/page io |
| `pdf2epub/parse/` | `parse_tags(raw, base_page, commentary=...) -> ParseResult`, `tags_parse_ok(raw)`, `serialize_page(page)` and friends |
| `pdf2epub/ocr/prompts/*.txt`, `tagspec.md` | Prompt templates (`string.Template`, `$name` placeholders) and the tag format |
| `tests/fixtures/tags/*.tags` | Realistic model outputs: `basic`, `commentary`, `messy`, and a 3-page sequence `seq_0001..0003` (paragraph, poem and note continued across pages, a lemma matching the previous page) |

Key model conventions:

* Inline lists hold plain `str` runs and inline objects. Offsets are never stored as truth;
  use `BlockText(block)` to get the flattened text and map offsets back. Poem text is
  `sadr + "\t" + ajuz`, verses joined by `"\n"`.
* `Quran.text` never contains the ornate brackets. The builder adds `ORNATE_OPEN` before and
  `ORNATE_CLOSE` after, in logical order.
* A note's anchor is a `Ref` node in the text. Printed markers come from the model
  (`generated=False`); the linker inserts `generated=True` refs for lemma and position notes.
  `Ref.note` and `Note.link.ref` point at each other.
* Ids: `p0037-b3` (block), `p0037-h1` (region), `p0037-n2` (note), `p0037-n2-b1` (block in a
  note), `p0037-r4` (ref). `assign_ids` renumbers and remaps links; use `next_ref_id` for new refs.
* Flags carry a `stage`; a stage calls `page.clear_flags(stage)` before re-running so re-runs
  are idempotent. Use `FlagCode` constants. Severity `warning` or `error` makes a page need
  review (configurable: `checks.block_on`).
* All text inside models is `clean_text`-ed NFC. Never store bidi control characters.

## 3. Module contracts

Signatures are binding; internals are free. Every public function has a docstring and
type hints. Async functions are marked.

### 3.1 `pdf2epub/ingest/` (spec §2 step 1, §10)

`ingest/render.py`
```python
def ingest_pdf(pdf_path: Path, wd: WorkDir, cfg: BookConfig, *,
               pdf_pages: set[int] | None = None, force: bool = False,
               progress: Callable[[int, int], None] | None = None) -> Manifest
```
* Opens the PDF with PyMuPDF. For each page (1-based; restricted to `pdf_pages` when given):
  if `cfg.ingest.extract` and the page shows exactly one raster image covering ≥ 90% of the
  page without rotation or clipping tricks, extract that image's original bytes untouched
  (`source="extract"`); otherwise render at `cfg.ingest.dpi` (`source="render"`).
* Clean-up (`ingest/cleanup.py`), each step switchable in `cfg.ingest`: deskew, crop scanner
  borders and empty margins (never cut printed text, including margin notes), split two-page
  spreads. In a right-to-left book the **right** half of a spread comes first.
* Writes `pages/NNNN.png` (grayscale when `cfg.ingest.grayscale`) and, when the source is in
  colour, `pages/NNNN.src.png` (same geometry, colour) for figure crops. Book page indices are
  assigned 1.. in reading order after splitting. Saves and returns the `Manifest` (with the
  PDF's sha256 and page count). Re-running skips pages whose image already exists unless
  `force`; the manifest is rebuilt consistently.

`ingest/cleanup.py` (pure functions on numpy arrays, uint8, gray or BGR)
```python
def to_gray(img: np.ndarray) -> np.ndarray
def estimate_skew(gray: np.ndarray, max_angle: float = 5.0) -> float   # degrees, + = counter-clockwise
def deskew(img: np.ndarray, max_angle: float = 5.0) -> tuple[np.ndarray, float]
def content_box(gray: np.ndarray) -> tuple[int, int, int, int]         # x0, y0, x1, y1 of all ink
def crop_borders(img: np.ndarray, pad: int = 20) -> tuple[np.ndarray, tuple[int, int, int, int]]
def find_spread_split(gray: np.ndarray) -> int | None                  # x of the gutter, or None
def split_spread(img: np.ndarray, x: int) -> tuple[np.ndarray, np.ndarray]  # (right page, left page)
def is_blank(gray: np.ndarray) -> bool
```
* Deskew only corrects small angles (scan tilt), never 90° turns. Dark scanner edges and
  black borders are removed before measuring content. A spread is detected only for
  landscape images with a clear low-ink gutter near the middle (and `spreads="auto"`).

### 3.2 `pdf2epub/layout/` (spec §6.3)

`layout/detect.py`
```python
def needs_layout(gray: np.ndarray) -> bool
def cv_regions(gray: np.ndarray) -> list[LayoutRegion]
def parse_layout_response(text: str, width: int, height: int, bbox_format: str = "rel1000") -> list[LayoutRegion]
async def detect_layout(entry: PageEntry, wd: WorkDir, cfg: BookConfig, client: "OcrClient") -> PageLayout
```
* `needs_layout`: a classical-CV test for pages whose notes are *not* simply at the foot:
  substantial ink in a side or top margin outside the main text column, rotated/slanted text
  blocks, or ruled frames. Conservative: pages with only a foot rule + notes return False.
* `cv_regions`: CV-only best effort (main block; foot notes below a horizontal rule; margin
  blocks), used when the VLM answer is unusable and for tests.
* `parse_layout_response`: tolerant JSON extraction (code fences, trailing prose, `bbox_2d`
  or `bbox`, coordinates in 0–1000 when `rel1000`, absolute pixels when `abs`); clamps boxes
  to the page, drops degenerate ones, assigns ids (`main`, `h1`, `h2`…, `header`, `pagenum`,
  `fig1`, `frame1`), maps `pos` aliases.
* `detect_layout`: per `cfg.layout.regions` — `never`: `PageLayout(source="none", split=False)`;
  `auto`: run `needs_layout`, and only if True ask the VLM (`ocr/prompts/layout.txt`);
  `always`: ask the VLM. `split=True` only when the page has hashiya regions that are not just
  a bottom region (margin, top, frame) or rotated text — pages with only foot notes are read
  in one call (spec §6.3 step 3). Writes `pages/NNNN.regions.json`. On a failed or unusable VLM
  answer: fall back to `cv_regions`, `split` as above, and record `source="cv"`; never raise
  for a bad answer. Rotation is clamped to a sane range and snapped to 0/90/180/270 when
  within 8° of one.

`layout/crop.py`
```python
def crop_region(img: np.ndarray, region: LayoutRegion, pad: int = 12) -> np.ndarray   # upright
def reading_order(regions: list[LayoutRegion]) -> list[LayoutRegion]
def write_region_crops(entry: PageEntry, layout: PageLayout, wd: WorkDir, pad: int = 12) -> PageLayout
```
* `crop_region` crops with padding (clamped), then rotates clockwise by `region.rotation`
  (exact transposes for multiples of 90°, `warpAffine` with white fill and expanded canvas
  otherwise) so the text is upright.
* `reading_order`: main text first (frames in right-to-left, top-to-bottom order), then hashiya:
  bottom, margin-right, margin-left, top, frame (spec §6.3 step 4); header and page-number
  regions are not read separately.

### 3.3 `pdf2epub/ocr/` (spec §4)

`ocr/backends.py`
```python
@dataclass
class Completion:
    text: str
    finish_reason: str = "stop"      # "stop" | "length" | "error"
    model: str = ""
    elapsed_s: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)

class Backend(Protocol):
    name: str
    async def complete(self, image_png: bytes, system: str, user: str, *, request_id: str,
                       max_tokens: int, temperature: float, extra: dict[str, Any]) -> Completion: ...
    async def aclose(self) -> None: ...

class OpenAIBackend:   # OpenAI-compatible /chat/completions (vLLM); image as base64 data URL
class FixtureBackend:  # returns <fixtures>/<request_id>.tags (or .txt / .json for layout); for tests and demos
class TesseractBackend:# offline dry runs: Tesseract text wrapped as <p> lines; layout requests -> '{"regions": []}'
def make_backend(ocr: OcrSettings) -> Backend
```
* `request_id` is `NNNN` for a whole page, `NNNN.<region_id>` for a region, `NNNN.layout`
  for a layout call. The fixture backend looks up exactly those names and raises
  `FileNotFoundError` with a clear message when one is missing.
* `OpenAIBackend`: `httpx.AsyncClient`, `Authorization: Bearer <key>` when a key is set,
  `timeout_s`, retries with exponential backoff and jitter on connect errors, timeouts, 429
  and 5xx (honour `Retry-After`), no retry on other 4xx. Payload: model, messages
  (system + user with `[{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}},
  {"type":"text","text":user}]`), `max_tokens`, `temperature`, plus `extra_body` merged in.

`ocr/client.py`
```python
class OcrClient:
    def __init__(self, cfg: BookConfig, wd: WorkDir, backend: Backend | None = None): ...
    async def read(self, image: np.ndarray | Path, system: str, user: str, *, request_id: str,
                   max_side: int | None = None) -> tuple[Completion, bool]   # (completion, cached)
    async def aclose(self) -> None
def encode_image(image: np.ndarray | Path, max_side: int, fmt: str = "png") -> tuple[bytes, tuple[int, int]]
def cache_key(image_bytes: bytes, model: str, prompt_version: str, system: str, user: str, params: dict) -> str
```
* Concurrency limited by `cfg.ocr.concurrency` (one semaphore per client). Images are
  downscaled so the longest side is at most `max_side` (never upscaled). The response cache
  lives in `wd.ocr_cache_dir/<key>.json` and is keyed by image bytes, model, prompt version,
  prompts and generation parameters; failed completions are never cached.

`ocr/prompts.py`
```python
PROMPT_VERSION: str   # short hash of all template files
def page_prompts(cfg: BookConfig) -> tuple[str, str]                       # (system, user)
def region_prompts(cfg: BookConfig, region: LayoutRegion) -> tuple[str, str]
def layout_prompts(cfg: BookConfig) -> tuple[str, str]
```
* Fills `$hashiya_rules` from `cfg.text.hashiya_forms` (margin, frame, layered, editor), and
  `$commentary_rules` from `cfg.text.commentary` ("mixed": wrap main-text words printed in
  parentheses or bold in `<matn>`, keeping the parentheses). Empty rules leave no blank lines.

`ocr/run.py`
```python
@dataclass
class PageOcr:
    tags: str
    info: OcrInfo
    layout: PageLayout | None
async def ocr_page(entry: PageEntry, wd: WorkDir, cfg: BookConfig, client: OcrClient, *,
                   force: bool = False) -> PageOcr
async def ocr_book(entries: list[PageEntry], wd: WorkDir, cfg: BookConfig, *, force: bool = False,
                   client: OcrClient | None = None,
                   progress: Callable[[int, int, PageOcr | None], None] | None = None) -> dict[int, PageOcr]
```
* `ocr_page`: if `pages/NNNN.tags` exists and not `force`, return it (with the saved
  `OcrInfo`). Otherwise run `layout.detect_layout`; if `split`, crop regions upright and read
  each with its region prompt; else read the whole page with the page prompt.
* Higher-resolution retry (spec §4 rule 6): when the output fails `tags_parse_ok` or ends
  with `finish_reason == "length"`, retry once with `cfg.ocr.retry_max_side`; keep the
  attempt that parses (else the last one). `OcrInfo.attempts` records it.
* Region merge: main-region tags first, then for each hashiya region in reading order
  `<hashiya pos="…" bbox="x0,y0,x1,y1" rotation="…" layer="…">` + its `<note>` output +
  `</hashiya>` (if a region's output already has a `<hashiya>` wrapper, unwrap it first).
  Independent frames become `<hashiya pos="frame" independent="1" …>` + block output.
* Writes `pages/NNNN.tags` and `pages/NNNN.ocr.json` (`{"info": OcrInfo, "attempts": [...]}`)
  atomically. A page whose calls all fail gets empty tags and an `OcrInfo` with
  `finish_reasons=["error"]`; `ocr_book` never aborts the whole book for one page.
* `ocr_book` runs pages concurrently (bounded by the client), calls `progress` after each.

`ocr/serve_modal.py`: a Modal app serving the model with vLLM's OpenAI-compatible server on an
L40S, scale to zero, weights cached in a Modal Volume, API key from a Modal Secret, 16-bit
weights (no 4-bit: spec §8 step 6), `--limit-mm-per-prompt` for one image, a sensible
`--max-model-len`. Usage instructions in the module docstring (`modal deploy …`, the URL to
pass as `--endpoint`). Must import without `modal` installed only when run (guard the import
so `pytest` collection never needs Modal).

### 3.4 `pdf2epub/checks/` (spec §4 bad pages, §5, §6.4–6.5)

Book-level entry point (written during integration, in `checks/__init__.py`):
```python
def run_checks(pages: list[Page], wd: WorkDir, cfg: BookConfig, *, tags: dict[int, str] | None = None) -> list[Page]
```
It calls, in order: `flags.check_health` per non-reviewed page, `quran.check_quran`,
`poetry.check_poetry`, `hashiya.link_notes`, then `flags.finalize_status` per page.
Reviewed pages are never re-checked or re-flagged; they may still be read as neighbours,
and the linker may add or move *generated* refs in them.

`checks/quran.py` (spec §5.1)
```python
@dataclass
class QuranMatch:
    status: Literal["verified", "corrected", "flagged"]
    score: float
    ref: str | None
    loc: QuranLoc | None
    text: str | None           # replacement text in the chosen variant (quoted part only)
    variant: str | None
    alternatives: list[str]    # other refs whose text is identical
class QuranIndex:
    @classmethod
    def load(cls, data_dir: Path | None = None) -> "QuranIndex"     # cached singleton per data_dir
    def match(self, text: str, *, style: str = "auto", threshold: float = 0.90, min_words: int = 3) -> QuranMatch
    def verse_text(self, sura: int, aya: int, variant: str) -> str
    def sura_name(self, sura: int) -> str                              # Arabic name, from quran-data.xml
def detect_quran_style(texts: list[str]) -> str    # book-level style from all OCR'd verse texts
def tag_untagged_verses(page: Page, brackets: list[str]) -> int   # wraps ﴿…﴾ spans not yet in a Quran span
def check_quran(pages: list[Page], cfg: BookConfig, *, log_path: Path | None = None) -> None
```
* Data: `pdf2epub/data/tanzil/quran-{uthmani,uthmani-plain,uthmani-min,simple,simple-plain,simple-clean}.txt`
  (`sura|aya|text`, comment lines start with `#`; keep the licence block untouched) and
  `quran-data.xml`. **Tanzil prepends the basmala to verse 1 of every sura except 1 and 9**;
  the index must treat that prefix separately so quotes of verse 1 without it still match,
  and a basmala quoted alone matches 1:1.
* Matching (spec §5.1 steps 2–4): matching-normalise (`normalize_for_match`; build indices
  from Uthmani and simple texts, optionally with `dagger_alef="alef"` and rasm, digits and
  verse-number ornaments in the OCR text ignored); word n-gram index over the whole Quran as
  one word sequence per variant, so partial verses and multi-verse quotes are found
  uniformly; rank candidate windows with rapidfuzz; refine window edges; exact after
  normalisation = verified, ≥ threshold = corrected, else flagged. Repeated passages (for
  example 55:13 and its repeats) are not errors: pick the first, list the others in
  `alternatives` and only raise `quran_ambiguous` (info) since the text is identical.
  Spans shorter than `min_words` are verified only on an exact, unambiguous match, and are
  never replaced otherwise (`quran_short`, info).
* Replacement (step 5): the chosen variant is `cfg.text.quran_style`, or when `auto`
  `detect_quran_style` over the whole book (Uthmani signs → `uthmani`; diacritics with pause
  marks → `simple`; diacritics without → `simple-plain`; none → `simple-clean`). Replace with
  exactly the quoted words of that variant (whole Tanzil words, never edited characters), keep
  the OCR text in `ocr_text`, set `ref` ("2:155", "2:155-156", "2:286-3:1"), `loc`, `score`,
  `variant`. Never change a span whose status is already `verified`/`corrected` unless its
  `ocr_text` is None and the text changed.
* Untagged verses: before matching, `tag_untagged_verses` wraps text between ornate
  brackets (either orientation) that the model left untagged, using `BlockText.wrap_range`,
  removing the brackets themselves from the text.
* Every replacement or flag appends one JSON line to `log_path` (page, block, ref, status,
  score, ocr text, new text). `quran_unmatched` flags (warning) target the block id.
* Spec §13: the wrong-verse rate must be 0 — prefer flagging to a doubtful correction.

`checks/poetry.py` (spec §5.2)
```python
def rhyme_letter(half: str) -> str | None        # الروي of a half-line, None if undeterminable
def check_poem(poem: Poem, cfg: BookConfig) -> list[Flag]
def check_poetry(pages: list[Page], cfg: BookConfig) -> None
```
* Structure: every verse of a two-column poem has both halves (`poem_missing_half`), halves of
  similar length (`poem_half_length`, ratio `cfg.checks.poem_half_ratio` on letters without
  diacritics), decorative separators read as text (`*`, `…`, `...`) inside a half.
* Rhyme: in a classical poem with ≥ 3 two-half verses, the dominant rawi of the عجز
  half-lines; a verse that breaks it is flagged (`poem_rhyme`). Handle diacritics, final
  long vowels of ishbaʿ/wasl (ا و ي ه after the rawi), tanwin alef, ة/ه. A poem continued
  from the previous page (`continued`, or a page opening with a poem and no heading after a
  page ending with a poem) is checked together with its previous part. Poems inside notes
  too.
* Meter (optional): only when `cfg.text.poem_meter` and a classifier is importable
  (`qawafi`); flags only (`poem_meter`), never fixes. Absent classifier: no-op.

`checks/hashiya.py` (spec §6.4–6.5)
```python
def extract_lemma(note: Note) -> str | None       # قوله … lemma from note text when the model gave none
def pair_markers(page: Page) -> None
def link_lemmas(pages: list[Page], i: int, cfg: BookConfig) -> None
def link_positions(page: Page) -> None
def link_continuations(pages: list[Page]) -> None
def link_notes(pages: list[Page], cfg: BookConfig) -> None
```
* `link_notes` first removes generated refs belonging to the notes it is about to relink
  (idempotent re-runs), then for each page, layer by layer from the top down (layer 1 targets
  the main text; layer L targets the text of layer L-1 notes on the same page):
  1. markers: pair refs and notes by `normalize_marker`, per page, in order; repeated
     symbols pair k-th with k-th. A note without a ref: look for the marker left as plain
     text in the target text ("(١)", "[١]", "١" right after a word) and turn it into a Ref;
     else `note_without_ref`. A ref without a note → `ref_without_note`.
  2. lemmas (قوله): lemma from `note.lemma` or `extract_lemma`; strip a trailing "إلخ"/"الخ";
     normalise lemma and target text identically (`normalize_with_map`); search this page,
     then `cfg.checks.lemma_lookback_pages` previous pages; when a lemma occurs more than once,
     choose occurrences for all lemma notes of the page jointly so anchors follow note order
     (monotone alignment, DP over candidates); exact, then fuzzy (rapidfuzz partial alignment,
     score ≥ `cfg.checks.lemma_threshold`); insert a generated Ref at the end of the matched
     words with `BlockText.insert_at`; else `lemma_not_found` / `lemma_low_score`.
  3. positions: a margin note with neither marker nor lemma links to the main-text line level
     with it, using region and main-block geometry when known (proportional estimate
     otherwise); status `guessed`, flag `position_guess` (info). No geometry at all → flagged.
  4. continuations: a note with `cont`, or the first note of a page's bottom region with no
     marker and no lemma while the previous page's last note has `cont_next` or no full
     stop, continues the previous page's last note: set `continues_from`/`continues_to`,
     `link.method="continuation"`, status `linked`; no anchor of its own. An orphan
     continuation → `continuation_orphan`.
  5. checks: numbering gaps per page (١، ٢، ٤ → `note_numbering_gap`; numbering may also run
     on from the previous page), anchors out of note order (`note_order`).
* Note kinds: `kind="editor"` from the model wins; otherwise `cfg.text.editor_markers`
  ("letters" → notes with letter markers are editor notes, "numbers", "symbols"), else
  `cfg.text.note_kind_default`.
* Every resolved link sets `Note.link` (status, method, page, block, ref, offset, score) and
  the anchor `Ref.note`.

`checks/flags.py` (spec §4 "Catching bad pages automatically")
```python
def find_repetition(text: str, min_repeats: int = 3) -> str | None
def model_word_count(page: Page) -> int
def tesseract_text(wd: WorkDir, entry_or_page: PageEntry | Page, cfg: BookConfig) -> str | None   # cached
def check_health(page: Page, cfg: BookConfig, *, tags: str | None = None,
                 tess_text: str | None = None) -> None
def finalize_status(page: Page, cfg: BookConfig) -> None
```
* `check_health` (stage `health`): `empty_output` (no text but the image has ink),
  `truncated` (`finish_reason == "length"` in `page.ocr`), `ocr_failed`, `repetition` (a line or
  an n-gram looping ≥ `repetition_min_repeats` times; also across the raw tags), `unclear_text`
  (one flag per Unclear span, targeting its block), `wordcount_mismatch` when Tesseract is
  available and the counts differ by more than `wordcount_tolerance` (skip tiny pages).
* `tesseract_text`: pytesseract with `--tessdata-dir` (`cfg.checks.tessdata_dir` or
  `default_tessdata_dir()`), language `cfg.checks.tesseract_lang`, cached in
  `pages/NNNN.tess.txt`; returns None (no flag) when Tesseract or the language data is missing.
* `finalize_status`: reviewed stays reviewed; otherwise `flagged` if any flag's severity is
  in `cfg.checks.block_on`, else `clean`.

### 3.5 `pdf2epub/assemble/` (spec §2 step 7, §5.2 page breaks, §6.4 continuations)

`assemble/pages.py`
```python
def page_label(page: Page, pages: list[Page]) -> str
def join_pages(pages: list[Page], cfg: BookConfig) -> list[tuple[Any, int]]    # (block, page index) stream
def split_chapters(stream: list[tuple[Any, int]], cfg: BookConfig) -> list[Chapter]
def assemble_book(pages: list[Page], cfg: BookConfig, *, slug: str, title: str,
                  author: str | None = None, identifier: str | None = None) -> Book
```
* Running headers and page numbers are not content. Every page contributes a `PageBreak`
  inline (index, printed label; `page_label` infers missing labels from neighbours, else uses
  the index) at the start of its first text block — inside the joined paragraph when the page
  starts mid-paragraph, inside the first verse's sadr for a joined poem. A page with no
  blocks gets its break on the next block.
* Join a paragraph across pages when the last block of page N is a paragraph and (it has
  `continues`, or page N+1's first block is a paragraph with `continued`, or it lacks terminal
  punctuation `. ! ؟ ? : … ».` and page N+1's first block is a paragraph). Insert a single
  space where needed.
* Join a poem across pages when page N+1 opens with a poem (no heading before it) and page N
  ends with a poem (spec §5.2).
* Chapters: split at headings of level ≤ `cfg.build.split_level` (None → the smallest level
  that occurs), ids `ch001`…; content before the first heading forms an untitled chapter;
  very long runs without headings are split every `cfg.build.max_chapter_pages` pages at a
  block boundary. `Chapter.title` is the heading's plain text; `Chapter.pages` the pages it covers.
* Independent regions (`independent=True`) become chapters of their own stream (named
  `frame`), in page order, after the main stream.
* `Book.digits` from `cfg.text.digits` or the majority digit style of the text;
  `Book.quran_variant` from the Quran spans; `identifier` defaults to a stable
  `urn:uuid:` derived from the slug.

`assemble/notes.py`
```python
def join_continued_notes(pages: list[Page]) -> dict[str, Note]       # merged notes by id
def chapter_notes(chapter: Chapter, notes: dict[str, Note]) -> list[Note]
```
* Joins a note continued across pages into the first part (content appended, ids of later
  parts mapped to the first), keeping its link. Collects for each chapter the notes whose
  anchor ref lies in it, in anchor order, plus their layer-2/3 notes (anchored inside those
  notes). Unlinked/flagged notes on a page are attached to the chapter holding that page,
  after the linked ones, so no note text is ever lost.

### 3.6 `pdf2epub/epub/` (spec §6.6, §7)

`epub/builder.py`
```python
def build_epub(book: Book, cfg: BookConfig, wd: WorkDir, out_path: Path) -> Path
```
`epub/render.py`: XHTML rendering of chapters, blocks and inlines (Jinja2 templates in
`epub/templates/`, CSS in `epub/css/book.css`, fonts in `epub/fonts/`).

`epub/validate.py`
```python
@dataclass
class EpubcheckReport:
    ok: bool
    errors: list[str]
    warnings: list[str]
    fatal: list[str]
    available: bool          # False when Java or the jar is missing
def run_epubcheck(epub_path: Path, jar: str | None = None) -> EpubcheckReport
```
Rules (spec §7 "EPUB rules" 1–8 are binding):
* EPUB 3, `mimetype` first and stored; `xml:lang="ar" lang="ar" dir="rtl"` on every `<html>`;
  `page-progression-direction="rtl"` on the spine; direction only via `dir`, never the CSS
  `direction`/`unicode-bidi` properties; digits as printed.
* One XHTML file per chapter; `nav.xhtml` with the table of contents (from headings), a
  `page-list` mapped to printed page numbers (`<span epub:type="pagebreak" role="doc-pagebreak"
  id="pg-NNNN" aria-label="…"/>` from `PageBreak` nodes), and landmarks; an NCX too for older
  readers; a credits page (Tanzil attribution with a link to tanzil.net — required by its
  licence — plus fonts and model credits); cover from the first page image when
  `cfg.build.cover == "first-page"`.
* Fonts: embed `Amiri` (or Noto Naskh Arabic per `cfg.build.body_font`) for the body and
  `Amiri Quran` for verses when `embed_fonts`; the book must still read correctly with the
  device font.
* Quran: `ORNATE_OPEN + text + ORNATE_CLOSE` in logical order, verse font; `quran_refs`:
  `label` (small "[البقرة: ١٥٥]" after the verse, digits per `book.digits`), `footnote` (a
  note), `none`. Quran spans still `flagged` only appear in draft builds and are marked.
* Poems: `two-column` = one table row per verse, sadr in the right cell, ajuz in the left,
  centred (tables, not flexbox/grid); `stacked` = sadr then indented ajuz, space between verses.
  Single-column poems are a centred column.
* Notes (spec §6.6): markers per `note_markers` (renumbered per chapter: author notes with
  digits in the book's digit style, editor notes with Arabic letters أ ب ج …; or printed);
  `popup` = `<a epub:type="noteref" href="#…">` + `<aside epub:type="footnote">` collected at
  the end of the chapter file with a back-link; `inline` = notes in a smaller font right after
  the block holding their anchors, markers still linked; `chapter-end` = an endnotes section
  at the end of each chapter with links both ways. Layer 2/3 notes always use chapter-end
  style. Lemma notes open with `cfg.text.lemma_template` in bold. `author_notes`/`editor_notes`
  `hide` drops those notes and their markers. Continuation notes are already merged.
* Matn spans bold with their printed parentheses; quotes keep their brackets; unclear spans
  render as plain text (draft builds mark them); figures as images from `Figure.image`
  (cropped by integration) with captions.
* Independent works (`stream != "main"`): extra sections after the main text, or a second
  EPUB (`<slug>-frame.epub`) when `cfg.build.frames == "separate-epub"`.
* Deterministic output (fixed zip timestamps, stable ids) so rebuilding the same book gives
  the same bytes.

### 3.7 `pdf2epub/review/` (spec §2 step 6, §6.5, §9 review screen)

```python
def create_app(wd: WorkDir) -> FastAPI
def serve(wd: WorkDir, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None
def save_training_record(wd: WorkDir, page: Page, original: Page | None) -> Path
```
* Local only (bind 127.0.0.1 by default). Pages list: status, flag counts, flagged first, with
  filter. Page view: the page image (with region boxes when known) beside the recognised text
  rendered from the page model; flagged words, Quran spans and note links highlighted;
  hovering a note highlights its anchor and vice versa, with a connecting line in the text panel.
* Actions (JSON API + a small vanilla-JS page in `review/static/`, no build step, no CDN):
  edit the page as tags (`serialize_page` → textarea → `parse_tags` → re-run the checks for
  that page and its neighbours → show the result); fix a note link by selecting words in the
  rendered main text (inserts a generated Ref with `link.method="manual"`, status `linked`);
  approve the page (status `reviewed`, flags kept for the record, saved to
  `pages/NNNN.reviewed.json`); revert a review.
* Every save writes a training record (`training/NNNN.json`: image path, corrected tags from
  `serialize_page`, note links, the original model tags) and appends to `logs/review.jsonl`
  with timestamps (the review-time metric).
* Safe file serving: images only from the work folder (`WorkDir.abs`), integer page indices.

### 3.8 `pdf2epub/eval/` (spec §9)

```python
def cer(ref: str, hyp: str) -> float
def wer(ref: str, hyp: str) -> float
def page_text(page: Page, *, notes: bool = False) -> str
def score_page(truth: Page, pred: Page) -> PageScores
def score_set(pairs: list[tuple[Page, Page]]) -> SetScores
def load_testset(path: Path) -> list[TestPage]
def evaluate(testset: Path, cfg: BookConfig, *, out_dir: Path, client: OcrClient | None = None) -> SetScores
def compare_report(results: dict[str, SetScores]) -> str      # markdown table, one column per model
```
* Test set folder: `pages/<id>.png` + `truth/<id>.json` (a reviewed Page) or `truth/<id>.tags`;
  optional `meta.json` (book of each page, for the "≥ 10 books" check).
* Metrics (spec §9 table): CER without and with diacritics (`normalize_for_score`), WER, Quran
  wrong-verse rate (ref differs from truth for matched spans) and auto-resolved rate, poem
  verse accuracy (both halves right — CER below a small tolerance — and in order), block-type
  accuracy (alignment of block-type sequences), note recall, note link accuracy (anchored to
  the right words: compare the normalised 3 words before each anchor), region detection
  (when truth regions exist), missed-error rate (pages with CER above a threshold or wrong
  links that were not flagged). Note text scored with the same CER, separately.
* `evaluate` runs OCR → parse → checks on each test page (no assembly) with the configured
  backend, caches predictions, writes `scores.json` and `report.md` into `out_dir`.

### 3.9 `pdf2epub/devtools/sample.py`

```python
@dataclass
class SampleBook:
    pdf: Path
    fixtures: Path        # NNNN.tags model outputs, for the fixture backend
    truth: Path           # NNNN.tags ground truth (same as fixtures unless noise is added)
    pages: int
def make_sample_book(out_dir: Path, *, scan_noise: bool = True, seed: int = 0) -> SampleBook
```
* Renders a small synthetic scanned book (6–8 pages) with PyMuPDF `insert_htmlbox` and the
  bundled Amiri fonts: headings, prose, block and inline Quran in ﴿ ﴾, a two-column poem
  continued across a page break, foot notes with markers, a قوله note, a note continued with
  =, a paragraph continued across pages, printed page numbers and a running header; then
  rasterises each page, adds mild scan noise and tilt, and writes an image-only PDF. Ground
  truth tags are written alongside so the whole pipeline runs offline with the fixture
  backend. Only real Quran text from Tanzil and well-known classical verses are used.

### 3.10 `pdf2epub/pipeline.py` and `pdf2epub/cli.py` (spec §10 commands)

Written during integration. `pdf2epub convert BOOK.pdf`, `review SLUG`, `redo SLUG --pages 37,112`,
`check SLUG`, `build SLUG`, `status SLUG`, `eval TESTSET`, `sample OUT_DIR`, `validate EPUB`.
`--layout two-column|stacked` (poem layout, as in the spec), `--notes popup|inline|chapter-end`,
`--commentary mixed|none`, `--regions auto|always|never`, `--quran-style`, `--quran-refs`,
`--editor-notes show|hide`, `--backend`, `--endpoint`, `--model`, `--work DIR`.

## 4. Conventions

* Python ≥ 3.10, type hints, `from __future__ import annotations`, docstrings on public
  functions, `logging.getLogger(__name__)` (no prints outside the CLI).
* Tests in `tests/test_<module>*.py`, pytest, no network, no GPU. Use the fixtures in
  `tests/fixtures/`; build pages with `parse_tags(Path(...).read_text(), Page(index=…))`.
  Tests that need Java/EPUBCheck or Tesseract are marked `@pytest.mark.slow` and skip cleanly
  when the tool is missing. Keep the default test run under a minute per module.
* Run tests with `/home/caf/outsider/pdf2epub/.venv/bin/python -m pytest`. Do not install
  packages; if one seems necessary, say so in your report.
* Vendored tools: `vendor/epubcheck/epubcheck.jar` (EPUBCheck 5.4.0, Java is installed),
  `vendor/tessdata/{ara,osd}.traineddata` (Tesseract binary at `/usr/sbin/tesseract`).

## 5. Rules settled by the contract review (binding)

These override anything above that they contradict.

1. **Which file a check reads and writes.** `run_checks` input for page *i* is
   `pages/NNNN.reviewed.json` when it exists, else `pages/NNNN.parsed.json` (never `NNNN.json`), so
   a re-check is a pure function of parse output and settings. Afterwards each page is saved back:
   reviewed pages to `reviewed.json`, the rest to `NNNN.json`.
2. **Reviewed pages are read-only** for every check (text, Quran spans, notes, flags). One
   exception: `link_notes` may add, move or delete *non-manual generated* refs inside a reviewed page
   when they anchor a note of a non-reviewed page. `run_checks(..., only=...)` makes the pages outside
   `only` read-only context under the same rule.
3. **Neighbours are by index.** The previous page of *p* is the page with index `p.index - 1` in the
   list; if it is absent there is no lookback, continuation, numbering run-on, poem or paragraph join,
   or label inference across the gap.
4. **No Quran span leaves `check_quran` as `unchecked`.** Short spans (< `quran_min_words`) are
   verified and replaced only when every matching location gives the same Tanzil words in the chosen
   variant; otherwise `flagged` with `quran_short` (warning). An unwrappable ﴿…﴾ raises
   `quran_unmatched`. `accepted` spans (approved in review) are never touched. With
   `quran_style="auto"` the style is resolved once per book and saved in
   `text.quran_style_detected`.
5. **Manual links are never removed or moved** by the linker; they record `NoteLink.anchor` and
   `occurrence` so they can be re-placed after a re-parse. The linker starts by deleting generated
   refs whose note does not exist or does not point back, and relinks notes whose `link.ref` does not
   resolve. It also removes a repeated "قوله: (…)" opening from lemma-note text (setting `lemma` and
   `anchor_type="lemma"`), so the builder's bold opening is never doubled.
6. **Failed OCR is never cached.** A page whose calls fail writes only `NNNN.ocr.json` (with the
   error), no `.tags`, and is retried by the next run. `force`/redo bypasses the response cache
   (`refresh=True`); `ocr.cache_salt` invalidates it. Blank pages (`PageEntry.blank`) are not sent to
   the model (`finish_reasons=["blank"]`).
7. **Word counts** for the Tesseract cross-check count every word the model read on the page (main
   text, notes, independent regions, header, page number, unclear spans); a word is a whitespace token
   with at least one letter or digit, counted the same way on both sides.
8. **Ingest indices are fixed by the first ingest** and kept in the manifest; later runs re-render
   only what they must and keep each PDF page's indices.
9. **Split pages** keep their header and page number: the layout answer carries the printed text of
   `header`/`page_number` regions (`LayoutRegion.text`), written into the merged tags as
   `<header>`/`<pagenum>`; layout `figure` regions become `<figure box>` in page units.
10. **Serialisation**: the review screen edits the `full=True` form (ids, generated anchors, geometry
    survive); training targets use the plain form.
11. **EPUB**: a ref renders as a noteref only when its note is shown in that chapter (a printed ref
    otherwise shows its sign as text, a generated one nothing); chapter titles are never empty; notes
    without an anchor are still shown, in a visible chapter-end list; `flagged`/`unchecked` Quran spans
    appear only in draft builds; `build_epub` returns the list of files written.
12. **Poems**: `Poem.single` comes only from `type="single"` or `text.poem_columns`; a missing left
    half-line in a two-column poem is a warning. A verse split across a page break is re-joined by
    assembly.
13. **Eval** reads per-book overrides from the test set's `meta.json`, runs checks per book, ranks
    models by whole-page CER, and accepts `.txt`/`.md` truths for text-only scoring.
