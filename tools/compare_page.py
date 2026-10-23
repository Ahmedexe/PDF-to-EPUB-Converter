"""Each scan beside two models' readings, with the words where they differ highlighted.

    python tools/compare_page.py FT_WORKDIR BASE_WORKDIR OUT.html [--offset 20] [--title "..."]

``--offset`` is added to page indices to show PDF page numbers (when the work folder holds an
extract of a larger PDF); ``--title`` names the book in the page's introduction.
"""
import base64
import html
import re
import sys
from pathlib import Path

import cv2
from rapidfuzz.distance import Levenshtein

from pdf2epub.arabic import normalize_for_match
from pdf2epub.workdir import WorkDir

ft, base, out = WorkDir(Path(sys.argv[1])), WorkDir(Path(sys.argv[2])), Path(sys.argv[3])
OFFSET = int(sys.argv[sys.argv.index("--offset") + 1]) if "--offset" in sys.argv else 0
BOOK = sys.argv[sys.argv.index("--title") + 1] if "--title" in sys.argv else ft.root.name
TAG = re.compile(r"(<[^>]+>)")


def words_of(tags: str) -> list[str]:
    return [w for part in TAG.split(tags) if not part.startswith("<") for w in part.split()]


def render(tags: str, marked: set[int]) -> str:
    out, k = [], 0
    for part in TAG.split(tags):
        if not part:
            continue
        if part.startswith("<"):
            out.append(convert_tag(part))
            continue
        pieces = re.split(r"(\s+)", part)
        for piece in pieces:
            if not piece or piece.isspace():
                out.append(piece)
                continue
            text = html.escape(piece)
            out.append(f'<mark>{text}</mark>' if k in marked else text)
            k += 1
    return "".join(out)


def convert_tag(tag: str) -> str:
    m = re.match(r"<(/?)([a-z0-9]+)([^>]*?)(/?)>", tag)
    if not m:
        return ""
    close, name, attrs, selfclose = m.groups()
    n = re.search(r'n="([^"]*)"', attrs)
    simple = {"p": "p", "h1": "h3", "h2": "h3", "h3": "h3", "h4": "h3", "poem": "div class=\"poem\"",
              "v": "div class=\"v\"", "s": "span class=\"s\"", "a": "span class=\"a\"",
              "quran": "span class=\"q\"", "quote": "span", "matn": "span class=\"matn\"",
              "unclear": "span class=\"unc\"", "header": "div class=\"hdr\"", "pagenum": "div class=\"pn\"",
              "hashiya": "div class=\"notes\""}
    if name == "ref":
        return f'<sup class="ref">{html.escape(n.group(1)) if n else "*"}</sup>'
    if name == "note":
        if close:
            return "</div>"
        cont = ' <span class="cont">(continued)</span>' if 'cont="prev"' in attrs else ""
        label = f'<b>({html.escape(n.group(1))})</b> ' if n else ""
        return f'<div class="note">{cont}{label}'
    if name == "sep":
        return "<hr>"
    if name == "figure":
        return '<div class="fig">[figure]</div>'
    if name in simple:
        el = simple[name]
        return f"</{el.split()[0]}>" if close else f"<{el}>"
    return ""


def image_uri(path: Path) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    scale = 1100 / max(h, w)
    img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


sections, total_diff, total_words = [], 0, 0
for i in ft.page_indices():
    if not ft.page_tags(i).exists():
        continue
    t_ft = ft.page_tags(i).read_text(encoding="utf-8")
    t_base = base.page_tags(i).read_text(encoding="utf-8") if base.page_tags(i).exists() else ""
    w_ft, w_base = words_of(t_ft), words_of(t_base)
    keep_ft = [k for k, w in enumerate(w_ft) if normalize_for_match(w)]
    keep_base = [k for k, w in enumerate(w_base) if normalize_for_match(w)]
    n_ft = [normalize_for_match(w_ft[k]) for k in keep_ft]
    n_base = [normalize_for_match(w_base[k]) for k in keep_base]
    mark_ft, mark_base = set(), set()
    for op in Levenshtein.opcodes(n_ft, n_base):
        if op.tag == "equal":
            continue
        mark_ft.update(keep_ft[k] for k in range(op.src_start, op.src_end))
        mark_base.update(keep_base[k] for k in range(op.dest_start, op.dest_end))
    total_diff += len(mark_ft) + len(mark_base)
    total_words += len(n_ft)
    page = ft.best_page(i)
    flags = sorted({f.code for f in page.flags if f.severity != "info"}) if page else []
    flag_html = "".join(f'<span class="flag">{html.escape(str(c).replace("_", " "))}</span>' for c in flags) \
        or '<span class="ok">passed every check</span>'
    banded = ' <span class="chip">re-read in bands</span>' if ft.page_regions(i).exists() else ""
    printed = page.printed_page if page and page.printed_page else "—"
    sections.append(f"""
<section class="page" id="p{i}">
  <header class="ph"><h2>PDF page {i + OFFSET} <span class="printed">printed page {html.escape(printed)}</span></h2>
    <div class="flags">{flag_html}{banded}</div></header>
  <div class="cols">
    <figure class="scan"><img loading="lazy" src="{image_uri(ft.page_image(i))}" alt="Scan of page {i + OFFSET}"></figure>
    <div class="read ft" dir="rtl" lang="ar"><div class="lbl" dir="ltr">Fine-tuned model (ft2) <span>{len(mark_ft)} words differ</span></div>{render(t_ft, mark_ft)}</div>
    <details class="read base" dir="rtl" lang="ar"><summary dir="ltr">Base model (Qwen3-VL-8B) <span>{len(mark_base)} words differ</span></summary>{render(t_base, mark_base)}</details>
  </div>
</section>""")

ready = sum(1 for i in ft.page_indices() if (p := ft.best_page(i)) and not [f for f in p.flags if f.severity != "info"])
page_html = f"""<title>{html.escape(BOOK)} Model Comparison</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>
/* Layout: one row per page — scan, fine-tuned reading, base reading folded under it. */
:root {{
  --bg: #f6f5f2; --panel: #ffffff; --ink: #1f2328; --muted: #5d6470; --line: #dcd9d2;
  --mark: #ffe08a; --mark-ink: #3a2c00; --flag: #b4471f; --ok: #2f7a4f; --accent: #2c5d8a;
  --ar: "Amiri", "Noto Naskh Arabic", "Traditional Arabic", serif;
  --ui: "IBM Plex Sans", system-ui, sans-serif;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #16181c; --panel: #1f2227; --ink: #e7e5e0; --muted: #9aa1ab; --line: #343840;
  --mark: #6b5414; --mark-ink: #fff3cf; --flag: #f08a63; --ok: #6cc391; --accent: #8db8e0; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #16181c; --panel: #1f2227; --ink: #e7e5e0; --muted: #9aa1ab; --line: #343840;
  --mark: #6b5414; --mark-ink: #fff3cf; --flag: #f08a63; --ok: #6cc391; --accent: #8db8e0; color-scheme: dark }}
body {{ background: var(--bg); color: var(--ink); font-family: var(--ui); padding-block: 24px 48px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
.intro {{ max-width: 70ch; margin: 0 auto 28px; display: grid; gap: 10px; line-height: 1.55; }}
.intro h1 {{ font-size: 1.6rem; margin: 0; text-wrap: balance; }}
.intro p {{ margin: 0; color: var(--muted); }}
.stats {{ display: flex; flex-wrap: wrap; gap: 8px 20px; font-variant-numeric: tabular-nums; }}
.stats b {{ font-size: 1.25rem; display: block; color: var(--ink); }}
.stats div {{ color: var(--muted); font-size: .85rem; }}
mark {{ background: var(--mark); color: var(--mark-ink); border-radius: 2px; padding: 0 1px; }}
.page {{ max-width: 1500px; margin: 0 auto 22px; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 14px; }}
.ph {{ display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 12px; }}
.ph h2 {{ margin: 0; font-size: 1rem; }}
.printed {{ color: var(--muted); font-weight: 400; margin-left: 8px; font-size: .9rem; }}
.flags {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.flag, .ok, .chip {{ font-size: .75rem; padding: 2px 8px; border-radius: 999px; border: 1px solid currentColor; letter-spacing: .02em; }}
.flag {{ color: var(--flag); }} .ok {{ color: var(--ok); }} .chip {{ color: var(--accent); }}
.cols {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr); gap: 16px; align-items: start; }}
.base {{ grid-column: 1 / -1; }}
@media (max-width: 860px) {{ .cols {{ grid-template-columns: minmax(0, 1fr); }} }}
.scan {{ margin: 0; }} .scan img {{ width: 100%; height: auto; border: 1px solid var(--line); background: #fff; }}
.read {{ font-family: var(--ar); font-size: 1.22rem; line-height: 1.9; min-width: 0; }}
.lbl, summary {{ font-family: var(--ui); font-size: .8rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px; cursor: default; }}
summary {{ cursor: pointer; padding: 6px 0; border-top: 1px dashed var(--line); }}
.lbl span, summary span {{ text-transform: none; letter-spacing: 0; margin-left: 8px; }}
.read p {{ margin: 0 0 .5em; }} .read h3 {{ font-size: 1.3rem; margin: .4em 0; text-align: center; }}
.hdr, .pn {{ font-family: var(--ui); font-size: .78rem; color: var(--muted); }}
.hdr::before {{ content: "header: "; }} .pn::before {{ content: "page number: "; }}
.poem {{ margin: .4em 0 .7em; }} .v {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0 1.5em; }}
.q {{ color: var(--accent); }} .q::before {{ content: "﴿"; }} .q::after {{ content: "﴾"; }}
.unc {{ text-decoration: underline wavy var(--flag); }}
sup.ref {{ color: var(--accent); font-size: .7em; }}
.notes {{ border-top: 1px solid var(--line); margin-top: .8em; padding-top: .5em; font-size: 1.02rem; }}
.note {{ margin-bottom: .3em; }} .cont {{ color: var(--muted); font-family: var(--ui); font-size: .75rem; }}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>
<div class="intro">
  <h1>Fine-tuned model on an unseen book</h1>
  <p>{len(sections)} pages of <span lang="ar" dir="rtl">{html.escape(BOOK)}</span>.
  Each page shows the scan, the fine-tuned model's reading, and the base model's reading folded underneath.
  Words where the two readings disagree are highlighted in both, so one of the two is wrong there (sometimes both).</p>
  <div class="stats">
    <div><b>{ready}/{len(sections)}</b>pages passed every check</div>
    <div><b>{total_words}</b>words read by the fine-tuned model</div>
    <div><b>{total_diff}</b>highlighted words (both readings)</div>
  </div>
</div>
{''.join(sections)}
"""
out.write_text(page_html, encoding="utf-8")
print(out, round(len(page_html) / 1e6, 1), "MB")
