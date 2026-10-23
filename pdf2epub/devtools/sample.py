"""A small synthetic scanned Arabic book with ground truth, for demos and end-to-end tests.

Each page is written once as tags (the ground truth, which is also what a
perfect model would return), rendered with the bundled Amiri fonts by PyMuPDF,
rasterised with mild scan noise and tilt, and saved as an image-only PDF.
The fixture OCR backend then replays the tags, so the whole pipeline runs
offline without a GPU::

    pdf2epub sample demo/
    pdf2epub convert demo/sample.pdf --backend fixture --fixtures demo/fixtures

Only Tanzil Quran text and well-known classical verses are used.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np
import pymupdf

from ..arabic import ORNATE_CLOSE, ORNATE_OPEN
from ..model import (
    Heading,
    Matn,
    Page,
    PageBreak,
    Paragraph,
    Poem,
    Quote,
    Quran,
    QuranBlock,
    Ref,
    Separator,
    Unclear,
)
from ..parse import parse_tags

FONTS = Path(__file__).resolve().parent.parent / "epub" / "fonts"

PAGES: list[str] = [
    # 1
    """<header>كتاب الصبر</header>
<h1>كتاب الصبر</h1>
<p>الحمد لله الذي جعل الصبر ضياء للمؤمنين، وأمر به عباده في كتابه فقال: <quran>إِنَّمَا يُوَفَّى الصَّابِرُونَ أَجْرَهُم بِغَيْرِ حِسَابٍ</quran>، والصلاة والسلام على نبينا محمد وعلى آله وصحبه أجمعين<ref n="١"/>.</p>
<p>أما بعد، فهذه رسالة لطيفة في فضل الصبر وأقسامه، جمعتها من كلام أهل العلم، ورتبتها على أبواب، وسميتها: تحفة الصابرين.</p>
<hashiya pos="bottom">
<note n="١">الصحب جمع صاحب، والمراد بهم من لقي النبي صلى الله عليه وسلم مؤمنا به ومات على ذلك.</note>
</hashiya>
<pagenum>١</pagenum>""",
    # 2
    """<header>باب فضل الصبر</header>
<h2>باب فضل الصبر</h2>
<p>قال الله تعالى:</p>
<quran>وَلَنَبْلُوَنَّكُم بِشَيْءٍ مِّنَ الْخَوْفِ وَالْجُوعِ وَنَقْصٍ مِّنَ الْأَمْوَالِ وَالْأَنفُسِ وَالثَّمَرَاتِ وَبَشِّرِ الصَّابِرِينَ</quran>
<p>وقال النبي صلى الله عليه وسلم: <quote>«والصبر ضياء»</quote><ref n="١"/>، أي نور يستضيء به صاحبه في ظلمات الشدائد، فلا يزال على الحق مهتديا، وقد قيل: إن الصبر مفتاح الفرج، وإن الإمامة في الدين تنال بالصبر واليقين، وهذا</p>
<hashiya pos="bottom">
<note n="١">رواه مسلم في كتاب الطهارة من حديث أبي مالك الأشعري رضي الله عنه.</note>
<note lemma="مفتاح الفرج">أي سبب لحصوله، كما يفتح الباب بالمفتاح =</note>
</hashiya>
<pagenum>٢</pagenum>""",
    # 3
    """<header>باب فضل الصبر</header>
<p cont="prev">معنى قول بعض السلف: الصبر من الإيمان بمنزلة الرأس من الجسد، وأنشد بعضهم:</p>
<poem><v><s>على قدر أهل العزم تأتي العزائم</s><a>وتأتي على قدر الكرام المكارم</a></v><v><s>وتعظم في عين الصغير صغارها</s><a>وتصغر في عين العظيم العظائم</a></v></poem>
<hashiya pos="bottom">
<note cont="prev">ولا يحصل الفرج إلا بعد الصبر على الشدة.</note>
</hashiya>
<pagenum>٣</pagenum>""",
    # 4
    """<header>باب أقسام الصبر</header>
<poem><v><s>يكلف سيف الدولة الجيش همه</s><a>وقد عجزت عنه الجيوش الخضارم</a></v><v><s>ويطلب عند الناس ما عند نفسه</s><a>وذلك ما لا تدعيه الضراغم</a></v></poem>
<h2>باب أقسام الصبر</h2>
<p>والصبر ثلاثة أقسام: صبر على طاعة الله، وصبر عن معصية الله، وصبر على أقدار الله المؤلمة<ref n="١"/>، قال تعالى: <quran>وَاصْبِرْ وَمَا صَبْرُكَ إِلَّا بِاللَّهِ</quran>.</p>
<sep text="* * *"/>
<p>وأعلاها الصبر على الطاعة، لأنها مقصودة لذاتها.</p>
<hashiya pos="bottom">
<note n="١">ذكر هذا التقسيم ابن القيم في عدة الصابرين.</note>
<note lemma="مقصودة لذاتها">أي أن الطاعة هي المطلوب الأول من العبد.</note>
</hashiya>
<pagenum>٤</pagenum>""",
    # 5
    """<header>خاتمة</header>
<h2>خاتمة</h2>
<p>نسأل الله أن يجعلنا من الصابرين الشاكرين، وأن يختم لنا بخير، فقد قال تعالى: <quran>وَالْعَصْرِ إِنَّ الْإِنسَانَ لَفِي خُسْرٍ إِلَّا الَّذِينَ آمَنُوا وَعَمِلُوا الصَّالِحَاتِ وَتَوَاصَوْا بِالْحَقِّ وَتَوَاصَوْا بِالصَّبْرِ</quran>.</p>
<p>وصلى الله على نبينا محمد وعلى آله وصحبه وسلم.</p>
<pagenum>٥</pagenum>""",
]


@dataclass
class SampleBook:
    pdf: Path
    fixtures: Path  # NNNN.tags model outputs for the fixture backend
    truth: Path  # NNNN.tags ground truth
    pages: int


def _inline_html(nodes: list[object]) -> str:
    out = []
    for node in nodes:
        if isinstance(node, str):
            out.append(escape(node))
        elif isinstance(node, Ref):
            if node.n:
                out.append(f'<span class="m">({escape(node.n)})</span>')
        elif isinstance(node, Quran):
            out.append(f'<span class="q">{ORNATE_OPEN}{escape(node.text)}{ORNATE_CLOSE}</span>')
        elif isinstance(node, (Quote, Matn)):
            inner = _inline_html(node.c)
            out.append(f"<b>{inner}</b>" if isinstance(node, Matn) else inner)
        elif isinstance(node, Unclear):
            out.append(escape(node.text))
        elif isinstance(node, PageBreak):
            continue
    return "".join(out)


def _block_html(block: object) -> str:
    if isinstance(block, Heading):
        return f"<h{min(block.level + 1, 4)}>{_inline_html(block.c)}</h{min(block.level + 1, 4)}>"
    if isinstance(block, Paragraph):
        indent = "" if block.continued else ' class="ind"'
        return f"<p{indent}>{_inline_html(block.c)}</p>"
    if isinstance(block, QuranBlock):
        return f'<p class="qb">{_inline_html(block.c)}</p>'
    if isinstance(block, Poem):
        rows = "".join(f"<tr><td>{_inline_html(v.sadr)}</td><td>{_inline_html(v.ajuz or [])}</td></tr>"
                       for v in block.verses)
        return f'<table class="poem">{rows}</table>'
    if isinstance(block, Separator):
        return f'<p class="sep">{escape(block.text or "* * *")}</p>'
    return ""


CSS = """
@font-face {font-family: amiri; src: url(Amiri-Regular.ttf);}
@font-face {font-family: amiri; src: url(Amiri-Bold.ttf); font-weight: bold;}
@font-face {font-family: amiriq; src: url(AmiriQuran.ttf);}
* {font-family: amiri; direction: rtl;}
body {font-size: 12.5pt; line-height: 1.5;}
p {text-align: justify; margin: 0 0 4pt 0;}
p.ind {text-indent: 14pt;}
p.qb {text-align: center; font-family: amiriq;}
p.sep {text-align: center;}
.q {font-family: amiriq;}
.m {font-size: 9pt;}
h2 {text-align: center; font-size: 17pt; margin: 4pt 0;}
h3 {text-align: center; font-size: 14pt; margin: 4pt 0;}
table.poem {width: 100%; margin: 4pt 0;}
table.poem td {text-align: center; width: 50%;}
.notes {font-size: 10pt;}
.hdr {text-align: center; font-size: 10pt;}
"""


def _note_html(note: object) -> str:
    from ..model import Note

    assert isinstance(note, Note)
    body = "".join(_inline_html(b.c) if isinstance(b, Paragraph) else _block_html(b) for b in note.blocks)
    if note.cont:
        body = "= " + body
    if note.cont_next:
        body = body + " ="
    if note.lemma:
        body = f"<b>قوله: ({escape(note.lemma)})</b> " + body
    if note.marker:
        body = f"({escape(note.marker)}) " + body
    return f"<p>{body}</p>"


def render_page(page: Page, archive: pymupdf.Archive, doc: pymupdf.Document) -> None:
    w, h = 420, 595
    pdf_page = doc.new_page(width=w, height=h)
    margin = 34
    if page.running_header:
        pdf_page.insert_htmlbox(pymupdf.Rect(margin, 18, w - margin, 38),
                                f'<p class="hdr">{escape(page.running_header)}</p>', css=CSS, archive=archive)
        pdf_page.draw_line((margin, 40), (w - margin, 40), width=0.5)
    notes = [n for r in page.hashiya for n in r.notes]
    notes_top = h - 40
    if notes:
        notes_html = '<div class="notes">' + "".join(_note_html(n) for n in notes) + "</div>"
        # Measure the notes, then place them at the foot under a short rule (right side, RTL).
        scratch = pymupdf.open()
        probe = scratch.new_page(width=w, height=h)
        spare, _scale = probe.insert_htmlbox(pymupdf.Rect(margin, 0, w - margin, 400), notes_html, css=CSS,
                                             archive=archive)
        scratch.close()
        needed = 400 - spare
        notes_top = h - 42 - needed
        pdf_page.insert_htmlbox(pymupdf.Rect(margin, notes_top, w - margin, h - 40), notes_html, css=CSS,
                                archive=archive)
        pdf_page.draw_line((w - margin - 110, notes_top - 4), (w - margin, notes_top - 4), width=0.6)
    body = "".join(_block_html(b) for b in page.blocks)
    pdf_page.insert_htmlbox(pymupdf.Rect(margin, 46, w - margin, notes_top - 10), body, css=CSS, archive=archive)
    if page.printed_page:
        pdf_page.insert_htmlbox(pymupdf.Rect(w / 2 - 30, h - 34, w / 2 + 30, h - 14),
                                f'<p class="hdr">{escape(page.printed_page)}</p>', css=CSS, archive=archive)


def _scan(pix: pymupdf.Pixmap, rng: random.Random, noise: bool) -> bytes:
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, 0].copy()
    if noise:
        angle = rng.uniform(-0.6, 0.6)
        m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=255)
        img = cv2.GaussianBlur(img, (3, 3), 0.6)
        nrng = np.random.default_rng(rng.randrange(1 << 30))
        img = np.clip(img.astype(np.int16) + nrng.normal(0, 9, img.shape), 0, 255).astype(np.uint8)
        img = np.clip(img.astype(np.int16) - 12, 0, 255).astype(np.uint8)  # slightly grey paper
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def make_sample_book(out_dir: Path, *, scan_noise: bool = True, seed: int = 0, dpi: int = 200) -> SampleBook:
    """Write ``sample.pdf``, ``fixtures/NNNN.tags`` and ``truth/NNNN.tags`` into ``out_dir``."""
    out_dir = Path(out_dir)
    fixtures, truth = out_dir / "fixtures", out_dir / "truth"
    fixtures.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    archive = pymupdf.Archive(str(FONTS))
    vector = pymupdf.open()
    for i, tags in enumerate(PAGES, 1):
        page = parse_tags(tags, Page(index=i)).page
        if page.flags:
            raise ValueError(f"sample page {i} does not parse cleanly: {page.flags}")
        render_page(page, archive, vector)
        (fixtures / f"{i:04d}.tags").write_text(tags + "\n", encoding="utf-8")
        (truth / f"{i:04d}.tags").write_text(tags + "\n", encoding="utf-8")
    scanned = pymupdf.open()
    for page in vector:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        png = _scan(pix, rng, scan_noise)
        new = scanned.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=png)
    pdf = out_dir / "sample.pdf"
    scanned.save(str(pdf), deflate=True)
    return SampleBook(pdf=pdf, fixtures=fixtures, truth=truth, pages=len(PAGES))
