# Page tag format (version 1)

The OCR model writes one page (or one region of a page) as light tagged text.
`pdf2epub.parse.parse_tags` turns it into the page model; `serialize_page`
writes it back. Training targets use exactly this format.

## Block tags (top level)

| Tag | Meaning |
| --- | --- |
| `<h1>` … `<h4>` | Headings; levels feed the table of contents |
| `<p>` | Paragraph. `cont="prev"` when the page's first paragraph continues the previous page (first line not indented, starts mid-sentence); `cont="next"` when the last paragraph runs onto the next page; `cont="both"` |
| `<quran>` | A displayed Quran passage, without the ﴿ ﴾ brackets |
| `<poem>` | Poem: `<v>` verses, each `<s>` (right-hand half-line, صدر) then `<a>` (left-hand half-line, عجز). A single-column verse has only `<s>`; `<poem type="single">` for a single-column poem |
| `<sep/>` | Decorative section divider (for example * * *) |
| `<figure box="x0 y0 x1 y1">caption</figure>` | Picture or diagram; box in 0–1000 page units |
| `<header>` | Running header or footer (dropped from the text, kept as metadata) |
| `<pagenum>` | Printed page number (dropped from the text, kept as metadata) |
| `<hashiya pos="…">` | A حاشية region holding `<note>` elements. `pos` is `bottom`, `margin-right`, `margin-left`, `top` or `frame`. Optional `layer="2"` for notes on the commentary, `independent="1"` for a separate work printed in a frame (then it holds block tags, not notes) |

## Inline tags (inside blocks and notes)

| Tag | Meaning |
| --- | --- |
| `<ref n="١"/>` | A note marker printed in the text, at the exact spot it is printed; `n` is the sign as printed without brackets |
| `<quran>` | A Quran passage inside running text |
| `<quote>` | Other bracketed quotation (hadith in « » or (( ))); the brackets stay inside, as printed |
| `<matn>` | Main-text words woven into a commentary (شرح ممزوج); printed parentheses stay inside |
| `<unclear>` | Words that cannot be read with confidence; the best reading inside, or empty |

## Notes

| Form | Tag |
| --- | --- |
| Numbered or symbol note | `<note n="١">text</note>` |
| قوله (lemma) note | `<note lemma="quoted main-text words">explanation</note>`; the opening "قوله: (…)" is not repeated in the text |
| Margin note with neither | `<note>text</note>` |
| Continues the previous page's note | `<note cont="prev">` or text starting with `=` |
| Continues on the next page | `<note cont="next">` or text ending with `=` |
| Editor's note (حواشي المحقق) | add `kind="editor"` |

Notes may contain `<quran>`, `<quote>`, `<poem>` and `<ref>` (for layered commentary).

## Example

```xml
<h2>باب الصبر</h2>
<p>قال الله تعالى:</p>
<quran>إِنَّمَا يُوَفَّى الصَّابِرُونَ أَجْرَهُم بِغَيْرِ حِسَابٍ</quran>
<p>وقال صلى الله عليه وسلم: <quote>«والصبر ضياء»</quote><ref n="١"/></p>
<poem><v><s>على قدر أهل العزم تأتي العزائم</s><a>وتأتي على قدر الكرام المكارم</a></v></poem>
<hashiya pos="bottom">
  <note n="١">رواه مسلم في كتاب الطهارة.</note>
  <note lemma="والصبر ضياء">أي يستضيء به صاحبه في ظلمات الشدائد.</note>
</hashiya>
<pagenum>٣٧</pagenum>
```
