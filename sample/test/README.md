Test set: 100 pages of الشعر والشعراء (sample/test/part.pdf), never used for training.

reference/NNNN.txt holds each page's text from the Shamela edition of the same print (book 23785), main text, a
line `---`, then the notes, located by content (tools/build_reference.py). On 2026-10-25 runs of words at the
start or end of a page's main text that are not printed on that page (the located span ran into the
neighbouring page) were trimmed, judged by the line-by-line readings of the completeness check: those readings
only decided whether text is on the page, never its spelling.
