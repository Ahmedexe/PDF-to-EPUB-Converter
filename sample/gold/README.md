Gold test set: 45 pages from five books (35 scored: pages 11-20 are excluded) and print styles, none used for training. Each page is
checked by a person against the scan in the review screen (`pdf2epub review gold`) and approved;
the approved pages (work/gold/pages/NNNN.reviewed.json) are the truth every model is scored on.
sources.json maps each gold page to its source PDF and page.

| Pages | Book | Style |
|---|---|---|
| 1-10 | الشعر والشعراء (test set) | letterpress, vowelled verse |
| 11-20 | عيون الأخبار vol. 2 (test2) | computer typesetting; **excluded** (two sets of foot notes per page, the second set is regularly skipped; not representative): no need to correct, not scored |
| 21-30 | التاريخ الإسلامي - العهد الأموي | modern typesetting |
| 31-40 | البيان والتبيين vol. 1 | Hārūn's letterpress |
| 41-45 | الأربعون النووية (held out) | digital, fully vowelled |
