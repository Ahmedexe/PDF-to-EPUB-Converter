"""Join pages into a book: continued paragraphs, poems and notes; chapters (spec §2 step 7)."""

from .notes import chapter_notes, join_continued_notes
from .pages import assemble_book, join_pages, page_label, split_chapters

__all__ = ["assemble_book", "chapter_notes", "join_continued_notes", "join_pages", "page_label", "split_chapters"]
