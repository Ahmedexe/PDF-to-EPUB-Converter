"""Test set and scoring (spec §9)."""

from .metrics import PageScores, SetScores, cer, page_text, score_page, score_set, wer
from .run import compare_report, evaluate, load_scores, load_testset

__all__ = ["PageScores", "SetScores", "cer", "compare_report", "evaluate", "load_scores", "load_testset", "page_text", "score_page", "score_set", "wer"]
