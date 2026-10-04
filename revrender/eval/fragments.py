"""Is a human-labelled fragment of a line in what a model read?

The earlier project's annotators wrote, for a revised line, the text that was struck out (``text_before``) and the text that replaced it
(``text_after``).  A model reading the line leaks the struck text when ``contains(reading, text_before)`` and keeps the rewrite when
``contains(reading, text_after)``.  Both sides go through ``data.ocr_lines.normalise`` first (LaTeX and Unicode spellings, no spaces); a
fragment counts as in the reading when it is a substring or when a window of the reading as long as the fragment matches it to
``LEVEL`` (difflib's ratio).  This is a fuzzy test of a few characters, not a character error rate.
"""
from difflib import SequenceMatcher

from ..data.ocr_lines import normalise

MIN_LENGTH = 2             # a fragment shorter than this (normalised) is too short to look for
LEVEL = .8                 # a window of the reading this similar to the fragment is the fragment


def similarity(a, b):
    """difflib's ratio of two readings after ``normalise``."""
    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def contains(reading, fragment, level=LEVEL):
    """True or False: the fragment is in the reading; None when it cannot be told (the fragment is too short or the reading empty)."""
    text, piece = normalise(reading), normalise(fragment)
    if len(piece) < MIN_LENGTH or not text:
        return None
    if piece in text:
        return True
    width = len(piece)
    return any(SequenceMatcher(None, text[i:i+width], piece).ratio() >= level for i in range(max(1, len(text)-width+1)))
