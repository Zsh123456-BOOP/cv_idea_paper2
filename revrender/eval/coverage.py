"""How much of the struck text, of the rewrite and of the valid text a model's reading of a revised line holds.

The earlier project's annotators typed, for a revised line, the whole line as it read before the strike (``text_before``) and the whole
line as it reads after it (``text_after``); most of the two is the same.  So the struck text is what ``before`` has and ``after`` has
not, the rewrite is what ``after`` has and ``before`` has not, and the valid text is what both have (``split``, difflib's blocks on the
``data.ocr_lines.normalise`` spelling).  A reading holds a character of the labelled text when a minimal edit alignment of the reading
to that text keeps it (``alignment``), so a character is held in order, not merely somewhere:

  leak     share of the struck characters the reading holds, aligned to ``before``: the lower the better
  rewrite  share of the rewritten characters the reading holds, aligned to ``after``
  valid    share of the valid characters the reading holds, aligned to ``after``: the higher the better (blanking harms it)

A share is None when the line has no such characters.  Reading exactly ``after`` scores leak 0.01, rewrite 1, valid 1; reading exactly
``before`` scores leak 1; an empty reading scores 0 everywhere; the reading of another line holds about 5% of anything (the characters
digits and operators share), which is the level below which a leak cannot be told from chance.
"""
import re
from difflib import SequenceMatcher

from ..data.ocr_lines import normalise


def prepare(text):
    """The labelled or read text in the spelling the shares are counted on (an annotator's leftover ``OCR：`` prefix dropped)."""
    return normalise(re.sub(r'^\s*OCR\s*[:：]', '', text or ''))


def similarity(a, b):
    """difflib's ratio of two readings after ``normalise``."""
    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def alignment(x, y):
    """The positions of x and of y that a minimal edit alignment of x to y keeps (equal characters); ties go to the diagonal."""
    n, m = len(x), len(y)
    cost = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        cost[i][0] = i
    for j in range(m+1):
        cost[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost[i][j] = min(cost[i-1][j]+1, cost[i][j-1]+1, cost[i-1][j-1]+(x[i-1] != y[j-1]))
    i, j, kept_x, kept_y = n, m, set(), set()
    while i and j:
        if x[i-1] == y[j-1] and cost[i][j] == cost[i-1][j-1]:
            kept_x.add(i-1)
            kept_y.add(j-1)
            i, j = i-1, j-1
        elif cost[i][j] == cost[i-1][j-1]+1:
            i, j = i-1, j-1
        elif cost[i][j] == cost[i-1][j]+1:
            i -= 1
        else:
            j -= 1
    return kept_x, kept_y


def split(before, after):
    """(struck, rewritten, valid): positions in ``before`` the strike removed, in ``after`` the rewrite added, in ``after`` kept from before.
    Both texts are already ``prepare``d."""
    struck, rewritten, valid = set(), set(), set()
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag == 'equal':
            valid.update(range(j1, j2))
        else:
            struck.update(range(i1, i2))
            rewritten.update(range(j1, j2))
    return struck, rewritten, valid


def coverage(reading, before, after):
    """{'leak', 'rewrite', 'valid'}: the shares (or None) of one reading of a line labelled ``before`` -> ``after``."""
    r, b, a = prepare(reading), prepare(before), prepare(after)
    struck, rewritten, valid = split(b, a)
    held_b, held_a = alignment(r, b)[1], alignment(r, a)[1]
    share = lambda positions, held: len(positions & held)/len(positions) if positions else None
    return {'leak': share(struck, held_b), 'rewrite': share(rewritten, held_a), 'valid': share(valid, held_a)}
