"""How much of the struck text, of the rewrite and of the valid text a model's reading of a revised line holds.

The earlier project's annotators typed, for a revised line, the whole line as it read before the strike (``text_before``) and the whole
line as it reads after it (``text_after``); most of the two is the same.  So the struck text is what ``before`` has and ``after`` has
not, the rewrite is what ``after`` has and ``before`` has not, and the valid text is what both have (``split``, difflib's blocks on the
``data.ocr_lines.normalise`` spelling).  A reading holds a character of the labelled text when a minimal edit alignment of the reading
to that text keeps it (``alignment``), so a character is held in order, not merely somewhere:

  leak     share of the struck characters the reading holds, aligned to ``before``: the lower the better
  rewrite  share of the rewritten characters the reading holds, aligned to ``after``
  valid    share of the valid characters the reading holds, aligned to ``after``: the higher the better (blanking harms it)

A share is None when the line has no such characters.  Equal-cost alignments differ in which copy of a repeated character they keep
(a struck "85" in "8585" and the valid "85" are the same characters), so a count is made twice, the alignment read from the end of the
line and from its start: ``coverage`` gives the first, ``counts`` also the characters held in both (strict) and in either (lenient)
-- a share is only as precise as that spread (about 3 points on a mean).  Reading exactly ``after`` scores leak about 0.01 (three lines
whose labels repeat a stutter), rewrite 1, valid 1; reading exactly ``before`` scores leak 1; an empty reading scores 0 everywhere; a
reading of another line holds 4% of the struck characters when it is as long as the labelled text and about 10% when it is as long as a
raw reading (struck text included): the level below which a leak cannot be told from chance (``eval.leakage.references`` measures it).
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


def counts(reading, before, after):
    """{'leak', 'rewrite', 'valid'}: for each, the number of labelled characters, how many the reading holds (alignment from the end of the
    line), and how many under both alignments (``strict``) and under either (``lenient``).  Empty for a name when the line has none."""
    r, b, a = prepare(reading), prepare(before), prepare(after)
    struck, rewritten, valid = split(b, a)
    held_b, held_a = alignment(r, b)[1], alignment(r, a)[1]
    mirrored_b = {len(b)-1-i for i in alignment(r[::-1], b[::-1])[1]}
    mirrored_a = {len(a)-1-i for i in alignment(r[::-1], a[::-1])[1]}
    out = {}
    for name, positions, held, mirrored in (('leak', struck, held_b, mirrored_b), ('rewrite', rewritten, held_a, mirrored_a), ('valid', valid, held_a, mirrored_a)):
        if positions:
            out[name] = {'total': len(positions), 'held': len(positions & held), 'strict': len(positions & held & mirrored), 'lenient': len(positions & (held | mirrored))}
    return out


def coverage(reading, before, after):
    """{'leak', 'rewrite', 'valid'}: the shares (or None) of one reading of a line labelled ``before`` -> ``after``."""
    got = counts(reading, before, after)
    return {name: got[name]['held']/got[name]['total'] if name in got else None for name in ('leak', 'rewrite', 'valid')}
