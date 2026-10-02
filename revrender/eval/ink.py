"""The ink view of a reading: the gold and the delivered masks both cut to ink.

A revision is annotated as a region, so part of its area is blank paper.  Read on ink alone, each gold revision becomes its
ink pixels (grey below ``INK_BELOW``; a revision with no ink is dropped and counted) and each delivered mask becomes its ink
pixels (one left with none stays, as a false positive).  Which masks are delivered is decided first, on the full masks, by the
model's own frozen point: the view changes how a reading is scored, never what a model delivers.
"""
import numpy as np

INK_BELOW = 200


def ink_maps(rows):
    """One boolean ink map per validation row (the row's grey channel)."""
    return [np.asarray(row['image'][0]) < INK_BELOW for row in rows]


def restrict(masks, truth, ink):
    """(masks cut to ink, gold cut to ink without the revisions that have none, how many revisions that dropped)."""
    cut = [np.asarray(t, bool) & ink for t in truth]
    kept = [t for t in cut if t.any()]
    return [np.asarray(m, bool) & ink for m in masks], kept, len(cut)-len(kept)
