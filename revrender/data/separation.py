"""What RevSep (``models.separation``) learns from a restorer pair, and where it is told the revision is.

    python -m revrender.data.separation hints --native SET/validation_native.json --detector DET --output-dir OUT
    python -m revrender.data.separation strokes --pairs V4 --gold-bank BANK --human-validation VAL.json --same-student PAIRS --output-dir OUT

Labels are per pixel 1, 0 or ``IGNORE`` (no loss), for its two heads: erase (a strike pixel on paper, made white) and cross (a strike
pixel over writing, redrawn).  A synthetic pair (``data.restoration_pairs``) knows both: its strike is what it darkened by more than
``REMOVED``, a crossing where the clean host has writing (grey < 200), to erase elsewhere; its faint edges (darkened by more than
``FAINT``, no more than ``REMOVED``) count for neither.  A human region knows what to erase -- exactly the pixels its target made
white -- but not which of the ink it kept lies under the strike: cross is 0 on paper and ``IGNORE`` on ink.

The hint is where the first paper's detector put the revision: its instance masks tightened to ink (``tighten``), cut to a crop
(``detector_hint``).  On the 201 human training regions they hold 85% of the annotators' erased pixels at 53% precision; 83% of their
other pixels lie within four pixels of the strike (the writing it touches) and what they miss is mostly specks of one to four pixels.
A synthetic pair has no detection, so its hint is simulated so (``simulated_hint``): its strike grown over the ink by one to
``reach`` pixels, less random 4-pixel blocks; a human region's detector hint is grown or shrunk by a pixel (``jittered_hint``).  With
reach 5 and drop .3, on 800 synthetic training pairs, it holds 85% of the strike at 69% precision against the strike on paper: cleaner
than the detector (whose writing is denser round the strike, and still 59% at reach 10), so the real regions teach the rest.

The stroke version (``models.separation`` with a stroke head) also learns the strike as an object: its centreline and half-width, and
the mask they draw, crossings included.  A synthetic pair knows it exactly (``strokes``: its strike's opacity redrawn as
``restoration_pairs.build`` drew it, from the record's seed, piece and placement, and checked against the pair's own pixels; the mask
is opacity at least one half, the centreline its skeleton, the half-width the distance to its edge there).  A human region only
knows what the annotator erased (``stroke_labels``): that mask, its skeleton and widths, and the kept ink next to it -- where the
strike may run under writing -- counted for none of them.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

from .. import pages
from ..util import digest, make_output_dir, require, write_manifest
from .restoration_pairs import REMOVED

IGNORE = 255        # a label no loss counts
UNDER = 3           # pixels: kept ink this close to a human region's erased strike may lie under it (stroke labels unknown there)
WIDTH_STEP = 4      # the stored half-width is in quarter pixels
FAINT = 8           # darkened by more than this but no more than REMOVED: a strike's faint edge, labelled for neither head
HUMAN = 'human_confirmed'   # the source of a human region's tiles in a pair set (``restoration_pairs.human_pairs``)


def pair_labels(x, y, human):
    """(erase, cross) labels (uint8) of a pair: the struck tile ``x`` and its target ``y``; ``human``: a human region's tile."""
    darkened = y.astype(np.int16) - x
    if human:
        return (darkened > 0).astype(np.uint8), np.where(x < 200, IGNORE, 0).astype(np.uint8)
    strike = darkened > REMOVED; faint = (darkened > FAINT) & ~strike; writing = y < 200
    return (np.where(faint, IGNORE, strike & ~writing).astype(np.uint8),
            np.where(faint, IGNORE, strike & writing).astype(np.uint8))


def simulated_hint(x, y, rng, reach=5, drop=.3):
    """A detector-like hint of a synthetic pair: its strike grown by 1..``reach`` pixels over the ink of ``x``, less 4-pixel blocks
    (each dropped with a probability drawn up to ``drop``)."""
    strike = (y.astype(np.int16) - x) > REMOVED
    if not strike.any(): return np.zeros(x.shape, np.uint8)
    grown = ndimage.binary_dilation(strike, iterations=int(rng.integers(1, reach+1)))
    h, w = x.shape; keep = rng.random(((h+3)//4, (w+3)//4)) >= rng.uniform(0, drop)
    return (grown & keep.repeat(4, 0).repeat(4, 1)[:h, :w] & (x < 200)).astype(np.uint8)


def jittered_hint(hint, x, rng):
    """A detector hint grown by a pixel, shrunk by one or left (a third each), on the ink of ``x``."""
    step = int(rng.integers(-1, 2)); h = hint.astype(bool)
    if step > 0: h = ndimage.binary_dilation(h)
    elif step < 0: h = ndimage.binary_erosion(h)
    return (h & (x < 200)).astype(np.uint8)


def detector_hint(instances, window, crop):
    """The detector's masks (``pages.read_output``'s instances of a page) inside ``window`` (l, t, r, b on the page), on the ink of
    ``crop`` (that window of the page, as the restorer gets it)."""
    l, t, r, b = window; hint = np.zeros((b-t, r-l), bool)
    for i in instances:
        L, T, R, B = i['box']; x0, y0, x1, y1 = max(l, L), max(t, T), min(r, R), min(b, B)
        if x1 > x0 and y1 > y0: hint[y0-t:y1-t, x0-l:x1-l] |= i['mask'][y0-T:y1-T, x0-L:x1-L]
    return (hint & (crop < 200)).astype(np.uint8)


def stroke_geometry(alpha):
    """(mask, centreline, half-width in pixels on the centreline) of a strike's opacity."""
    from skimage.morphology import skeletonize
    mask = alpha >= .5; centre = skeletonize(mask) if mask.any() else mask
    return mask.astype(np.uint8), centre.astype(np.uint8), np.where(centre, ndimage.distance_transform_edt(mask), 0.).astype(np.float32)


def stroke_labels(x, y, human, truth=None):
    """(mask, centreline, half-width) labels of a pair: a synthetic one's from its stored geometry (``truth``: the three arrays of
    ``strokes``), a human tile's from what its target whitened, with the kept ink within ``UNDER`` of it ``IGNORE`` in mask and line."""
    if not human:
        mask, centre, width = truth
        return mask.astype(np.uint8), centre.astype(np.uint8), width.astype(np.float32)/WIDTH_STEP
    mask, centre, width = stroke_geometry(((y.astype(np.int16)-x) > 0).astype(np.float32))
    unknown = (x < 200) & ~mask.astype(bool) & ndimage.binary_dilation(mask.astype(bool), iterations=UNDER)
    return np.where(unknown, IGNORE, mask).astype(np.uint8), np.where(unknown, IGNORE, centre).astype(np.uint8), width


def pair_alpha(record, bank):
    """The opacity of a pair's gold strike as ``restoration_pairs.build`` drew it (``bank``: the gold pieces it chose among)."""
    from PIL import Image, ImageDraw
    from .restoration_pairs import SIZE, material_alpha
    if record['clean']: return np.zeros((SIZE, SIZE), np.float32)
    if record['split'] == 'val':
        lay = Image.new('L', (SIZE, SIZE), 0); l, t, w, h = record['placement']
        ImageDraw.Draw(lay).line([(l, t+h//2), (l+w//2, t), (l+w-1, t+h-1)], fill=255, width=max(2, h//4))
        return np.asarray(lay, np.float32)/255
    alpha, piece = material_alpha(bank, random.Random(record['seed']), record['placement'])
    require(piece == record['pieces']['gold_pixels'], f"{record['id']}: not the piece the pair was drawn with")
    return alpha


def strokes(pairs, gold_bank, human_validation, same_student, destination):
    """The strike geometry of every pair of a pair set (its gold arm): stroke.npy (mask), centre.npy (centreline), width.npy (half-width
    in quarter pixels on the centreline), row for row with its records; each redrawn strike must give the pair's own pixels."""
    from .paste_strikes import Bank
    from .prior_annotations import twins
    from .restoration_pairs import canonical, draw_pair
    other = twins(same_student); records = json.loads((pairs/'records.json').read_text())
    x = np.load(pairs/'gold_pixels.npy', mmap_mode='r'); y = np.load(pairs/'targets.npy', mmap_mode='r')
    excluded = {canonical(r['group_uuid'], other) for r in json.loads(human_validation.read_text())} | {r['group_uuid'] for r in records if r['split'] == 'val'}
    bank = Bank(gold_bank); bank.pieces = [p for p in bank.pieces if canonical(p['writer'], other) not in excluded]
    out = make_output_dir(destination); n = len(records); shape = x.shape
    arrays = {k: np.lib.format.open_memmap(out/f'{k}.npy', mode='w+', dtype=np.uint8, shape=shape) for k in ('stroke', 'centre', 'width')}
    for i, r in enumerate(records):
        alpha = pair_alpha(r, bank); clean = np.asarray(y[i])
        if not r['clean']:
            image, _ = draw_pair(clean, alpha, float(np.percentile(clean[clean < 200], 30)))
            require(np.array_equal(image, x[i]), f"{r['id']}: the redrawn strike does not give the pair's pixels")
        mask, centre, width = stroke_geometry(alpha)
        arrays['stroke'][i] = mask; arrays['centre'][i] = centre; arrays['width'][i] = np.clip(np.rint(width*WIDTH_STEP), 0, 255)
    for a in arrays.values(): a.flush()
    inputs = {str(p): digest(p) for p in [pairs/'records.json', pairs/'gold_pixels.npy', pairs/'targets.npy', human_validation, same_student,
                                          gold_bank/'run_manifest.json']}
    write_manifest(out, config={'pairs': str(pairs), 'gold_bank': str(gold_bank), 'human_validation': str(human_validation),
                                'same_student': str(same_student)}, inputs=inputs, pairs=n, width_step=WIDTH_STEP,
                   strikes=sum(not r['clean'] for r in records), files={f'{k}.npy': digest(out/f'{k}.npy') for k in arrays})


def native_hints(native, detector, destination):
    """The detector hints of a native set's crops (``data.review_set``'s ``validation_native.json``: page and box of each), hint_<k>
    on crop k: a hinted restorer's human validation (``baselines.restoration --human-validation-hints``)."""
    rows = json.loads(native.read_text()); arrays = np.load(native.with_suffix('.npz')); hints = {}; read = {}
    for k, r in enumerate(rows):
        if r['page'] not in read: read[r['page']] = pages.read_output(detector, r['page'])[1]
        hints[f'hint_{k}'] = detector_hint(read[r['page']], r['box'], arrays[f'image_{k}'])
    out = make_output_dir(destination); np.savez_compressed(out/'hints.npz', **hints)
    inputs = {str(p): digest(p) for p in [native, native.with_suffix('.npz'), detector/'run_manifest.json']}
    write_manifest(out, config={'native': str(native), 'detector': str(detector)}, inputs=inputs, crops=len(rows),
                   empty=sum(not h.any() for h in hints.values()), files={'hints.npz': digest(out/'hints.npz')})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    h = sub.add_parser('hints', help=native_hints.__doc__)
    for name in ('native', 'detector', 'output-dir'): h.add_argument('--'+name, type=Path, required=True)
    k = sub.add_parser('strokes', help=strokes.__doc__)
    for name in ('pairs', 'gold-bank', 'human-validation', 'same-student', 'output-dir'): k.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == 'hints': native_hints(args.native, args.detector, args.output_dir)
    else: strokes(args.pairs, args.gold_bank, args.human_validation, args.same_student, args.output_dir)


if __name__ == '__main__':
    main()
