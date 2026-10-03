"""Where the handwriting is on a whole-question image, and the boxes for the handwriting the OCR model did not box.

The OCR model finds most lines but not all: it returns a block of several lines as one region (``data.ocr_lines`` can only split it in
equal heights), skips a line now and then and stops short of a stroke that runs past the end of a line.  The page itself says where the
handwriting is.  A question's pages are all cut to one size, so what is printed (the question, its labels, the dashed answer box) is the
same ink on every page of the group and is what at least half of the group's pages have at a pixel (``printed_templates``);
what is left inside the answer box, off the figures, is the student's (``handwriting``).  ``split_by_ink`` puts the cuts of a
multi-line region where the ink has gaps; ``leftover_boxes`` boxes the handwriting no line box covers, a piece that runs over several
lines cut into lines at the gaps of its ink.  (Cutting a piece out in the shape of its ink, the rest of the box whitened, was tried
on the validation lines and read a little worse than the plain box, 167 against 171 of 265 gold revisions: pieces are plain boxes.)

    python -m revrender.data.handwriting --pages DIR [--pages DIR ...] --output templates.npz
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ..eval.ink import INK_BELOW
from ..util import require

PRINTED_SHARE = .5         # a pixel is printed if at least this share of the group's pages have ink there
TEMPLATE_PAD = 5           # thickening of the printed mask: scans of one sheet differ by a pixel or two
BOX_NEED = .35             # a side of the answer box is the ink row / column that is this much dashed line
BOX_MARGIN = 30            # without a detected side: this far from the page edge
MIN_PAGES = 8              # a template needs this many pages of the group
BLOCK = 4                  # joining of leftover strokes works on blocks of 4 x 4 pixels
MIN_INK = 150              # a leftover blob with less ink than this (pixels) is dust
MAX_BOX_HEIGHT = 140       # a taller leftover blob is cut into lines of at most this height, at the gaps of its ink
BOX_INSET = 12             # ink this close to the answer box's own line is the line's, not the student's
THIN = 7                   # marks this thin, alone or repeated (a rule, dashes, a stray line), are not writing
WORD_GAP = 80              # leftover blobs on one row this close (pixels) are the words of one line


def group_key(image_id, size):
    """Pages of one question cut to one size share their printed template: ``<question>_<width>_<height>``."""
    return f'{int(image_id.split("_")[1].lstrip("s"))}_{size[0]}_{size[1]}'


def ink_map(grey):
    return np.asarray(grey) < INK_BELOW


def printed_templates(files_by_key, share=PRINTED_SHARE, min_pages=MIN_PAGES):
    """{key: bool mask}: the printed template of each group of at least ``min_pages`` page files."""
    out = {}
    for key, files in files_by_key.items():
        if len(files) < min_pages:
            continue
        count = None
        for path in files:
            with Image.open(path) as opened:
                ink = ink_map(opened.convert('L'))
            count = ink.astype(np.uint16) if count is None else count + ink
        out[key] = ndimage.binary_dilation(count/len(files) >= share, np.ones((TEMPLATE_PAD, TEMPLATE_PAD), bool))
    return out


def answer_box(ink):
    """(top, bottom, left, right) of the dashed answer box, read from where the ink projections are mostly dashed line; a side that is
    not found falls back to ``BOX_MARGIN`` from the page edge."""
    h, w = ink.shape
    rows, cols = ink.sum(1), ink.sum(0)

    def side(profile, lo, hi, need, fallback):
        k = lo + int(np.argmax(profile[lo:hi]))
        return k if profile[k] >= need else fallback

    return (side(rows, 0, int(.25*h), BOX_NEED*w, BOX_MARGIN), side(rows, int(.75*h), h, BOX_NEED*w, h-BOX_MARGIN),
            side(cols, 0, int(.25*w), BOX_NEED*h, BOX_MARGIN), side(cols, int(.75*w), w, BOX_NEED*h, w-BOX_MARGIN))


def handwriting(grey, template=None, figures=()):
    """The student's ink: dark pixels inside the answer box, off the printed template and off the figure boxes (pixel boxes)."""
    ink = ink_map(grey)
    top, bottom, left, right = answer_box(ink)
    inside = np.zeros_like(ink)
    inside[top+BOX_INSET:bottom-BOX_INSET, left+BOX_INSET:right-BOX_INSET] = True
    ink &= inside
    if template is not None and template.shape == ink.shape:
        ink &= ~template
    for l, t, r, b in figures:
        ink[t:b, l:r] = False
    return ink


def boxes_mask(boxes, shape, pad=2):
    """A boolean map of the union of pixel boxes (left, top, right, bottom), each grown by ``pad``."""
    mask = np.zeros(shape, bool)
    for l, t, r, b in boxes:
        mask[max(0, t-pad):b+pad, max(0, l-pad):r+pad] = True
    return mask


def emptiest_row(profile, lo, hi, centre):
    """The middle of the emptiest stretch of rows in [lo, hi]; of several equally empty stretches, the one nearest ``centre``."""
    values = profile[lo:hi+1]
    empty = list(values <= values.min()+1e-6)+[False]
    best, start = None, None
    for i, flag in enumerate(empty):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            middle = lo+(start+i-1)//2
            best = middle if best is None or abs(middle-centre) < abs(best-centre) else best
            start = None
    return best


def split_by_ink(box, parts, ink, search=.5):
    """A region box cut into ``parts`` lines where the ink has gaps: the cut between lines j and j+1 is the middle of the emptiest
    stretch of rows within ``search`` of a line height around the equal split.  Each line is the ink's own extent in its band (None for a
    band without ink)."""
    l, t, r, b = box
    window = ink[t:b, l:r]
    profile = ndimage.uniform_filter1d(window.sum(1).astype(float), 5)
    height = (b-t)/parts
    cuts = [0]
    for j in range(1, parts):
        centre = int(round(j*height))
        lo, hi = max(cuts[-1]+1, int(centre-search*height)), min(window.shape[0]-1, int(centre+search*height))
        cuts.append(emptiest_row(profile, lo, hi, centre) if hi > lo else centre)
    cuts.append(window.shape[0])
    lines = []
    for lo, hi in zip(cuts, cuts[1:]):
        ys, xs = np.nonzero(window[lo:hi])
        lines.append(None if not len(ys) else [l+int(xs.min()), t+lo+int(ys.min()), l+int(xs.max())+1, t+lo+int(ys.max())+1])
    return lines


def lines_by_ink(window, limit=MAX_BOX_HEIGHT):
    """Row ranges (top, bottom) that cut a tall piece of ink into lines at its emptiest rows, each at most ``limit`` tall."""
    height = window.shape[0]
    if height <= limit:
        return [(0, height)]
    profile = ndimage.uniform_filter1d(window.sum(1).astype(float), 5)
    cuts, start = [], 0
    while height-start > limit:
        lo, hi = start+limit//2, start+limit
        cut = emptiest_row(profile, lo, hi-1, (lo+hi)//2)
        cuts.append((start, cut))
        start = cut
    cuts.append((start, height))
    return cuts


def looks_written(piece):
    """Whether a piece of leftover ink is writing or a scribble and not a rule, dashes or a stray line: it holds a mark that is 14 pixels
    or more in both directions (a character, a blot), or at least two marks of which the typical one is more than a stroke wide."""
    marks, count = ndimage.label(ndimage.binary_dilation(piece, np.ones((3, 3), bool)))
    if not count:
        return False
    areas = ndimage.sum(piece, marks, range(1, count+1))
    thin = [min(r[0].stop-r[0].start, r[1].stop-r[1].start) for r, a in zip(ndimage.find_objects(marks), areas) if a >= 15]
    return bool(thin) and (max(thin) >= 14 or (len(thin) >= 2 and float(np.median(thin)) > THIN))


def row_pieces(blobs):
    """Blobs [l, t, r, b] on one row (half of the shorter one's height shared) and a word's gap apart or less are one line: merged."""
    boxes = [list(b) for b in blobs]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                a, b = boxes[i], boxes[j]
                shared = min(a[3], b[3])-max(a[1], b[1])
                gap = max(a[0], b[0])-min(a[2], b[2])
                if shared >= .5*min(a[3]-a[1], b[3]-b[1]) and gap <= WORD_GAP:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def leftover_boxes(ink, boxes, pad=4, join=(1, 6)):
    """Boxes (left, top, right, bottom in pixels) for the handwriting no box covers.  The leftover ink is joined into blobs (strokes of a
    word), the blobs of one row into pieces (words of a line); a piece that is little ink or does not look written (``looks_written``) is
    dropped, a piece over several lines is cut into lines at the gaps of its ink, and each box is grown by ``pad``."""
    left = ink & ~boxes_mask(boxes, ink.shape)
    h, w = left.shape
    h4, w4 = -(-h//BLOCK), -(-w//BLOCK)
    padded = np.zeros((h4*BLOCK, w4*BLOCK), bool)
    padded[:h, :w] = left
    blocks = padded.reshape(h4, BLOCK, w4, BLOCK).any((1, 3))
    labels, _ = ndimage.label(ndimage.binary_dilation(blocks, np.ones(join, bool)))
    blobs = []
    for region in ndimage.find_objects(labels):
        y0, y1, x0, x1 = region[0].start*BLOCK, min(h, region[0].stop*BLOCK), region[1].start*BLOCK, min(w, region[1].stop*BLOCK)
        ys, xs = np.nonzero(left[y0:y1, x0:x1])
        if len(ys):
            blobs.append([x0+int(xs.min()), y0+int(ys.min()), x0+int(xs.max())+1, y0+int(ys.max())+1])
    out = []
    for lft, top, rgt, bottom in row_pieces(blobs):
        piece = left[top:bottom, lft:rgt]
        for lo, hi in lines_by_ink(piece):
            band = piece[lo:hi]
            if band.sum() < MIN_INK or not looks_written(band):
                continue
            by, bx = np.nonzero(band)
            l, r = lft+int(bx.min()), lft+int(bx.max())+1
            t, b = top+lo+int(by.min()), top+lo+int(by.max())+1
            out.append([max(0, l-pad), max(0, t-pad), min(w, r+pad), min(h, b+pad)])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of <image_id>.jpg (repeatable)')
    parser.add_argument('--output', type=Path, required=True, help='templates.npz: one printed mask per (question, size) group')
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'create-only output exists', FileExistsError)
    groups = defaultdict(list)
    for directory in args.pages:
        for path in sorted(Path(directory).glob('*.jpg')):
            with Image.open(path) as opened:
                groups[group_key(path.stem, opened.size)].append(path)
    templates = printed_templates(groups)
    np.savez_compressed(args.output, **templates)
    print({'groups': len(groups), 'with a template': len(templates), 'pages': sum(len(v) for v in groups.values())})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
