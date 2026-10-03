"""Whole-question pages cleaned: each page kept to its question's own dashed frame, with the dust and the haze of the scan taken off.

A page is cut out of the answer sheet around one question's dashed frame, with a margin that runs over the questions above and below: their
dashed lines, a line or two of their handwriting, the printed footer, the black alignment blocks at the corners.  None of it belongs to the
question, and the model reads some of it (a block at the corner as a blackout, a neighbour's crossed-out word as this question's).

The pages of one question cut to one size (``handwriting.group_key``) are cut from the same window of the same sheet, so their dashed
lines lie at the same rows to a pixel or two.  A line at the same row on (nearly) every page of the group is the sheet's: the group's own
frame is the pair of neighbouring such lines with the widest gap between them, the questions above and below being closer.  A row that
only some pages have is handwriting that looks dashed (a ruled-out line, an underline) and takes no part.  Per page the frame's rows are the
page's own dashed rows nearest to the group's, its sides the outermost dashed columns (never inside the group's).  ``clean`` whites out
everything beyond ``FRAME_MARGIN`` pixels of the frame, takes off the specks (a piece of ink of at most ``SPECK_AREA`` pixels with no other
ink within ``SPECK_ALONE`` pixels) and bleaches the faint haze the JPEG leaves on the paper away from the ink.  The page keeps its size, so
the OCR boxes, the printed templates and the gold stay valid; the frame's box says where to crop.

    python -m revrender.data.clean_pages --pages DIR [--pages DIR ...] [--ids FILE] --output-dir OUT

OUT/<image_id>.png are the cleaned pages and OUT/frames.json the box (left, top, right, bottom) each page was kept to ({image_id: box or
null}: a page of a group with too few pages for the rows to agree is only de-dusted).  ``--ids`` names the pages to write; the frame of a
group is always read from all the pages given.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .. import pages
from ..eval.ink import INK_BELOW
from ..util import make_output_dir, to_jsonable, write_json, write_manifest
from .handwriting import group_key

DASH_BELOW = 170           # a dash of the frame is this dark or darker
LINE_SHARE = .3            # a row (column) is a dashed line when this share of the page's middle is dash
LINE_DASHES = 15           # ... and it has at least this many dashes of an even length
SIDE_SHARE = .25           # the same for the vertical sides, which are read between the frame's corners
SIDE_DASHES = 6            # (a side is shorter than a line, so it has fewer dashes)
CORNER = 12                # rows next to the horizontal lines are left out of the sides: the corners are round
MIN_GROUP = 8              # a group of fewer pages has no frame
SHARED = .3                # a row is the sheet's when this share of the group's pages have a dashed line within ROW_JITTER of it
ROW_JITTER = 3             # pages of a group differ by a pixel or two
SNAP = 8                   # a page's own dashed row this close to the group's replaces it
MIN_HEIGHT = 60            # a frame lower than this is not one
FRAME_MARGIN = 6           # pixels of page kept beyond the frame's lines (less when a neighbour's line is closer)
SPECK_AREA = 8             # a piece of ink this small (pixels) ...
SPECK_ALONE = 6            # ... with no other ink this close is dust
HAZE_FROM = 235            # paper this light or lighter ...
HAZE_FAR = 3               # ... this far from any ink is bleached to white


def dash_pattern(line, minimum):
    """Whether a row or column of dark pixels is a dashed line: at least ``minimum`` dashes of an even length with gaps between them."""
    steps = np.diff(np.concatenate([[0], np.asarray(line, np.int8), [0]]))
    starts, ends = np.nonzero(steps == 1)[0], np.nonzero(steps == -1)[0]
    dashes, gaps = ends-starts, starts[1:]-ends[:-1]
    return len(dashes) >= minimum and len(gaps) >= minimum-1 and 4 < dashes.mean() < 60 and gaps.mean() > 2 and dashes.std()/dashes.mean() < .6


def dashed_lines(dark, share, axis, minimum):
    """Indices along ``axis`` (0: rows, 1: columns) of the dashed lines of a boolean map of dark pixels: of each run of candidate lines
    (those with ``share`` of the middle of the page dark) the middle one of those with the most dark pixels, if it is dashed."""
    length = dark.shape[1-axis]
    middle = slice(int(.1*length), int(.9*length))
    fraction = dark[:, middle].mean(1) if axis == 0 else dark[middle, :].mean(0)
    found = np.nonzero(fraction >= share)[0]
    out = []
    for group in np.split(found, np.nonzero(np.diff(found) > 3)[0]+1) if len(found) else []:
        strongest = group[fraction[group] >= .9*fraction[group].max()]
        k = int(strongest[len(strongest)//2])
        if dash_pattern(dark[k] if axis == 0 else dark[:, k], minimum):
            out.append(k)
    return out


def page_lines(grey):
    """(the page's dashed rows, its dashed columns between its outermost two rows): the lines the frame is looked for in."""
    dark = np.asarray(grey) < DASH_BELOW
    rows = dashed_lines(dark, LINE_SHARE, 0, LINE_DASHES)
    inner = dark[rows[0]+CORNER:rows[-1]-CORNER] if len(rows) >= 2 and rows[-1]-rows[0] > 2*CORNER else dark
    return rows, dashed_lines(inner, SIDE_SHARE, 1, SIDE_DASHES)


def shared_rows(row_lists):
    """The rows the sheet has, from the dashed rows of every page of a group: those within ``ROW_JITTER`` of each other on at least
    ``SHARED`` of the pages, each at the median of its cluster."""
    ys = np.array(sorted(y for rows in row_lists for y in rows))
    out, i = [], 0
    while i < len(ys):
        j = i
        while j+1 < len(ys) and ys[j+1]-ys[i] <= 2*ROW_JITTER:
            j += 1
        if j+1-i >= SHARED*len(row_lists):
            out.append(float(np.median(ys[i:j+1])))
        i = j+1
    return out


def group_frame(row_lists, side_lists):
    """A group's frame from its pages' lines: {'top', 'bottom', 'above', 'below', 'left', 'right'} (above and below: the nearest
    neighbouring line or None; left and right: the medians of the pages' outermost dashed columns, None without any) or None."""
    rows = shared_rows(row_lists)
    if len(rows) < 2:
        return None
    gap, k = max((b-a, i) for i, (a, b) in enumerate(zip(rows, rows[1:])))
    if gap < MIN_HEIGHT:
        return None
    sides = [(min(s), max(s)) for s in side_lists if s]
    return {'top': rows[k], 'bottom': rows[k+1], 'above': rows[k-1] if k else None, 'below': rows[k+2] if k+2 < len(rows) else None,
            'left': float(np.median([s[0] for s in sides])) if sides else None, 'right': float(np.median([s[1] for s in sides])) if sides else None}


def page_box(shape, frame, rows, sides):
    """The box (left, top, right, bottom, inclusive) a page of this (height, width) is kept to: the group's frame, its rows snapped to the
    page's own, ``FRAME_MARGIN`` pixels around, less above or below where a neighbour's line is closer than twice that.  The sides are the
    page's outermost dashed columns if there are two, never inside the group's."""
    height, width = shape
    near = lambda y: min((r for r in rows if abs(r-y) <= SNAP), key=lambda r: abs(r-y), default=int(round(y)))
    top, bottom = near(frame['top']), near(frame['bottom'])
    up = min(FRAME_MARGIN, max(2, int(top-frame['above'])//2-2)) if frame['above'] is not None else FRAME_MARGIN
    down = min(FRAME_MARGIN, max(2, int(frame['below']-bottom)//2-2)) if frame['below'] is not None else FRAME_MARGIN
    left, right = (min(sides), max(sides)) if len(sides) >= 2 else (width, 0)
    left = min(left, frame['left']) if frame['left'] is not None else left
    right = max(right, frame['right']) if frame['right'] is not None else right
    left, right = (0, width-1) if left > right else (left-FRAME_MARGIN, right+FRAME_MARGIN)
    return max(0, int(left)), max(0, top-up), min(width-1, int(right)), min(height-1, bottom+down)


def clean(grey, box):
    """(the cleaned page, the number of specks taken off): ``grey`` kept inside ``box`` (left, top, right, bottom; None keeps it all),
    then de-dusted and the haze bleached."""
    page = np.array(grey, np.uint8)
    if box is not None:
        left, top, right, bottom = box
        keep = np.zeros(page.shape, bool)
        keep[top:bottom+1, left:right+1] = True
        page[~keep] = 255
    ink = page < INK_BELOW
    pieces, count = ndimage.label(ink, np.ones((3, 3), bool))
    specks = 0
    if count:
        names = np.arange(1, count+1)
        small = ndimage.sum(ink, pieces, names) <= SPECK_AREA
        near = ndimage.binary_dilation(np.isin(pieces, names[~small]), np.ones((3, 3), bool), iterations=SPECK_ALONE)
        alone = small & (ndimage.maximum(near, pieces, names) == 0)
        specks = int(alone.sum())
        dust = np.isin(pieces, names[alone])
        page[ndimage.binary_dilation(dust, np.ones((3, 3), bool), iterations=2) & (page < 245)] = 255
    far = ~ndimage.binary_dilation(page < INK_BELOW, np.ones((3, 3), bool), iterations=HAZE_FAR)
    page[far & (page >= HAZE_FROM)] = 255
    return page, specks


def read_grey(path):
    with Image.open(path) as opened:
        return np.asarray(opened.convert('L'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of <image_id>.jpg (repeatable)')
    parser.add_argument('--ids', type=Path, help='a file of image ids, one per line: the pages to write (default: every page)')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    files = pages.page_files(args.pages)
    wanted = sorted(files) if args.ids is None else [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    pages.page_files(args.pages, wanted)
    out = make_output_dir(args.output_dir)
    lines, shapes = {}, {}
    for image_id, path in files.items():
        grey = read_grey(path)
        lines[image_id], shapes[image_id] = page_lines(grey), grey.shape
    groups = defaultdict(list)
    for image_id in files:
        groups[group_key(image_id, shapes[image_id][::-1])].append(image_id)
    frames = {key: group_frame([lines[i][0] for i in ids], [lines[i][1] for i in ids]) if len(ids) >= MIN_GROUP else None
              for key, ids in groups.items()}
    boxes, totals = {}, {'pages': 0, 'no_frame': 0, 'specks': 0}
    for image_id in wanted:
        frame = frames[group_key(image_id, shapes[image_id][::-1])]
        box = page_box(shapes[image_id], frame, *lines[image_id]) if frame is not None else None
        page, specks = clean(read_grey(files[image_id]), box)
        Image.fromarray(page).save(out/f'{image_id}.png', optimize=True)
        boxes[image_id] = list(box) if box is not None else None
        totals['pages'] += 1
        totals['no_frame'] += box is None
        totals['specks'] += specks
    write_json(out/'frames.json', boxes)
    write_manifest(out, config=to_jsonable(args), inputs={}, ink_below=INK_BELOW, frame_margin=FRAME_MARGIN, speck_area=SPECK_AREA,
                   speck_alone=SPECK_ALONE, haze_from=HAZE_FROM, haze_far=HAZE_FAR, groups=len(groups),
                   groups_without_frame=sum(f is None for f in frames.values()), **totals)
    print(totals)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
