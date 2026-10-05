"""Real struck lines cut out of paper 1's gold: what ``data.paste_strikes`` draws its synthetic strikes with besides its own strokes.

    python -m revrender.data.strike_bank --gold GOLD_PAGES --pages DIR [--pages DIR ...] --mapping index_manifest.json
        --same-student pairs.json --output-dir OUT

``--gold`` is paper 1's gold put back on its pages (``data.page_gold``, not cut to ink: the band drawn along the annotator's line).
Of every instance of a writer of paper 1's training split -- never one of a writer whose other id (``--same-student``) is in another
split -- the stroke is cut along the band's centre line: the stroke's width is measured on the ink under that line (twice the median
distance to the ink's edge), and the piece is the non-white pixels (``tighten.non_white``) within half that width (and ``REACH``) of
it, the grey of the scan there, paper elsewhere.  The band itself is wider than the stroke (paper 1 widened thin strokes and the
gold went through the 96 x 768 strip and back), so cutting by the band takes the writing beside the stroke as well; cut by the
centre line, only the pixels where the stroke crosses writing are left of it.  A mark is left out (and counted) when it is not one
clean stroke: wider than ``MAX_WIDTH`` of its line (a blackout -- blackouts are taken as unreadable and not drawn), shorter than
``ELONGATION`` widths, with more than ``MAX_ENDS`` ends (a scribble over a word) or with more than ``MAX_FILL`` times the ink one
stroke of its width and length holds (it carries writing).  ``line_height`` is the height of the annotated line, so a piece is
scaled to a host line by the ratio of the two lines' heights; ``ink_level`` is the grey of the piece's own ink (the median of its
darkest third), so it can be drawn in another writer's ink.

OUT/bank.npz holds ``piece_<n>`` (uint8 grey); OUT/pieces.json lists the pieces (id, image_id, instance, writer, kind 'line', size,
box on the page, line_height, ink_level, pixels, width, length, ends, fill); OUT/left_out.json the marks left out (their window on the
page) and why; OUT/sheet_kept.png and
OUT/sheet_left_out.png show a sample of each.  The pieces are student work: OUT stays out of Git.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scipy import ndimage

from .. import pages as pages_module
from ..eval.ink import INK_BELOW
from ..tighten import non_white
from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .prior_annotations import twins, writer_splits

THIN = .45                 # a mark whose typical height is under this share of its line's height ...
LONG = 2.5                 # ... and that is this many times longer than high is a struck line (the page prefill's rule)
MIN_WIDTH = 2.             # a stroke is taken at least this wide (pixels)
REACH = 1.                 # the cut takes this much beyond half the stroke's width (its faint edge)
MAX_WIDTH = .2             # a struck line is at most this share of its line's height wide
ELONGATION = 3.            # and its centre line at least this many of its widths long
MAX_ENDS = 4               # a centre line with more ends than this is a scribble over a word (a cross has four)
MAX_FILL = 1.6             # ink near the centre line beyond this many times one stroke of that width and length is writing
MARGIN = 6                 # the window round a mark its stroke is looked for in (pixels)
SHEET = 48                 # pieces on a sheet


def kind_of(mask, line_height):
    """'line' for a thin, long mark (a bool mask on its line), else 'blot'."""
    ys, xs = np.nonzero(mask)
    columns = np.unique(xs)
    typical = float(np.median([int(ys[xs == c].max()-ys[xs == c].min())+1 for c in columns]))
    return 'line' if typical <= THIN*line_height and xs.max()-xs.min()+1 >= LONG*typical else 'blot'


def training_writers(mapping, same_student):
    """The writers whose marks the bank takes: paper 1's training split, without a student whose other id is in another split."""
    splits, other = writer_splits(mapping, same_student), twins(same_student)
    return {w for w, s in splits.items() if s == 'train' and splits.get(other.get(w), 'train') == 'train'}


def window(grey, instance):
    """(grey window round the instance's mask, the mask on it, the window's left and top on the page): the mask's box and ``MARGIN``
    round it, inside the page."""
    l, t, _, _ = instance['box']
    ys, xs = np.nonzero(instance['mask'])
    if not len(ys):
        return None
    y0, x0 = max(0, t+ys.min()-MARGIN), max(0, l+xs.min()-MARGIN)
    y1, x1 = min(grey.shape[0], t+ys.max()+1+MARGIN), min(grey.shape[1], l+xs.max()+1+MARGIN)
    mask = np.zeros((y1-y0, x1-x0), bool)
    mask[t+ys-y0, l+xs-x0] = True
    return grey[y0:y1, x0:x1], mask, (int(x0), int(y0))


def ends_of(centre):
    """The ends of a one-pixel centre line: its pixels with exactly one neighbour on it."""
    neighbours = ndimage.convolve(centre.astype(np.uint8), np.ones((3, 3), np.uint8), mode='constant')-centre
    return int((centre & (neighbours == 1)).sum())


def strike(grey, mask, line_height):
    """(grey piece, its pixels, measures with the piece's left and top in the window) of a struck line cut along the centre line of its
    gold band, or (None, None, measures with the reason it is left out)."""
    from skimage.morphology import skeletonize
    ink, soft = grey < INK_BELOW, non_white(grey)
    centre = skeletonize(mask)
    length = int(centre.sum())
    if length < 3:
        return None, None, {'length': length, 'reason': 'no centre line'}
    depth = ndimage.distance_transform_edt(ink)
    on = centre & ink
    width = max(MIN_WIDTH, 2*float(np.median(depth[on]))) if on.any() else MIN_WIDTH
    near = ndimage.distance_transform_edt(~centre) <= width/2+REACH
    measures = {'width': round(width, 2), 'length': length, 'ends': ends_of(centre), 'fill': round(float((near & ink).sum())/(length*width), 2)}
    reason = ('wide' if width > MAX_WIDTH*line_height else 'short' if length < ELONGATION*width else
              'scribble' if measures['ends'] > MAX_ENDS else 'writing' if measures['fill'] > MAX_FILL else None)
    inside = near & soft
    if reason or not inside.any():
        return None, None, {**measures, 'reason': reason or 'no ink'}
    ys, xs = np.nonzero(inside)
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    inside = inside[y0:y1, x0:x1]
    return np.where(inside, grey[y0:y1, x0:x1], 255).astype(np.uint8), inside, {**measures, 'at': (int(x0), int(y0))}


def ink_level(piece, inside):
    """The grey of a piece's own ink: the median of its darkest third."""
    values = np.sort(piece[inside])
    return int(np.median(values[:max(1, len(values)//3)]))


def sheet(images, captions, path):
    """Images on white tiles with a caption each, to look at."""
    tile_w, tile_h, cols = 240, 90, 6
    out = Image.new('L', (cols*tile_w, max(1, (len(images)+cols-1)//cols)*tile_h), 255)
    draw = ImageDraw.Draw(out)
    for k, (image, caption) in enumerate(zip(images, captions)):
        image = Image.fromarray(image)
        image.thumbnail((tile_w-10, tile_h-20))
        x, y = (k % cols)*tile_w, (k//cols)*tile_h
        out.paste(image, (x+5, y+16))
        draw.text((x+5, y+2), caption, fill=0)
    out.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--gold', type=Path, required=True, help="paper 1's gold on its pages (data.page_gold, not cut to ink)")
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    parser.add_argument('--mapping', type=Path, required=True, help="paper 1's split mapping (index_manifest.json)")
    parser.add_argument('--same-student', type=Path, required=True, help='json of students with two writer ids')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    allowed = training_writers(args.mapping, args.same_student)
    ids = sorted(p.stem for p in args.gold.glob('*.json') if p.stem not in ('run_manifest', 'summary') and p.stem.split('_')[0] in allowed)
    require(ids, 'no gold page of a training writer')
    scans = pages_module.page_files(args.pages, ids)
    bank, pieces, left_out, left_windows = {}, [], [], []
    for image_id in ids:
        record, instances = pages_module.read_output(args.gold, image_id)
        if not instances:
            continue
        with Image.open(scans[image_id]) as opened:
            grey = np.asarray(opened.convert('L'))
        require(grey.shape == tuple(record['size'][::-1]), f'{image_id}: the scan is not the size of the gold')
        for k, instance in enumerate(instances):
            made = window(grey, instance)
            if made is None:
                continue
            line = record['lines'][instance['line']]['box']
            (x0, y0) = made[2]
            piece, inside, measures = strike(made[0], made[1], line[3]-line[1])
            if piece is None:
                left_out.append({'image_id': image_id, 'instance': k, 'window': [x0, y0, x0+made[0].shape[1], y0+made[0].shape[0]], **measures})
                left_windows.append(np.where(made[1], made[0], np.minimum(made[0], 235)).astype(np.uint8))
                continue
            ax, ay = measures.pop('at')
            measures['box'] = [x0+ax, y0+ay, x0+ax+piece.shape[1], y0+ay+piece.shape[0]]
            n = f'piece_{len(pieces)}'
            bank[n] = piece
            pieces.append({'id': n, 'image_id': image_id, 'instance': k, 'writer': image_id.split('_')[0], 'kind': 'line',
                           'size': [int(piece.shape[1]), int(piece.shape[0])], 'line_height': int(line[3]-line[1]),
                           'ink_level': ink_level(piece, inside), 'pixels': int(inside.sum()), **measures})
    require(pieces, 'no piece')
    out = make_output_dir(args.output_dir)
    np.savez_compressed(out/'bank.npz', **bank)
    write_json(out/'pieces.json', pieces)
    write_json(out/'left_out.json', left_out)
    rng = random.Random(SEED)
    kept = rng.sample(pieces, min(SHEET, len(pieces)))
    sheet([bank[p['id']] for p in kept], [f'{p["size"][0]}x{p["size"][1]} w{p["width"]} f{p["fill"]}' for p in kept], out/'sheet_kept.png')
    gone = rng.sample(range(len(left_out)), min(SHEET, len(left_out)))
    sheet([left_windows[i] for i in gone], [f'{left_out[i]["reason"]} w{left_out[i].get("width")} e{left_out[i].get("ends")} f{left_out[i].get("fill")}'
                                             for i in gone], out/'sheet_left_out.png')
    summary = {'pages': len(ids), 'pieces': len(pieces), 'writers': len({p['writer'] for p in pieces}), 'left_out': len(left_out),
               'left_out_by_reason': {r: sum(m['reason'] == r for m in left_out) for r in sorted({m['reason'] for m in left_out})}}
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in [args.mapping, args.same_student, args.gold/'run_manifest.json']}, **summary)
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
