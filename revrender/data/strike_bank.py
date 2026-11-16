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
from .allowed import allowed as allowed_page
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


def traced_pixels(grey, centre, width):
    """Redraw the annotation's trajectory, without copying intersecting letter pixels.

    Width and darkness are robust global measurements of the scanned stroke. The
    geometry comes only from the annotation band, never the skeleton of the ink.
    This deliberately does not claim to recover the true layer at intersections.
    """
    distance = ndimage.distance_transform_edt(~centre)
    values = grey[centre & (grey < INK_BELOW)]
    level = float(np.median(values)) if len(values) else 40.
    opacity = np.clip(width/2+.5-distance, 0, 1)
    return np.rint(255-(255-level)*opacity).astype(np.uint8), opacity > 0


def strike(grey, mask, line_height, render='pixels'):
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
    if render == 'traced':
        rendered, inside = traced_pixels(grey, centre, width)
    else:
        rendered = np.where(inside, grey, 255).astype(np.uint8)
    ys, xs = np.nonzero(inside)
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    inside = inside[y0:y1, x0:x1]
    return rendered[y0:y1, x0:x1], inside, {**measures, 'at': (int(x0), int(y0))}


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
    parser.add_argument('--render', choices=('pixels', 'traced'), default='pixels',
                        help='copy scanned pixels, or redraw the gold trajectory without letter fragments')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    allowed = training_writers(args.mapping, args.same_student)
    ids = sorted(p.stem for p in args.gold.glob('*.json') if p.stem not in ('run_manifest', 'summary') and p.stem.split('_')[0] in allowed
                 and allowed_page(p.stem))                                   # never a labmate's page (data.allowed)
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
            piece, inside, measures = strike(made[0], made[1], line[3]-line[1], args.render)
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


# ---- synthesis materials (plan of 2026-11-16): a strike as the opacity of its own ink, cleaned, and measured -------------------------
INK_ALPHA = .3             # a pixel of a strike at least this opaque is its ink (to measure it)
REACH = 60                 # a human strike's gap over writing is closed up to this many pixels from its pixels on each side
LINE_SHARE = .1            # a straight part of a human strike holds at least this share of its pixels
LINES = 6                  # and a strike has at most this many
STUB = 4.                  # a piece off a line's band that touches it and holds at most this many widths squared is writing the eraser took
HOLE_REACH = 1.            # a human strike's holes are looked for within this many of its widths
TRIM = .5                 # a cleaned bank line keeps this many pixels beyond half its local median width from its centre
WINDOW_WIDTHS = 3.         # the local median width is taken over this many widths along the line on each side
MIN_COVER = .7             # a bank line with less of its centre covered by ink is broken (dots, pieces) and left out


def opacity(grey, level, where=None):
    """The opacity of ink of grey ``level`` in a grey image (1 at the level and darker), 0 outside ``where``."""
    alpha = np.clip((255.-grey.astype(np.float32))/(255.-min(float(level), 200.)), 0, 1)
    return alpha if where is None else np.where(where, alpha, 0.).astype(np.float32)


def axis_frame(points):
    """(t, s) of points (y, x) along and across their principal axis, and the two variances."""
    centred = points-points.mean(0)
    values, vectors = np.linalg.eigh(np.cov(centred.T) if len(points) > 2 else np.eye(2))
    along, across = vectors[:, 1], vectors[:, 0]
    return centred @ along, centred @ across, values[::-1]


def line_measures(alpha):
    """How much a strike looks like one clean line: its length along its axis, its typical width (the median ink across it), ``fill`` (its
    ink over one line of that width and length: above 1 it carries more, e.g. writing) and ``cover`` (the share of its length that has
    ink).  ``linear``: elongated, one run of ink across it at most places and at least three widths long; only then are fill and cover
    meant (check 1 and 2 of the plan)."""
    ys, xs = np.nonzero(alpha >= INK_ALPHA)
    if len(ys) < 6:
        return {'linear': False, 'pixels': int(len(ys))}
    t, s, var = axis_frame(np.stack([ys, xs], 1).astype(float))
    bins = np.round(t-t.min()).astype(int); length = int(bins.max())+1
    count = np.bincount(bins, minlength=length)
    runs = []
    for b in np.flatnonzero(count):
        across = np.unique(np.round(s[bins == b]).astype(int)); runs.append(1+int((np.diff(across) > 1).sum()))
    width = float(np.median(count[count > 0]))
    measures = {'pixels': int(len(ys)), 'length': length, 'width': round(width, 2), 'fill': round(float(len(ys))/(length*width), 3),
                'cover': round(float((count > 0).mean()), 3), 'elongation': round(float(var[0]/max(var[1], .25)), 1),
                'one_run': round(float(np.mean(np.array(runs) == 1)), 3)}
    measures['linear'] = bool(measures['elongation'] >= 9 and measures['one_run'] >= .7 and length >= 3*width)
    return measures


def along_line(seeds, ink, theta, rho, half, reach):
    """The pixels between seeds along the line x cos(theta) + y sin(theta) = rho (within ``half`` of it), on each row of the line's band
    reachable from a seed on both sides through ink alone, at most ``reach`` pixels from each."""
    h, w = ink.shape; normal = np.array([np.cos(theta), np.sin(theta)]); direction = np.array([-np.sin(theta), np.cos(theta)])
    ys, xs = np.nonzero(seeds); along = xs*direction[0]+ys*direction[1]
    t = np.arange(int(np.floor(along.min()))-1, int(np.ceil(along.max()))+2)
    filled = np.zeros_like(ink)
    for s in np.arange(-half, half+1):
        x = np.rint(rho*normal[0]+s*normal[0]+t*direction[0]).astype(int); y = np.rint(rho*normal[1]+s*normal[1]+t*direction[1]).astype(int)
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h); xi, yi = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
        seed = inside & seeds[yi, xi]; dark = inside & ink[yi, xi]
        if seed.sum() < 2:
            continue
        n = np.arange(len(t))
        last_seed = np.maximum.accumulate(np.where(seed, n, -1)); last_break = np.maximum.accumulate(np.where(~dark, n, -1))
        next_seed = np.minimum.accumulate(np.where(seed, n, len(t))[::-1])[::-1]; next_break = np.minimum.accumulate(np.where(~dark, n, len(t))[::-1])[::-1]
        good = (last_seed > last_break) & (next_seed < next_break) & (n-last_seed <= reach) & (next_seed-n <= reach) & dark & (last_seed >= 0)
        filled[yi[good], xi[good]] = True
    return filled


def straight_parts(ys, xs, apart, least):
    """Up to ``LINES`` lines (theta, rho: x cos theta + y sin theta = rho) through at least ``least`` of the points, strongest first (a Hough
    transform at a degree and a pixel); a line within ``apart`` pixels and 6 degrees of a stronger one is the same line."""
    angles = np.deg2rad(np.arange(-90, 90)); rho = np.rint(np.outer(xs, np.cos(angles))+np.outer(ys, np.sin(angles))).astype(int)
    low = rho.min(); votes = np.zeros((rho.max()-low+1, len(angles)), int)
    np.add.at(votes, (rho-low, np.broadcast_to(np.arange(len(angles)), rho.shape)), 1)
    found = []
    while len(found) < LINES:
        r, a = np.unravel_index(np.argmax(votes), votes.shape)
        if votes[r, a] < least:
            break
        found.append((float(angles[a]), float(r+low)))
        for da in range(-6, 7):                       # the angle wraps round: -90 degrees is 90 with rho reversed
            aa = a+da; flip = aa < 0 or aa >= len(angles); aa %= len(angles)
            centre = -(r+low)-low if flip else r
            votes[max(0, centre-apart):max(0, centre+apart+1), aa] = 0
    return found


def completed(grey, strike, width):
    """A human strike (the pixels the annotator erased) with its gaps closed where it crossed writing.  The strike's straight parts are
    found as lines through its pixels (a Hough transform: a dashed line is one line); along each, between its pixels, a gap is closed
    on every row of its band where the scan has ink all the way from a strike pixel on one side to one on the other (there the strike
    lay on the writing), at most ``REACH`` pixels from each.  Nothing is closed across a line (two parallel strokes stay two), nor where
    the paper shows between (a real break); scribbled pixels off every line are left as they are.  Returns it and the least share of a
    line's length the closed strike covers (1 without lines)."""
    ink = grey < INK_BELOW; full = strike.copy(); total = int(strike.sum()); found = []
    if total < 6:
        return full, 1.
    half = max(1, int(round(width/2)))
    ys, xs = np.nonzero(strike); yy, xx = np.mgrid[:grey.shape[0], :grey.shape[1]]
    bands = np.zeros_like(strike)
    for theta, rho in straight_parts(ys, xs, max(3, int(width)), max(6, LINE_SHARE*total)):
        near = np.abs(xs*np.cos(theta)+ys*np.sin(theta)-rho) <= half+.5
        if near.sum() < 6:
            continue
        seeds = np.zeros_like(strike); seeds[ys[near], xs[near]] = True
        seeds = ndimage.binary_dilation(seeds, np.ones((3, 3), bool)) & ink       # a curved stroke leaves a row of its band for the next
        full |= along_line(seeds, ink, theta, rho, half, REACH)
        along = xx*-np.sin(theta)+yy*np.cos(theta); at = xs[near]*-np.sin(theta)+ys[near]*np.cos(theta)
        band = (np.abs(xx*np.cos(theta)+yy*np.sin(theta)-rho) <= half+1) & (along >= at.min()-half) & (along <= at.max()+half)
        bands |= band; found.append((band & (along >= at.min()) & (along <= at.max()), along))
    if bands.any():                        # stubs of the writing the eraser took with a line: off its band, touching it, small
        labels, n = ndimage.label(full & ~bands, np.ones((3, 3)))
        touching = np.unique(labels[ndimage.binary_dilation(bands) & (labels > 0)])
        sizes = np.bincount(labels.ravel(), minlength=n+1)
        stubs = [k for k in touching if sizes[k] <= STUB*width*width]
        full &= ~np.isin(labels, stubs)
    rest = full & ~bands                   # the scribbled part: its small holes where it lay on writing closed
    if rest.any():
        disk = ndimage.generate_binary_structure(2, 1); r = max(1, half)
        full |= ndimage.binary_closing(np.pad(rest, r+1), disk, iterations=r)[r+1:-r-1, r+1:-r-1] & ink
    covers = []
    for band, along in found:              # how much of each line's length has the strike once closed
        steps = np.round(along[band]).astype(int); inked = np.round(along[band & full]).astype(int)
        covers.append(len(np.unique(inked))/max(1, len(np.unique(steps))))
    return full, min(covers, default=1.)


def holes(grey, strike, width):
    """The writing a strike still has holes on: ink within its closing (``HOLE_REACH`` widths) that is not the strike, over the strike."""
    r = max(1, int(round(HOLE_REACH*width)))
    closed = ndimage.binary_closing(np.pad(strike, r+1), ndimage.generate_binary_structure(2, 1), iterations=r)[r+1:-r-1, r+1:-r-1]
    return float((closed & ~strike & (grey < INK_BELOW)).sum())/max(1, strike.sum())


def human_material(grey, strike):
    """(opacity before, opacity after, ink level, ``holes`` left, the least ``cover`` of its lines) of a human strike: before is its erased pixels as they are, after with its gaps closed
    (``completed``), the closed pixels as opaque as the scan is there up to the strike's own ink (where writing lies under it the scan is
    darker: the strike's own grey is taken)."""
    level = ink_level(grey, strike)
    before = opacity(grey, level, strike)
    width = max(MIN_WIDTH, 2*float(np.median(ndimage.distance_transform_edt(strike)[strike]))) if strike.any() else MIN_WIDTH
    after, cover = completed(grey, strike, width)
    return before, opacity(grey, level, after), level, holes(grey, after, width), cover


def trimmed_line(grey, level):
    """(opacity before, opacity after, measures) of a bank line (``strike``'s piece): after, the bumps the writing it crossed left on it
    cut off -- at each place along its axis, ink farther from its centre than half the local median width (over ``WINDOW_WIDTHS`` widths
    each side) and ``TRIM`` is dropped -- and its darker crossings at its own grey (the opacity stops at 1).  ``measures['broken']``: less
    than ``MIN_COVER`` of its length has ink (dots, pieces): left out."""
    before = opacity(grey, level, grey < 255)
    ys, xs = np.nonzero(before >= INK_ALPHA)
    if len(ys) < 6:
        return before, before, {'broken': True}
    t, s, _ = axis_frame(np.stack([ys, xs], 1).astype(float))
    bins = np.round(t-t.min()).astype(int); length = int(bins.max())+1
    count = np.bincount(bins, minlength=length).astype(float)
    centre = np.full(length, np.nan)
    for b in np.flatnonzero(count):
        centre[b] = np.median(s[bins == b])
    have = ~np.isnan(centre)
    centre = np.interp(np.arange(length), np.flatnonzero(have), centre[have])
    width = float(np.median(count[count > 0])); half = max(2, int(round(WINDOW_WIDTHS*width)))
    centre = ndimage.median_filter(centre, size=2*half+1, mode='nearest')
    local = np.array([np.median(count[max(0, b-half):b+half+1][count[max(0, b-half):b+half+1] > 0]) for b in range(length)])
    # every pixel of the piece (its faint edge too) is judged by where it lies along and across the line
    py, px = np.nonzero(grey < 255)
    mean = np.stack([ys, xs], 1).astype(float).mean(0)
    _, vectors = np.linalg.eigh(np.cov(np.stack([ys, xs], 1).astype(float).T))
    pt = (np.stack([py, px], 1)-mean) @ vectors[:, 1]; ps = (np.stack([py, px], 1)-mean) @ vectors[:, 0]
    pb = np.clip(np.round(pt-t.min()).astype(int), 0, length-1)
    keep = np.abs(ps-centre[pb]) <= local[pb]/2+TRIM
    mask = np.zeros(grey.shape, bool); mask[py[keep], px[keep]] = True
    after = opacity(grey, level, mask)
    return before, after, {'broken': bool((count > 0).mean() < MIN_COVER), 'cover': round(float((count > 0).mean()), 3)}
