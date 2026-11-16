"""A strike region's polygon drawn tight round what was struck (user 2026-11-16: the box should hug the struck writing and its strike,
so the annotator need not resize it), from the strike pixels a model would erase: no training.

    python -m revrender.tools.annotate.tight_region score --dataset REGIONS.json --library PACK_DIR --output OUT.json
        [--writers SUMMARY.json]

``tight_polygon``: within the box widened by ``MARGIN`` pixels, the ink (grey below ``INK``) connected to the strike pixels (8-connected,
the strike dilated by ``TOUCH``) -- the strike and the strokes it crosses -- and the strike itself; their convex hull widened by ``PAD``,
as the four corners of its smallest turned rectangle when that rectangle is nearly the hull (``RECTANGLE`` of its area or more: most of
the annotator's polygons have four corners), else the hull cut to at most ``CORNERS`` corners, each cut pushed out so nothing selected
is left outside.  No strike pixel: the box as it is.

``score``: each confirmed region of a merged dataset that a detection of the library (``pixel_library infer``'s pack) started, the
annotator's polygon against the detection's box and against the tight polygon from the annotator's own strike pixels on that box (the
best a model's strike can give): the share of the annotator's strike pixels inside, and the ink inside it against the ink inside the
annotator's polygon (intersection over union, and the ink it adds or leaves out); ``--writers``: a prefill set's summary, its held-out
writers reported apart; ``--predicted``: the library was inferred with ``--tight`` (and without the annotator's records on these pages),
and the polygons scored are its own, from the model's strike.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from ...data.pixel_masks import decode, region_pixels
from ...util import make_output_dir, require, write_json

INK = 200
MARGIN = 8
TOUCH = 2
PAD = 2
RECTANGLE = .8
CORNERS = 8


def selected(grey, box, strike):
    """(mask, left, top): the strike and the ink it touches, in the box widened by ``MARGIN``; ``strike`` is a page-sized mask."""
    h, w = grey.shape; l, t, r, b = box
    L, T, R, B = max(0, l-MARGIN), max(0, t-MARGIN), min(w, r+MARGIN), min(h, b+MARGIN)
    ink = grey[T:B, L:R] < INK; seed = strike[T:B, L:R].copy()
    if not seed.any(): return seed, L, T
    labels, _ = ndimage.label(ink, np.ones((3, 3)))
    touched = np.unique(labels[ndimage.binary_dilation(seed, iterations=TOUCH) & ink]); touched = touched[touched > 0]
    return np.isin(labels, touched) | seed, L, T


def turned_rectangle(points):
    """The four corners of the smallest rectangle, at any turn, round the hull's points, and its area."""
    best = None
    for a, z in zip(points, np.roll(points, -1, 0)):
        edge = z-a; n = np.hypot(*edge)
        if n == 0: continue
        u = edge/n; v = np.array([-u[1], u[0]]); p, q = points@u, points@v
        area = (p.max()-p.min())*(q.max()-q.min())
        if best is None or area < best[0]:
            best = (area, [u*p.min()+v*q.min(), u*p.max()+v*q.min(), u*p.max()+v*q.max(), u*p.min()+v*q.max()])
    return np.array(best[1]), best[0]


def cross(a, z):
    return a[0]*z[1]-a[1]*z[0]


def area(points):
    x, y = points[:, 0], points[:, 1]; return abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))/2


def widened(points, d):
    """A convex polygon (counter-clockwise) with every edge moved out by d."""
    out = []
    for k in range(len(points)):
        p0, p1, p2 = points[k-1], points[k], points[(k+1) % len(points)]
        n1 = np.array([p1[1]-p0[1], p0[0]-p1[0]]); n2 = np.array([p2[1]-p1[1], p1[0]-p2[0]])
        n1, n2 = n1/np.hypot(*n1), n2/np.hypot(*n2); m = n1+n2
        out.append(p1 + m*d/max(np.dot(m, n1), 1e-6))
    return np.array(out)


def cut(points, corners):
    """At most ``corners`` of a convex polygon's corners: drop, one at a time, the corner whose dropping loses the least area, then push the
    edges out until the dropped corners are inside again."""
    keep = list(range(len(points)))
    while len(keep) > corners:
        loss = [area(points[[keep[k-1], keep[k], keep[(k+1) % len(keep)]]]) for k in range(len(keep))]
        keep.pop(int(np.argmin(loss)))
    poly = points[keep]
    if area(poly) and cross(poly[1]-poly[0], poly[2]-poly[1]) < 0: poly = poly[::-1]
    reach = 0.
    for k in range(len(poly)):
        a, z = poly[k], poly[(k+1) % len(poly)]; n = np.array([z[1]-a[1], a[0]-z[0]]); n /= np.hypot(*n)
        reach = max(reach, float(((points-a)@n).max()))
    return widened(poly, reach) if reach > 0 else poly


def tight_polygon(grey, box, strike):
    """The tight polygon (page coordinates, integer corners) round the strike and what it crosses; the box's corners when nothing is struck."""
    mask, L, T = selected(grey, box, strike)
    l, t, r, b = box
    if not mask.any(): return [[l, t], [r, t], [r, b], [l, b]]
    ys, xs = np.nonzero(mask)
    pts = np.unique(np.concatenate([np.stack([xs+L+dx, ys+T+dy], 1) for dx in (0, 1) for dy in (0, 1)]), axis=0).astype(float)
    try:
        hull = pts[ConvexHull(pts).vertices]
    except QhullError:
        hull = np.array([[pts[:, 0].min(), pts[:, 1].min()], [pts[:, 0].max(), pts[:, 1].min()], [pts[:, 0].max(), pts[:, 1].max()], [pts[:, 0].min(), pts[:, 1].max()]])
        hull[2:, :] += .5; hull[1:3, 0] += .5
    rect, rect_area = turned_rectangle(hull)
    poly = rect if area(hull) >= RECTANGLE*rect_area else cut(hull, CORNERS)
    if cross(poly[1]-poly[0], poly[2]-poly[1]) < 0: poly = poly[::-1]
    poly = widened(poly, PAD); h, w = grey.shape
    return [[int(np.clip(round(x), 0, w)), int(np.clip(round(y), 0, h))] for x, y in poly]


def bounds(polygon):
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]; return [min(xs), min(ys), max(xs), max(ys)]


def inside(polygon, shape):
    """A page-sized mask of the pixels whose centres lie in the polygon."""
    l, t, r, b = bounds(polygon); out = np.zeros(shape, bool)
    if r > l and b > t: out[t:b, l:r] = region_pixels(polygon, [l, t, r, b])
    return out


def compare(polygon, truth, ink, strike):
    mine = inside(polygon, ink.shape); a, z = mine & ink, truth & ink
    union = (a | z).sum()
    return {'strike_inside': float((mine & strike).sum()/strike.sum()) if strike.any() else 1., 'ink_iou': float((a & z).sum()/union) if union else 1.,
            'ink_added': float((a & ~z).sum()/max(z.sum(), 1)), 'ink_left_out': float((z & ~a).sum()/max(z.sum(), 1)),
            'corners': len(polygon)}


def score(args):
    rows = [r for r in json.loads(args.dataset.read_text()) if r.get('pixels_confirmed') and not r.get('deleted') and r.get('region')]
    pack = json.loads((args.library/'pack.json').read_text()); pages = pack['pages']
    detections = {pages[r['page']]['id']+'#'+r['id'].split('#')[-1]: r for r in pack['rows']}
    image_of = {p['id']: args.library/p['image'] for p in pages}
    held = set(json.loads(args.writers.read_text())['validation_writers']) if args.writers else set()
    results = []
    for page in sorted({r['page'] for r in rows if r['id'] in detections}):
        with Image.open(image_of[page]) as im: grey = np.asarray(im.convert('L'))
        ink = grey < INK
        for r in (r for r in rows if r['page'] == page and r['id'] in detections):
            l, t, rr, b = r['box']; strike = np.zeros(grey.shape, bool)
            strike[t:b, l:rr] = decode(r['runs'], (b-t, rr-l)) & region_pixels(r['region'], r['box'])
            truth = inside(r['region'], grey.shape); d = detections[r['id']]; dl, dt, dr, db = d.get('detection_box') or d['box']
            on_box = np.zeros(grey.shape, bool); on_box[dt:db, dl:dr] = strike[dt:db, dl:dr]
            tight = d['region'] if args.predicted else tight_polygon(grey, [dl, dt, dr, db], on_box)
            results.append({'id': r['id'], 'writer': r['writer'], 'held_out': r['writer'] in held, 'mode': r.get('mode'),
                            'detection': compare([[dl, dt], [dr, dt], [dr, db], [dl, db]], truth, ink, strike),
                            'tight': compare(tight, truth, ink, strike)})
    require(results, 'no region a detection started')

    def summary(rs):
        out = {'regions': len(rs)}
        for kind in ('detection', 'tight'):
            v = lambda k: np.array([x[kind][k] for x in rs])
            out[kind] = {'strike_inside': float(v('strike_inside').mean()), 'all_strike_inside': float((v('strike_inside') >= .999).mean()),
                         'ink_iou': float(np.median(v('ink_iou'))), 'ink_iou_mean': float(v('ink_iou').mean()),
                         'ink_iou_at_least_0.9': float((v('ink_iou') >= .9).mean()),
                         'ink_added': float(np.median(v('ink_added'))), 'ink_left_out': float(np.median(v('ink_left_out')))}
        return out
    report = {'all': summary(results), 'held_out': summary([x for x in results if x['held_out']]) if held else None,
              'constants': {'INK': INK, 'MARGIN': MARGIN, 'TOUCH': TOUCH, 'PAD': PAD, 'RECTANGLE': RECTANGLE, 'CORNERS': CORNERS}, 'regions': results}
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'regions'}, indent=1), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score')
    for name in ('dataset', 'library', 'output'): s.add_argument('--'+name, type=Path, required=True)
    s.add_argument('--writers', type=Path, help="a prefill set's summary.json: its held-out writers apart")
    s.add_argument('--predicted', action='store_true', help="the library was inferred with --tight: its own polygons (from the model's strike) are scored")
    args = p.parse_args(argv); {'score': score}[args.command](args)


if __name__ == '__main__':
    main()
