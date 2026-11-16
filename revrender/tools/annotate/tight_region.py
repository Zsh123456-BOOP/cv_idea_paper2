"""A strike region's polygon drawn tight round what was struck (user 2026-11-16: the box should hug the struck writing and its strike,
so the annotator need not resize it), from the strike pixels a model would erase: no training.

    python -m revrender.tools.annotate.tight_region score --dataset REGIONS.json --library PACK_DIR --output OUT.json
        [--writers SUMMARY.json] [--predicted]
    python -m revrender.tools.annotate.tight_region learn --dataset REGIONS.json --library PACK_DIR --writers SUMMARY.json --output-dir OUT

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

``learn`` (the second version, plan of 2026-11-16): the library inferred without ``--tight`` on the dataset's pages, so each detection
keeps its box and the model's strike on it; a small classifier (gradient boosting) learns from the training writers' pieces (``pieces``;
label: more than half the piece inside the annotator's polygon, from the annotator's strike and the model's) which to draw round, and both
versions are scored on the held-out writers from the model's strike, with the rule written before.
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
    return shape(mask, L, T, grey.shape, box)


def shape(mask, L, T, size, box):
    """The polygon round a window's selected pixels (window at L, T): the hull, its turned rectangle when nearly the hull, else the hull
    cut to ``CORNERS``, widened by ``PAD``; the box's corners when nothing is selected."""
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
    poly = widened(poly, PAD); h, w = size
    return [[int(np.clip(round(x), 0, w)), int(np.clip(round(y), 0, h))] for x, y in poly]


FEATURES = ('covered', 'off_axis', 'beyond_ends', 'within_ends', 'size', 'from_centre_x', 'from_centre_y')


def pieces(grey, box, strike):
    """(the strike in the box widened by ``MARGIN``, each piece of the ink left when it is taken away (8-connected), each piece's features,
    left, top) -- the second version's candidates (plan of 2026-11-16): the share of the piece under or within ``TOUCH`` pixels of the
    strike; its centre's distance from the strike's main axis and beyond its ends along it, over the box's height, and whether it is
    within the ends; its size over the box's ink; its centre from the box's centre, over the box's width and height."""
    h, w = grey.shape; l, t, r, b = box
    L, T, R, B = max(0, l-MARGIN), max(0, t-MARGIN), min(w, r+MARGIN), min(h, b+MARGIN)
    ink = grey[T:B, L:R] < INK; seed = strike[T:B, L:R] & ink
    if not seed.any(): return seed, [], [], L, T
    labels, n = ndimage.label(ink & ~seed, np.ones((3, 3))); near = ndimage.binary_dilation(seed, iterations=TOUCH)
    sy, sx = np.nonzero(seed); points = np.stack([sx, sy], 1).astype(float); mean = points.mean(0)
    u = np.linalg.svd(points-mean, full_matrices=False)[2][0] if len(points) > 1 else np.array([1., 0.]); v = np.array([-u[1], u[0]])
    along = (points-mean)@u; height, width, box_ink = max(1, b-t), max(1, r-l), max(1, int(ink[t-T:b-T, l-L:r-L].sum()))
    masks, rows = [], []
    for k, where in enumerate(ndimage.find_objects(labels), 1):
        piece = labels == k; ys, xs = np.nonzero(piece); c = np.array([xs.mean(), ys.mean()]); a = (c-mean)@u
        beyond = max(0., along.min()-a, a-along.max())
        masks.append(piece)
        rows.append([float((piece & near).sum()/piece.sum()), float(abs((c-mean)@v))/height, beyond/height, float(beyond == 0), float(piece.sum())/box_ink,
                     float(c[0]+L-(l+r)/2)/width, float(c[1]+T-(t+b)/2)/height])
    return seed, masks, rows, L, T


def tight_polygon_learned(grey, box, strike, model):
    """The second version: the strike and the pieces the classifier keeps, drawn as the first version draws."""
    seed, masks, rows, L, T = pieces(grey, box, strike)
    keep = model.predict(np.array(rows)) if rows else []
    chosen = seed.copy()
    for m, k in zip(masks, keep):
        if k: chosen |= m
    return shape(chosen, L, T, grey.shape, box)


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


def matched(args):
    """(page grey, ink, dataset row, the annotator's strike on the page, the detection's box, the model's strike on it) for each confirmed
    region a detection of the library started, page by page."""
    rows = [r for r in json.loads(args.dataset.read_text()) if r.get('pixels_confirmed') and not r.get('deleted') and r.get('region')]
    pack = json.loads((args.library/'pack.json').read_text()); pages = pack['pages']
    detections = {pages[r['page']]['id']+'#'+r['id'].split('#')[-1]: r for r in pack['rows']}
    image_of = {p['id']: args.library/p['image'] for p in pages}
    for page in sorted({r['page'] for r in rows if r['id'] in detections}):
        with Image.open(image_of[page]) as im: grey = np.asarray(im.convert('L'))
        for r in (r for r in rows if r['page'] == page and r['id'] in detections):
            l, t, rr, b = r['box']; human = np.zeros(grey.shape, bool)
            human[t:b, l:rr] = decode(r['runs'], (b-t, rr-l)) & region_pixels(r['region'], r['box'])
            d = detections[r['id']]; box = d.get('detection_box') or d['box']; dl, dt, dr, db = d['box']
            model = np.zeros(grey.shape, bool); model[dt:db, dl:dr] = decode(d['runs'], (db-dt, dr-dl))
            yield grey, r, human, box, model


def learn(args):
    """The second version trained on the training writers' pieces (the annotator's strike and the model's), then both versions on the
    held-out writers from the model's strike, and the rule written before (plan of 2026-11-16): the share with ink IoU of 0.9 or more
    rises with the writer-resampled 95% interval above 0, and the annotator's strike inside falls by at most a point."""
    import pickle
    from sklearn.ensemble import GradientBoostingClassifier
    from ...util import SEED
    held = set(json.loads(args.writers.read_text())['validation_writers']); X, y, kept = [], [], []
    for grey, r, human, box, model in matched(args):
        if r['writer'] in held: kept.append((grey, r, human, box, model)); continue
        truth = inside(r['region'], grey.shape)
        l, t, rr, b = box
        for strike in (human & np.pad(np.ones((b-t, rr-l), bool), ((t, grey.shape[0]-b), (l, grey.shape[1]-rr))), model):
            _, masks, rows, L, T = pieces(grey, box, strike)
            for m, f in zip(masks, rows):
                X.append(f); y.append(int(truth[T:T+m.shape[0], L:L+m.shape[1]][m].mean() > .5))
    require(X and kept, 'no pieces to learn from, or no held-out region')
    clf = GradientBoostingClassifier(random_state=SEED).fit(np.array(X), np.array(y))
    results = []
    for grey, r, human, box, model in kept:
        truth = inside(r['region'], grey.shape); ink = grey < INK
        results.append({'id': r['id'], 'writer': r['writer'], 'first': compare(tight_polygon(grey, box, model), truth, ink, human),
                        'second': compare(tight_polygon_learned(grey, box, model, clf), truth, ink, human)})
    good = lambda x, k: float(x[k]['ink_iou'] >= .9)
    writers = sorted({x['writer'] for x in results}); by = {w: [x for x in results if x['writer'] == w] for w in writers}
    diff = lambda chosen: float(np.mean([good(x, 'second')-good(x, 'first') for x in chosen]))
    rng = np.random.default_rng(SEED); draws = [diff([x for w in rng.choice(writers, len(writers)) for x in by[w]]) for _ in range(1000)]
    interval = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    strike_drop = float(np.mean([x['first']['strike_inside']-x['second']['strike_inside'] for x in results]))
    summary = {k: {'iou_0.9_share': float(np.mean([good(x, k) for x in results])), 'ink_iou_mean': float(np.mean([x[k]['ink_iou'] for x in results])),
                   'strike_inside': float(np.mean([x[k]['strike_inside'] for x in results]))} for k in ('first', 'second')}
    decision = {'difference': diff(results), 'interval': interval, 'strike_inside_drop': strike_drop,
                'switchable': bool(interval[0] > 0 and strike_drop <= .01),
                'rule': 'the share with ink IoU >= 0.9 rises, writer-resampled 95% lower end > 0, and the strike inside falls by at most a point'}
    out = make_output_dir(args.output_dir); (out/'classifier.pkl').write_bytes(pickle.dumps(clf))
    write_json(out/'report.json', {'pieces': len(X), 'kept_share': float(np.mean(y)), 'held_out_regions': len(results), 'writers': len(writers),
                                   'features': FEATURES, 'importances': clf.feature_importances_.tolist(), 'summary': summary, 'decision': decision, 'regions': results})
    print(json.dumps({'pieces': len(X), 'held_out_regions': len(results), 'summary': summary, 'decision': decision}, indent=1), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score')
    for name in ('dataset', 'library', 'output'): s.add_argument('--'+name, type=Path, required=True)
    s.add_argument('--writers', type=Path, help="a prefill set's summary.json: its held-out writers apart")
    s.add_argument('--predicted', action='store_true', help="the library was inferred with --tight: its own polygons (from the model's strike) are scored")
    n = sub.add_parser('learn')
    for name in ('dataset', 'library', 'writers', 'output-dir'): n.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(argv); {'score': score, 'learn': learn}[args.command](args)


if __name__ == '__main__':
    main()
