"""Where restorers go wrong on the fixed validation regions, and how real strikes differ from the synthetic ones they learn on.

    python -m revrender.eval.restoration_errors zones --evaluation EVAL --output OUT.json
    python -m revrender.eval.restoration_errors strokes --review-set SET --pairs PAIRS [--sample 1000] --output OUT.json

``zones`` reads a unified evaluation (``eval.unified_restoration``: its frozen pixel cases and each model's restored crops) and splits
the errors by where they fall: a strike pixel left in place next to writing (within ``NEAR`` pixels: where a strike crosses or
touches a character) or away from it; a writing pixel wrongly erased next to the strike or away from it.  It also splits each case by
how the box remover's strip scales it (``data.gold.CANVAS``: enlarged or shrunk), the one input the restorers do not share, and gives
every model's read-back per validation strike beside the scan's, and how whole each strike is erased (``fragments``: of the strike's
components of at least ``PIECE`` ink pixels, the pixel shares in ones mostly erased (more than 80%), partly (20 to 80%) and barely).

``strokes`` measures each strike of the human regions (``data.review_set``'s native training and validation crops) and of a seeded
sample of the synthetic pairs (``data.restoration_pairs``: the pixels the strike darkened by more than ``REMOVED``): its width and the
writing's (twice the distance to the background along the ridge of each mask), the width ratio, the share of its pixels touching
writing, how much darker it is than the writing, its length in heights of the writing, and its straightness (the minor over the major
axis of its pixels).  The ratios are scale-free, so a pair drawn on an enlarged host compares with a native crop -- except the share
touching writing, counted within one pixel (an enlarged host shows less of it).  Nothing is trained.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..data.gold import CANVAS
from ..data.restoration_pairs import REMOVED
from ..util import SEED, digest, require, write_json, provenance

NEAR = 2            # pixels: a strike pixel this close to writing touches or crosses it
PIECE = 15          # ink pixels: the smallest strike component the fragment shares count


def removed_of(image, restored):
    return ((restored.astype(np.int16)-image) > REMOVED) & (image < 200)


def zone_counts(image, gold, domain, restored):
    ink = (image < 200) & domain; stroke = gold.astype(bool) & ink; text = ink & ~stroke
    near_text = ndimage.binary_dilation(text, iterations=NEAR); near_stroke = ndimage.binary_dilation(stroke, iterations=NEAR)
    removed = removed_of(image, restored) & domain
    parts = {'stroke_near_writing': stroke & near_text, 'stroke_away': stroke & ~near_text,
             'writing_near_stroke': text & near_stroke, 'writing_away': text & ~near_stroke}
    # left in place for strike pixels, erased for writing pixels: both are errors
    return {k: (int(m.sum()), int(((~removed) if k.startswith('stroke') else removed)[m].sum())) for k, m in parts.items()}


def fragment_counts(image, gold, domain, restored):
    """Strike pixels in components (8-connected, at least ``PIECE``) mostly / partly / barely erased."""
    stroke = gold.astype(bool) & (image < 200) & domain; removed = removed_of(image, restored) & domain
    labels, n = ndimage.label(stroke, np.ones((3, 3))); counts = np.zeros(3, np.int64)
    if not n: return counts
    size = np.bincount(labels.ravel(), minlength=n+1); erased = np.bincount(labels.ravel(), removed.ravel(), minlength=n+1)
    for k in range(1, n+1):
        if size[k] < PIECE: continue
        share = erased[k]/size[k]; counts[0 if share > .8 else 1 if share >= .2 else 2] += size[k]
    return counts


def zones(args):
    require(not args.output.exists(), 'output exists')
    root = args.evaluation
    cases = json.loads((root/'inputs/pixels.json').read_text()); archive = np.load(root/'inputs/pixels.npz')
    specs = json.loads((root/'inputs/models.json').read_text())
    from .unified_restoration import row_scores
    reads = row_scores(root, args.splits)
    result = {}
    for spec in specs:
        restored = np.load(root/'runs'/spec['name']/'restored_pixels.npz')
        totals = {}; by_scale = {'enlarged': np.zeros(3, np.int64), 'shrunk': np.zeros(3, np.int64)}; pieces = np.zeros(3, np.int64)
        for k, case in enumerate(cases):
            if case['split'] not in args.splits: continue
            image, gold, domain = archive[f'image_{k}'], archive[f'gold_{k}'], archive[f'domain_{k}']
            for name, (n, wrong) in zone_counts(image, gold, domain, restored[f'image_{k}']).items():
                totals.setdefault(name, [0, 0]); totals[name][0] += n; totals[name][1] += wrong
            pieces += fragment_counts(image, gold, domain, restored[f'image_{k}'])
            ink = (image < 200) & domain; truth = gold.astype(bool) & ink; removed = removed_of(image, restored[f'image_{k}']) & domain
            scale = min(CANVAS[1]/image.shape[1], CANVAS[0]/image.shape[0])
            by_scale['enlarged' if scale >= 1 else 'shrunk'] += [(removed & truth).sum(), (removed & ~truth & ink).sum(), (~removed & truth).sum()]
        f1 = {k: float(2*v[0]/max(1, 2*v[0]+v[1]+v[2])) for k, v in by_scale.items()}
        result[spec['name']] = {'error_rate': {k: v[1]/max(1, v[0]) for k, v in totals.items()}, 'pixels': {k: v[0] for k, v in totals.items()},
                                'f1_by_strip_scale': f1, 'cases_by_strip_scale': {k: sum(1 for c, case in enumerate(cases) if case['split'] in args.splits
                                    and (min(CANVAS[1]/archive[f'image_{c}'].shape[1], CANVAS[0]/archive[f'image_{c}'].shape[0]) >= 1) == (k == 'enlarged'))
                                    for k in by_scale},
                                'fragments': dict(zip(('mostly_erased', 'partly_erased', 'barely_erased'), (pieces/max(1, pieces.sum())).tolist())),
                                'read_back': reads.get(spec['name'], {})}
    result['scan'] = {'read_back': reads.get('scan', {})}
    write_json(args.output, {'definitions': {'near': NEAR, 'removed_above': REMOVED, 'piece': PIECE, 'mostly': .8, 'barely': .2}, 'models': result,
                             'provenance': provenance(config={'evaluation': str(root), 'splits': list(args.splits)}, inputs={str(root/'inputs/models.json'): digest(root/'inputs/models.json')})})
    for name, r in result.items():
        if 'error_rate' in r: print(name, {k: round(100*v, 1) for k, v in r['error_rate'].items()}, {k: round(100*v, 1) for k, v in r['f1_by_strip_scale'].items()})


def width(mask):
    """Twice the mean distance to the background along the ridge of a mask: its stroke width in pixels."""
    if not mask.any(): return None
    distance = ndimage.distance_transform_edt(mask)
    ridge = mask & (distance >= ndimage.maximum_filter(distance, 3))
    return float(2*distance[ridge].mean())


def measure(image, stroke):
    """The measures of one strike (``stroke``: its pixels) over the writing of its crop."""
    ink = image < 200; stroke = stroke & ink; text = ink & ~stroke
    if stroke.sum() < 10 or text.sum() < 10: return None
    w, t = width(stroke), width(text)
    rows = np.flatnonzero(text.any(1)); height = max(1, rows.max()-rows.min()+1)
    ys, xs = np.nonzero(stroke); coords = np.stack([xs, ys]).astype(np.float64); coords -= coords.mean(1, keepdims=True)
    major, minor = sorted(np.linalg.eigvalsh(np.cov(coords)), reverse=True) if len(xs) > 2 else (1., 0.)
    return {'stroke_width': w, 'writing_width': t, 'width_ratio': w/max(t, 1e-6),
            'touching_writing': float((stroke & ndimage.binary_dilation(text)).sum()/stroke.sum()),
            'darker_than_writing': float(np.median(image[text]).astype(float)-np.median(image[stroke])),
            'length_in_heights': float((xs.max()-xs.min()+1)/height), 'straightness': float(np.sqrt(max(minor, 0)/max(major, 1e-6)))}


def summary_of(values):
    out = {}
    for key in values[0]:
        v = np.array([x[key] for x in values], float)
        out[key] = [float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75))]
    return {'strikes': len(values), 'quartiles': out}


def strokes(args):
    require(not args.output.exists(), 'output exists')
    real = []
    for name in ('training_native', 'validation_native'):
        rows = json.loads((args.review_set/f'{name}.json').read_text()); arrays = np.load(args.review_set/f'{name}.npz')
        for k in range(len(rows)):
            m = measure(arrays[f'image_{k}'], arrays[f'mask_{k}'].astype(bool))
            if m: real.append(m)
    records = json.loads((args.pairs/'records.json').read_text())
    x = np.load(args.pairs/'gold_pixels.npy', mmap_mode='r'); y = np.load(args.pairs/'targets.npy', mmap_mode='r')
    chosen = [i for i, r in enumerate(records) if r['split'] == 'train' and not r['clean']]
    chosen = random.Random(SEED).sample(chosen, min(args.sample, len(chosen)))
    synthetic = [m for i in chosen if (m := measure(np.asarray(x[i]), (np.asarray(y[i]).astype(np.int16)-x[i]) > REMOVED))]
    require(real and synthetic, 'nothing to measure')
    result = {'real': summary_of(real), 'synthetic': summary_of(synthetic)}
    write_json(args.output, {**result, 'provenance': provenance(config={'review_set': str(args.review_set), 'pairs': str(args.pairs), 'sample': args.sample},
                             inputs={str(args.review_set/'run_manifest.json'): digest(args.review_set/'run_manifest.json'),
                                     str(args.pairs/'run_manifest.json'): digest(args.pairs/'run_manifest.json')})})
    for key in result['real']['quartiles']:
        print(f"{key:20s} real {result['real']['quartiles'][key][1]:7.2f}  synthetic {result['synthetic']['quartiles'][key][1]:7.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('zones'); p.add_argument('--evaluation', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--splits', nargs='+', default=['val'], help='the pixel cases and strikes counted (val, dev)')
    p = sub.add_parser('strokes')
    for key in ('review-set', 'pairs', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--sample', type=int, default=1000)
    args = parser.parse_args(argv)
    {'zones': zones, 'strokes': strokes}[args.command](args)


if __name__ == '__main__':
    main()
