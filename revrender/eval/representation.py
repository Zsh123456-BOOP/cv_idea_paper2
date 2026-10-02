"""How a mask's representation bounds what it can draw, read on validation.

  grids      how well a mask carried on each model family's grid could draw each gold revision even if every cell were right:
             the revision's gold is averaged into the grid's cells (area averaging) and brought back the way the models bring
             their masks back (bilinear, cut at 0.5), and its IoU with the gold is read --
               image/4                        the whole line at a quarter of its resolution (MaskDINO, DI-MaskDINO, Mask2Former,
                                              SOLOv2, CondInst and RevRender's coarse mask)
               box 28x28                      a square grid in the revision's own (tight, gold) box (Mask R-CNN's mask head)
               box 224x224                    PointRend's grid after its subdivision
               box 14x56, 28x112, 56x224      RevRender's render levels (1:4, as a line)
             per revision, and by stroke-width third
  thickness  each frozen model's delivered masks eroded by 1 px or dilated by 1-3 px (3x3; the model and its point otherwise
             unchanged): component F1 at IoU 0.5 and the pixel IoU of the masks' union, so what the pixel score rewards can be
             read against what the instance score rewards

Nothing is retrained or re-chosen: every point is the frozen one, and the unchanged masks must reproduce the freeze's TP / FP.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ..data.bundle import gold_source_sha, load_bundle
from ..data.gold import CANVAS
from ..util import digest, provenance, require, to_jsonable
from . import matching
from .fates import delivered, greedy, iou_matrix
from .freeze import read_frozen

SCHEMA = 'revrender-representation-v1'
GRIDS = {'image/4': ('image', 24, 192), 'box 28x28': ('box', 28, 28), 'box 224x224': ('box', 224, 224),
         'box 14x56': ('box', 14, 56), 'box 28x112': ('box', 28, 112), 'box 56x224': ('box', 56, 224)}
STEPS = (-1, 0, 1, 2, 3)
IOU_CUTS = (.5, .75)
WIDTHS = ('thin', 'medium', 'thick')


def stroke_width(component):
    """Mean stroke width: area over skeleton length (at least one pixel)."""
    from skimage.morphology import skeletonize
    component = np.array(component, bool)          # bundle masks are read-only memory maps
    return float(component.sum())/max(int(np.count_nonzero(skeletonize(component))), 1)


def width_edges(widths):
    """The two stroke widths that cut the revisions into thirds."""
    return [float(v) for v in np.quantile(np.asarray(widths), [1/3, 2/3])]


def width_third(width, edges):
    return WIDTHS[sum(width > e for e in edges)]


def tight_box(mask):
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max())+1, int(xs.min()), int(xs.max())+1


def through_grid(mask, kind, grid_height, grid_width):
    """The mask averaged into a grid (over the whole line, or over its tight box) and brought back bilinearly, cut at 0.5."""
    y0, y1, x0, x1 = (0, mask.shape[0], 0, mask.shape[1]) if kind == 'image' else tight_box(mask)
    crop = np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=np.float32)
    small = Image.fromarray(crop, mode='F').resize((grid_width, grid_height), Image.BOX)
    back = np.asarray(small.resize((x1-x0, y1-y0), Image.BILINEAR))
    out = np.zeros(mask.shape, bool)
    out[y0:y1, x0:x1] = back >= .5
    return out


def iou(a, b):
    union = float((a | b).sum())
    return float((a & b).sum())/union if union else 0.


def grid_reading(truths, widths, edges):
    """Per grid: mean IoU and the shares at 0.5 / 0.75 over every gold revision and by width third; and the per-revision IoUs."""
    per = {name: [] for name in GRIDS}
    for truth in truths:
        for component in truth:
            for name, (kind, grid_height, grid_width) in GRIDS.items():
                per[name].append(iou(through_grid(component, kind, grid_height, grid_width), component))
    thirds = [width_third(w, edges) for w in widths]
    out = {}
    for name, values in per.items():
        entry = {'mean_iou': sum(values)/len(values)}
        for cut in IOU_CUTS:
            entry[f'share_iou_ge_{cut}'] = sum(v >= cut for v in values)/len(values)
        entry['by_width'] = {}
        for third in WIDTHS:
            vs = [v for v, k in zip(values, thirds) if k == third]
            entry['by_width'][third] = {'n': len(vs), 'mean_iou': sum(vs)/len(vs),
                                        **{f'share_iou_ge_{cut}': sum(v >= cut for v in vs)/len(vs) for cut in IOU_CUTS}}
        out[name] = entry
    return out, per


def reshaped(mask, step):
    """The mask eroded by ``-step`` pixels (step < 0) or dilated by ``step`` pixels (step > 0), 3x3."""
    if step == 0:
        return mask
    if step > 0:
        return ndimage.binary_dilation(mask, np.ones((3, 3), bool), iterations=step)
    return ndimage.binary_erosion(mask, np.ones((3, 3), bool), iterations=-step)


def read_thickness(job):
    """One frozen model at every thickness step: TP / FP at IoU 0.5 and the pixel counts of the masks' union."""
    label, path, bundle_sha, source_sha, order, truths = job
    provide, _, point = read_frozen(path, bundle_sha, source_sha)
    totals = {s: {'tp': 0, 'fp': 0, 'inter': 0, 'union': 0, 'pred': 0, 'gold': 0} for s in STEPS}
    for line, truth in zip(order, truths):
        masks, scores = provide(line)
        chosen = delivered(masks, np.asarray(scores, np.float64), point)
        gold_union = np.logical_or.reduce(truth) if truth else np.zeros(CANVAS, bool)
        for s in STEPS:
            shaped = [m for m in (reshaped(masks[i], s) for i in chosen) if m.any()]     # a mask that vanishes is not delivered
            won = greedy(iou_matrix(shaped, truth), range(len(shaped)), range(len(truth)), .5) if shaped and truth else {}
            union = np.logical_or.reduce(shaped) if shaped else np.zeros(CANVAS, bool)
            t = totals[s]
            t['tp'] += len(won)
            t['fp'] += len(shaped)-len(won)
            t['inter'] += int((union & gold_union).sum())
            t['union'] += int((union | gold_union).sum())
            t['pred'] += int(union.sum())
            t['gold'] += int(gold_union.sum())
    gold = sum(map(len, truths))
    if point['expected'] is not None:
        require((totals[0]['tp'], totals[0]['fp']) == (point['expected']['true_positive'], point['expected']['false_positive']),
                f'{label}: the unchanged masks do not reproduce the freeze')
    return label, {str(s): {'f1': 2*t['tp']/(t['tp']+t['fp']+gold), 'true_positive': t['tp'], 'false_positive': t['fp'],
                            'pixel_iou': t['inter']/t['union'], 'pixel_precision': t['inter']/t['pred'] if t['pred'] else None,
                            'pixel_recall': t['inter']/t['gold']} for s, t in totals.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--freeze', action='append', default=[], help='label=freeze.json: read the thickness of this model too')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'create-only output exists', FileExistsError)
    labels = [e.split('=', 1)[0] for e in args.freeze]
    require(len(set(labels)) == len(labels), 'duplicate freeze label')
    source = gold_source_sha(args.bundle, args.bundle_sha)
    bundle = load_bundle(args.bundle, args.bundle_sha)
    dev, order, truths = matching.validation_order(bundle['dev'])
    truths = [[np.array(g, bool) for g in truth] for truth in truths]
    widths = [stroke_width(g) for truth in truths for g in truth]
    edges = width_edges(widths)
    grids, per = grid_reading(truths, widths, edges)
    for name, entry in grids.items():
        print(json.dumps({'grid': name, 'mean_iou': round(entry['mean_iou'], 3), 'iou>=.5': round(entry['share_iou_ge_0.5'], 3),
                          'iou>=.75': round(entry['share_iou_ge_0.75'], 3)}), flush=True)
    thickness = {}
    if args.freeze:
        jobs = [(label, path, args.bundle_sha, source, order, truths) for label, path in (e.split('=', 1) for e in args.freeze)]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            for label, reading in pool.map(read_thickness, jobs):
                thickness[label] = reading
                print(json.dumps({'thickness': label, **{s: [round(r['f1'], 3), round(r['pixel_iou'], 3)] for s, r in reading.items()}}),
                      flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps({'schema': SCHEMA, 'test_touched': False, 'bundle_sha256': args.bundle_sha,
                            'gold_revisions': len(widths), 'width_edges': edges, 'grids': grids,
                            'grid_definitions': {k: list(v) for k, v in GRIDS.items()}, 'per_revision_iou': per,
                            'widths': widths, 'thickness_steps_px': list(STEPS), 'thickness': thickness,
                            'provenance': provenance(to_jsonable(args), {str(args.bundle): args.bundle_sha, **{
                                e.split('=', 1)[1]: digest(e.split('=', 1)[1]) for e in args.freeze}})},
                           indent=1, sort_keys=True)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
