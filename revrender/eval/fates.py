"""Found and drawn right: what happened to each gold revision at a model's frozen point.

At a frozen point (mask threshold, NMS, score cut) the delivered masks (kept by NMS, at or above the cut, non-empty) are
matched to the gold components one to one at IoU 0.50.  Every gold component then has one fate:

  matched   a delivered mask drew it right (IoU >= 0.50, one to one)
  extent    not matched, but a delivered mask reaches it: IoU >= 0.10, or the mask covers >= 20% of the component
  missed    nothing delivered reaches it

  found                    = (matched + extent) / gold       the model located the revision
  drawn right              = matched / gold                  located and drawn right (IoU >= 0.50); the recall
  drawn right, of found    = matched / (matched + extent)    of those located, the share whose extent is right

Every delivered mask is ``true`` (matched), ``on_gold`` (not matched, but >= 20% of its pixels lie on the gold union)
or ``spurious``.  ``--ink-only`` reads the same fates a second time with the gold and the delivered masks cut to ink
(eval.ink).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..data.bundle import gold_source_sha, load_bundle
from ..util import digest, provenance, require, to_jsonable
from . import ink as ink_view
from . import matching
from .freeze import read_frozen

EXTENT_IOU, EXTENT_COVER, ON_GOLD = .1, .2, .2


def iou_matrix(masks, truth):
    """IoU of every mask against every gold component (rows: masks, columns: gold)."""
    if not masks or not truth:
        return np.zeros((len(masks), len(truth)))
    a = np.stack([m.reshape(-1) for m in masks]).astype(np.float32)
    b = np.stack([t.reshape(-1) for t in truth]).astype(np.float32)
    inter = a @ b.T
    return inter/np.maximum(a.sum(1)[:, None]+b.sum(1)[None]-inter, 1)


def greedy(iou, rows, columns, threshold):
    """IoU-ordered one-to-one matching at or above the threshold: {gold index: mask index}."""
    pairs = sorted(((iou[r, g], r, g) for r in rows for g in columns if iou[r, g] >= threshold),
                   key=lambda t: (-t[0], t[1], t[2]))
    used_r, used_g, out = set(), set(), {}
    for _, r, g in pairs:
        if r not in used_r and g not in used_g:
            used_r.add(r)
            used_g.add(g)
            out[g] = r
    return out


def delivered(masks, scores, point):
    keep = matching.suppressed(masks, scores, point['nms'])
    return [i for i in keep if float(scores[i]) >= point['score_threshold'] and masks[i].any()]


def line_record(masks, scores, truth, point, ink=None):
    """Gold-side fates and prediction-side kinds of one line at one frozen point (``ink``: read on ink alone)."""
    chosen = delivered(masks, scores, point)
    shown = [masks[i] for i in chosen]
    dropped = 0
    if ink is not None:
        shown, truth, dropped = ink_view.restrict(shown, truth, ink)
    iou = iou_matrix(shown, truth) if shown and truth else np.zeros((len(shown), len(truth)))
    won = greedy(iou, range(len(shown)), range(len(truth)), .5) if shown and truth else {}
    gold_union = np.logical_or.reduce(truth) if truth else None
    gold = []
    for g, component in enumerate(truth):
        cover = max((float((m & component).sum())/float(component.sum()) for m in shown), default=0.)
        best = float(iou[:, g].max()) if shown else 0.
        fate = 'matched' if g in won else ('extent' if best >= EXTENT_IOU or cover >= EXTENT_COVER else 'missed')
        gold.append({'fate': fate, 'iou': float(iou[won[g], g]) if g in won else best})
    hits = set(won.values())
    predicted = []
    for r, m in enumerate(shown):
        on_gold = gold_union is not None and float((m & gold_union).sum()) >= ON_GOLD*float(m.sum())
        predicted.append('true' if r in hits else ('on_gold' if on_gold else 'spurious'))
    record = {'tp': len(won), 'fp': len(shown)-len(won), 'gold': gold, 'predicted': predicted}
    if ink is not None:
        record['without_ink'] = {'gold': dropped, 'delivered': sum(not m.any() for m in shown)}
    return record


def summarise(records):
    gold = Counter(g['fate'] for r in records for g in r['gold'])
    predicted = Counter(p for r in records for p in r['predicted'])
    total = sum(gold.values())
    located = gold['matched']+gold['extent']
    tp, fp = sum(r['tp'] for r in records), sum(r['fp'] for r in records)
    out = {'gold': dict(gold), 'predicted': dict(predicted), 'gold_components': total,
           'found': located/total if total else 0.,
           'drawn_right': gold['matched']/total if total else 0.,
           'drawn_right_of_found': gold['matched']/located if located else 0.,
           'true_positive': tp, 'false_positive': fp, 'f1': 2*tp/(tp+fp+total) if tp+fp+total else 0.}
    if any('without_ink' in r for r in records):
        out['without_ink'] = {k: sum(r['without_ink'][k] for r in records) for k in ('gold', 'delivered')}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--freeze', action='append', required=True, help='label=freeze.json')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--ink-only', action='store_true', help='also read the fates on ink alone (the gold and the masks cut to ink)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'create-only output exists', FileExistsError)
    bundle = load_bundle(args.bundle, args.bundle_sha)
    dev, order, truths = matching.validation_order(bundle['dev'])
    inks = ink_view.ink_maps(dev) if args.ink_only else [None]*len(dev)
    source = gold_source_sha(args.bundle, args.bundle_sha)
    report = {}
    for entry in args.freeze:
        label, path = entry.split('=', 1)
        provide, _, point = read_frozen(path, args.bundle_sha, source)
        records, ink_records = [], []
        for line, truth, ink in zip(order, truths, inks):
            masks, scores = provide(line)
            scores = np.asarray(scores, np.float64)
            records.append(line_record(masks, scores, truth, point))
            if args.ink_only:
                ink_records.append(line_record(masks, scores, truth, point, ink))
        summary = summarise(records)
        best = point['expected']
        require(best is None or (summary['true_positive'], summary['false_positive']) == (best['true_positive'], best['false_positive']),
                f'{label}: the recount disagrees with its freeze')
        report[label] = dict(summary, freeze_sha256=digest(path))
        if args.ink_only:
            report[label]['ink'] = summarise(ink_records)
        print(json.dumps({'label': label, 'found': round(summary['found'], 4), 'drawn_right': round(summary['drawn_right'], 4),
                          'drawn_right_of_found': round(summary['drawn_right_of_found'], 4),
                          **({'ink_found': round(report[label]['ink']['found'], 4), 'ink_drawn_right': round(report[label]['ink']['drawn_right'], 4)}
                             if args.ink_only else {})}), flush=True)
    with args.output.open('x') as f:
        f.write(json.dumps({'models': report, 'test_touched': False, 'definitions': {
            'extent_iou': EXTENT_IOU, 'extent_cover': EXTENT_COVER, 'on_gold': ON_GOLD,
            **({'ink_below': ink_view.INK_BELOW} if args.ink_only else {})},
            'provenance': provenance(to_jsonable(args), {str(args.bundle): args.bundle_sha, **{
                e.split('=', 1)[1]: digest(e.split('=', 1)[1]) for e in args.freeze}})}, indent=1, sort_keys=True)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
