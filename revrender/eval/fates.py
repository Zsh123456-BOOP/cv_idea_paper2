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
or ``spurious``.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..data.bundle import load_bundle
from ..util import digest, provenance, require, to_jsonable
from . import matching

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


def line_record(masks, scores, truth, point):
    """Gold-side fates and prediction-side kinds of one line at one frozen point."""
    chosen = delivered(masks, scores, point)
    iou = iou_matrix([masks[i] for i in chosen], truth) if chosen and truth else np.zeros((len(chosen), len(truth)))
    won = greedy(iou, range(len(chosen)), range(len(truth)), .5) if chosen and truth else {}
    gold_union = np.logical_or.reduce(truth) if truth else None
    gold = []
    for g, component in enumerate(truth):
        cover = max((float((masks[i] & component).sum())/float(component.sum()) for i in chosen), default=0.)
        best = float(iou[:, g].max()) if chosen else 0.
        fate = 'matched' if g in won else ('extent' if best >= EXTENT_IOU or cover >= EXTENT_COVER else 'missed')
        gold.append({'fate': fate, 'iou': float(iou[won[g], g]) if g in won else best})
    hits = set(won.values())
    predicted = []
    for r, i in enumerate(chosen):
        on_gold = gold_union is not None and float((masks[i] & gold_union).sum()) >= ON_GOLD*float(masks[i].sum())
        predicted.append('true' if r in hits else ('on_gold' if on_gold else 'spurious'))
    return {'tp': len(won), 'fp': len(chosen)-len(won), 'gold': gold, 'predicted': predicted}


def summarise(records):
    gold = Counter(g['fate'] for r in records for g in r['gold'])
    predicted = Counter(p for r in records for p in r['predicted'])
    total = sum(gold.values())
    located = gold['matched']+gold['extent']
    return {'gold': dict(gold), 'predicted': dict(predicted), 'gold_components': total,
            'found': located/total if total else 0.,
            'drawn_right': gold['matched']/total if total else 0.,
            'drawn_right_of_found': gold['matched']/located if located else 0.,
            'true_positive': sum(r['tp'] for r in records), 'false_positive': sum(r['fp'] for r in records)}


def read_freeze_point(frozen):
    best = frozen['validation']['own_best']
    return {'nms': float(best['nms']), 'score_threshold': float(best['score_threshold'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--freeze', action='append', required=True, help='label=freeze.json')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'create-only output exists', FileExistsError)
    bundle = load_bundle(args.bundle, args.bundle_sha)
    dev, order, truths = matching.validation_order(bundle['dev'])
    report = {}
    for entry in args.freeze:
        label, path = entry.split('=', 1)
        frozen = json.loads(Path(path).read_text())
        require(frozen['test_touched'] is False and frozen['bundle_sha256'] == args.bundle_sha,
                f'{label} is not a validation freeze of this bundle')
        provide, _ = matching.provider_for(frozen['validation_archive'], frozen['model'], frozen['mask_threshold'])
        point = read_freeze_point(frozen)
        records = []
        for line, truth in zip(order, truths):
            masks, scores = provide(line)
            records.append(line_record(masks, np.asarray(scores, np.float64), truth, point))
        summary = summarise(records)
        best = frozen['validation']['own_best']
        require((summary['true_positive'], summary['false_positive']) == (best['true_positive'], best['false_positive']),
                f'{label}: the recount disagrees with its freeze')
        report[label] = dict(summary, freeze_sha256=digest(path))
        print(json.dumps({'label': label, 'found': round(summary['found'], 4), 'drawn_right': round(summary['drawn_right'], 4),
                          'drawn_right_of_found': round(summary['drawn_right_of_found'], 4)}), flush=True)
    with args.output.open('x') as f:
        f.write(json.dumps({'models': report, 'test_touched': False, 'definitions': {
            'extent_iou': EXTENT_IOU, 'extent_cover': EXTENT_COVER, 'on_gold': ON_GOLD},
            'provenance': provenance(to_jsonable(args), {str(args.bundle): args.bundle_sha, **{
                e.split('=', 1)[1]: digest(e.split('=', 1)[1]) for e in args.freeze}})}, indent=1, sort_keys=True)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
