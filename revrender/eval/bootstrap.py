"""Paired writer bootstrap: confidence intervals for the differences between frozen models on the validation split.

Each variant is read at its own frozen point (mask threshold, NMS and score cut), per line:

  component  TP, FP and gold count at IoU 0.50
  pixels     intersection, predicted and gold pixels of the kept union
  AP         the per-line COCO records (score, match at IoU 0.50:0.05:0.95) after the frozen NMS, <= 100 per line

The writers are resampled with replacement (seed 42, 5,000 draws by default); every variant gets the same draw, so each
named pair's difference is paired.  Reported per pair and metric: the full-sample difference, the 2.5 / 97.5
percentiles and the share of draws in which the difference is not above zero.  Before any draw the per-line totals must
reproduce each freeze's TP, FP and pixel IoU exactly.

A variant is ``label=freeze:<freeze.json>``.  Nothing here chooses or changes an operating point.
"""
import argparse
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..data.bundle import load_bundle, manifest_at
from ..util import SEED, digest, require
from . import ap as ap_module
from . import matching
from .fates import iou_matrix

SCHEMA = 'hme-revision-writer-bootstrap-v1'
METRICS = ('f1', 'AP', 'AP50', 'AP75', 'pixel_iou')
POINTS = 101


def parse_variant(text):
    label, _, spec = text.partition('=')
    kind, _, path = spec.partition(':')
    require(label and kind == 'freeze' and path, 'variant must be label=freeze:<path>')
    return label, kind, Path(path)


def variant_point(path, bundle_sha, source_sha=None):
    """The archive, how to read it, and the frozen point of one variant."""
    record = json.loads(path.read_text())
    require(record.get('test_touched') is False, f'{path}: not a validation record')
    on_source = source_sha is not None and record['bundle_sha256'] == source_sha
    require(record['bundle_sha256'] == bundle_sha or on_source, f'{path}: frozen on another bundle')
    best = record['validation']['own_best']
    return {'archive': record['validation_archive'], 'packed': record['model'] == 'point_context',
            'mask_threshold': float(record['mask_threshold']), 'nms': float(best['nms']),
            'score_threshold': float(best['score_threshold']), 'expected': None if on_source else best}


def line_counts(masks, scores, truth, point):
    """One line at one frozen point: component, pixel and AP records."""
    keep = matching.suppressed(masks, scores, point['nms'])
    matched, kept, ordered = matching.prefix_outcomes([masks[i] for i in keep],
                                                      np.array([float(scores[i]) for i in keep]), truth)
    admitted = int(np.sum(np.asarray(ordered, np.float64) >= point['score_threshold']))
    tp, fp = int(matched[admitted]), int(kept[admitted]-matched[admitted])
    union = None
    for i in keep:
        if float(scores[i]) >= point['score_threshold']:
            union = masks[i] if union is None else np.logical_or(union, masks[i])
    inter = predicted = gold_pixels = 0
    if union is not None or truth:
        shape = truth[0].shape if truth else union.shape
        union = np.zeros(shape, bool) if union is None else union
        gold = np.zeros(shape, bool)
        for component in truth:
            gold |= component
        inter, predicted, gold_pixels = int((union & gold).sum()), int(union.sum()), int(gold.sum())
    nonempty = [i for i in keep if masks[i].any()]
    if nonempty:
        iou = iou_matrix([masks[i] for i in nonempty], truth)
        records = [ap_module.match_line(iou, [scores[i] for i in nonempty], t) for t in ap_module.IOUS]
        det_scores = np.array([s for s, _ in records[0]], np.float64)
        det_tp = np.array([[hit for _, hit in r] for r in records], bool).T
    else:
        det_scores, det_tp = np.zeros(0), np.zeros((0, len(ap_module.IOUS)), bool)
    return {'tp': tp, 'fp': fp, 'gold': len(truth), 'inter': inter, 'pred': predicted, 'gold_px': gold_pixels,
            'scores': det_scores, 'hits': det_tp}


def read_variant(job):
    """Every validation line of one variant; runs in a worker process."""
    label, path, bundle_sha, source_sha, order, truths = job
    point = variant_point(path, bundle_sha, source_sha)
    if point['packed']:
        provide, available = matching.packed_provider(Path(point['archive']), point['mask_threshold'])
    else:
        provide, available = matching.native_provider(Path(point['archive']))
    require(set(order) <= available, f'{label}: archive does not cover validation')
    lines = []
    for line, truth in zip(order, truths):
        masks, scores = provide(line)
        lines.append(line_counts(masks, np.asarray(scores, np.float64), truth, point))
    tp, fp = sum(x['tp'] for x in lines), sum(x['fp'] for x in lines)
    inter, pred, gold_px = (sum(x[k] for x in lines) for k in ('inter', 'pred', 'gold_px'))
    expected = point['expected']
    if expected is not None:
        require((tp, fp) == (expected['true_positive'], expected['false_positive']),
                f'{label}: per-line totals {tp}/{fp} are not the freeze {expected["true_positive"]}/{expected["false_positive"]}')
        require(abs(inter/(pred+gold_px-inter)-expected['pixel_iou']) < 1e-12, f'{label}: pixel IoU does not reproduce')
    records = [[] for _ in ap_module.IOUS]
    for x in lines:
        for k in range(len(ap_module.IOUS)):
            records[k] += list(zip(x['scores'].tolist(), x['hits'][:, k].tolist()))
    gold = sum(x['gold'] for x in lines)
    full_ap = [ap_module.average_precision(r, gold) for r in records]
    return label, lines, {'point': {k: point[k] for k in ('mask_threshold', 'nms', 'score_threshold')},
                          'archive': point['archive'], 'source': str(path), 'source_sha256': digest(path),
                          'full_ap': full_ap, 'reproduced_freeze': expected is not None}


def interpolated_ap(scores, hits, gold):
    """The 101-point AP for every IoU column at once; ties keep the given order."""
    if not gold or not len(scores):
        return np.zeros(hits.shape[1])
    order = np.argsort(-scores, kind='stable')
    hits = hits[order]
    tp = np.cumsum(hits, axis=0, dtype=np.float64)
    fp = np.cumsum(~hits, axis=0, dtype=np.float64)
    recall = tp/gold
    precision = tp/np.maximum(tp+fp, np.finfo(np.float64).eps)
    precision = np.maximum.accumulate(precision[::-1], axis=0)[::-1]
    grid = np.linspace(0, 1, POINTS)
    out = np.empty(hits.shape[1])
    for k in range(hits.shape[1]):
        index = np.searchsorted(recall[:, k], grid, side='left')
        out[k] = np.where(index < len(recall), precision[np.minimum(index, len(recall)-1), k], 0.).mean()
    return out


def by_writer(lines, writers, names):
    """Per-writer sums and detection arrays of one variant."""
    out = []
    for name in names:
        rows = [x for x, w in zip(lines, writers) if w == name]
        out.append({'tp': sum(x['tp'] for x in rows), 'fp': sum(x['fp'] for x in rows), 'gold': sum(x['gold'] for x in rows),
                    'inter': sum(x['inter'] for x in rows), 'pred': sum(x['pred'] for x in rows),
                    'gold_px': sum(x['gold_px'] for x in rows),
                    'scores': np.concatenate([x['scores'] for x in rows]) if rows else np.zeros(0),
                    'hits': np.concatenate([x['hits'] for x in rows]) if rows else np.zeros((0, 10), bool)})
    return out


def metrics_of(groups, draw):
    """The five metrics of one variant on one draw of writer indices."""
    tp = sum(groups[i]['tp'] for i in draw)
    fp = sum(groups[i]['fp'] for i in draw)
    gold = sum(groups[i]['gold'] for i in draw)
    inter = sum(groups[i]['inter'] for i in draw)
    union = sum(groups[i]['pred']+groups[i]['gold_px'] for i in draw)-inter
    aps = interpolated_ap(np.concatenate([groups[i]['scores'] for i in draw]),
                          np.concatenate([groups[i]['hits'] for i in draw]), gold)
    return {'f1': 2*tp/(tp+fp+gold) if tp+fp+gold else 0., 'AP': float(aps.mean()), 'AP50': float(aps[0]),
            'AP75': float(aps[5]), 'pixel_iou': inter/union if union else 0.}


def summarise(full, draws):
    """Full-sample difference, percentile interval and the share of draws at or below zero."""
    values = np.asarray(draws)
    return {'difference': full, 'ci95': [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            'share_not_above_zero': float(np.mean(values <= 0)), 'draws': len(values)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--bundle', type=Path, required=True, help='the validation bundle (gold v2)')
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--variant', action='append', required=True, help='label=freeze:<path>')
    parser.add_argument('--pair', action='append', required=True, help='A-B: the difference A minus B')
    parser.add_argument('--resamples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=SEED, choices=(SEED,))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'create-only output exists', FileExistsError)
    variants = [parse_variant(v) for v in args.variant]
    labels = [v[0] for v in variants]
    require(len(set(labels)) == len(labels), 'duplicate variant label')
    pairs = [tuple(p.split('-', 1)) for p in args.pair]
    require(all(len(p) == 2 and p[0] in labels and p[1] in labels and p[0] != p[1] for p in pairs),
            'a pair names an unknown variant')
    require(args.resamples >= 100, 'too few resamples')
    manifest = manifest_at(args.bundle, args.bundle_sha)
    bundle = load_bundle(args.bundle, args.bundle_sha)
    dev = sorted(bundle['dev'], key=lambda r: not r['is_positive'])
    require(not any(r.get('split') == 'test' for r in dev), 'a test row reached the validation split')
    order = [r['sample_id'].split(':', 1)[1] for r in dev]
    truths = [list(r['instances']) for r in dev]
    writers = [r['group_uuid'] for r in dev]
    source_sha = (manifest.get('gold_v2') or {}).get('from_bundle_sha256')
    names = sorted(set(writers))
    jobs = [(label, path, args.bundle_sha, source_sha, order, truths) for label, _, path in variants]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        read = {label: (lines, info) for label, lines, info in pool.map(read_variant, jobs)}
    groups = {label: by_writer(read[label][0], writers, names) for label in labels}
    everyone = list(range(len(names)))
    full = {label: metrics_of(groups[label], everyone) for label in labels}
    for label in labels:
        reference = read[label][1]['full_ap']
        require(abs(full[label]['AP']-float(np.mean(reference))) < 1e-3 and abs(full[label]['AP50']-reference[0]) < 1e-3,
                f'{label}: the vectorised AP does not reproduce the reference AP')
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(names), size=(args.resamples, len(names)))
    per_draw = {label: [] for label in labels}
    for draw in draws:
        for label in labels:
            per_draw[label].append(metrics_of(groups[label], draw))
    result_pairs = {}
    for a, b in pairs:
        result_pairs[f'{a}-{b}'] = {m: summarise(full[a][m]-full[b][m], [x[m]-y[m] for x, y in zip(per_draw[a], per_draw[b])])
                                    for m in METRICS}
    intervals = {label: {m: [float(np.percentile([x[m] for x in per_draw[label]], q)) for q in (2.5, 97.5)] for m in METRICS}
                 for label in labels}
    result = {'schema': SCHEMA, 'bundle_sha256': args.bundle_sha, 'lines': len(order), 'writers': len(names),
              'gold_components': sum(map(len, truths)), 'resamples': args.resamples, 'seed': args.seed,
              'variants': {label: {'full_sample': full[label], 'ci95': intervals[label],
                                   'full_ap_order': read[label][1]['full_ap'],
                                   **{k: v for k, v in read[label][1].items() if k != 'full_ap'}} for label in labels},
              'pairs': result_pairs, 'test_touched': False, 'python': platform.python_version(), 'numpy': np.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(result, indent=1, sort_keys=True)+'\n')
    for name, pair in result_pairs.items():
        print(json.dumps({'pair': name, **{m: [round(pair[m]['difference'], 4), [round(v, 4) for v in pair[m]['ci95']],
                                               round(pair[m]['share_not_above_zero'], 4)] for m in METRICS}},
                         ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
