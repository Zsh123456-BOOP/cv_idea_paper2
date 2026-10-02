"""COCO-style mask AP, read beside the component F1.

Per line, detections in descending score each take the best still-unmatched gold at or above the IoU threshold; the
curve over all lines is made monotone and sampled at 101 recall points; at most 100 detections per line.  AP is the
mean over IoU 0.50:0.05:0.95; AP50 and AP75 are the single thresholds.  The detections are the model's masks after its
own frozen mask NMS (``AP``, ``AP50``, ``AP75``) or with no NMS (``*_raw``).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..data.bundle import gold_source_sha, load_bundle
from ..util import digest, provenance, require, to_jsonable
from . import ink as ink_view
from . import matching
from .fates import iou_matrix
from .freeze import read_frozen

SCHEMA = 'hme-revision-mask-ap-v1'
IOUS = tuple(round(.5+.05*i, 2) for i in range(10))
MAX_DETS = 100


def match_line(iou, scores, threshold):
    """COCO's per-image matching: detections in descending score take the best unmatched gold with IoU >= threshold."""
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:MAX_DETS]
    taken, out = set(), []
    for i in order:
        best, best_iou = None, threshold-1e-12
        for g in range(iou.shape[1]):
            if g not in taken and iou[i, g] >= best_iou and iou[i, g] >= threshold:
                best, best_iou = g, iou[i, g]
        if best is not None:
            taken.add(best)
        out.append((float(scores[i]), best is not None))
    return out


def average_precision(records, gold):
    """COCO 101-point interpolated AP from (score, is_tp) over all lines."""
    if not gold:
        return 0.
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records], dtype=np.float64)
    fp = np.cumsum([not r[1] for r in records], dtype=np.float64)
    recall = tp/gold
    precision = tp/np.maximum(tp+fp, np.finfo(np.float64).eps)
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])
    points = np.linspace(0, 1, 101)
    index = np.searchsorted(recall, points, side='left')
    sampled = [precision[i] if i < len(precision) else 0. for i in index]
    return float(np.mean(sampled))


def mask_ap(provide, order, truths, nms, inks=None):
    """AP, AP50, AP75 and the per-IoU values at one NMS level (``nms`` = 1.0 reads the raw masks).

    ``inks`` (one ink map per line, eval.ink) reads them on ink alone: the masks are kept by the NMS on the full masks, then both
    they and the gold are cut to ink, and the gold that has no ink is not counted.
    """
    gold = 0
    records = {t: [] for t in IOUS}
    for k, (line, truth) in enumerate(zip(order, truths)):
        masks, scores = provide(line)
        keep = [i for i in matching.suppressed(masks, scores, nms) if masks[i].any()]
        shown = [masks[i] for i in keep]
        if inks is not None:
            shown, truth, _ = ink_view.restrict(shown, truth, inks[k])
        gold += len(truth)
        if not keep:
            continue
        iou = iou_matrix(shown, truth)
        for t in IOUS:
            records[t] += match_line(iou, [scores[i] for i in keep], t)
    values = {str(t): average_precision(records[t], gold) for t in IOUS}
    return {'AP': sum(values.values())/len(values), 'AP50': values['0.5'], 'AP75': values['0.75'], 'by_iou': values}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--freeze', action='append', required=True, help='label=freeze.json')
    parser.add_argument('--bundle', type=Path, required=True, help='the validation bundle (gold v2)')
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--ink-only', action='store_true', help='also read AP on ink alone (the gold and the masks cut to ink)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'create-only output exists', FileExistsError)
    source = gold_source_sha(args.bundle, args.bundle_sha)
    bundle = load_bundle(args.bundle, args.bundle_sha)
    dev, order, truths = matching.validation_order(bundle['dev'])
    inks = ink_view.ink_maps(dev) if args.ink_only else None
    gold = sum(map(len, truths))
    report = {}
    for entry in args.freeze:
        label, path = entry.split('=', 1)
        provide, _, point = read_frozen(path, args.bundle_sha, source)
        kept = mask_ap(provide, order, truths, point['nms'])
        raw = mask_ap(provide, order, truths, 1.)
        report[label] = {'model': point['model'], 'mask_threshold': point['mask_threshold'], 'nms': point['nms'],
                         'AP': kept['AP'], 'AP50': kept['AP50'], 'AP75': kept['AP75'],
                         'AP_raw': raw['AP'], 'AP50_raw': raw['AP50'],
                         'by_iou': {'nms': kept['by_iou'], 'raw': raw['by_iou']}, 'freeze_sha256': digest(path)}
        if args.ink_only:
            report[label]['ink'] = mask_ap(provide, order, truths, point['nms'], inks)
        print(json.dumps({'label': label, **{k: round(report[label][k], 4) for k in ('AP', 'AP50', 'AP75', 'AP_raw', 'AP50_raw')},
                          **({'ink_AP': round(report[label]['ink']['AP'], 4)} if args.ink_only else {})}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps({'schema': SCHEMA, 'gold_components': gold, 'models': report, 'test_touched': False,
                            'bundle_sha256': args.bundle_sha, **({'ink_below': ink_view.INK_BELOW} if args.ink_only else {}),
                            'provenance': provenance(to_jsonable(args), {str(args.bundle): args.bundle_sha, **{
                                e.split('=', 1)[1]: digest(e.split('=', 1)[1]) for e in args.freeze}})},
                           indent=1, sort_keys=True)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
