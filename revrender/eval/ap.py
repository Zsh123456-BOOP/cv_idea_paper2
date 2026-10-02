"""COCO-style mask AP, read beside the component F1.

Per line, detections in descending score each take the best still-unmatched gold at or above the IoU threshold; the
curve over all lines is made monotone and sampled at 101 recall points; at most 100 detections per line.  AP is the
mean over IoU 0.50:0.05:0.95; AP50 and AP75 are the single thresholds.  The detections are the model's masks after its
own frozen mask NMS (``AP``, ``AP50``, ``AP75``) or with no NMS (``*_raw``).
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from ..data.bundle import load_bundle, manifest_at
from ..util import digest, require
from . import matching
from .fates import iou_matrix

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


def mask_ap(provide, order, truths, nms):
    """AP, AP50, AP75 and the per-IoU values at one NMS level (``nms`` = 1.0 reads the raw masks)."""
    gold = sum(map(len, truths))
    records = {t: [] for t in IOUS}
    for line, truth in zip(order, truths):
        masks, scores = provide(line)
        keep = [i for i in matching.suppressed(masks, scores, nms) if masks[i].any()]
        if not keep:
            continue
        iou = iou_matrix([masks[i] for i in keep], truth)
        for t in IOUS:
            records[t] += match_line(iou, [scores[i] for i in keep], t)
    values = {str(t): average_precision(records[t], gold) for t in IOUS}
    return {'AP': sum(values.values())/len(values), 'AP50': values['0.5'], 'AP75': values['0.75'], 'by_iou': values}


def read_freeze(path, bundle_sha, source_sha=None):
    frozen = json.loads(Path(path).read_text())
    require(frozen['test_touched'] is False and frozen['bundle_sha256'] in (bundle_sha, source_sha),
            f'{path} is not a validation freeze of this bundle or its gold-v2 source')
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--freeze', action='append', required=True, help='label=freeze.json')
    parser.add_argument('--bundle', type=Path, required=True, help='the validation bundle (gold v2)')
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'create-only output exists', FileExistsError)
    manifest = manifest_at(args.bundle, args.bundle_sha)
    source = (manifest.get('gold_v2') or {}).get('from_bundle_sha256')
    bundle = load_bundle(args.bundle, args.bundle_sha)
    dev, order, truths = matching.validation_order(bundle['dev'])
    gold = sum(map(len, truths))
    report = {}
    for entry in args.freeze:
        label, path = entry.split('=', 1)
        frozen = read_freeze(path, args.bundle_sha, source)
        provide, _ = matching.provider_for(frozen['validation_archive'], frozen['model'], frozen['mask_threshold'])
        nms = frozen['validation']['own_best']['nms']
        kept = mask_ap(provide, order, truths, nms)
        raw = mask_ap(provide, order, truths, 1.)
        report[label] = {'model': frozen['model'], 'mask_threshold': frozen['mask_threshold'], 'nms': nms,
                         'AP': kept['AP'], 'AP50': kept['AP50'], 'AP75': kept['AP75'],
                         'AP_raw': raw['AP'], 'AP50_raw': raw['AP50'],
                         'by_iou': {'nms': kept['by_iou'], 'raw': raw['by_iou']}, 'freeze_sha256': digest(path)}
        print(json.dumps({'label': label, **{k: round(report[label][k], 4) for k in ('AP', 'AP50', 'AP75', 'AP_raw', 'AP50_raw')}}),
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps({'schema': SCHEMA, 'gold_components': gold, 'models': report, 'test_touched': False,
                            'bundle_sha256': args.bundle_sha, 'python': platform.python_version()},
                           indent=1, sort_keys=True)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
