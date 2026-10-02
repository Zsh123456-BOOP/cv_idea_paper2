"""Read RevRender's dev predictions at a frozen point: TP / FP / FN, F1, AP, AP50, AP75 and pixel IoU, and compare two
prediction archives line by line.

    rvr-report --archive dev_predictions.npz --freeze freeze_rvr.json --bundle <bundle> --bundle-sha <sha>
               [--compare other_dev_predictions.npz]

``--compare`` checks the second archive against the first: same lines in the same order, identical scores, identical
packed masks at every mask threshold; if not, how many lines and bits differ.  This is the inference-equivalence check
of the paper-1 model after the port.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..data.bundle import load_bundle
from ..util import require
from . import ap as ap_module
from . import matching


def read_point(provide, order, truths, point):
    """Counts, F1, pixel IoU and AP at one (NMS, score cut) point of one provider."""
    gold = sum(map(len, truths))
    tables, _ = matching.tabulate_at(provide, order, truths, point['nms'])
    reading = matching.reading(tables, point['score_threshold'], gold)
    reading.update(matching.pixel_iou(provide, order, truths, point))
    reading.update({k: v for k, v in ap_module.mask_ap(provide, order, truths, point['nms']).items() if k != 'by_iou'})
    reading['gold_components'] = gold
    return reading


def compare_archives(first, second):
    """Line-by-line comparison of two packed prediction archives."""
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        out = {'sample_ids_equal': a['sample_ids'].tolist() == b['sample_ids'].tolist(),
               'mask_thresholds_equal': a['mask_thresholds'].tolist() == b['mask_thresholds'].tolist(),
               'mask_shape_equal': a['mask_shape'].tolist() == b['mask_shape'].tolist(),
               'scores_identical': bool(np.array_equal(a['scores'], b['scores'])),
               'packed_masks_identical': bool(np.array_equal(a['packed_masks'], b['packed_masks']))}
        if not out['scores_identical']:
            out['lines_with_different_scores'] = int((a['scores'] != b['scores']).any(1).sum())
            out['max_abs_score_difference'] = float(np.abs(a['scores'].astype(np.float64)-b['scores']).max())
        if not out['packed_masks_identical']:
            different = a['packed_masks'] != b['packed_masks']
            out['lines_with_different_masks'] = int(different.any(axis=(0, 2, 3)).sum())
            out['differing_bytes'] = int(different.sum())
            out['differing_bits'] = int(np.unpackbits(a['packed_masks'] ^ b['packed_masks']).sum())
        out['identical'] = all(out[k] for k in ('sample_ids_equal', 'mask_thresholds_equal', 'mask_shape_equal',
                                                'scores_identical', 'packed_masks_identical'))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--compare', type=Path, help='a second archive to compare against --archive')
    args = parser.parse_args(argv)
    frozen = json.loads(args.freeze.read_text())
    require(frozen['test_touched'] is False and frozen['model'] == 'point_context', 'not a RevRender validation freeze')
    bundle = load_bundle(args.bundle, args.bundle_sha)
    dev, order, truths = matching.validation_order(bundle['dev'])
    provide, available = matching.packed_provider(args.archive, float(frozen['mask_threshold']))
    require(set(order) == available, 'the archive does not cover the validation lines')
    best = frozen['validation']['own_best']
    point = {'nms': float(frozen['nms']), 'score_threshold': float(frozen['score_threshold'])}
    result = read_point(provide, order, truths, point)
    expected = {k: best[k] for k in ('true_positive', 'false_positive', 'false_negative', 'f1', 'pixel_iou')}
    result['matches_freeze'] = all(abs(result[k]-v) < 1e-12 for k, v in expected.items())
    if args.compare:
        result['compared_with'] = {str(args.compare): compare_archives(args.archive, args.compare)}
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result['matches_freeze'] and all(c['identical'] for c in result.get('compared_with', {}).values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
