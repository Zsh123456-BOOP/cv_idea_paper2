"""Re-score frozen models on the validation split: TP / FP / FN, F1, pixel IoU, AP, AP50, AP75 at each model's frozen point,
and compare against recorded numbers and against another archive.

    rvr-report --bundle B --bundle-sha S --freeze rvr=freeze_rvr.json --freeze maskdino=freeze_maskdino.json ...
               [--archive rvr=dev_predictions.npz]          read RevRender's predictions from a new archive, same frozen point
               [--compare rvr=old_dev_predictions.npz]      check that archive against another, line by line
               [--reference-ap ap.json] [--reference-table main_table.json]

Each freeze names the model's archive, mask threshold, mask NMS and score cut.  The reading is recomputed from the archive
and must equal the freeze's own record (``matches_freeze``); AP and the table columns are checked against the reference
files if given.  ``--compare`` checks a prediction archive against another: same lines in the same order, identical scores,
identical packed masks at every mask threshold; if not, how many lines and bits differ.  Together these are the
equivalence checks of the port: the paper-1 model on a new inference run, and the seven baselines on their frozen predictions.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..data.bundle import load_bundle
from ..util import require
from . import ap as ap_module
from . import matching
from .freeze import read_frozen

TABLE_FIELDS = {'component_f1': 'f1', 'true_positive': 'true_positive', 'false_positive': 'false_positive',
                'pixel_iou': 'pixel_iou'}
AP_FIELDS = ('AP', 'AP50', 'AP75', 'AP_raw', 'AP50_raw')


def read_point(provide, order, truths, point):
    """Counts, F1, pixel IoU and AP at one (NMS, score cut) point of one provider."""
    gold = sum(map(len, truths))
    tables, _ = matching.tabulate_at(provide, order, truths, point['nms'])
    reading = matching.reading(tables, point['score_threshold'], gold)
    reading.update(matching.pixel_iou(provide, order, truths, point))
    kept = ap_module.mask_ap(provide, order, truths, point['nms'])
    raw = ap_module.mask_ap(provide, order, truths, 1.)
    reading.update({k: v for k, v in kept.items() if k != 'by_iou'}, AP_raw=raw['AP'], AP50_raw=raw['AP50'])
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


def labelled(values):
    out = {}
    for value in values or []:
        label, _, path = value.partition('=')
        require(label and path, f'expected label=path, got {value!r}')
        out[label] = Path(path)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--freeze', action='append', required=True, help='label=freeze.json (repeatable)')
    parser.add_argument('--archive', action='append', help="label=path: read RevRender's predictions from this archive")
    parser.add_argument('--compare', action='append', help='label=path: compare the archive against this one')
    parser.add_argument('--reference-ap', type=Path, help='a mask-AP record: models.<label>.AP / AP50 / AP75 / AP_raw / AP50_raw')
    parser.add_argument('--reference-table', type=Path, help='a table record: rows[].model with component_f1, TP, FP, pixel_iou')
    args = parser.parse_args(argv)
    freezes, archives, compares = labelled(args.freeze), labelled(args.archive), labelled(args.compare)
    reference_ap = json.loads(args.reference_ap.read_text())['models'] if args.reference_ap else {}
    reference_rows = ({r['model']: r for r in json.loads(args.reference_table.read_text())['rows']}
                      if args.reference_table else {})
    bundle = load_bundle(args.bundle, args.bundle_sha)
    dev, order, truths = matching.validation_order(bundle['dev'])
    report, failed = {}, False
    for label, path in freezes.items():
        provide, available, point = read_frozen(path, args.bundle_sha, archive=archives.get(label))
        require(set(order) <= available, f'{label}: the archive does not cover the validation lines')
        result = read_point(provide, order, truths, point)
        best, archive = point['expected'], point['archive']
        checks = {'matches_freeze': all(abs(result[k]-best[k]) < 1e-12 for k in
                                        ('true_positive', 'false_positive', 'false_negative', 'f1', 'pixel_iou'))}
        if label in reference_ap:
            checks['matches_reference_ap'] = all(abs(result[k]-reference_ap[label][k]) < 1e-9 for k in AP_FIELDS)
        if label in reference_rows:
            checks['matches_reference_table'] = all(abs(result[field]-reference_rows[label][name]) < 1e-9
                                                    for name, field in TABLE_FIELDS.items())
        if label in compares:
            checks['archive_identical'] = compare_archives(archive, compares[label])
        report[label] = {'reading': result, 'checks': checks}
        failed |= not all((v['identical'] if isinstance(v, dict) else v) for v in checks.values())
        print(json.dumps({'label': label, 'tp_fp_fn': [result['true_positive'], result['false_positive'], result['false_negative']],
                          'f1': round(result['f1'], 4), 'AP': round(result['AP'], 4), 'AP75': round(result['AP75'], 4),
                          'pixel_iou': round(result['pixel_iou'], 4), **{k: (v['identical'] if isinstance(v, dict) else v)
                                                                         for k, v in checks.items()}}), flush=True)
    print(json.dumps({'all_checks_pass': not failed, 'models': sorted(report)}))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
