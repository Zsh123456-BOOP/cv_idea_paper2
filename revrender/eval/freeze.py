"""Freeze a model's operating point on validation: mask threshold, mask NMS and score cut, chosen once.

RevRender (``rvr-freeze``): the frontier of (NMS, score cut) is read at each mask threshold in 0.3 / 0.4 / 0.5; the best
F1 wins, ties to fewer false positives, then the lower threshold.  Baselines archive binary masks at the threshold their
own upstream code applies; ``rvr-native-freeze`` reads their frontier at that one threshold, and
baselines.native.threshold re-runs a frozen checkpoint at 0.3 / 0.4 / 0.5 and uses the same rule.

The record (schema hme-revision-c832-freeze-v2) is what mask AP, the bootstrap, the found/drawn analysis and the
tables read.  Nothing here reads a test row: the bundle holds none, and a record that says otherwise is refused.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ..data.bundle import load_bundle
from ..util import digest, require, write_json
from . import matching

SCHEMA = 'hme-revision-c832-freeze-v2'
NATIVE_SCHEMA = 'hme-revision-c832-freeze-v1'
CANDIDATES = (.3, .4, .5)
BUDGETS = (41, 55, 65, 70, 83, 111)
RULE = 'chosen on validation among 0.3/0.4/0.5 (user, 2026-09-27)'


def pick(readings):
    """Best F1; ties to fewer false positives, then the lower threshold."""
    return min(readings, key=lambda r: (-r['f1'], r['false_positive'], r['mask_threshold']))


def read_candidates(provider_for_level, order, truths, candidates=CANDIDATES, fewer_false_positives=True):
    """The frontier and the best point at each candidate mask threshold: ([readings], {threshold: frontier}).

    Best = highest F1; ``fewer_false_positives`` breaks F1 ties by false positives (the native freeze of paper 1 did not).
    """
    gold = sum(map(len, truths))
    readings, frontiers = [], {}
    for level in candidates:
        provide, available = provider_for_level(level)
        require(set(order) == available, 'validation archive coverage mismatch')
        points = matching.frontier(provide, order, truths, gold)
        best = (matching.best_point(points) if fewer_false_positives
                else dict(max(points, key=lambda p: p['f1'])))
        best.update(matching.pixel_iou(provide, order, truths, best), mask_threshold=level)
        readings.append(best)
        frontiers[level] = points
    return readings, frontiers


def budget_table(points, budgets=BUDGETS):
    return {str(b): (matching.true_positives_at(points, b) or {}).get('true_positive') for b in budgets}


def freeze_record(model, bundle, bundle_sha, summary, checkpoint, archive, best, readings, frontiers, truths,
                  schema=SCHEMA, rule=RULE, summary_path=None):
    points = frontiers[best['mask_threshold']]
    manifest = bundle['manifest']
    return {'schema': schema, 'split': 'C832', 'model': model, 'candidate': manifest['candidate'],
            'test_touched': False, 'common_test_sample_ids_sha256': manifest.get('common_test_sample_ids_sha256'),
            'bundle_sha256': bundle_sha, 'chosen_step': summary['best_step'], 'checkpoint': str(checkpoint),
            'checkpoint_sha256': digest(checkpoint),
            'selection_record_sha256': digest(summary_path) if summary_path else None,
            'validation_archive': str(archive), 'mask_threshold': best['mask_threshold'], 'mask_threshold_rule': rule,
            'score_threshold': best['score_threshold'], 'nms': best['nms'],
            'validation': {'gold_components': sum(map(len, truths)), 'own_best': best,
                           'by_mask_threshold': {str(r['mask_threshold']): r for r in readings},
                           'tp_within_fp_budget': budget_table(points)},
            'frozen_at': datetime.now(timezone.utc).isoformat()}


def freeze_revrender(bundle_dir, bundle_sha, run, output):
    """Freeze one formal RevRender run (its best.pt and dev_predictions.npz)."""
    require(not Path(output).exists(), 'create-only output exists', FileExistsError)
    run = Path(run)
    summary = json.loads((run/'summary.json').read_text())
    manifest = json.loads((run/'run_manifest.json').read_text())
    require(summary['stage'] == 'formal' and summary['checkpoint_reload_verified'] and (run/'complete.json').is_file(),
            'incomplete formal run')
    require(manifest['profile']['dataset_sha256'] == bundle_sha, 'wrong run dataset')
    checkpoint, archive = run/'best.pt', run/'dev_predictions.npz'
    bundle = load_bundle(bundle_dir, bundle_sha)
    dev, order, truths = matching.validation_order(bundle['dev'])
    readings, frontiers = read_candidates(lambda level: matching.packed_provider(archive, level), order, truths)
    best = pick(readings)
    record = freeze_record('point_context', bundle, bundle_sha, summary, checkpoint, archive, best, readings, frontiers,
                           truths, summary_path=run/'summary.json')
    write_json(output, record)
    return record


def freeze_native(model, bundle_dir, bundle_sha, run, output):
    """Freeze one formal native baseline run at its upstream mask threshold 0.5 (its archive holds binary masks)."""
    require(not Path(output).exists(), 'create-only output exists', FileExistsError)
    run = Path(run)
    summary = json.loads((run/'summary.json').read_text())
    manifest = json.loads((run/'run_manifest.json').read_text())
    require(summary['status'] == 'complete' and not summary['smoke_only'] and (run/'train_complete.json').is_file(),
            'incomplete native formal run')
    require(manifest['method'] == model and summary['method'] == model, 'wrong native model')
    require(manifest['dataset_manifest_sha256'] == bundle_sha, 'wrong native dataset')
    checkpoint, archive = run/'best_model.pth', run/'best_dev_instances'
    bundle = load_bundle(bundle_dir, bundle_sha)
    dev, order, truths = matching.validation_order(bundle['dev'])
    readings, frontiers = read_candidates(lambda level: matching.native_provider(archive), order, truths, candidates=(.5,),
                                         fewer_false_positives=False)
    best = readings[0]
    record = freeze_record(model, bundle, bundle_sha, summary, checkpoint, archive, best, readings, frontiers, truths,
                           schema=NATIVE_SCHEMA, rule='the upstream binarisation at 0.5', summary_path=run/'summary.json')
    write_json(output, record)
    return record


def _arguments(argv, native):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    if native:
        parser.add_argument('--model', required=True, choices=('maskdino', 'pointrend', 'mask2former', 'condinst',
                                                               'maskrcnn', 'solov2', 'dimaskdino'))
    for name in ('bundle', 'run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = _arguments(argv, native=False)
    record = freeze_revrender(args.bundle, args.bundle_sha, args.run, args.output)
    best = record['validation']['own_best']
    print(json.dumps({'mask_threshold': best['mask_threshold'], 'f1': round(best['f1'], 4),
                      'tp_fp': [best['true_positive'], best['false_positive']], 'pixel_iou': round(best['pixel_iou'], 4)}))
    return 0


def main_native(argv=None):
    args = _arguments(argv, native=True)
    record = freeze_native(args.model, args.bundle, args.bundle_sha, args.run, args.output)
    best = record['validation']['own_best']
    print(json.dumps({'model': args.model, 'f1': round(best['f1'], 4), 'tp_fp': [best['true_positive'], best['false_positive']]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
