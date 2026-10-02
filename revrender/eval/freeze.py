"""Freeze a model's operating point on validation: mask threshold, mask NMS and score cut, chosen once.

RevRender (``rvr-freeze``): the frontier of (NMS, score cut) is read at each mask threshold in 0.3 / 0.4 / 0.5; the best
F1 wins, ties to fewer false positives, then the lower threshold.  Baselines archive binary masks at the threshold their
own upstream code applies; ``rvr-freeze --model <name>`` reads their frontier at that one threshold, and
baselines.native.threshold re-runs a frozen checkpoint at 0.3 / 0.4 / 0.5 and uses the same rule.

The record (schema revrender-freeze-v2; paper-1's records read the same) is what mask AP, the bootstrap, the found/drawn analysis and the
tables read.  Nothing here reads a test row: the bundle holds none, and a record that says otherwise is refused.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ..data.bundle import load_bundle
from ..util import digest, provenance, require, schema_is, write_json
from . import matching

SCHEMA = 'revrender-freeze-v2'                  # written by this package
NATIVE_SCHEMA = 'revrender-freeze-native-v1'
FREEZE_SCHEMAS = (SCHEMA, NATIVE_SCHEMA, r'hme-revision-c\d+-freeze-v[12]')   # accepted when reading (paper-1 records included)
CANDIDATES = (.3, .4, .5)
BUDGETS = (41, 55, 65, 70, 83, 111)
NATIVE_FREEZE_SCHEMAS = (NATIVE_SCHEMA, r'hme-revision-c\d+-freeze-v1')
RULE = 'chosen on validation among 0.3/0.4/0.5 (user, 2026-09-27)'


def frozen_point(path, bundle_sha, source_sha=None, archive=None):
    """What a validation freeze tells a reader: the archive, the model, the mask threshold, the NMS level and score cut, and
    ``expected``, the freeze's own reading, which a recount must reproduce (None for a freeze made on the gold-v2 source
    bundle, whose lines differ).  ``archive`` names another archive of the same model, read at the same point;
    ``bundle_sha=None`` skips the check against a bundle (a reader that only needs the point, like revrender.pages)."""
    record = json.loads(Path(path).read_text())
    on_source = source_sha is not None and record['bundle_sha256'] == source_sha
    require(record.get('test_touched') is False and (bundle_sha is None or record['bundle_sha256'] == bundle_sha or on_source),
            f'{path}: not a validation freeze of this bundle')
    best = record['validation']['own_best']
    return {'archive': str(archive or record['validation_archive']), 'model': record['model'],
            'mask_threshold': float(record['mask_threshold']), 'nms': float(best['nms']),
            'score_threshold': float(best['score_threshold']), 'expected': None if on_source else best}


def read_frozen(path, bundle_sha, source_sha=None, archive=None):
    """(provider, line ids the archive covers, point) of a validation freeze."""
    point = frozen_point(path, bundle_sha, source_sha, archive)
    provide, available = matching.provider_for(point['archive'], point['model'], point['mask_threshold'])
    return provide, available, point


def is_freeze(schema, native_only=False):
    """Whether a record's schema tag names a freeze (a native-baseline one at its upstream threshold if native_only)."""
    return any(schema_is(schema, p) for p in (NATIVE_FREEZE_SCHEMAS if native_only else FREEZE_SCHEMAS))


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
                  schema=SCHEMA, rule=RULE, summary_path=None, bundle_dir='', budgets=BUDGETS):
    points = frontiers[best['mask_threshold']]
    manifest = bundle['manifest']
    return {'schema': schema, 'split': 'validation', 'model': model, 'candidate': manifest['candidate'],
            'test_touched': False, 'common_test_sample_ids_sha256': manifest.get('common_test_sample_ids_sha256'),
            'bundle_sha256': bundle_sha, 'chosen_step': summary['best_step'], 'checkpoint': str(checkpoint),
            'checkpoint_sha256': digest(checkpoint),
            'selection_record_sha256': digest(summary_path) if summary_path else None,
            'validation_archive': str(archive), 'mask_threshold': best['mask_threshold'], 'mask_threshold_rule': rule,
            'provenance': provenance({'bundle': str(bundle_dir), 'run': str(Path(checkpoint).parent), 'model': model},
                                     {str(Path(bundle_dir)): bundle_sha, str(checkpoint): digest(checkpoint)}),
            'score_threshold': best['score_threshold'], 'nms': best['nms'],
            'validation': {'gold_components': sum(map(len, truths)), 'own_best': best,
                           'by_mask_threshold': {str(r['mask_threshold']): r for r in readings},
                           'tp_within_fp_budget': budget_table(points, budgets)},
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
                           truths, summary_path=run/'summary.json', bundle_dir=bundle_dir)
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
                           schema=NATIVE_SCHEMA, rule='the upstream binarisation at 0.5', summary_path=run/'summary.json',
                           bundle_dir=bundle_dir, budgets=matching.FALSE_POSITIVE_BUDGETS)
    write_json(output, record)
    return record


def main(argv=None):
    """``--model point_context`` (default) freezes a RevRender run; a native model name freezes a native baseline run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--model', default='point_context', choices=('point_context', 'maskdino', 'pointrend', 'mask2former',
                                                                     'condinst', 'maskrcnn', 'solov2', 'dimaskdino'))
    for name in ('bundle', 'run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    args = parser.parse_args(argv)
    if args.model == 'point_context':
        record = freeze_revrender(args.bundle, args.bundle_sha, args.run, args.output)
    else:
        record = freeze_native(args.model, args.bundle, args.bundle_sha, args.run, args.output)
    best = record['validation']['own_best']
    print(json.dumps({'model': args.model, 'mask_threshold': best['mask_threshold'], 'f1': round(best['f1'], 4),
                      'tp_fp': [best['true_positive'], best['false_positive']], 'pixel_iou': round(best['pixel_iou'], 4)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
