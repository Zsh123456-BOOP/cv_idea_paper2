"""Freeze and re-score on a synthetic bundle: predictions that are the gold, read back through the whole evaluation path."""
import json

import numpy as np
import pytest

from revrender import infer
from revrender.data import bundle as B
from revrender.eval import freeze, matching, report
from revrender.eval.inloop import MASK_THRESHOLDS
from revrender.util import RefusedError, digest, write_json


def make_run(root, bundle, sha, perfect=True):
    data = B.load_bundle(bundle, sha)
    rows = infer.validation_rows(data)
    by_id = {r['sample_id']: r for r in data['dev']}
    bits = np.zeros((len(MASK_THRESHOLDS), len(rows), 32, 96*768//8), np.uint8)
    scores = np.full((len(rows), 32), .05, np.float32)
    for i, row in enumerate(rows):
        for k, component in enumerate(by_id[row['sample_id']]['instances']):
            for t in range(len(MASK_THRESHOLDS)):
                bits[t, i, k] = np.packbits(component.reshape(-1))
            scores[i, k] = .9 if perfect else .2
    root.mkdir()
    infer.save_predictions(root/'dev_predictions.npz', bits, scores, rows)
    (root/'best.pt').write_bytes(b'weights')
    write_json(root/'summary.json', {'stage': 'formal', 'checkpoint_reload_verified': True, 'best_step': 123})
    write_json(root/'run_manifest.json', {'profile': {'dataset_sha256': sha}})
    write_json(root/'complete.json', {'files': {}})
    return root


def test_a_perfect_run_freezes_at_f1_one_and_rescores_identically(bundle, tmp_path):
    path, sha = bundle
    run = make_run(tmp_path/'run', path, sha)
    record = freeze.freeze_revrender(path, sha, run, tmp_path/'freeze.json')
    best = record['validation']['own_best']
    assert best['f1'] == 1.0 and best['false_positive'] == 0 and best['pixel_iou'] == 1.0
    assert record['mask_threshold'] == .3 and record['model'] == 'point_context'            # ties: the lower threshold
    assert record['test_touched'] is False and record['checkpoint_sha256'] == digest(run/'best.pt')
    assert record['provenance']['seed'] == 42 and 'commit' in record['provenance']['git']
    assert record['validation']['gold_components'] == sum(r['instance_count'] for r in B.load_bundle(path, sha)['dev'])
    with pytest.raises(FileExistsError):
        freeze.freeze_revrender(path, sha, run, tmp_path/'freeze.json')
    code = report.main(['--bundle', str(path), '--bundle-sha', sha, '--freeze', f'rvr={tmp_path/"freeze.json"}',
                        '--compare', f'rvr={run/"dev_predictions.npz"}'])
    assert code == 0


def test_a_run_of_another_dataset_or_an_unfinished_run_is_not_frozen(bundle, tmp_path):
    path, sha = bundle
    run = make_run(tmp_path/'run', path, sha)
    write_json(run/'../other.json', {})
    (run/'summary.json').unlink()
    write_json(run/'summary.json', {'stage': 'learning', 'checkpoint_reload_verified': True, 'best_step': 1})
    with pytest.raises(RefusedError, match='incomplete'):
        freeze.freeze_revrender(path, sha, run, tmp_path/'f.json')
    (run/'summary.json').unlink()
    write_json(run/'summary.json', {'stage': 'formal', 'checkpoint_reload_verified': True, 'best_step': 1})
    (run/'run_manifest.json').unlink()
    write_json(run/'run_manifest.json', {'profile': {'dataset_sha256': 'other'}})
    with pytest.raises(RefusedError, match='wrong run dataset'):
        freeze.freeze_revrender(path, sha, run, tmp_path/'f2.json')


def test_the_report_flags_a_reading_that_differs_from_its_freeze(bundle, tmp_path):
    path, sha = bundle
    run = make_run(tmp_path/'run', path, sha)
    freeze.freeze_revrender(path, sha, run, tmp_path/'freeze.json')
    worse = make_run(tmp_path/'worse', path, sha, perfect=False)
    code = report.main(['--bundle', str(path), '--bundle-sha', sha, '--freeze', f'rvr={tmp_path/"freeze.json"}',
                        '--archive', f'rvr={worse/"dev_predictions.npz"}'])
    assert code == 1            # at the frozen score cut (0.9) the second run delivers nothing


def test_the_validation_order_puts_revised_lines_first_and_strips_the_bundle_prefix(bundle):
    path, sha = bundle
    data = B.load_bundle(path, sha)
    dev, order, truths = matching.validation_order(data['dev'])
    positives = [r['is_positive'] for r in dev]
    assert positives == sorted(positives, reverse=True)
    assert order[0] == dev[0]['sample_id'].split(':', 1)[1] and all(len(t) == len(r['instances']) for t, r in zip(truths, dev))


def test_the_analysis_commands_read_a_frozen_run_end_to_end(bundle, tmp_path):
    from revrender.eval import ap, bootstrap, fates, pixel_errors, representation
    path, sha = bundle
    run = make_run(tmp_path/'run', path, sha)
    freeze.freeze_revrender(path, sha, run, tmp_path/'freeze.json')
    common = ['--bundle', str(path), '--bundle-sha', sha]
    frozen = f'rvr={tmp_path/"freeze.json"}'
    assert fates.main([*common, '--freeze', frozen, '--ink-only', '--output', str(tmp_path/'fates.json')]) == 0
    read = json.loads((tmp_path/'fates.json').read_text())['models']['rvr']
    assert read['f1'] == 1. and read['ink']['f1'] == 1. and read['ink']['without_ink'] == {'gold': 0, 'delivered': 0}
    assert ap.main([*common, '--freeze', frozen, '--ink-only', '--output', str(tmp_path/'ap.json')]) == 0
    assert json.loads((tmp_path/'ap.json').read_text())['models']['rvr']['ink']['AP50'] == 1.
    assert bootstrap.main([*common, '--variant', f'a=freeze:{tmp_path/"freeze.json"}', '--variant', f'b=freeze:{tmp_path/"freeze.json"}',
                           '--pair', 'a-b', '--resamples', '100', '--workers', '1', '--ink-only',
                           '--output', str(tmp_path/'bootstrap.json')]) == 0
    boot = json.loads((tmp_path/'bootstrap.json').read_text())
    assert boot['ink_only'] is True and boot['variants']['a']['full_sample']['f1'] == 1.
    assert pixel_errors.main([*common, '--freeze', frozen, '--workers', '1', '--output', str(tmp_path/'classes.json')]) == 0
    classes = json.loads((tmp_path/'classes.json').read_text())['models']['rvr']
    assert classes['revision ink']['wrong'] == 0 and classes['far ink']['wrong'] == 0       # the run draws exactly the gold
    pytest.importorskip('skimage')
    assert representation.main([*common, '--freeze', frozen, '--workers', '1', '--output', str(tmp_path/'representation.json')]) == 0
    shaped = json.loads((tmp_path/'representation.json').read_text())
    assert shaped['thickness']['rvr']['0']['f1'] == 1. and shaped['thickness']['rvr']['0']['pixel_iou'] == 1.
    assert shaped['thickness']['rvr']['3']['pixel_iou'] < 1. and set(shaped['grids']) == set(representation.GRIDS)
