"""The parts of train.py that need no GPU: schedule, validation cadence, gates, evidence."""
import json

import numpy as np
import pytest

from revrender import config, train
from revrender.util import RefusedError, write_json

TRAIN = config.load_recipe()['train']


def test_the_schedule_warms_up_then_follows_a_cosine_to_five_percent():
    assert train.learning_rate_at(1, TRAIN) == pytest.approx(3e-4/500)
    assert train.learning_rate_at(500, TRAIN) == pytest.approx(3e-4)
    assert train.learning_rate_at(28300, TRAIN) == pytest.approx(3e-4*.05)
    middle = train.learning_rate_at(14400, TRAIN)
    assert 3e-4*.05 < middle < 3e-4 and middle == pytest.approx(3e-4*(.05+.95*.5), rel=1e-3)


def due(step):
    return train.validation_is_due(step, 28300, 500, 14150, 4)


def test_validation_is_sparse_before_selection_starts_and_every_500_after():
    assert [s for s in range(1, 4001) if due(s)] == [2000, 4000]
    assert due(14000) and not due(14250) and due(14500) and due(15000)
    assert due(28300) and not due(28299)


def test_the_learning_gate_needs_less_loss_recall_precision_multi_hits_and_quiet_clean_rows():
    ok = dict(initial=100., final=60., tp=6, fp=2, gold=8, multi_tp=2, clean_fired=1)
    assert train.learning_gate(**ok)
    for field, value in (('final', 80.), ('tp', 5), ('fp', 4), ('multi_tp', 1), ('clean_fired', 2)):
        assert not train.learning_gate(**{**ok, field: value}), field


def row(count):
    return {'instances': np.zeros((count, 4, 4), bool)}


def test_learning_rows_take_two_single_two_multi_and_four_clean_lines():
    rows = [row(1), row(0), row(2), row(1), row(3), row(0), row(0), row(0), row(0), row(1)]
    picked = train.learning_rows(rows)
    assert [len(r['instances']) for r in picked] == [1, 1, 2, 3, 0, 0, 0, 0]
    with pytest.raises(RefusedError, match='missing TRAIN'):
        train.learning_rows(rows[:4])


def test_union_learning_gate_still_covers_two_distinct_multistroke_crops():
    rows=[{**row(1),'source_instances':n} for n in (1,1,2,3)]+[row(0) for _ in range(4)]
    picked=train.learning_rows(rows)
    assert [r.get('source_instances',0) for r in picked[:4]]==[1,1,2,3]
    assert sum(len(r['instances']) for r in picked)==4
    assert train.learning_gate(100,50,4,0,4,2,0)
    assert not train.learning_gate(100,50,3,0,4,1,0)


def make_stage(path, profile, stage='smoke', steps=4, failed=False, exact_replay=True, passed=True):
    path.mkdir(parents=True)
    write_json(path/'run_manifest.json', {'profile': profile, 'frozen_test_touched': False})
    write_json(path/'summary.json', {'exact_replay': exact_replay, 'passed': passed})
    write_json(path/'trace.json', [])
    write_json(path/'complete.json', {'stage': stage, 'steps': steps, 'files': {
        n: train.digest(path/n) for n in ('run_manifest.json', 'summary.json', 'trace.json')}})
    if failed:
        write_json(path/'failure.json', {})


def test_gate_evidence_must_be_complete_unchanged_and_of_the_same_profile(tmp_path):
    profile = {'git_commit': 'abc', 'recipe': {'a': 1}}
    make_stage(tmp_path/'ok', profile)
    complete, summary = train.validate_evidence(tmp_path/'ok', profile)
    assert complete['stage'] == 'smoke' and summary['exact_replay']
    with pytest.raises(RefusedError, match='profile'):
        train.validate_evidence(tmp_path/'ok', {**profile, 'git_commit': 'def'})
    make_stage(tmp_path/'failed', profile, failed=True)
    with pytest.raises(RefusedError, match='failure'):
        train.validate_evidence(tmp_path/'failed', profile)
    (tmp_path/'ok'/'summary.json').write_text('{}')
    with pytest.raises(RefusedError, match='changed'):
        train.validate_evidence(tmp_path/'ok', profile)


def test_the_cli_refuses_an_unpaired_corrected_bundle_and_a_formal_run_without_gates(tmp_path):
    base = ['--bundle', str(tmp_path), '--bundle-sha', 's', '--backbone-weights', str(tmp_path), '--output-dir', str(tmp_path/'o')]
    with pytest.raises(SystemExit):
        train.parse(['--stage', 'smoke', '--corrected-bundle', str(tmp_path), *base])
    with pytest.raises(SystemExit):
        train.parse(['--stage', 'formal', *base])
    with pytest.raises(SystemExit):
        train.parse(['--stage', 'formal', '--replay-of', str(tmp_path), '--gate-root', str(tmp_path), *base])
    assert train.parse(['--stage', 'smoke', *base]).stage == 'smoke'
