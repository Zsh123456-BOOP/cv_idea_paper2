"""The metrics must not invent gold, punish verbatim compliance, or infer damage."""
import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from revrender.eval.recognition_score import SCHEMA, score, validate
from revrender.util import RefusedError


def key(identifier, kind, deletion, *, verbatim=None, damage='no', readable=True):
    return {'id': identifier, 'kind': kind, 'char': 'x', 'readable': readable,
            'box': [.1, .1, .3, .6], 'whitening_damage': damage,
            'judgments': {'deletion': dict(zip('BCD', deletion)),
                          'verbatim': dict(zip('BCD', verbatim or deletion))}}


def example():
    keys = [key('a', 'cancelled', ['outside', 'missing', 'inside']),
            key('b', 'cancelled', ['wrong_outside', 'inside', 'wrong_inside']),
            key('c', 'cancelled', ['outside', 'missing', 'uncertain']),
            key('u', 'cancelled', ['outside', 'outside', 'inside'], readable=False),
            key('v1', 'valid', ['outside', 'wrong_outside', 'outside'], damage='yes'),
            key('v2', 'valid', ['outside', 'wrong_outside', 'outside'], damage='no'),
            key('v3', 'valid', ['wrong_outside', 'missing', 'outside'], damage='yes'),
            key('v4', 'valid', ['outside', 'missing', 'outside'], damage='uncertain'),
            key('v5', 'valid', ['outside', 'inside', 'outside'], damage='yes')]
    row = {'revision': 'yes', 'reviewed': True, 'key_region_complete': True, 'keys': keys}
    return {'schema': SCHEMA, 'dataset_id': 'sample', 'rows': {'r1': row}}


def test_blank_or_unconfirmed_is_pending_not_zero_error():
    empty = score({'rows': {}}, 80)
    assert empty['cancelled_ink_leakage']['B']['rate'] is None
    assert empty['valid_ink_cut_off']['deletion']['rate'] is None
    data = example()
    data['rows']['r1']['reviewed'] = False
    assert score(data, 80)['coverage']['confirmed_positive_rows'] == 0


def test_leakage_is_paired_and_tracks_cancelled_ink_even_when_misread():
    out = score(example(), 80)
    assert out['cancelled_ink_leakage'] == {
        'B': {'count': 2, 'denominator': 2, 'rate': 1},
        'C': {'count': 0, 'denominator': 2, 'rate': 0},
        'D': {'count': 0, 'denominator': 2, 'rate': 0}}
    assert out['coverage']['unmatched_cancelled_keys'] == 1
    assert out['coverage']['unreadable_keys'] == 1
    # Verbatim output never affects the deletion-task leakage metric.
    data = example()
    for k in data['rows']['r1']['keys']:
        k['judgments']['verbatim'] = dict.fromkeys('BCD', 'outside')
    assert score(data, 80)['cancelled_ink_leakage'] == out['cancelled_ink_leakage']


def test_cut_off_requires_readable_valid_ink_a_correct_baseline_and_visible_damage():
    out = score(example(), 80)['valid_ink_cut_off']
    assert out['deletion']['count'] == 2
    assert out['deletion']['denominator'] == 3
    # Correctly read glyph inside tags is still read in verbatim, but lost from valid deletion output.
    assert out['verbatim']['count'] == 1
    assert out['verbatim']['denominator'] == 3
    assert out['deletion']['unassessed_valid_keys'] == 1


@pytest.mark.parametrize('revision', ['no', 'uncertain', 'duplicate'])
def test_false_positives_uncertain_and_duplicates_do_not_enter_metrics(revision):
    data = example()
    data['rows']['r1']['revision'] = revision
    assert score(data, 80)['cancelled_ink_leakage']['B']['denominator'] == 0


def test_annotation_validation_rejects_wrong_dataset_bad_boxes_and_incomplete_region():
    data = example()
    assert validate(data, {'r1'}, 'sample') is data
    with pytest.raises(RefusedError, match='another pack'):
        validate(data, {'r1'}, 'another')
    invalid = copy.deepcopy(data)
    invalid['rows']['r1']['keys'][0]['box'][2] = 2
    with pytest.raises(RefusedError, match='character box'):
        validate(invalid, {'r1'}, 'sample')
    invalid = copy.deepcopy(data)
    invalid['rows']['r1']['key_region_complete'] = False
    with pytest.raises(RefusedError, match='fully reviewed'):
        validate(invalid, {'r1'}, 'sample')
    invalid = copy.deepcopy(data)
    invalid['rows']['r1']['keys'][0]['judgments']['deletion']['D'] = ''
    with pytest.raises(RefusedError, match='missing judgments'):
        validate(invalid, {'r1'}, 'sample')


@pytest.mark.skipif(not shutil.which('node'), reason='browser readout parity requires Node')
def test_browser_readout_matches_canonical_python_metric_for_all_judgments():
    source = Path(__file__).parents[1]/'revrender/tools/recognition_metrics.js'
    script = "const m=require(process.argv[1]);let s='';process.stdin.on('data',x=>s+=x);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(m.recognitionMetrics(JSON.parse(s),80))));"
    for data in [example(), {'rows': {}}]:
        output = subprocess.run(['node', '-e', script, str(source)], input=json.dumps(data), text=True,
                                capture_output=True, check=True)
        assert json.loads(output.stdout) == score(data, 80)
