"""Native baselines, the parts that need no detectron2: metrics, early stopping, schedule, rows, journal options."""
import math
import random

import numpy as np
import pytest

from revrender.baselines.native import metrics, train
from revrender.data import journal


def record(name, positive):
    mask = np.zeros((4, 4), bool)
    if positive:
        mask[:2, :2] = True
    return dict(sample_id=name, mask=mask, instances=mask[None] if positive else np.zeros((0, 4, 4), bool), is_positive=positive)


def test_clean_rows_change_the_work_point_and_small_true_instances_survive():
    pos, clean = record('p', True), record('c', False)
    pp = metrics.prepare_record(pos, dict(masks=pos['instances'], scores=[.8]))
    cp = metrics.prepare_record(clean, dict(masks=np.ones((1, 4, 4), bool), scores=[.6]))
    result = metrics.select_operating_point([pp, cp])
    assert result['component_f1'] == 1 and result['threshold'] == .8 and result['clean_fp'] == 0 and result['gold_components'] == 1


def test_duplicate_instances_are_false_positives():
    pos, clean = record('p', True), record('c', False)
    pp = metrics.prepare_record(pos, dict(masks=np.repeat(pos['instances'], 2, axis=0), scores=[.8, .8]))
    cp = metrics.prepare_record(clean, dict(masks=clean['instances'], scores=[]))
    result = metrics.select_operating_point([pp, cp])
    assert (result['tp'], result['fp'], result['fn']) == (1, 1, 0) and result['pixel_f1'] == 1


def test_misaligned_predictions_and_duplicate_rows_are_refused():
    pos = record('p', True)
    with pytest.raises(ValueError, match='identity'):
        metrics.prepare_record(pos, dict(sample_id='wrong', masks=pos['instances'], scores=[1]))
    pp = metrics.prepare_record(pos, dict(masks=pos['instances'], scores=[1]))
    with pytest.raises(ValueError, match='unique'):
        metrics.select_operating_point([pp, pp])


def test_early_stopping_counts_validations_in_a_row_without_a_better_one():
    stopper = train.EarlyStop(3)
    assert stopper.update(1, (.5, .4)) and not stopper.update(2, (.5, .4)) and not stopper.stop
    assert stopper.update(3, (.6, .1)) and stopper.stale == 0
    for step in (4, 5):
        assert not stopper.update(step, (.59, .9))
    assert not stopper.stop
    stopper.update(6, (.6, .1))
    assert stopper.stop and stopper.best_step == 3


def test_native_schedule_matches_the_common_budget():
    assert train.learning_rate(500, 28300) == pytest.approx(3e-4)
    assert train.learning_rate(250, 28300) == pytest.approx(1.5e-4)
    assert train.learning_rate(28300, 28300) == pytest.approx(3e-4*.05)


def test_the_cycle_visits_every_row_once_per_pass():
    cycle = train.Cycle(list(range(10)), random.Random(42))
    first = [cycle.take() for _ in range(10)]
    second = [cycle.take() for _ in range(10)]
    assert sorted(first) == sorted(second) == list(range(10)) and first != second


def pool_row(line, writer, positive):
    mask = np.zeros((96, 768), np.uint8)
    if positive:
        mask[10:20, 10:60] = 255
    return {'sample_id': 'e:'+line, 'line_id': line, 'group_uuid': writer, 'image': np.full((96, 768), 200, np.uint8), 'mask': mask}


def test_extra_pool_rows_become_native_rows_in_a_fixed_order():
    rows = {'fit_positive': [pool_row('b', 'w', True), pool_row('a', 'w', True)], 'fit_clean': [pool_row('c', 'w', False)]}
    out = train.native_rows(rows)
    assert [r['line_id'] for r in out] == ['a', 'b', 'c'] and [r['is_positive'] for r in out] == [True, True, False]
    assert out[0]['image'].shape == (3, 96, 768) and out[0]['mask'].dtype == bool and len(out[0]['instances']) == 1
    rows['fit_clean'][0]['mask'][0, 0] = 255
    with pytest.raises(Exception, match='disagrees'):
        train.native_rows(rows)


def test_writer_fractions_are_nested_and_deterministic():
    writers = [f'writer{i}' for i in range(40)]
    sets = {f: journal.kept_writers(writers, f) for f in journal.FRACTIONS}
    assert sets[.125] <= sets[.25] <= sets[.5] <= sets[1.] and len(sets[.125]) == math.ceil(.125*40) == 5
    assert sets[.5] == journal.kept_writers(list(reversed(writers)), .5)
