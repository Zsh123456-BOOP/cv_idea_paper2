import numpy as np
import pytest

from revrender.data import synth_views
from revrender.eval.privileged import rule_a
from revrender.eval.selective import answered_aurc, threshold


def test_window_stats_counts_cover_share_and_writing():
    clean = np.full((20, 20), 255, np.uint8); clean[5:15, 5:15] = 40          # writing
    scan = clean.copy(); scan[9:11, 0:20] = 30                                   # a strike across it
    st = synth_views.window_stats(scan, clean)
    assert st['strike'] == 20 and st['kept'] == 100 and abs(st['strike_share']-20/120) < 1e-9      # a crossing is no lighter when erased
    assert 0 < st['covered'] < 1


def test_host_stretches_stay_short():
    clean = np.full((10, 60), 255, np.uint8)
    for x in range(0, 60, 10): clean[2:8, x:x+5] = 0                             # six characters of 30 ink pixels
    found = synth_views.host_stretches(clean)
    assert found and all(int((clean[:, a:b] < 200).sum()) <= synth_views.SHORT for a, b in found)
    assert all(int((clean[:, a:b] < 200).sum()) >= synth_views.LEAST for a, b in found)


def test_real_subset_is_nested_and_near_its_share():
    pytest.importorskip('torch')
    from revrender.train_clean_view import real_subset
    regions = [(f'r{k}', f'w{k % 23}', []) for k in range(400)]
    small, big = real_subset(regions, .25), real_subset(regions, .5)
    assert set(small) <= set(big) and big[:len(small)] == small
    assert abs(sum(w in set(small) for _, w, _ in regions)/100-1) < .1


def diff(lo, hi):
    return {'exact': {'interval': [lo, hi]}}


def test_rule_a_main_and_floor():
    assert rule_a(diff(.01, .1), diff(-.05, .05), diff(-.02, .06)) == 'useful'
    assert rule_a(diff(.01, .1), diff(-.08, -.01), diff(-.02, .06)) == 'heavy_better_elsewhere_worse'
    assert rule_a(diff(-.08, -.01), diff(-.05, .05), diff(-.02, .06)) == 'worse'
    assert rule_a(diff(-.01, .1), diff(-.05, .05), diff(-.02, .06)) == 'cannot_tell'


def test_answered_aurc_and_threshold():
    scores = {'a': .9, 'b': .8, 'c': .7, 'd': .6}; right = {'a': True, 'b': True, 'c': False, 'd': False}
    assert answered_aurc(scores, right, 4) == pytest.approx((0+0+1/3+2/4)/4)
    assert answered_aurc(scores, right, 8) == pytest.approx(answered_aurc(scores, right, 4)/2)      # refusals count as never answered
    assert threshold(scores, right, .3, False) == .8 and threshold(scores, right, .0, True) is None
