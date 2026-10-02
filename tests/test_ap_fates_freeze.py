"""Mask AP, found / drawn right, the freeze rule and the in-loop reading."""
import json

import numpy as np
import pytest

from revrender.eval import ap, fates, freeze, inloop, matching
from revrender.eval.report import compare_archives


def test_perfect_ranking_scores_one_and_a_leading_false_positive_costs():
    assert ap.average_precision([(.9, True), (.8, True)], 2) == pytest.approx(1.)
    assert ap.average_precision([(.9, False), (.8, True), (.7, True)], 2) < 1.
    assert ap.average_precision([(.9, True)], 2) == pytest.approx(51/101)      # recall reaches only 0.5
    assert ap.average_precision([], 3) == 0.


def test_matching_is_greedy_by_score_and_one_to_one():
    iou = np.array([[.8, .0], [.7, .6], [.55, .0]])
    assert ap.match_line(iou, [.9, .8, .7], .5) == [(.9, True), (.8, True), (.7, False)]
    assert ap.match_line(iou, [.9, .8, .7], .75) == [(.9, True), (.8, False), (.7, False)]
    assert ap.IOUS[0] == .5 and ap.IOUS[-1] == .95 and len(ap.IOUS) == 10


def test_mask_ap_reads_ap50_ap75_and_the_mean():
    gold = np.zeros((12, 24), bool)
    gold[2:7, 2:10] = True
    exact = gold.copy()
    loose = np.zeros_like(gold)
    loose[2:7, 2:14] = True                                              # IoU 8/12 = .67: right at .5, wrong at .75
    provide = lambda line: ([exact, loose], np.array([.9, .8])) if line == 'a' else ([loose], np.array([.7]))
    out = ap.mask_ap(provide, ['a', 'b'], [[gold], [gold]], 1.)
    # detections by score: exact (TP), duplicate on line a (FP), loose on line b (TP at .5; FP at .75)
    assert out['AP50'] == pytest.approx((51 + 50*2/3)/101)
    assert out['AP75'] == pytest.approx(51/101)
    assert out['AP'] == pytest.approx(np.mean(list(out['by_iou'].values())))


def test_the_vectorised_bootstrap_ap_equals_the_reference_on_every_column():
    from revrender.eval import bootstrap
    rng = np.random.default_rng(0)
    scores = rng.random(60)
    hits = rng.random((60, 10)) < np.linspace(.8, .2, 10)
    for gold in (25, 40):
        mine = bootstrap.interpolated_ap(scores, hits, gold)
        theirs = [ap.average_precision(list(zip(scores.tolist(), hits[:, k].tolist())), gold) for k in range(10)]
        assert np.allclose(mine, theirs, atol=1e-12)
    assert bootstrap.interpolated_ap(np.zeros(0), np.zeros((0, 10), bool), 5).tolist() == [0.]*10


def test_bootstrap_line_counts_reproduce_the_frozen_rules():
    from revrender.eval import bootstrap
    gold = np.zeros((8, 16), bool)
    gold[2:6, 2:8] = True
    hit, miss, low = np.zeros_like(gold), np.zeros_like(gold), np.zeros_like(gold)
    hit[2:6, 2:7] = True
    miss[0:2, 12:15] = True
    low[5:8, 10:16] = True
    point = {'nms': .5, 'score_threshold': .5}
    x = bootstrap.line_counts([hit, miss, low], np.array([.9, .6, .2]), [gold], point)
    assert (x['tp'], x['fp'], x['gold']) == (1, 1, 1)
    reference = matching.pixel_iou(lambda line: ([hit, miss, low], np.array([.9, .6, .2])), ['a'], [[gold]], dict(point))
    assert x['inter']/(x['pred']+x['gold_px']-x['inter']) == pytest.approx(reference['pixel_iou'])
    assert x['scores'].tolist() == [.9, .6, .2] and x['hits'][:, 0].tolist() == [True, False, False]


def test_writer_draws_pair_every_variant_and_sum_per_writer():
    from revrender.eval import bootstrap
    lines = [{'tp': 1, 'fp': 0, 'gold': 1, 'inter': 4, 'pred': 4, 'gold_px': 4, 'scores': np.array([.9]), 'hits': np.ones((1, 10), bool)},
             {'tp': 0, 'fp': 1, 'gold': 1, 'inter': 0, 'pred': 3, 'gold_px': 5, 'scores': np.array([.8]), 'hits': np.zeros((1, 10), bool)}]
    groups = bootstrap.by_writer(lines, ['w1', 'w2'], ['w1', 'w2'])
    whole = bootstrap.metrics_of(groups, [0, 1])
    assert whole['f1'] == pytest.approx(2/(1+1+2)) and whole['pixel_iou'] == pytest.approx(4/12)
    only_good = bootstrap.metrics_of(groups, [0, 0])
    assert only_good['f1'] == 1. and only_good['AP'] == pytest.approx(1.)
    s = bootstrap.summarise(.1, [.2, -.1, .05, .3])
    assert s['share_not_above_zero'] == .25 and s['difference'] == .1


def test_bootstrap_refuses_a_malformed_variant():
    from revrender.eval import bootstrap
    with pytest.raises(Exception, match='label=freeze'):
        bootstrap.parse_variant('a=other:x')


def test_fates_separate_found_from_drawn_right():
    shape = (20, 40)
    gold = [np.zeros(shape, bool) for _ in range(3)]
    gold[0][2:8, 2:12] = True
    gold[1][2:8, 20:30] = True
    gold[2][12:18, 2:12] = True
    right = gold[0].copy()                                       # matched: IoU 1
    wide = np.zeros(shape, bool)
    wide[2:8, 20:60-20+0] = True                                 # reaches gold 1 (IoU 10/20 = .5) ... too wide at 40 cols
    wide[2:8, 20:40] = True                                      # IoU .5 exactly -> matched; use a wider miss instead
    wide = np.zeros(shape, bool)
    wide[2:8, 20:40] = True                                      # covers gold[1] fully, IoU = 60/120 = .5 -> matched
    reach = np.zeros(shape, bool)
    reach[12:18, 2:5] = True                                     # covers 30% of gold[2], IoU .3: reached, not drawn right
    stray = np.zeros(shape, bool)
    stray[14:19, 30:38] = True                                   # touches nothing
    masks = [right, wide, reach, stray]
    scores = np.array([.9, .8, .7, .6])
    record = fates.line_record(masks, scores, gold, {'nms': 1., 'score_threshold': .5})
    assert [g['fate'] for g in record['gold']] == ['matched', 'matched', 'extent']
    assert record['predicted'] == ['true', 'true', 'on_gold', 'spurious']
    summary = fates.summarise([record])
    assert summary['found'] == 1.0 and summary['drawn_right'] == pytest.approx(2/3)
    assert summary['drawn_right_of_found'] == pytest.approx(2/3)
    cut = fates.line_record(masks, scores, gold, {'nms': 1., 'score_threshold': .85})
    assert [g['fate'] for g in cut['gold']] == ['matched', 'missed', 'missed']
    partial = fates.summarise([cut])
    assert partial['found'] == pytest.approx(1/3) and partial['drawn_right'] == pytest.approx(1/3) and partial['drawn_right_of_found'] == 1.0


def test_pick_best_f1_then_fewer_false_positives_then_lower_threshold():
    r = lambda t, f1, fp: {'mask_threshold': t, 'f1': f1, 'false_positive': fp}
    assert freeze.pick([r(.3, .68, 87), r(.4, .686, 85), r(.5, .66, 93)])['mask_threshold'] == .4
    assert freeze.pick([r(.3, .70, 80), r(.4, .70, 70)])['mask_threshold'] == .4
    assert freeze.pick([r(.3, .70, 70), r(.5, .70, 70)])['mask_threshold'] == .3


def test_read_candidates_reads_each_threshold_and_refuses_a_partial_archive():
    shape = (12, 24)
    gold = np.zeros(shape, bool)
    gold[2:7, 2:10] = True
    good = gold.copy()
    lines = {'a': ([good], np.array([.9]))}
    providers = lambda level: ((lambda line: lines[line]), {'a'})
    readings, frontiers = freeze.read_candidates(providers, ['a'], [[gold]], candidates=(.3, .5))
    assert [r['mask_threshold'] for r in readings] == [.3, .5] and readings[0]['f1'] == 1.0
    with pytest.raises(Exception, match='coverage'):
        freeze.read_candidates(lambda level: (lambda line: lines['a'], {'b'}), ['a'], [[gold]], candidates=(.3,))


def predictions(n_lines=2, q=32, h=4, w=8, seed=0):
    rng = np.random.default_rng(seed)
    thresholds = list(inloop.MASK_THRESHOLDS)
    probability = rng.random((n_lines, q, h, w))
    bits = np.stack([np.packbits((probability >= t).reshape(n_lines, q, -1), axis=-1) for t in thresholds])
    return bits, rng.random((n_lines, q)).astype(np.float32), probability


def test_the_in_loop_reading_finds_a_perfect_prediction(tmp_path):
    h, w = 4, 8
    truth = np.zeros((1, h, w), bool)
    truth[0, 1:3, 2:6] = True
    bits = np.zeros((7, 1, 32, 4), np.uint8)
    packed = np.packbits(np.broadcast_to(truth[0].reshape(-1), (32, h*w)).copy(), axis=-1)
    for k in range(7):
        bits[k, 0, 0] = packed[0]
    scores = np.zeros((1, 32), np.float32)
    scores[0, 0] = .9
    reading = inloop.evaluate(bits, scores, [truth], 1, packed=True)
    assert reading['instance']['0.50']['f1'] == 1.0 and reading['pixel_iou'] == 1.0
    assert len(reading['grid']) == 35


def test_in_loop_matching_pairs_are_one_to_one_by_iou():
    ious = np.array([[.9, .6], [.7, .1]])
    used_p, used_g = inloop.match_pairs(ious, [0, 1])
    assert used_p == {0, 1} or used_g == {0}     # prediction 0 takes gold 0; prediction 1 (IoU .7 with gold 0) is left over
    assert used_g == {0}


def test_two_archives_compare_line_by_line(tmp_path):
    bits, scores, _ = predictions()
    base = dict(packed_masks=bits, scores=scores, mask_shape=np.array([2, 32, 4, 8]),
                mask_thresholds=np.array(inloop.MASK_THRESHOLDS), sample_ids=np.array(['a:x', 'a:y']))
    np.savez(tmp_path/'one.npz', **base)
    np.savez(tmp_path/'two.npz', **base)
    assert compare_archives(tmp_path/'one.npz', tmp_path/'two.npz')['identical']
    changed = bits.copy()
    changed[2, 1, 3, 0] ^= 1
    np.savez(tmp_path/'three.npz', **{**base, 'packed_masks': changed})
    result = compare_archives(tmp_path/'one.npz', tmp_path/'three.npz')
    assert not result['identical'] and result['lines_with_different_masks'] == 1 and result['differing_bits'] == 1
