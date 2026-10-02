"""Component matching and F1: greedy one-to-one at IoU >= 0.5, mask NMS, score cuts, the frontier, budgets."""
import numpy as np
import pytest

from revrender.eval import matching as M

SHAPE = (12, 24)


def box(top, left, h=5, w=8):
    mask = np.zeros(SHAPE, bool)
    mask[top:top+h, left:left+w] = True
    return mask


def reference_counts(masks, scores, truth, cut, iou_threshold=.5):
    """The matching written out directly: keep scores >= cut, non-empty; IoU-ordered greedy one-to-one."""
    kept = [i for i in range(len(masks)) if scores[i] >= cut and masks[i].any()]
    pairs = []
    for p in kept:
        for g, t in enumerate(truth):
            inter = int((masks[p] & t).sum())
            if inter:
                iou = inter/int((masks[p] | t).sum())
                if iou >= iou_threshold:
                    pairs.append((iou, p, g))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_p, used_g = set(), set()
    for _, p, g in pairs:
        if p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
    return len(used_g), len(kept)-len(used_g)


def random_line(rng):
    truth = [box(int(rng.integers(0, 7)), int(rng.integers(0, 16))) for _ in range(int(rng.integers(0, 3)))]
    masks, scores = [], []
    for _ in range(int(rng.integers(1, 5))):
        if rng.random() < .5 and truth:
            masks.append(truth[int(rng.integers(0, len(truth)))].copy())
        else:
            masks.append(box(int(rng.integers(0, 7)), int(rng.integers(0, 16))))
        scores.append(round(float(rng.random()), 1))            # coarse on purpose: ties happen
    return masks, np.array(scores), truth


def test_the_prefix_table_equals_matching_the_filtered_list_at_every_threshold():
    rng = np.random.default_rng(11)
    for _ in range(40):
        masks, scores, truth = random_line(rng)
        matched, kept, ordered = M.prefix_outcomes(masks, scores, truth)
        for cut in sorted(set(ordered.tolist())):
            admitted = int(np.searchsorted(-ordered, -cut, side='right'))
            tp, fp = reference_counts(masks, scores, truth, cut)
            assert (int(matched[admitted]), int(kept[admitted]-matched[admitted])) == (tp, fp)


def test_a_lower_scored_candidate_can_steal_a_match_and_the_table_knows():
    gold = box(2, 2)
    weak = np.zeros(SHAPE, bool)
    weak[2:7, 2:8] = True                      # IoU 0.75 with the gold
    exact = gold.copy()                        # IoU 1.0, scored lower
    matched, kept, ordered = M.prefix_outcomes([weak, exact], np.array([.9, .1]), [gold])
    tables = [(matched, kept, ordered)]
    assert M.totals_at(tables, .5) == (1, 0)    # only the weak one: it matches
    assert M.totals_at(tables, .05) == (1, 1)   # exact takes the gold, the weak one is now a false positive


def test_the_one_pass_sweep_equals_reading_every_threshold_separately():
    rng = np.random.default_rng(7)
    for _ in range(8):
        tables, cuts, gold_total = [], set(), 0
        for _ in range(int(rng.integers(2, 6))):
            masks, scores, truth = random_line(rng)
            gold_total += len(truth)
            table = M.prefix_outcomes(masks, scores, truth)
            tables.append(table)
            cuts.update(float(v) for v in table[2])
        fast = {round(r['score_threshold'], 6): r for r in M.sweep_every_cut(tables, gold_total)}
        assert set(fast) == {round(c, 6) for c in cuts}
        for cut in cuts:
            slow = M.reading(tables, cut, gold_total)
            for field in ('true_positive', 'false_positive', 'precision', 'recall', 'f1', 'false_negative'):
                assert fast[round(cut, 6)][field] == pytest.approx(slow[field]), (cut, field)


def test_recall_is_against_the_declared_gold_not_the_kept_set():
    gold = box(0, 0)
    table = M.prefix_outcomes([gold], np.array([.9]), [gold])
    got = M.reading([table], .5, 221)
    assert got['true_positive'] == 1 and got['false_negative'] == 220 and got['recall'] == pytest.approx(1/221)


def test_ties_in_f1_keep_the_lower_threshold():
    gold = box(0, 0)
    table = M.prefix_outcomes([gold], np.array([.5]), [gold])
    assert M.best_over([table], (0.3, 0.5, 0.7), 1)['score_threshold'] == 0.3


def test_suppression_off_is_on_the_grid_and_nms_runs_in_score_order():
    assert M.NMS_LEVELS[0] == 1.0
    a, b = box(2, 2), box(2, 3)
    assert M.suppressed([a, b], np.array([.9, .8]), 1.) == [0, 1]
    assert M.suppressed([a, b], np.array([.9, .8]), .5) == [0]
    assert M.suppressed([a, b], np.array([.1, .9]), .5) == [1]          # score order, not archive order
    assert M.suppressed([a, box(8, 18)], np.array([.9, .8]), .1) == [0, 1]   # a disjoint candidate survives
    empty = np.zeros(SHAPE, bool)
    assert M.suppressed([a, empty], np.array([.9, .8]), .3) == [0, 1]    # empties stay: the FP count is unchanged


def test_suppression_only_ever_removes_candidates():
    rng = np.random.default_rng(1)
    for _ in range(20):
        masks, scores, _ = random_line(rng)
        previous = None
        for level in (1., .9, .7, .5, .3):
            kept = M.suppressed(masks, scores, level)
            if previous is not None:
                assert set(kept) <= set(previous)
            previous = kept


def point(precision, recall, **extra):
    return {'precision': precision, 'recall': recall, 'f1': 0., 'true_positive': extra.get('tp', 0),
            'false_positive': extra.get('fp', 0), 'score_threshold': extra.get('cut', .5), 'nms': extra.get('nms', 1.)}


def test_a_dominated_point_is_dropped_from_the_frontier():
    gold, stray = box(1, 1), box(8, 16)
    lines = {'a': ([gold, stray], np.array([.9, .2]), [gold]), 'b': ([stray], np.array([.6]), [])}
    provide = lambda line: (lines[line][0], lines[line][1])
    points = M.frontier(provide, ['a', 'b'], [lines['a'][2], lines['b'][2]], 1)
    recalls = [p['recall'] for p in points]
    assert recalls == sorted(set(recalls)) and all(b > a for a, b in zip(recalls, recalls[1:]))
    assert points[0]['precision'] == 1.0 and points[-1]['recall'] == 1.0
    for p in points:                                     # no kept point is dominated by another
        assert not any(q['precision'] > p['precision'] and q['recall'] >= p['recall'] for q in points)


def test_recall_at_precision_never_reads_below_the_floor():
    points = [point(.9, .2), point(.72, .48), point(.60, .57)]
    assert M.recall_at_precision(points, .72)['recall'] == .48
    assert M.recall_at_precision(points, .60)['recall'] == .57
    assert M.recall_at_precision(points, .95) is None


def test_the_budget_reading_never_overspends():
    points = [point(.9, .2, tp=50, fp=6), point(.7, .5, tp=110, fp=47), point(.6, .57, tp=126, fp=83)]
    assert M.true_positives_at(points, 41)['true_positive'] == 50
    assert M.true_positives_at(points, 83)['true_positive'] == 126
    assert M.true_positives_at(points, 2) is None
    assert M.FALSE_POSITIVE_BUDGETS == (41, 55, 83, 111)


def test_pixel_iou_is_read_at_a_stated_point():
    gold, strong, stray = box(1, 1), box(1, 1), box(8, 16)
    provide = lambda line: ([strong, stray], np.array([.9, .2]))
    tight = M.pixel_iou(provide, ['a'], [[gold]], {'nms': 1.0, 'score_threshold': .5})
    loose = M.pixel_iou(provide, ['a'], [[gold]], {'nms': 1.0, 'score_threshold': .1})
    assert tight['pixel_iou'] == 1.0
    assert loose['pixel_iou'] < 1.0 and loose['pixel_recall'] == 1.0


def test_a_clean_line_with_nothing_kept_contributes_nothing():
    provide = lambda line: ([np.zeros(SHAPE, bool)], np.array([.1]))
    assert M.pixel_iou(provide, ['a'], [[]], {'nms': 1.0, 'score_threshold': .5}) == {
        'pixel_iou': 0.0, 'pixel_precision': 0.0, 'pixel_recall': 0.0}


def test_a_shape_mismatch_between_candidates_and_gold_is_refused():
    provide = lambda line: ([np.zeros((4, 4), bool)], np.array([.5]))
    with pytest.raises(ValueError, match='against gold'):
        M.tabulate_at(provide, ['a'], [[box(0, 0)]], 1.)


def test_the_packed_provider_reads_one_mask_threshold_per_line(tmp_path):
    rng = np.random.default_rng(0)
    thresholds = [.2, .3, .4]
    probability = rng.random((3, 2, 32, 4, 8)) > .5                  # 3 thresholds, 2 lines, 32 queries, 4x8 pixels
    bits = np.packbits(probability.reshape(3, 2, 32, -1), axis=-1)
    path = tmp_path/'p.npz'
    np.savez(path, packed_masks=bits, scores=rng.random((2, 32)).astype(np.float32), mask_shape=np.array([2, 32, 4, 8]),
             mask_thresholds=np.array(thresholds), sample_ids=np.array(['s:x:1', 's:y:2']))
    provide, available = M.packed_provider(path, .3)
    assert available == {'x:1', 'y:2'}
    masks, scores = provide('y:2')
    assert len(masks) == 32 and masks[0].shape == (4, 8) and np.array_equal(masks[5], probability[1, 1, 5])
    with pytest.raises(Exception, match='not stored once'):
        M.packed_provider(path, .35)


def test_the_native_provider_unpacks_instance_archives(tmp_path):
    a, b = box(1, 1), box(6, 10)
    stack = np.stack([a, b])
    np.savez(tmp_path/'0000.npz', sample_id=np.asarray('s:x:1'), scores=np.array([.9, .4]), shape=np.array(stack.shape),
             masks_packed=np.packbits(stack.reshape(2, -1), axis=1))
    provide, available = M.native_provider(tmp_path)
    assert available == {'x:1'}
    masks, scores = provide('x:1')
    assert np.array_equal(masks[1], b) and scores.tolist() == [.9, .4]
