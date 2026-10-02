"""The analysis readings: the ink view, the grid ceilings and thickness, the six-class pixel errors.

``tests/golden/analysis_tools.json`` holds what the paper-1 implementations returned on the synthetic lines of
``synthetic_lines`` (recorded once from them; the inputs are regenerated here from seed 42)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from revrender.eval import ap, bootstrap, fates, freeze, ink, matching, pixel_errors, representation

GOLDEN = Path(__file__).parent/'golden'/'analysis_tools.json'
SHAPE = (96, 768)


def fingerprint(mask):
    return hashlib.sha256(np.packbits(np.asarray(mask, bool)).tobytes()).hexdigest()


def rectangle(rng):
    mask = np.zeros(SHAPE, bool)
    top, left = int(rng.integers(5, 60)), int(rng.integers(10, 600))
    mask[top:top+int(rng.integers(6, 30)), left:left+int(rng.integers(40, 160))] = True
    return mask


def synthetic_lines(count=6, seed=42):
    """Per line: gold components, candidate masks with scores, an ink map and the grey image it came from."""
    rng = np.random.default_rng(seed)
    lines = []
    for k in range(count):
        truth = [rectangle(rng) for _ in range(k % 4)]
        masks, scores = [], []
        for component in truth:
            moved = np.roll(component, (int(rng.integers(-8, 9)), int(rng.integers(-24, 25))), axis=(0, 1))
            masks.append(moved)
            scores.append(float(rng.uniform(.3, .95)))
        for _ in range(1 + k % 2):
            masks.append(rectangle(rng))
            scores.append(float(rng.uniform(.1, .6)))
        grey = np.where(rng.random(SHAPE) < .55, 60, 240).astype(np.uint8)
        lines.append({'truth': truth, 'masks': masks, 'scores': np.array(scores), 'grey': grey, 'ink': grey < ink.INK_BELOW})
    return lines


def test_the_ink_view_cuts_both_sides_and_drops_gold_without_ink():
    ink_map = np.zeros((8, 16), bool)
    ink_map[2:6, 2:6] = True
    gold_a = np.zeros((8, 16), bool)
    gold_a[2:6, 2:10] = True
    gold_blank = np.zeros((8, 16), bool)
    gold_blank[6:8, 10:16] = True
    mask = np.zeros((8, 16), bool)
    mask[2:6, 2:8] = True
    cut_masks, cut_truth, dropped = ink.restrict([mask], [gold_a, gold_blank], ink_map)
    assert dropped == 1 and len(cut_truth) == 1
    assert cut_masks[0].sum() == 16 and cut_truth[0].sum() == 16
    rows = [{'image': np.stack([np.where(ink_map, 50, 250).astype(np.uint8)]*3)}]
    assert ink.ink_maps(rows)[0].tolist() == ink_map.tolist()


def ink_example():
    ink_map = np.zeros((8, 16), bool)
    ink_map[2:6, 2:6] = True
    gold = np.zeros((8, 16), bool)
    gold[2:6, 2:10] = True                       # half of it is blank paper
    blank = np.zeros((8, 16), bool)
    blank[6:8, 10:16] = True                     # nothing but paper
    hit = np.zeros((8, 16), bool)
    hit[2:6, 2:8] = True                         # region IoU .75, ink IoU 1
    stray = np.zeros((8, 16), bool)
    stray[0:2, 12:16] = True                     # on paper: empty once cut to ink
    return ink_map, [gold, blank], [hit, stray]


def test_fates_on_ink_read_the_delivered_masks_the_model_chose_and_count_what_the_cut_removed():
    ink_map, truth, masks = ink_example()
    point = {'nms': 1., 'score_threshold': .5}
    scores = np.array([.9, .6])
    region = fates.line_record(masks, scores, truth, point)
    assert (region['tp'], region['fp']) == (1, 1) and [g['fate'] for g in region['gold']] == ['matched', 'missed']
    on_ink = fates.line_record(masks, scores, truth, point, ink_map)
    assert (on_ink['tp'], on_ink['fp']) == (1, 1)                              # the emptied mask is still a false positive
    assert on_ink['gold'][0]['iou'] == 1. and len(on_ink['gold']) == 1
    assert on_ink['without_ink'] == {'gold': 1, 'delivered': 1}
    summary = fates.summarise([on_ink])
    assert summary['f1'] == pytest.approx(2*1/(1+1+1)) and summary['without_ink'] == {'gold': 1, 'delivered': 1}
    assert 'without_ink' not in fates.summarise([region])
    # the score cut is applied to the full masks: below it nothing is delivered, whatever the ink says
    assert fates.line_record(masks, scores, truth, {'nms': 1., 'score_threshold': .95}, ink_map)['tp'] == 0


def test_ap_on_ink_counts_an_emptied_mask_as_a_false_positive_and_only_the_gold_with_ink():
    ink_map, truth, masks = ink_example()
    provide = lambda line: (masks, np.array([.9, .6]))
    inks = [ink_map]
    assert ap.mask_ap(provide, ['a'], [truth], 1., inks)['AP50'] == pytest.approx(1.)          # hit first, then the emptied mask
    behind = lambda line: (masks, np.array([.6, .9]))
    assert ap.mask_ap(behind, ['a'], [truth], 1., inks)['AP50'] == pytest.approx(.5)            # the emptied mask first
    assert ap.mask_ap(provide, ['a'], [truth], 1.)['AP50'] == pytest.approx(51/101)           # region: 1 of 2 gold found
    again = ap.mask_ap(provide, ['a'], [truth], 1., None)
    assert again == ap.mask_ap(provide, ['a'], [truth], 1.)


def test_bootstrap_line_counts_on_ink_agree_with_the_fates_and_count_pixels_on_ink():
    ink_map, truth, masks = ink_example()
    scores = np.array([.9, .6])
    point = {'nms': 1., 'score_threshold': .5}
    region = bootstrap.line_counts(masks, scores, truth, point)
    assert (region['tp'], region['fp'], region['gold']) == (1, 1, 2)
    on_ink = bootstrap.line_counts(masks, scores, truth, point, ink_map)
    assert (on_ink['tp'], on_ink['fp'], on_ink['gold']) == (1, 1, 1)
    assert (on_ink['inter'], on_ink['pred'], on_ink['gold_px']) == (16, 16, 16)
    assert on_ink['scores'].tolist() == [.9, .6] and on_ink['hits'][:, 0].tolist() == [True, False]


def test_a_freeze_is_read_on_its_bundle_or_on_the_source_it_was_corrected_from(tmp_path):
    record = {'test_touched': False, 'bundle_sha256': 'old', 'model': 'maskdino', 'validation_archive': 'a', 'mask_threshold': .5,
              'validation': {'own_best': {'nms': .5, 'score_threshold': .3, 'true_positive': 1}}}
    path = tmp_path/'f.json'
    path.write_text(json.dumps(record))
    assert freeze.frozen_point(path, 'new', 'old')['expected'] is None          # the gold-v2 source bundle: nothing to recount against
    point = freeze.frozen_point(path, 'old')
    assert point['expected']['true_positive'] == 1 and (point['nms'], point['score_threshold']) == (.5, .3)
    assert freeze.frozen_point(path, 'old', archive='b')['archive'] == 'b'
    with pytest.raises(Exception, match='not a validation freeze'):
        freeze.frozen_point(path, 'new', 'other')
    path.write_text(json.dumps(dict(record, test_touched=True)))
    with pytest.raises(Exception, match='not a validation freeze'):
        freeze.frozen_point(path, 'old')


def test_the_ink_view_reproduces_the_paper_one_readings_on_synthetic_lines():
    golden = json.loads(GOLDEN.read_text())['ink_view']
    point = {'nms': 1., 'score_threshold': 0.}
    lines = synthetic_lines()
    for line, expected in zip(lines, golden['lines']):
        record = fates.line_record(line['masks'], line['scores'], line['truth'], point, line['ink'])
        assert (record['tp'], record['fp']) == tuple(expected['tp_fp'])
        assert [g['fate'] for g in record['gold']] == expected['fates']
        assert [record['without_ink']['gold'], record['without_ink']['delivered']] == expected['without_ink']
    provide = {str(k): (line['masks'], line['scores']) for k, line in enumerate(lines)}
    order = list(provide)
    out = ap.mask_ap(lambda key: provide[key], order, [line['truth'] for line in lines], 1., [line['ink'] for line in lines])
    assert out['AP'] == golden['AP'] and out['AP50'] == golden['AP50'] and out['AP75'] == golden['AP75']


def test_grids_and_thickness_reproduce_the_paper_one_readings():
    golden = json.loads(GOLDEN.read_text())['representation']
    lines = synthetic_lines()
    components = [g for line in lines for g in line['truth']][:3]
    for component, expected in zip(components, golden['grids']):
        for name, (kind, grid_height, grid_width) in representation.GRIDS.items():
            back = representation.through_grid(component, kind, grid_height, grid_width)
            assert fingerprint(back) == expected[name]['sha'] and representation.iou(back, component) == expected[name]['iou']
    for component, expected in zip(components[:2], golden['reshaped']):
        for step in representation.STEPS:
            assert fingerprint(representation.reshaped(component, step)) == expected[str(step)]


def test_the_six_pixel_classes_partition_the_line_and_reproduce_the_paper_one_counts():
    golden = json.loads(GOLDEN.read_text())['pixel_classes']
    for line, expected in zip(synthetic_lines(), golden):
        gold = np.logical_or.reduce(line['truth']) if line['truth'] else np.zeros(SHAPE, bool)
        classes = pixel_errors.classes_of(line['ink'], gold)
        assert sum(int(c.sum()) for c in classes.values()) == gold.size
        assert {name: int(c.sum()) for name, c in classes.items()} == expected['pixels']
        covered = line['masks'][0]
        assert {name: int(pixel_errors.wrong(c, name, covered).sum()) for name, c in classes.items()} == expected['wrong']


def test_width_thirds_cut_the_revisions_into_three_groups():
    widths = [1., 2., 3., 4., 5., 6.]
    edges = representation.width_edges(widths)
    assert [representation.width_third(w, edges) for w in widths] == ['thin', 'thin', 'medium', 'medium', 'thick', 'thick']


def test_stroke_width_is_area_over_skeleton_length():
    pytest.importorskip('skimage')
    bar = np.zeros((20, 60), bool)
    bar[8:12, 5:55] = True                       # 4 px thick, 50 px long
    assert representation.stroke_width(bar) == pytest.approx(4., rel=.1)
    assert representation.stroke_width(np.zeros((5, 5), bool)) == 0.


def test_thickness_steps_grow_and_shrink_the_mask():
    mask = np.zeros((20, 40), bool)
    mask[8:12, 10:30] = True
    sizes = [int(representation.reshaped(mask, s).sum()) for s in representation.STEPS]
    assert sizes == sorted(sizes) and sizes[1] == int(mask.sum()) and sizes[0] < sizes[1]
    assert matching.MATCH_IOU == .5
