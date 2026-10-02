"""Revision-specific baselines: the pure parts (components, conversions, proposals, schedules)."""
import numpy as np
import pytest

from revrender.baselines import component_cls, mathscrub, paper, removal
from revrender.baselines.common import read_archive, write_line_archive


def strip_with_ink():
    grey = np.full((96, 768), 252, np.uint8)
    grey[30:40, 100:140] = 20          # a block
    grey[60:62, 300:420] = 20          # a stroke
    grey[10:12, 10:12] = 20            # 4 px: below MIN_PIXELS
    return grey


def test_component_classifier_keeps_components_of_at_least_twelve_ink_pixels():
    labels, kept = component_cls.components(strip_with_ink())
    assert len(kept) == 2
    boxes = sorted((b[1].start, b[1].stop) for _, b in kept)
    assert boxes == [(100, 140), (300, 420)]


def test_crops_have_the_full_line_height_and_a_fixed_width():
    grey = strip_with_ink()
    labels, kept = component_cls.components(grey)
    for label, box in kept:
        g, m = component_cls.crop(grey, labels == label, box)
        assert g.shape == m.shape == (96, component_cls.CROP_W) and m.any()


def test_conversions_fill_dilate_and_box_a_component():
    ring = np.zeros((96, 768), bool)
    ring[20:40, 100:140] = True
    ring[25:35, 110:130] = False                          # a hole
    box = (slice(20, 40), slice(100, 140))
    assert component_cls.convert(ring, box, 'fill').sum() == 20*40
    assert component_cls.convert(ring, box, 'dil2').sum() > ring.sum()
    assert component_cls.convert(ring, box, 'box').sum() == 20*40
    with pytest.raises(Exception, match='unknown'):
        component_cls.convert(ring, box, 'bogus')


def test_a_component_is_positive_when_a_fifth_of_its_ink_lies_on_gold():
    comp = np.zeros((96, 768), bool)
    comp[10:20, 10:20] = True
    gold = np.zeros_like(comp)
    gold[10:12, 10:20] = True                              # 20 of 100 pixels
    assert component_cls.positive(comp, gold)
    gold[:] = False
    gold[10:11, 10:20] = True
    assert not component_cls.positive(comp, gold)


def test_removal_maps_become_scored_instances_and_closing_merges_small_gaps():
    removal_map = np.zeros((96, 768), np.float32)
    removal_map[30:40, 100:120] = .8
    removal_map[30:40, 122:140] = .4                        # 2 px gap
    masks, scores = removal.instances(removal_map, .3, 1)
    assert len(masks) == 2 and sorted(scores.round(2).tolist()) == [.4, .8]
    masks, scores = removal.instances(removal_map, .3, 7)
    assert len(masks) == 1 and scores[0] == pytest.approx(removal_map[masks[0]].mean())
    assert removal.windows() == [0, 192, 384]


def test_word_proposals_and_labels_follow_the_published_rule():
    grey = strip_with_ink()
    boxes = paper.word_proposals(grey, paper.MIN_INK)
    assert boxes and all(x2 > x1 and y2 > y1 for x1, y1, x2, y2 in boxes)
    gold = np.zeros((96, 768), bool)
    gold[30:40, 100:140] = True
    labels = [paper.candidate_label(gold, b) for b in boxes]
    assert sum(labels) >= 1 and paper.candidate_label(np.zeros((96, 768), bool), boxes[0]) == 0
    assert paper.otsu_ink(grey)[35, 120] and not paper.otsu_ink(grey)[0, 0]


def test_the_shape_features_have_seven_values():
    crop = np.full((40, 120), 250, np.uint8)
    crop[10:30, 20:100] = 30
    assert paper.seven_features(crop).shape == (7,)


def test_mathscrub_components_labels_and_geometry():
    image = np.repeat(strip_with_ink()[None], 3, axis=0)
    mask = np.zeros((96, 768), bool)
    mask[30:40, 100:140] = True
    page = mathscrub.extract_components({'sample_id': 's', 'image': image, 'mask': mask}, with_labels=True)
    labelled = {c['label'] for c in page['components']}
    assert labelled == {0, 1} and mathscrub.LABEL_FRACTION == .5
    view = mathscrub.resize_bilinear(np.arange(12, dtype=np.float32).reshape(3, 4))
    assert view.shape == (128, 128) and view[0, 0] == 0 and view[-1, -1] == 11
    stats = mathscrub.fit_geometry_statistics([{'components': [{'area': 10, 'max_side': 5}, {'area': 30, 'max_side': 9}]}])
    assert stats['count'] == 2 and stats['fitted_on'] == 'train_components_only'
    assert mathscrub.learning_rate_factor(0, 10, 100) == pytest.approx(.1)
    assert mathscrub.learning_rate_factor(10, 10, 100) == pytest.approx(1.)
    assert mathscrub.learning_rate_factor(100, 10, 100) == pytest.approx(0., abs=1e-12)


def test_an_archive_written_for_a_baseline_is_read_back_and_scored(tmp_path):
    gold = np.zeros((96, 768), bool)
    gold[30:40, 100:140] = True
    stray = np.zeros_like(gold)
    stray[60:70, 400:440] = True
    archive = tmp_path/'dev_instances'
    archive.mkdir()
    write_line_archive(archive, 0, 'a:x', [gold, stray], [.9, .3])
    write_line_archive(archive, 1, 'a:y', [], [])
    points, point = read_archive(archive, ['x', 'y'], [[gold], []])
    assert point['true_positive'] == 1 and point['f1'] > .6
    assert point['mask_threshold'] == .5 and 'pixel_iou' in point
