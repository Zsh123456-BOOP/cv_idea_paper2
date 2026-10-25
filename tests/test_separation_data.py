import json

import numpy as np
import pytest

from revrender.data.separation import IGNORE, detector_hint, jittered_hint, native_hints, pair_labels, simulated_hint
from revrender.util import SEED


def synthetic_pair():
    clean = np.full((32, 32), 250, np.uint8); clean[10:22, 8:12] = 120          # a stroke of writing
    struck = clean.copy(); struck[15:17, 2:30] = 30                            # a strike across it
    struck[14, 2:30] = np.minimum(clean[14, 2:30], 225)                         # its faint edge: darkened by 25 on paper
    return struck, clean


def test_a_synthetic_pair_erases_the_strike_on_paper_redraws_it_over_writing_and_ignores_its_faint_edge():
    x, y = synthetic_pair(); erase, cross = pair_labels(x, y, human=False)
    assert erase[15, 3] == 1 and cross[15, 3] == 0                              # strike on paper
    assert cross[15, 9] == 1 and erase[15, 9] == 0                              # strike over the writing
    assert erase[14, 3] == IGNORE and cross[14, 3] == IGNORE                    # faint edge
    assert erase[5, 5] == 0 and cross[5, 5] == 0 and cross[12, 9] == 0          # paper and untouched writing


def test_a_human_tile_erases_what_its_target_whitened_and_never_claims_a_crossing_on_ink():
    x = np.full((8, 8), 250, np.uint8); x[2:6, 2:6] = 40; y = x.copy(); y[3, 2:6] = 255
    erase, cross = pair_labels(x, y, human=True)
    assert erase[3].tolist() == [0, 0, 1, 1, 1, 1, 0, 0] and erase.sum() == 4
    assert (cross[x < 200] == IGNORE).all() and (cross[x >= 200] == 0).all()


def test_the_simulated_hint_covers_the_strike_and_nearby_ink_only_and_repeats_with_its_seed():
    x, y = synthetic_pair()
    a = simulated_hint(x, y, np.random.default_rng(SEED), reach=2, drop=0.)
    b = simulated_hint(x, y, np.random.default_rng(SEED), reach=2, drop=0.)
    assert np.array_equal(a, b) and a[15, 3] == 1 and a[15, 9] == 1
    assert not a[x >= 200].any() and not a[2:8].any()                           # ink only, nothing far from the strike
    assert not simulated_hint(y, y, np.random.default_rng(SEED)).any()          # a clean pair: no hint


def test_jitter_keeps_a_hint_on_the_ink_within_a_pixel():
    x = np.full((20, 20), 250, np.uint8); x[5:15, 5:15] = 30; hint = np.zeros_like(x); hint[8:12, 8:12] = 1
    for seed in range(6):
        h = jittered_hint(hint, x, np.random.default_rng(seed)).astype(bool)
        assert h[9:11, 9:11].all() and not h[:7].any() and not h[x >= 200].any()


def test_the_detector_hint_is_every_instance_mask_inside_the_window_on_its_ink():
    crop = np.full((10, 20), 250, np.uint8); crop[:, 5:15] = 40
    first = {'box': [95, 48, 110, 56], 'mask': np.ones((8, 15), bool)}          # overlaps the window's left part
    other = {'box': [300, 0, 310, 10], 'mask': np.ones((10, 10), bool)}         # elsewhere on the page
    hint = detector_hint([first, other], (100, 50, 120, 60), crop)
    assert hint.shape == crop.shape and hint[:6, 5:10].all() and not hint[6:].any()
    assert not hint[:, :5].any() and not hint[:, 10:].any()                     # paper, and outside the first instance


def test_native_hints_cut_each_crop_from_its_pages_detection(tmp_path):
    from revrender.pages import write_page
    det = tmp_path/'det'; det.mkdir(); (det/'run_manifest.json').write_text('{}')
    write_page(det, 'p1', (100, 60), [], [{'line': 0, 'score': 1., 'box': [10, 10, 30, 20], 'mask': np.ones((10, 20), bool)}])
    image = np.full((10, 20), 40, np.uint8)
    (tmp_path/'v.json').write_text(json.dumps([{'id': 'r', 'page': 'p1', 'box': [20, 10, 40, 20]}])); np.savez(tmp_path/'v.npz', image_0=image)
    native_hints(tmp_path/'v.json', det, tmp_path/'out')
    hint = np.load(tmp_path/'out'/'hints.npz')['hint_0']
    assert hint[:, :10].all() and not hint[:, 10:].any()
    assert json.loads((tmp_path/'out'/'run_manifest.json').read_text())['crops'] == 1


def test_a_hinted_restorer_gets_its_hint_tiled_with_the_crop():
    from revrender.baselines.restoration import restore_native
    grey = np.full((40, 300), .9, np.float32); hint = np.zeros_like(grey); hint[10:20, 50:250] = 1
    assert np.array_equal(restore_native(lambda x, h: h, grey, hint), hint)
    assert np.array_equal(restore_native(lambda x: x, grey), grey)


def test_a_human_tiles_stroke_labels_skip_the_kept_ink_next_to_the_strike():
    pytest.importorskip('skimage')
    from revrender.data.separation import stroke_labels
    x = np.full((20, 40), 250, np.uint8); x[9:11, 2:38] = 30; x[4:16, 18:21] = 60; y = x.copy(); y[9:11, 2:38] = 255; y[9:11, 18:21] = 60
    mask, centre, half = stroke_labels(x, y, human=True)
    assert mask[9, 5] == 1 and centre[9:11, 5].sum() == 1 and half[centre == 1].max() >= 1
    assert mask[9, 19] == IGNORE and centre[9, 19] == IGNORE and mask[2, 19] == 0      # under the kept writing: unknown; far: no


def test_the_stroke_geometry_holds_the_whole_strike_and_its_half_width():
    pytest.importorskip('skimage')
    from revrender.data.separation import stroke_geometry
    alpha = np.zeros((20, 40), np.float32); alpha[8:13, 3:37] = 1
    mask, centre, half = stroke_geometry(alpha)
    assert mask.sum() == 5*34 and centre[10, 10] == 1 and centre[:8].sum() == 0 and abs(half[10, 10]-3) < 1e-6


def test_writing_labels_are_the_hosts_ink_or_the_kept_ink_with_the_struck_writing_marked():
    from revrender.data.separation import writing_labels
    y = np.full((20, 40), 250, np.uint8); y[4:16, 18:21] = 60; x = y.copy(); x[9:11, 2:38] = 30
    labels, near = writing_labels(x, y, human=False)
    assert labels[5, 19] == 1 and labels[9, 5] == 0 and near[6, 5] and not near[0, 0]
    h = x.copy(); target = x.copy(); target[9:11, 2:38] = 255; target[9:11, 18:21] = 60
    labels, near = writing_labels(h, target, human=True)
    assert labels[4, 19] == 1 and labels[10, 5] == IGNORE or labels[10, 5] == 0
