"""Synthetic revisions on clean lines: a struck-out stretch pasted into a line, a whole line struck, and the masks of the strokes."""
import random

import numpy as np
from PIL import Image

from revrender.data import paste_strikes as S


def line(words, height=40, width=240, gap=30, colour=(20, 20, 20)):
    """A white line with a block of ink for each word width."""
    out = np.full((height, width, 3), 255, np.uint8)
    x = 10
    for w in words:
        out[10:30, x:x+w] = colour
        x += w+gap
    return out


def test_words_are_runs_of_ink_columns_closer_than_a_gap_joined():
    host = line([30, 40, 20], gap=30)
    assert S.word_spans(S.ink_of(host), host.shape[0]) == [(10, 40), (70, 110), (140, 160)]
    close = line([30, 40], gap=6)
    assert S.word_spans(S.ink_of(close), close.shape[0]) == [(10, 86)]
    assert S.word_spans(S.ink_of(np.full((40, 80, 3), 255, np.uint8)), 40) == []


def test_a_stretch_taken_from_a_donor_holds_whole_words_and_is_trimmed_to_the_ink():
    donor = line([30, 40, 20], gap=30)
    rng = random.Random(1)
    for _ in range(20):
        piece = S.take_segment(donor, rng, 200)
        assert piece is not None and piece.shape[0] <= 24 and S.ink_of(piece).any()
    assert S.take_segment(np.full((40, 80, 3), 255, np.uint8), rng, 200) is None


def test_every_style_draws_strokes_in_the_ink_colour_and_a_mask_of_them():
    for style in S.STYLES:
        layer, opacity = S.strokes((120, 30), style, random.Random(3), (10, 20, 200), 3, 6)
        assert layer.shape == (30, 132, 3) and opacity.shape == (30, 132) and opacity.max() > .9
        dark = layer[opacity > .9]
        assert len(dark) and np.abs(dark.mean(0)-np.array([10, 20, 200])).max() < 20
        assert (opacity >= S.MASK_ALPHA).sum() > 0 and layer.min() >= 0 and layer.max() <= 255


def test_a_pasted_struck_stretch_widens_the_line_keeps_the_hosts_writing_and_masks_the_strokes():
    host, donor = line([50, 60]), line([40, 30])
    for seed in range(12):
        out = S.compose(host, [donor], random.Random(seed), 'paste', style='line')
        image, mask = np.asarray(out['image']), out['mask']
        assert out['kind'] == 'paste' and out['pasted'] == 1 and image.shape[1] > host.shape[1] and mask.shape == image.shape[:2] and mask.any()
        assert S.ink_of(image).sum() >= S.ink_of(host).sum()          # nothing of the host was lost, the pasted writing and strokes are added
        assert out['boxes'] and all(0 <= v <= 1000 for box in out['boxes'] for v in box)


def test_a_whole_line_strike_crosses_the_line_and_leaves_the_writing_under_it():
    host = line([50, 60, 40])
    out = S.compose(host, [], random.Random(5), 'whole', style='wave')
    mask = out['mask']
    assert out['kind'] == 'whole' and np.asarray(out['image']).shape == host.shape and mask.any(0).mean() > .95
    assert (np.asarray(out['image']).min(2)[S.ink_of(host)] < 200).all()       # the ink of the host is still dark under the strokes


def test_what_cannot_be_drawn_comes_out_clean_and_the_same_recipe_draws_the_same_sample():
    host = line([50, 60])
    blank = np.full((40, 100, 3), 255, np.uint8)
    assert S.compose(host, [], random.Random(1), 'paste')['kind'] == 'clean'
    assert S.compose(host, [blank], random.Random(1), 'paste')['kind'] == 'clean'
    assert S.compose(blank, [host], random.Random(1), 'whole')['kind'] == 'clean'
    clean = S.compose(host, [host], random.Random(1), 'clean')
    assert np.array_equal(np.asarray(clean['image']), host) and not clean['mask'].any() and clean['boxes'] == []
    a, b = S.compose(host, [host], random.Random(9), 'paste'), S.compose(host, [host], random.Random(9), 'paste')
    assert np.array_equal(np.asarray(a['image']), np.asarray(b['image'])) and np.array_equal(a['mask'], b['mask'])


def test_boxes_join_close_strokes_and_give_thousandths():
    mask = np.zeros((20, 200), bool)
    mask[5:10, 10:50] = True
    mask[5:10, 55:60] = True
    mask[12:18, 150:180] = True
    assert S.boxes_of(mask) == [[50, 250, 300, 500], [750, 600, 900, 900]]
    assert S.boxes_of(np.zeros((20, 200), bool)) == []


def test_a_stacked_stretch_sits_above_or_below_a_word_of_the_host_in_its_columns_and_keeps_the_hosts_writing():
    host, donor = line([50, 60, 40]), line([40, 30])
    seen = set()
    for seed in range(30):
        out = S.compose(host, [donor], random.Random(seed), 'stack', style='double')
        image, mask = np.asarray(out['image']), out['mask']
        assert out['kind'] == 'stack' and image.shape[0] > host.shape[0] and image.shape[1] >= host.shape[1] and mask.any()
        rows = np.flatnonzero(mask.any(1))
        seen.add('above' if rows.mean() < image.shape[0]/2 else 'below')
        assert S.ink_of(image).sum() >= S.ink_of(host).sum()
    assert seen == {'above', 'below'}
    assert S.compose(host, [], random.Random(1), 'stack')['kind'] == 'clean'


def fake_bank(tmp_path):
    """A bank of two real-looking marks: a struck line (a thin dark band) and a blackout (a dark disc), from 40 px lines."""
    stroke = np.full((6, 90), 255, np.uint8)
    stroke[2:4] = 40
    disc = np.full((24, 24), 255, np.uint8)
    yy, xx = np.mgrid[:24, :24]
    disc[(yy-12)**2+(xx-12)**2 < 120] = 30
    np.savez_compressed(tmp_path/'bank.npz', piece_0=stroke, piece_1=disc)
    pieces = [{'id': 'piece_0', 'kind': 'line', 'size': [90, 6], 'line_height': 40, 'ink_level': 40},
              {'id': 'piece_1', 'kind': 'blot', 'size': [24, 24], 'line_height': 40, 'ink_level': 30}]
    (tmp_path/'pieces.json').write_text(__import__('json').dumps(pieces))
    return S.Bank(tmp_path)


def test_a_real_mark_is_drawn_in_the_hosts_ink_across_its_region_with_room(tmp_path):
    bank = fake_bank(tmp_path)
    for style in S.GOLD_STYLES:
        layer, opacity = S.strokes((120, 30), style, random.Random(3), (10, 20, 200), 3, 6, bank, 40)
        assert layer.shape == (30, 132, 3) and opacity.shape == (30, 132) and opacity.max() > .9
        dark = layer[opacity > .9]
        assert np.abs(dark.mean(0)-np.array([10, 20, 200])).max() < 20
    line_layer, line_opacity = S.strokes((120, 30), 'gold_line', random.Random(3), (0, 0, 0), 3, 6, bank, 40)
    assert (line_opacity > .5).any(0).mean() > .95                    # a struck line runs the whole width, room included
    assert S.choose_style(random.Random(1), bank, 1.) in S.GOLD_STYLES and S.choose_style(random.Random(1), bank, 0.) in S.STYLES


def test_removing_a_struck_line_keeps_the_writing_it_crosses_and_removing_a_blackout_takes_all_it_covers(tmp_path):
    bank = fake_bank(tmp_path)
    host = line([50, 60, 40])
    for seed in range(6):
        out = S.compose(host, [], random.Random(seed), 'whole', bank=bank, gold_share=1.)
        writing = np.asarray(Image.fromarray(host).convert('L')) < S.WRITING_BELOW
        assert out['style'] in S.GOLD_STYLES and out['remove'].any()
        if out['style'] == 'gold_line':
            assert not (out['remove'] & writing).any() and (out['mask'] & ~writing <= out['remove']).all()
        else:
            assert (out['mask'] <= out['remove']).all()                 # a blackout's faint edge goes too
    for seed in range(6):
        out = S.compose(host, [line([40, 30])], random.Random(seed), 'paste', style='line')
        image, remove = np.asarray(out['image']), out['remove']
        assert out['style'] == 'line' and remove.any()
        restored = image.copy()
        restored[remove] = 255
        assert S.ink_of(restored).sum() >= S.ink_of(host).sum()      # the host and the struck writing are still there
    clean = S.compose(host, [], random.Random(1), 'clean', bank=bank, gold_share=1.)
    assert not clean['remove'].any() and clean['style'] is None


def test_a_second_revision_on_a_sample_moves_the_first_with_the_writing_and_is_numbered_next(tmp_path):
    bank = fake_bank(tmp_path)
    host = line([50, 60, 40])
    for seed in range(8):
        first = S.compose(host, [line([40, 30])], random.Random(seed), 'paste', bank=bank, gold_share=.5)
        assert set(np.unique(first['labels'])) == {0, 1} and np.array_equal(first['labels'] == 1, first['remove'])
        second = S.compose(np.asarray(first['image']), [line([30, 30])], random.Random(seed+100), 'stack' if seed % 2 else 'paste',
                           bank=bank, gold_share=.5, labels=first['labels'])
        labels = second['labels']
        assert labels.shape == np.asarray(second['image']).shape[:2] and set(np.unique(labels)) == {0, 1, 2}
        assert (labels == 1).sum() == first['remove'].sum()             # the first revision moved whole, nothing of it lost
        assert ((labels == 2) <= second['remove']).all() and (labels == 2).any()
