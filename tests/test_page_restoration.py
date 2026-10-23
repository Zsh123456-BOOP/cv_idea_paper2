import numpy as np
from PIL import Image

from revrender.data.pixel_masks import encode
from revrender.data.restoration_pairs import question_gold
from revrender.eval.page_restoration import erase_boxes, score


def _question():
    page = np.full((60, 120), 255, np.uint8); page[10:50, 10:110:6] = 50          # writing everywhere
    strike = np.zeros((20, 40), bool); strike[9:11, :] = True; page[10:30, 10:50][strike] = 20
    mask = np.zeros((20, 40), bool); mask[9:11, :] = True
    found = dict(id='a', box=[10, 10, 50, 30], region=[[10, 10], [50, 10], [50, 30], [10, 30]], runs=encode(mask), confirmed=True, detector_index=0)
    hand = dict(found, id='b', box=[60, 30, 100, 50], region=[[60, 30], [100, 30], [100, 50], [60, 50]])
    del hand['detector_index']; page[30:50, 60:100][strike] = 20
    return Image.fromarray(page), [found, hand]


def test_a_box_model_never_reaches_a_hand_added_strike_and_a_whole_page_eraser_is_charged_outside_regions():
    page, revisions = _question()
    grey, gold, regions = question_gold(page, revisions)
    _, hand, _ = question_gold(page, [r for r in revisions if 'detector_index' not in r])
    assert gold.sum() == 2*80 and hand.sum() == 80
    perfect = np.where(gold, 1., grey/255.)
    tp, fp, fn, writing, fp_out, writing_out, hand_tp, hand_all = score(grey, gold, regions, hand, perfect)
    assert tp == 160 and fp == 0 and fn == 0 and hand_tp == hand_all == 80
    boxes = erase_boxes(lambda x: np.ones_like(x), grey, revisions)               # erases everything it is shown
    tp, fp, fn, writing, fp_out, writing_out, hand_tp, hand_all = score(grey, gold, regions, hand, boxes)
    assert hand_tp == 0 and fp_out == 0 and tp == 80
    everything = np.ones(grey.shape, np.float32)
    assert score(grey, gold, regions, hand, everything)[4] == (grey < 200)[~regions].sum()
