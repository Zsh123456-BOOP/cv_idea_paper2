"""Pages kept to their question's own frame: the frame is the sheet's (the rows every page of the group has), the neighbours' lines, the
marks at the corners and the dust are taken off, handwriting is not."""
import json

import numpy as np
import pytest
from PIL import Image

from revrender.data import clean_pages as C

W, H = 600, 200
TOP, BOTTOM, ABOVE, BELOW = 22, 170, 10, 190          # the frame's lines and its neighbours', on every page of the group


def dashes_across(grey, y, x0=20, x1=580, dash=10, gap=6, value=60):
    for x in range(x0, x1, dash+gap):
        grey[y-1:y+2, x:x+dash] = value


def dashes_down(grey, x, y0, y1, dash=10, gap=6, value=60):
    for y in range(y0, y1, dash+gap):
        grey[y:y+dash, x-1:x+2] = value


def sheet(seed, extra_row=None, mark=True):
    """One page of the group: the frame and its neighbours, handwriting, a black mark outside, dust, haze."""
    rng = np.random.default_rng(seed)
    grey = np.full((H, W), 255, np.uint8)
    for y in (ABOVE, TOP, BOTTOM, BELOW):
        dashes_across(grey, y)
    for x in (24, 576):
        dashes_down(grey, x, TOP+12, BOTTOM-12)
    grey[60:76, 100:300] = 20                              # handwriting (a thick bar)
    grey[90:94, 320:420] = 30                              # more of it
    grey[196:199, 60:300] = 40                             # a neighbour's line of writing under its top
    if mark:
        grey[176:199, 0:18] = 0                            # a black alignment block outside the frame
    if extra_row is not None:
        dashes_across(grey, extra_row, 60, 500, value=50)  # handwriting that looks dashed, on this page only
    for k in range(6):                                     # dust: six 2 x 2 specks in the blank, far from each other
        grey[100+9*k:102+9*k, 150+70*k:152+70*k] = 30
    grey[110:150, 40:120] = np.where(rng.random((40, 80)) < .5, 248, 255)     # haze on the paper
    return grey


def group(tmp_path, count=10, name='w{}_3_1'):
    folder = tmp_path/'pages'
    folder.mkdir()
    for i in range(count):
        Image.fromarray(sheet(i, extra_row=130 if i in (2, 5) else None)).save(folder/f'{name.format(i)}.png')
    return folder


def lines_of(grey):
    return C.page_lines(grey)


def test_the_sheet_s_rows_are_the_frame_and_a_row_only_some_pages_have_is_not():
    pages = [sheet(i, extra_row=130 if i in (2, 5) else None) for i in range(10)]
    rows = [lines_of(p)[0] for p in pages]
    assert rows[0] == [ABOVE, TOP, BOTTOM, BELOW] and rows[2] == [ABOVE, TOP, 130, BOTTOM, BELOW]
    frame = C.group_frame(rows, [lines_of(p)[1] for p in pages])
    assert (frame['top'], frame['bottom'], frame['above'], frame['below']) == (TOP, BOTTOM, ABOVE, BELOW)     # not the ruled-out row 130
    assert abs(frame['left']-24) <= 1 and abs(frame['right']-576) <= 1


def test_the_box_is_the_frame_with_a_margin_that_stops_short_of_a_close_neighbour():
    pages = [sheet(i) for i in range(10)]
    lines = [lines_of(p) for p in pages]
    frame = C.group_frame([l[0] for l in lines], [l[1] for l in lines])
    left, top, right, bottom = C.page_box((H, W), frame, *lines[0])
    assert top == TOP-4 and bottom == BOTTOM+6                          # 12 px to the line above: 4; 20 px below: the full 6
    assert top > ABOVE+1 and bottom < BELOW-1                           # neighbours' lines stay outside
    assert abs(left-(24-C.FRAME_MARGIN)) <= 1 and abs(right-(576+C.FRAME_MARGIN)) <= 1
    snapped = C.page_box((H, W), frame, [TOP+2, BOTTOM-1], lines[0][1])  # a page whose own rows are a pixel or two off
    assert snapped[1] == TOP+2-5 and snapped[3] == BOTTOM-1+6            # the margin is worked out from the page's own row: 14 px to the line above gives 5
    assert C.page_box((H, W), frame, [], [])[1:4:2] == (TOP-4, BOTTOM+6)  # without rows of its own the page takes the group's


def test_clean_whites_out_what_lies_beyond_the_box_and_takes_off_dust_and_haze_and_nothing_else():
    pages = [sheet(i) for i in range(10)]
    lines = [lines_of(p) for p in pages]
    frame = C.group_frame([l[0] for l in lines], [l[1] for l in lines])
    box = C.page_box((H, W), frame, *lines[3])
    cleaned, specks = C.clean(pages[3], box)
    left, top, right, bottom = box
    assert (cleaned[:top] == 255).all() and (cleaned[bottom+1:] == 255).all() and (cleaned[:, :left] == 255).all()   # neighbours, writing under, the mark
    assert specks == 6 and (cleaned[100:160, 140:560][pages[3][100:160, 140:560] == 30] == 255).all()   # all the dust is white
    inside = np.zeros((H, W), bool)
    inside[top:bottom+1, left:right+1] = True
    original = pages[3]
    ink = (original < 200) & inside
    assert ((cleaned < 200) & inside).sum() == ink.sum()-4*specks        # the dust, and only it, is gone from the ink inside
    assert (cleaned[60:76, 100:300] == 20).all() and (cleaned[90:94, 320:420] == 30).all()            # handwriting untouched
    assert (cleaned[110:150, 40:120] == 255).all()                       # the haze (248) far from ink is bleached
    near = cleaned[56:58, 100:300]
    assert (near == original[56:58, 100:300]).all()                      # next to ink nothing changes
    assert (C.clean(original, None)[0][185:, :] != 255).any()            # no box: only dust and haze are taken


def test_the_tool_writes_the_cleaned_pages_and_their_boxes_and_asks_a_group_for_enough_pages(tmp_path):
    folder = group(tmp_path)
    out = tmp_path/'clean'
    assert C.main(['--pages', str(folder), '--output-dir', str(out)]) == 0
    boxes = json.loads((out/'frames.json').read_text())
    assert sorted(boxes) == [f'w{i}_3_1' for i in range(10)] and all(b is not None for b in boxes.values())
    page = np.asarray(Image.open(out/'w4_3_1.png'))
    assert page.shape == (H, W) and (page[:boxes['w4_3_1'][1]] == 255).all()
    manifest = json.loads((out/'run_manifest.json').read_text())
    assert (manifest['pages'], manifest['no_frame'], manifest['groups'], manifest['groups_without_frame']) == (10, 0, 1, 0)
    assert manifest['specks'] == 8*6+2*4              # six specks a page, but two on each of the two pages with a ruled row next to them stay: ink is near
    with pytest.raises(FileExistsError):
        C.main(['--pages', str(folder), '--output-dir', str(out)])
    (tmp_path/'ids.txt').write_text('w1_3_1\n')
    C.main(['--pages', str(folder), '--ids', str(tmp_path/'ids.txt'), '--output-dir', str(tmp_path/'one')])
    assert sorted(p.name for p in (tmp_path/'one').glob('*.png')) == ['w1_3_1.png']
    assert json.loads((tmp_path/'one'/'frames.json').read_text())['w1_3_1'] == boxes['w1_3_1']          # the frame is read from all the pages
    few = tmp_path/'few'
    few.mkdir()
    for i in range(3):
        Image.fromarray(sheet(i)).save(few/f'v{i}_4_1.png')
    C.main(['--pages', str(few), '--output-dir', str(tmp_path/'dust')])
    assert json.loads((tmp_path/'dust'/'frames.json').read_text()) == {f'v{i}_4_1': None for i in range(3)}
