"""Where the handwriting is: printed templates, the answer box, the student's ink, ink-guided cuts, leftover pieces."""
import numpy as np
from PIL import Image

from revrender.data import handwriting as H


def sheet(seed, width=300, height=200):
    """A page of one question: the printed label and a dashed frame (the same on every page) and a stroke that is the student's own."""
    rng = np.random.default_rng(seed)
    grey = np.full((height, width), 255, np.uint8)
    for x in range(20, width-20, 14):                    # dashed top and bottom of the answer box
        grey[20:22, x:x+9] = 40
        grey[height-22:height-20, x:x+9] = 40
    for y in range(20, height-20, 14):
        grey[y:y+9, 20:22] = 40
        grey[y:y+9, width-22:width-20] = 40
    grey[26:36, 30:70] = 30                              # the printed label
    top, left = int(rng.integers(60, 140)), int(rng.integers(40, 160))
    grey[top:top+8, left:left+int(rng.integers(40, 90))] = 25
    return grey


def test_pages_of_a_question_share_a_key_and_the_printed_template_is_what_most_of_them_have(tmp_path):
    files = {}
    for k in range(10):
        Image.fromarray(sheet(k)).save(tmp_path/f'w{k}_3_1.jpg')
    assert H.group_key('w0_3_1', (300, 200)) == '3_300_200' and H.group_key('w0_s7_2', (10, 20)) == '7_10_20'
    files['3_300_200'] = sorted(tmp_path.glob('*.jpg'))
    templates = H.printed_templates(files)
    template = templates['3_300_200']
    assert template[30, 40] and template[20, 25] and not template[110, 100]           # the label and the frame are printed; the stroke is not
    assert H.printed_templates({'k': files['3_300_200'][:3]}) == {}                    # a group of fewer than eight pages has no template


def test_the_student_ink_is_what_is_left_inside_the_answer_box_off_the_template_and_the_figures(tmp_path):
    for k in range(10):
        Image.fromarray(sheet(k)).save(tmp_path/f'w{k}_3_1.jpg')
    template = H.printed_templates({'g': sorted(tmp_path.glob('*.jpg'))})['g']
    grey = sheet(3)
    top, bottom, left, right = H.answer_box(H.ink_map(grey))
    assert (top, bottom, left, right) == (20, 178, 20, 278) or (abs(top-20) <= 2 and abs(bottom-178) <= 2 and abs(left-20) <= 2 and abs(right-278) <= 2)
    ink = H.handwriting(grey, template)
    assert ink.sum() > 100 and not ink[26:36, 30:70].any() and not ink[:22].any()             # label and frame are gone, the stroke stays
    figure = H.handwriting(grey, template, figures=[[0, 0, 300, 200]])
    assert not figure.any()


def test_a_region_of_several_lines_is_cut_where_the_ink_has_gaps_not_in_equal_heights():
    ink = np.zeros((200, 400), bool)
    ink[10:40, 20:300] = True                            # a tall first line
    ink[120:135, 20:250] = True                          # a short second line
    ink[150:180, 20:300] = True                          # a third
    cuts = H.split_by_ink([0, 0, 400, 200], 3, ink)
    assert [c[1] for c in cuts] == [10, 120, 150] and [c[3] for c in cuts] == [40, 135, 180]      # each band is its line's own extent
    assert H.split_by_ink([0, 0, 400, 200], 2, np.zeros((200, 400), bool)) == [None, None]


def test_handwriting_no_box_covers_gets_a_box_and_a_tall_piece_is_cut_into_lines():
    ink = np.zeros((300, 400), bool)
    ink[20:40, 30:200] = True                            # covered by a line box
    ink[100:120, 50:250] = True                          # a line the OCR missed
    ink[200:215, 50:150] = True                          # a tall scribble over three lines, tight gaps of 2 px
    ink[217:232, 50:150] = True
    ink[234:290, 50:150] = True
    ink[250:252, 270:395] = True                         # a thin long rule
    ink[10:12, 10:13] = True                             # dust
    boxes = H.leftover_boxes(ink, [[20, 15, 220, 45]])
    assert any(b[1] <= 100 and b[3] >= 120 and b[0] <= 50 and b[2] >= 250 for b in boxes)
    assert not any(b[0] >= 260 for b in boxes) and not any(b[1] <= 11 for b in boxes)                  # no rule, no dust
    assert sum(1 for b in boxes if b[1] >= 195) >= 1 and all(b[3]-b[1] <= H.MAX_BOX_HEIGHT+2*4 for b in boxes)


def test_a_tall_piece_is_cut_at_the_emptiest_rows_and_a_short_one_is_left_whole():
    window = np.zeros((300, 50), bool)
    window[:100] = True
    window[103:200] = True
    window[210:300] = True
    assert H.lines_by_ink(window[:100]) == [(0, 100)]
    cuts = H.lines_by_ink(window)
    assert cuts[0][0] == 0 and cuts[-1][1] == 300 and all(b-a <= H.MAX_BOX_HEIGHT for a, b in cuts) and len(cuts) >= 3
    assert any(98 <= a <= 104 for a, _ in cuts[1:])


def test_marks_that_are_only_thin_strokes_are_not_writing():
    rule = np.zeros((20, 200), bool)
    rule[9:11, :] = True
    dashes = np.zeros((20, 200), bool)
    for x in range(0, 190, 14):
        dashes[9:11, x:x+9] = True
    stray = np.zeros((100, 20), bool)
    stray[:, 8:11] = True
    words = np.zeros((30, 200), bool)
    for x in (10, 60, 110):
        words[5:25, x:x+14] = True
    blot = np.zeros((60, 80), bool)
    blot[10:50, 10:70] = True
    assert not H.looks_written(rule) and not H.looks_written(dashes) and not H.looks_written(stray) and not H.looks_written(np.zeros((9, 9), bool))
    assert H.looks_written(words) and H.looks_written(blot)


def test_blobs_on_one_row_are_the_words_of_a_line_and_blobs_on_other_rows_stay_apart():
    near = H.row_pieces([[10, 100, 60, 130], [100, 105, 150, 135]])
    assert near == [[10, 100, 150, 135]]
    assert len(H.row_pieces([[10, 100, 60, 130], [100, 140, 150, 170]])) == 2                  # another row
    assert len(H.row_pieces([[10, 100, 60, 130], [200, 100, 250, 130]])) == 2                  # a gap wider than a word
    assert len(H.row_pieces([[10, 100, 60, 130], [70, 100, 120, 130], [130, 102, 180, 128]])) == 1
