"""Real struck lines cut out of paper 1's gold for the synthetic strikes: which writers, the cut along the centre line, what is left out."""
import json

import numpy as np
import pytest
from PIL import Image

from revrender import pages as pages_module
from revrender.data import strike_bank as B


def test_redrawing_a_trajectory_does_not_copy_a_crossing_letter():
    centre = np.zeros((32, 100), bool)
    centre[16, 10:90] = True
    plain = np.full(centre.shape, 255, np.uint8)
    plain[15:18, 10:90] = 50
    crossed = plain.copy()
    crossed[3:29, 45:49] = 10
    a, ma = B.traced_pixels(plain, centre, 3.)
    b, mb = B.traced_pixels(crossed, centre, 3.)
    assert np.array_equal(a, b) and np.array_equal(ma, mb)
    assert b[5, 46] == 255 and b[16, 46] == 50
    assert ma[16, 10:90].all()


def test_redrawing_keeps_a_real_cross_trajectory():
    centre = np.zeros((30, 40), bool)
    centre[15, 5:35] = True
    centre[5:25, 20] = True
    grey = np.where(centre, 60, 255).astype(np.uint8)
    result, inside = B.traced_pixels(grey, centre, 2.)
    assert inside[centre].all() and result[15, 20] == 60
    assert not inside[2, 2]


def test_a_mark_is_a_struck_line_when_thin_and_long_else_a_blackout():
    line = np.zeros((40, 200), bool)
    line[18:22, 10:190] = True
    blob = np.zeros((40, 200), bool)
    blob[5:35, 50:90] = True
    assert B.kind_of(line, 40) == 'line' and B.kind_of(blob, 40) == 'blot'


def test_only_training_writers_whose_other_id_is_not_elsewhere_give_marks(tmp_path):
    mapping, same = tmp_path/'map.json', tmp_path/'same.json'
    mapping.write_text(json.dumps({'rows': [{'group_uuid': w, 'split': s} for w, s in (('a', 'train'), ('b', 'train'), ('c', 'dev'), ('d', 'train'), ('e', 'test'))]}))
    same.write_text(json.dumps({'pairs': [{'x': 'b', 'y': 'c'}, {'x': 'd', 'y': 'e'}]}))
    assert B.training_writers(mapping, same) == {'a'}


def test_a_struck_line_is_cut_to_its_stroke_and_not_to_the_wider_band():
    pytest.importorskip('skimage')
    grey = np.full((40, 120), 250, np.uint8)
    grey[19:22, 10:110] = 40                  # the strike, 3 pixels thick
    grey[8:32, 50:54] = 20                    # a letter it crosses
    band = np.zeros_like(grey, bool)
    band[14:27, 8:112] = True                 # the gold band, much wider than the stroke
    piece, inside, measures = B.strike(grey, band, 40)
    assert piece is not None, measures
    assert 2 <= measures['width'] <= 4 and inside.shape[0] <= 7          # the stroke and its edge, not the band
    assert inside.sum() < band.sum()/2 and piece.min() == 20             # only where it crosses is the letter in it


def test_what_is_not_one_clean_stroke_is_left_out():
    pytest.importorskip('skimage')
    from PIL import ImageDraw
    grey = np.full((40, 120), 250, np.uint8)
    wide = np.zeros_like(grey, bool)
    wide[5:35, 20:100] = True
    grey[5:35, 20:100] = 30                   # a blackout
    assert B.strike(grey, wide, 40)[2]['reason'] == 'wide'
    canvas = Image.new('L', (120, 60), 0)
    draw = ImageDraw.Draw(canvas)
    for a, b in (((10, 10), (110, 50)), ((10, 50), (110, 10)), ((10, 30), (110, 30)), ((60, 2), (60, 58))):
        draw.line([a, b], fill=255, width=5)  # a scribble: strokes every way across a word
    band = np.asarray(canvas) > 0
    grey = np.where(band, 40, 250).astype(np.uint8)
    assert B.strike(grey, band, 120)[2]['reason'] == 'scribble'
    dot = np.zeros((40, 120), bool)
    dot[18:22, 50:56] = True
    assert B.strike(np.where(dot, 40, 250).astype(np.uint8), dot, 40)[2]['reason'] in ('short', 'no centre line')


def test_the_bank_holds_the_training_writers_struck_lines_with_their_lines_heights(tmp_path):
    pytest.importorskip('skimage')
    scans, gold = tmp_path/'scans', tmp_path/'gold'
    scans.mkdir()
    gold.mkdir()
    grey = np.full((100, 300), 250, np.uint8)
    grey[30:33, 20:200] = 40
    grey[55:85, 220:250] = 30
    strike, blot = np.zeros((40, 280), bool), np.zeros((40, 280), bool)
    strike[8:15, 8:182] = True
    blot[5:35, 210:240] = True
    for name in ('a_5_1', 'c_5_1'):
        Image.fromarray(grey).convert('RGB').save(scans/f'{name}.png')
        pages_module.write_page(gold, name, (300, 100), [{'box': [10, 20, 290, 60], 'source': 'gold', 'ocr_type': None, 'ocr_text': ''}],
                                [{'line': 0, 'score': 1., 'box': [10, 20, 290, 60], 'mask': strike}, {'line': 0, 'score': 1., 'box': [10, 50, 290, 90], 'mask': blot}])
    (gold/'run_manifest.json').write_text('{}')
    mapping, same = tmp_path/'map.json', tmp_path/'same.json'
    mapping.write_text(json.dumps({'rows': [{'group_uuid': 'a', 'split': 'train'}, {'group_uuid': 'c', 'split': 'dev'}]}))
    same.write_text(json.dumps({'pairs': []}))
    out = tmp_path/'bank'
    assert B.main(['--gold', str(gold), '--pages', str(scans), '--mapping', str(mapping), '--same-student', str(same), '--output-dir', str(out)]) == 0
    pieces = json.loads((out/'pieces.json').read_text())
    assert [(p['image_id'], p['kind'], p['line_height']) for p in pieces] == [('a_5_1', 'line', 40)]
    left, top, right, bottom = pieces[0]['box']
    assert 18 <= left <= 21 and 28 <= top <= 30 and 30 <= bottom <= 35 and 190 <= right <= 196     # stroke rows 30-32; the band stops at column 192
    assert [m['reason'] for m in json.loads((out/'left_out.json').read_text())] == ['wide']
    with np.load(out/'bank.npz') as arrays:
        assert arrays['piece_0'].shape[0] <= 7 and arrays['piece_0'].min() == 40
    assert (out/'sheet_kept.png').exists() and (out/'sheet_left_out.png').exists()
