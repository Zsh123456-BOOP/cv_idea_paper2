"""Real revision marks cut out of paper 1's gold for the synthetic revisions: which writers, how a mark is cut, its kind and ink."""
import json

import numpy as np
from PIL import Image

from revrender import pages as pages_module
from revrender.data import strike_bank as B


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


def test_a_mark_is_cut_to_its_mask_on_paper_with_the_grey_of_its_own_ink(tmp_path):
    grey = np.full((60, 100), 250, np.uint8)
    grey[20:24, 10:80] = 50                   # the strike
    grey[10:40, 30:34] = 20                   # a letter it crosses, outside the mask but where they meet
    mask = np.zeros((40, 90), bool)
    mask[10:14, 5:75] = True                  # on the box [5, 10, 95, 50]
    piece, inside = B.cut(grey, {'box': [5, 10, 95, 50], 'mask': mask})
    assert piece.shape == (4, 70) and inside.all() and piece.max() == 50 and B.ink_level(piece, inside) == 50          # the crossing letter does not set it
    assert B.cut(grey, {'box': [0, 0, 4, 4], 'mask': np.zeros((4, 4), bool)}) is None


def test_the_bank_holds_the_training_writers_marks_with_their_lines_heights(tmp_path):
    scans, gold = tmp_path/'scans', tmp_path/'gold'
    scans.mkdir()
    gold.mkdir()
    grey = np.full((100, 300), 250, np.uint8)
    grey[30:33, 20:200] = 40
    grey[60:80, 220:240] = 30
    strike, blot = np.zeros((40, 280), bool), np.zeros((40, 280), bool)
    strike[10:13, 10:180] = True
    blot[30:40, 210:230] = True
    for name in ('a_5_1', 'c_5_1'):
        Image.fromarray(grey).convert('RGB').save(scans/f'{name}.jpg', quality=98)
        pages_module.write_page(gold, name, (300, 100), [{'box': [10, 20, 290, 60], 'source': 'gold', 'ocr_type': None, 'ocr_text': ''}],
                                [{'line': 0, 'score': 1., 'box': [10, 20, 290, 60], 'mask': strike}, {'line': 0, 'score': 1., 'box': [10, 50, 290, 90], 'mask': blot}])
    (gold/'run_manifest.json').write_text('{}')
    mapping, same = tmp_path/'map.json', tmp_path/'same.json'
    mapping.write_text(json.dumps({'rows': [{'group_uuid': 'a', 'split': 'train'}, {'group_uuid': 'c', 'split': 'dev'}]}))
    same.write_text(json.dumps({'pairs': []}))
    out = tmp_path/'bank'
    assert B.main(['--gold', str(gold), '--pages', str(scans), '--mapping', str(mapping), '--same-student', str(same), '--output-dir', str(out)]) == 0
    pieces = json.loads((out/'pieces.json').read_text())
    assert [(p['image_id'], p['kind'], p['line_height']) for p in pieces] == [('a_5_1', 'line', 40), ('a_5_1', 'blot', 40)]
    with np.load(out/'bank.npz') as arrays:
        assert arrays['piece_0'].shape == (3, 170) and arrays['piece_0'].max() < 120
    assert (out/'sheet_line.png').exists() and (out/'sheet_blot.png').exists()
