"""The user's annotations read back: final text and struck stretches per line, marks paired to lines and cut to ink, counts per split."""
import json

import numpy as np
import pytest
from PIL import Image

from revrender.data import page_labels as L


def scan(folder, image_id):
    grey = np.full((80, 300), 255, np.uint8)
    grey[18:26, 20:120] = 0                       # writing on the first line, struck through
    grey[50:60, 20:200] = 0                       # writing on the second line
    folder.mkdir(exist_ok=True)
    Image.fromarray(grey).convert('RGB').save(folder/f'{image_id}.jpg', quality=95)


def export(path, pages):
    path.write_text(json.dumps({'schema': L.SCHEMA, 'pack': 'paper2-test', 'pages': pages}, ensure_ascii=False), encoding='utf-8')


LINES = [{'poly': [[10, 10], [290, 10], [290, 35], [10, 35]], 'text': 'x=[[3]]5', 'source': 'ocr', 'line_id': 'q:L1'},
         {'poly': [[10, 40], [290, 40], [290, 70], [10, 70]], 'text': 'y=2', 'source': 'ocr', 'line_id': 'q:L2'},
         {'poly': [[10, 72], [290, 72], [290, 79], [10, 79]], 'text': '第1页', 'source': 'ocr', 'not_writing': True}]
MARKS = [{'type': 'line', 'shape': 'line', 'path': [[20, 22], [118, 22]]}, {'type': 'blot', 'shape': 'box', 'x': 150, 'y': 72, 'w': 10, 'h': 5}]


def test_marks_go_to_the_line_that_holds_them_left_to_right():
    per_line, outside = L.assign(LINES, MARKS + [{'type': 'blot', 'shape': 'box', 'x': 100, 'y': 15, 'w': 5, 'h': 5}])
    assert per_line == [[0, 2], [], []] and outside == [1]


def test_a_struck_line_is_rasterised_thin_and_cut_to_ink():
    ink = np.zeros((80, 300), bool)
    ink[18:26, 20:120] = True
    mask = L.rasterise(MARKS[0], (300, 80), ink)
    assert mask.any() and not mask[:, 130:].any() and mask.sum() <= (L.STROKE_PX+1)*100


def test_the_tool_reads_the_questions_marked_done_and_counts_per_split(tmp_path):
    scans = tmp_path/'scans'
    scan(scans, 'q_5_1')
    scan(scans, 'r_6_1')
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps({'pages': [{'image_id': 'q_5_1', 'split': 'test'}, {'image_id': 'r_6_1', 'split': 'val'}]}))
    first, second = tmp_path/'a.json', tmp_path/'b.json'
    export(first, {'q_5_1': {'lines': LINES, 'marks': MARKS, 'done': False}, 'r_6_1': {'lines': LINES[:1], 'marks': [], 'done': True}})
    export(second, {'q_5_1': {'lines': LINES, 'marks': MARKS, 'done': True}})          # the later file wins
    out = tmp_path/'out'
    assert L.main(['--export', str(first), '--export', str(second), '--selection', str(selection), '--pages', str(scans), '--output-dir', str(out)]) == 0
    rows = [json.loads(line) for line in (out/'labels.jsonl').read_text(encoding='utf-8').splitlines()]
    q = [r for r in rows if r['image_id'] == 'q_5_1']
    assert q[0]['final'] == 'x=5' and q[0]['struck'] == [{'text': '3', 'legible': 'yes'}] and q[0]['marks'] == [0] and q[0]['paired']
    assert q[1]['struck'] == [] and q[1]['paired'] and q[2]['not_writing'] and not q[2]['paired']
    summary = json.loads((out/'summary.json').read_text())
    assert summary['test']['lines'] == 2 and summary['test']['struck_stretches'] == 1 and summary['test']['marks_outside_lines'] == 1
    assert summary['val']['unpaired_lines'] == 1                                   # a [[ ]] with no mark is counted, never fixed
    with np.load(out/'marks'/'q_5_1.npz') as masks:
        assert np.unpackbits(masks['mask_0']).sum() > 0
    marks = json.loads((out/'marks'/'q_5_1.json').read_text())
    assert [m['line'] for m in marks['marks']] == [0, None]
    with pytest.raises(Exception, match='not an export'):
        bad = tmp_path/'bad.json'
        bad.write_text('{"schema": "x"}')
        L.main(['--export', str(bad), '--selection', str(selection), '--pages', str(scans), '--output-dir', str(tmp_path/'o2')])
