"""The user's revision annotations read back: the pixels each stroke removes, revisions and their regions, the scan with them removed."""
import json

import numpy as np
import pytest
from PIL import Image

from revrender.data import page_labels as L


def scan(folder, image_id):
    """A word (a block of ink) struck through by a thinner stroke that runs past it, and a separate blackened blob."""
    grey = np.full((80, 300), 255, np.uint8)
    grey[15:35, 40:100] = 30                      # the word
    grey[24:27, 20:140] = 0                       # the strike: crosses the word, runs out on both sides
    grey[50:70, 200:240] = 10                     # a blackout
    folder.mkdir(exist_ok=True)
    Image.fromarray(grey).convert('RGB').save(folder/f'{image_id}.png')


def export(path, pages):
    path.write_text(json.dumps({'schema': L.SCHEMA, 'pack': 'paper2-revisions-test', 'pages': pages}), encoding='utf-8')


STATE = {'strokes': [{'id': 's1', 'kind': 'line', 'path': [[20, 25], [38, 25]]}, {'id': 's3', 'kind': 'line', 'path': [[102, 25], [139, 25]]},
                     {'id': 's5', 'kind': 'blot', 'path': [[198, 48], [242, 48], [242, 72], [198, 72]]}],
         'groups': [{'id': 'g2', 'strokes': ['s1', 's3'], 'region': [[17, 12], [143, 12], [143, 38], [17, 38]], 'region_auto': True},
                    {'id': 'g6', 'strokes': ['s5'], 'region': [[195, 45], [245, 45], [245, 75], [195, 75]], 'region_auto': False}], 'next': 6}


def test_a_struck_line_takes_the_stroke_it_was_drawn_on_and_not_the_writing_it_stopped_at():
    grey = np.full((80, 300), 255, np.uint8)
    grey[15:35, 40:100] = 30
    grey[24:27, 20:140] = 0
    ink = grey < 200
    from scipy import ndimage
    depth = ndimage.distance_transform_edt(ink)
    pixels = L.stroke_pixels(STATE['strokes'][0], ink, ink, depth, (300, 80))
    assert pixels[24:27, 20:38].all() and not pixels[:, 40:100].any() and not pixels[:20, :].any()
    assert 2 <= L.stroke_width([[20, 25], [38, 25]], ink, depth, (300, 80)) <= 4


def test_the_tool_removes_the_revisions_keeps_the_writing_and_counts_per_split(tmp_path):
    scans = tmp_path/'scans'
    scan(scans, 'q_5_1')
    scan(scans, 'r_6_1')
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps({'pages': [{'image_id': 'q_5_1', 'split': 'test'}, {'image_id': 'r_6_1', 'split': 'val'}]}))
    first, second = tmp_path/'a.json', tmp_path/'b.json'
    export(first, {'q_5_1': {**STATE, 'done': False}, 'r_6_1': {'strokes': [], 'groups': [], 'next': 0, 'done': True}})
    export(second, {'q_5_1': {**STATE, 'done': True}})                                  # the later file wins
    out = tmp_path/'out'
    assert L.main(['--export', str(first), '--export', str(second), '--selection', str(selection), '--pages', str(scans), '--output-dir', str(out)]) == 0
    removed = np.asarray(Image.open(out/'removed'/'q_5_1.png').convert('L'))
    assert (removed[24:27, 20:38] > 240).all() and (removed[24:27, 102:140] > 240).all()      # the strike is gone on both sides
    assert (removed[15:35, 42:98] < 100).all()                                               # the word it crossed is whole
    assert (removed[52:68, 202:238] > 240).all()                                             # the blackout is gone
    record = json.loads((out/'q_5_1.json').read_text())
    assert [r['kind'] for r in record['revisions']] == ['line', 'blot'] and record['revisions'][0]['ink_in_region'] > 0
    with np.load(out/'q_5_1.npz') as masks:
        assert set(masks.files) == {'stroke_0', 'stroke_1', 'stroke_2', 'revision_0', 'revision_1', 'region_0', 'region_1'}
    summary = json.loads((out/'summary.json').read_text())
    assert summary['test']['revisions_line'] == 1 and summary['test']['revisions_blot'] == 1 and summary['test']['regions_moved_by_hand'] == 1
    assert summary['test']['strokes_without_ink'] == 0 and summary['val']['questions'] == 1
    with pytest.raises(Exception, match='not an export'):
        bad = tmp_path/'bad.json'
        bad.write_text('{"schema": "x"}')
        L.main(['--export', str(bad), '--selection', str(selection), '--pages', str(scans), '--output-dir', str(tmp_path/'o2')])
