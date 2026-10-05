"""The remover read inside the boxes a person drew: struck-line boxes only, cropped with the reading's margin, written like a page."""
import json

import numpy as np
from PIL import Image

from revrender import box_remover as X
from revrender import infer
from revrender import pages as pages_module
from revrender.eval import revision_boxes as R


def test_struck_line_boxes_are_read_and_written_like_a_page(tmp_path, monkeypatch):
    scans = tmp_path/'scans'
    scans.mkdir()
    grey = np.full((100, 300), 250, np.uint8)
    grey[30:33, 20:120] = 30
    Image.fromarray(grey).convert('RGB').save(scans/'q_5_1.png')
    box = lambda l, t, r, b: [[l, t], [r, t], [r, b], [l, b]]
    labels = tmp_path/'labels.json'
    labels.write_text(json.dumps({'schema': R.SCHEMA, 'pages': {'q_5_1': {'done': True, 'revisions': [
        {'id': 'r1', 'kind': 'line', 'region': box(20, 25, 120, 40), 'text': 'ab', 'unreadable': False},
        {'id': 'r2', 'kind': 'blot', 'region': box(200, 20, 230, 50), 'text': '', 'unreadable': True}]}}}))
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps({'pages': [{'image_id': 'q_5_1', 'split': 'val'}]}))
    seen = []

    def strokes_everywhere(model, strips, point, batch_size=None):
        seen.extend(strips)
        return [[(np.ones((96, 768), bool), .9)] for _ in strips]
    monkeypatch.setattr(infer, 'load_checkpoint', lambda path, variant=None, expected_sha=None: (None, {'step': 7}))
    monkeypatch.setattr(infer, 'predict_lines', strokes_everywhere)
    checkpoint = tmp_path/'step_7.pt'
    checkpoint.write_bytes(b'x')
    out = tmp_path/'out'
    assert X.main(['--checkpoint', str(checkpoint), '--point', '.4', '.5', '.1', '--labels', str(labels), '--selection', str(selection), '--split', 'val',
                   '--pages', str(scans), '--output-dir', str(out)]) == 0
    record, instances = pages_module.read_output(out, 'q_5_1')
    assert len(seen) == 1 and seen[0].shape == (96, 768)                  # the blackout's box is not read
    assert record['lines'][0]['box'] == [14, 19, 126, 46] and record['lines'][0]['source'] == 'box'
    assert len(instances) == 1 and instances[0]['mask'].shape == (27, 112) and instances[0]['score'] == .9
