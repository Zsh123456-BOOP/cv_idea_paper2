"""A model's revisions read against the user's boxes: found, missed, removed outside every box, and what a reader holds of the struck text."""
import json

import numpy as np
from PIL import Image

from revrender import pages as pages_module
from revrender.eval import revision_boxes as R


def test_found_missed_and_false_places_and_the_held_share(tmp_path):
    scans, out_pages = tmp_path/'scans', tmp_path/'model'
    scans.mkdir()
    out_pages.mkdir()
    grey = np.full((100, 300), 255, np.uint8)
    grey[20:40, 20:60] = 0          # revision 1: the model removes most of it
    grey[20:40, 100:140] = 0        # revision 2: the model misses it
    grey[60:80, 200:260] = 0        # valid writing: the model removes a piece of it
    Image.fromarray(grey).convert('RGB').save(scans/'q_5_1.png')
    hit, wrong = np.zeros((20, 40), bool), np.zeros((20, 60), bool)
    hit[:, :30] = True
    wrong[5:15, 10:30] = True
    pages_module.write_page(out_pages, 'q_5_1', (300, 100), [{'box': [0, 0, 300, 100], 'source': 'ocr', 'ocr_type': 'text', 'ocr_text': ''}],
                            [{'line': 0, 'score': .8, 'box': [20, 20, 60, 40], 'mask': hit}, {'line': 0, 'score': .3, 'box': [200, 60, 260, 80], 'mask': wrong}])
    (out_pages/'run_manifest.json').write_text('{}')
    box = lambda l, t, r, b: [[l, t], [r, t], [r, b], [l, b]]
    labels = tmp_path/'labels.json'
    labels.write_text(json.dumps({'schema': R.SCHEMA, 'pages': {'q_5_1': {'done': True, 'revisions': [
        {'id': 'r1', 'kind': 'line', 'region': box(18, 18, 62, 42), 'text': '40a', 'unreadable': False},
        {'id': 'r2', 'kind': 'blot', 'region': box(98, 18, 142, 42), 'text': '', 'unreadable': True}]}}}))
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps({'pages': [{'image_id': 'q_5_1', 'split': 'val'}]}))
    out = tmp_path/'read'
    assert R.main(['boxes', '--labels', str(labels), '--selection', str(selection), '--split', 'val', '--pages-output', str(out_pages),
                   '--pages', str(scans), '--cuts', '.2', '.5', '--output-dir', str(out)]) == 0
    summary = json.loads((out/'summary.json').read_text())
    low, high = summary['by_cut']['0.2'], summary['by_cut']['0.5']
    assert (low['found'], low['false_places'], low['recall_line'], low['recall_blot']) == (1, 1, 1., 0.)
    assert (high['found'], high['false_places'], high['precision']) == (1, 0, 1.) and summary['best_cut'] == .5
    rows = json.loads((out/'pack'/'pack.json').read_text())['rows']
    assert [r['id'] for r in rows] == ['q_5_1#r1']                       # an unreadable revision is not read
    erased = np.asarray(Image.open(out/'pack'/rows[0]['images']['erased']).convert('L'))
    assert erased[8:28, 6:36].min() == 255 and erased[8:28, 40:46].max() == 0
    readings = tmp_path/'p.jsonl'
    readings.write_text('\n'.join(json.dumps({'id': 'q_5_1#r1', 'condition': c, 'text': t}) for c, t in (('raw_neutral', '4'), ('erased_neutral', '40a'))))
    assert R.main(['read', '--pack', str(out/'pack'), '--readings', str(readings), '--output', str(tmp_path/'read.json')]) == 0
    held = json.loads((tmp_path/'read.json').read_text())['held']
    assert held['erased_neutral/line']['held'] == 1. and abs(held['raw_neutral/all']['held']-1/3) < 1e-9
    assert R.held('4?a', '4xa') == 1. and R.held('?', 'x') is None
