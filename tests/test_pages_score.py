"""Page-level scoring of an output against the gold put back on the page."""
import json

import numpy as np
import pytest

from revrender import pages
from revrender.eval import pages_score

BOX = [10, 10, 190, 70]


def block(top, left, rows=14, cols=60):
    mask = np.zeros((60, 180), bool)
    mask[top:top+rows, left:left+cols] = True
    return {'line': 0, 'score': .9, 'box': BOX, 'mask': mask}


def write(root, name, per_page):
    out = root/name
    out.mkdir()
    for image_id, instances in per_page.items():
        pages.write_page(out, image_id, (200, 80), [{'box': BOX, 'source': 'gold', 'ocr_type': None, 'ocr_text': ''}], instances)
    (out/'run_manifest.json').write_text('{}')
    return out


def test_a_drawn_revision_is_matched_a_stray_mask_is_false_and_an_unseen_revision_is_missed(tmp_path):
    gold = write(tmp_path, 'gold', {'a': [block(10, 20)], 'b': [block(10, 20)], 'c': []})
    delivered = write(tmp_path, 'out', {'a': [block(10, 20), block(36, 100)], 'b': [], 'c': [block(30, 10)]})
    summary, chosen = pages_score.score(delivered, gold)
    assert chosen == ['a', 'b', 'c']
    assert summary['gold_components'] == 2 and summary['gold'] == {'matched': 1, 'missed': 1}
    assert (summary['found'], summary['drawn_right']) == (.5, .5)
    assert summary['true_positive'] == 1 and summary['false_positive'] == 2


def test_only_the_pages_of_both_outputs_are_scored_and_the_result_is_written_once(tmp_path):
    gold = write(tmp_path, 'gold', {'a': [block(10, 20)], 'z': [block(10, 20)]})
    delivered = write(tmp_path, 'out', {'a': [block(10, 20)], 'y': []})
    target = tmp_path/'scores.json'
    args = ['--pages-output', str(delivered), '--gold-output', str(gold), '--output', str(target)]
    assert pages_score.main(args) == 0
    record = json.loads(target.read_text())
    assert record['pages'] == 1 and record['scores']['drawn_right'] == 1.
    with pytest.raises(FileExistsError):
        pages_score.main(args)
    with pytest.raises(Exception, match='no page is in both'):
        pages_score.main(['--pages-output', str(delivered), '--gold-output', str(write(tmp_path, 'other', {'q': []})),
                          '--output', str(tmp_path/'none.json')])
