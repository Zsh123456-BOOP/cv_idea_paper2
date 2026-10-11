"""The struck text page: the regions it takes from a pixel review export, their crops, and the offline page it packs."""
import json
import re
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from revrender.data.pixel_masks import SCHEMA, encode
from revrender.tools.annotate import struck_text as S


def revision(rid, box, erase, **extra):
    l, t, r, b = box
    mask = np.zeros((b-t, r-l), bool)
    mask[erase] = True
    return {'id': rid, 'box': box, 'region': [[l, t], [r, t], [r, b], [l, b]], 'runs': encode(mask), 'confirmed': True,
            'annotationMode': 'erase', **extra}


@pytest.fixture
def made(tmp_path):
    scans, review = tmp_path/'scans', tmp_path/'review'
    (review/'images').mkdir(parents=True)
    scans.mkdir()
    pages = {'w1_2_1': 'train', 'w2_3_1': 'val', 'w3_4_1': 'test'}
    for page in pages:
        grey = np.full((60, 200), 255, np.uint8)
        grey[20:40, 50:150] = 30
        Image.fromarray(grey).convert('RGB').save(scans/f'{page}.jpg', quality=95)
        Image.fromarray(grey).save(review/'images'/f'{page}.png')
    (review/'pack.json').write_text(json.dumps({'pages': [{'id': p, 'split': s} for p, s in pages.items()]}))
    export = {'schema': SCHEMA, 'pages': {
        'w1_2_1': {'revisions': [revision('w1_2_1#b', [60, 25, 120, 38], (slice(0, 2), slice(None))),
                                 revision('w1_2_1#a', [50, 5, 100, 22], (slice(None), slice(0, 3))),
                                 revision('w1_2_1#gone', [50, 5, 100, 22], (slice(None), slice(0, 3)), deleted=True),
                                 {**revision('w1_2_1#open', [50, 5, 100, 22], (slice(None), slice(0, 3))), 'confirmed': False}]},
        'w2_3_1': {'revisions': [revision('w2_3_1#a', [55, 20, 140, 40], (slice(5, 8), slice(None)), unreadable=True)]},
        'w3_4_1': {'revisions': [revision('w3_4_1#a', [55, 20, 140, 40], (slice(5, 8), slice(None)))]}}}
    (tmp_path/'export.json').write_text(json.dumps(export))
    return tmp_path


def run_crops(root):
    S.main(['crops', '--export', str(root/'export.json'), '--review', str(root/'review'), '--pages', str(root/'scans'),
            '--split', 'train', '--split', 'val', '--output-dir', str(root/'crops')])
    return json.loads((root/'crops'/'regions.json').read_text())


def test_crops_take_confirmed_regions_of_the_chosen_splits_top_to_bottom(made):
    records = run_crops(made)
    assert [r['id'] for r in records] == ['w1_2_1#a', 'w1_2_1#b', 'w2_3_1#a']        # test pages, deleted and open rows left out
    assert [r['split'] for r in records] == ['train', 'train', 'val']
    assert records[2]['unreadable'] is True
    assert records[0]['crop'] == [20, 0, 130, 34]                                       # CROP margin, kept inside the page
    pack = json.loads((made/'crops'/'pack.json').read_text())
    assert pack['schema'] == 'revrender-line-pack' and [r['id'] for r in pack['rows']] == [r['id'] for r in records]
    assert pack['rows'][0]['boxes'] == [[273, 147, 727, 647]]
    manifest = json.loads((made/'crops'/'run_manifest.json').read_text())
    assert manifest['regions'] == 3 and manifest['left_out'] == 1


def test_the_erased_crop_whitens_exactly_the_annotators_pixels(made):
    records = run_crops(made)
    r = records[1]                                                                       # box [60, 25, 120, 38], first two rows erased
    clean = np.asarray(Image.open(made/'crops'/r['images']['clean']))
    erased = np.asarray(Image.open(made/'crops'/r['images']['erased']))
    l, t = r['crop'][:2]
    changed = clean != erased
    assert changed.sum() == 2*60
    ys, xs = np.nonzero(changed)
    assert (ys.min()+t, ys.max()+t, xs.min()+l, xs.max()+l) == (25, 26, 60, 119)
    assert (erased[changed] == 255).all()


def test_the_page_carries_regions_starting_texts_and_questions(made):
    records = run_crops(made)
    readings = made/'predictions.jsonl'
    readings.write_text('\n'.join(json.dumps({'condition': 'raw_struck', 'id': r['id'], 'text': text})
                                  for r, text in zip(records, ['$\\cancel{x=5}$', '□2', '□□□□□□□□'])))
    (made/'questions.jsonl').write_text('\n'.join(json.dumps({
        'question_global_id': q, 'printed_problem_no': n, 'source_exam_title': 'exam', 'problem_statement_markdown': 'solve $x$',
        'standard_solution_markdown': '$$x=4$$', 'final_answer': '$4$'}) for q, n in (('m_p18', 18), ('m_p19', 19))))
    (made/'groups.jsonl').write_text('\n'.join(json.dumps({'question_global_id': q, 'image_ids': [i]})
                                               for q, i in (('m_p18', 'w1_2_1'), ('m_p19', 'w2_3_1'))))
    S.main(['page', '--crops', str(made/'crops'), '--readings', str(readings), '--questions', str(made/'questions.jsonl'),
            '--answer-groups', str(made/'groups.jsonl'), '--output-dir', str(made/'page')])
    html = (made/'page'/'补文字.html').read_text(encoding='utf-8')
    pack = json.loads(re.search(r'window\.PACK = (\{.*?\});</script>', html, re.S).group(1))
    assert [r['prefill'] for r in pack['regions']] == ['x=5', '?2', '???']
    assert pack['pages']['w2_3_1']['question'] == 'm_p19' and set(pack['questions']) == {'m_p18', 'm_p19'}
    assert pack['regions'][0]['erased'].startswith('data:image/png;base64,')
    assert '<!--PACK-->' not in html and 'function mathParts' in html
    again = made/'page2'
    S.main(['page', '--crops', str(made/'crops'), '--readings', str(readings), '--questions', str(made/'questions.jsonl'),
            '--answer-groups', str(made/'groups.jsonl'), '--output-dir', str(again)])
    assert json.loads((again/'run_manifest.json').read_text())['id'] == pack['id']    # same regions, same browser storage


@pytest.mark.skipif(not shutil.which('node'), reason='needs node')
def test_the_page_script_parses(tmp_path):
    html = S.slot(S.slot(S.EDITOR.read_text(encoding='utf-8'), 'RULES', S.RULES.read_text(encoding='utf-8')), 'PACK', 'window.PACK = {};')
    scripts = re.findall(r'<script>(.*?)</script>', html, re.S)
    for k, code in enumerate(scripts):
        path = tmp_path/f'{k}.js'
        path.write_text(code)
        subprocess.run(['node', '--check', str(path)], check=True)
