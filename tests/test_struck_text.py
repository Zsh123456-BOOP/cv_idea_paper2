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
from revrender.tools.annotate.pixel_review import HERE


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


def test_the_page_is_a_text_mode_review_pack_with_the_annotators_pixels(made):
    records = run_crops(made)
    readings = made/'predictions.jsonl'
    readings.write_text('\n'.join(json.dumps({'condition': 'raw_struck', 'id': r['id'], 'text': text})
                                  for r, text in zip(records, ['$\\cancel{x=5}$', '□2', '□□□□□□□□'])))
    (made/'questions.jsonl').write_text('\n'.join(json.dumps({
        'question_global_id': q, 'printed_problem_no': n, 'source_exam_title': 'exam', 'problem_statement_markdown': 'solve $x$',
        'standard_solution_markdown': '$$x=4$$', 'final_answer': '$4$'}) for q, n in (('m_p18', 18), ('m_p19', 19))))
    (made/'groups.jsonl').write_text('\n'.join(json.dumps({'question_global_id': q, 'image_ids': [i]})
                                               for q, i in (('m_p18', 'w1_2_1'), ('m_p19', 'w2_3_1'))))
    S.main(['page', '--crops', str(made/'crops'), '--export', str(made/'export.json'), '--readings', str(readings),
            '--questions', str(made/'questions.jsonl'), '--answer-groups', str(made/'groups.jsonl'), '--output-dir', str(made/'page')])
    pack = json.loads((made/'page'/'pack'/'pack.json').read_text())
    assert pack['task'] == 'strike_text' and pack['text_source_pack'] == S.identity(records, 'struck-text')
    assert [r['prefill'] for r in pack['rows']] == ['x=5', '?2', '???'] == [r['text'] for r in pack['rows']]
    export = json.loads((made/'export.json').read_text())
    assert pack['rows'][1]['runs'] == export['pages']['w1_2_1']['revisions'][0]['runs']          # the annotator's pixels, not a prediction
    assert [p['exam_question'] for p in pack['pages']] == ['m_p18', 'm_p19'] and set(pack['questions']) == {'m_p18', 'm_p19'}
    initial = json.loads((made/'page'/'initial.json').read_text())
    assert initial['pack'] == pack['id'] and initial['pages']['w2_3_1']['revisions'][0]['unreadable'] is True
    assert [r['text'] for r in initial['pages']['w1_2_1']['revisions']] == ['x=5', '?2']
    html = (made/'page'/'site'/'标注.html').read_text()
    assert 'const TextMode' in html and 'const SymbolPicker' in html and 'window.ReviewLibraryStore=' in html


@pytest.mark.skipif(not shutil.which('node'), reason='needs node')
def test_the_editor_scripts_parse_and_formulas_are_told_apart(tmp_path):
    for name in ('symbol_picker.js', 'pixel_text.js', 'pixel_editor.js'):
        subprocess.run(['node', '--check', str(HERE/name)], check=True)
    code = f"const P = require({json.dumps(str(HERE/'symbol_picker.js'))}); console.log(JSON.stringify([P.insideMath('a $x'), P.insideMath('a $x$ '), "
    code += "P.insideMath('\\\\$ 5'), new Set(P.SYMBOLS.map(s => s[0])).size === P.SYMBOLS.length, P.BAR.every(b => P.SYMBOLS.some(s => s[0] === b))]));"
    assert json.loads(subprocess.run(['node', '-e', code], capture_output=True, text=True, check=True).stdout) == [True, False, False, True, True]


def test_erased_crops_hold_only_the_polygon_with_the_strike_white(made):
    run_crops(made)
    export = json.loads((made/'export.json').read_text())
    export['pages']['w1_2_1']['revisions'][0]['region'] = [[60, 25], [100, 25], [100, 38], [60, 38]]     # narrower than its box
    (made/'export2.json').write_text(json.dumps(export))
    S.main(['erased', '--crops', str(made/'crops'), '--export', str(made/'export2.json'), '--review', str(made/'review'),
            '--output-dir', str(made/'erased')])
    pack = json.loads((made/'erased'/'pack.json').read_text())
    row = next(r for r in pack['rows'] if r['id'] == 'w1_2_1#b')
    crop = np.asarray(Image.open(made/'erased'/row['images']['erased']))
    assert crop.shape == (13+2*S.EDGE, 40+2*S.EDGE)                      # the polygon and EDGE of paper
    assert (crop[:S.EDGE] == 255).all() and (crop[:, -S.EDGE:] == 255).all()
    assert (crop[S.EDGE:S.EDGE+2, S.EDGE:S.EDGE+40] == 255).all()          # the annotator's erased rows
    assert (crop[S.EDGE+2:S.EDGE+13, S.EDGE:S.EDGE+40] == 30).all()        # the writing left under the strike


def test_redraft_keeps_the_page_identity_and_remembers_the_starting_texts(made, tmp_path):
    from revrender.tools.annotate.pixel_offline import EmbeddedReview
    test_the_page_is_a_text_mode_review_pack_with_the_annotators_pixels(made)
    source = made/'page'/'site'/'标注.html'
    before = json.loads(EmbeddedReview(source.read_text()).values['review-pack'])
    readings = tmp_path/'erased.jsonl'
    readings.write_text('\n'.join(json.dumps({'condition': 'erased_neutral', 'id': r['id'], 'text': t})
                                  for r, t in zip(before['rows'], ['x=5', '[空]', '12'])))
    S.main(['update', '--source', str(source), '--readings', str(readings), '--output-dir', str(tmp_path/'new')])
    after = json.loads(EmbeddedReview((tmp_path/'new'/'标注.html').read_text()).values['review-pack'])
    assert (after['id'], after['image_revision']) == (before['id'], before['image_revision'])
    assert [r['prefill'] for r in after['rows']] == ['x=5', '', '12']
    assert [r.get('previous_prefill') for r in after['rows']] == [None, ['?2'], ['???']]


def test_update_carries_judgment_drafts_and_keeps_the_page(made, tmp_path):
    from revrender.tools.annotate.pixel_offline import EmbeddedReview
    test_the_page_is_a_text_mode_review_pack_with_the_annotators_pixels(made)
    source = made/'page'/'site'/'标注.html'
    before = json.loads(EmbeddedReview(source.read_text()).values['review-pack'])
    draft = {'schema': S.JUDGMENT_SCHEMA, 'values': {'struck': ['对', '错'], 'after': ['对', '错'], 'type': ['纠错']},
             'judgments': {before['rows'][0]['id']: {'struck': '错', 'after': '对', 'type': '纠错', 'reason': 'r'}}}
    (tmp_path/'j.json').write_text(json.dumps(draft, ensure_ascii=False))
    S.main(['update', '--source', str(source), '--judgments', str(tmp_path/'j.json'), '--output-dir', str(tmp_path/'new')])
    after = json.loads(EmbeddedReview((tmp_path/'new'/'标注.html').read_text()).values['review-pack'])
    assert (after['id'], after['image_revision'], after['rows']) == (before['id'], before['image_revision'], before['rows'])
    assert after['judgment_drafts'] == draft['judgments'] and after['judgment_values']['type'] == ['纠错']


def test_one_pass_keeps_open_regions_the_annotators_text_and_pages_left_without_regions(made):
    export = json.loads((made/'export.json').read_text())
    export['pages']['w1_2_1']['revisions'][3]['text'] = ''                     # an open region whose text the annotator emptied
    export['pages']['w3_4_1'] = {'revisions': [], 'deleted_revisions': [revision('w3_4_1#a', [55, 20, 140, 40], (slice(5, 8), slice(None)), deleted=True)]}
    (made/'export.json').write_text(json.dumps(export))
    S.main(['crops', '--export', str(made/'export.json'), '--review', str(made/'review'), '--pages', str(made/'scans'), '--split', 'train',
            '--split', 'test', '--unconfirmed', '--output-dir', str(made/'crops')])
    records = json.loads((made/'crops'/'regions.json').read_text())
    assert [r['id'] for r in records] == ['w1_2_1#a', 'w1_2_1#open', 'w1_2_1#b']
    assert [e['page'] for e in json.loads((made/'crops'/'empty_pages.json').read_text())] == ['w3_4_1']
    (made/'predictions.jsonl').write_text('\n'.join(json.dumps({'condition': 'erased_neutral', 'id': r['id'], 'text': 'x'}) for r in records))
    (made/'questions.jsonl').write_text(json.dumps({'question_global_id': 'q', 'printed_problem_no': 1, 'source_exam_title': 'e',
                                                    'problem_statement_markdown': '', 'standard_solution_markdown': '', 'final_answer': ''}))
    (made/'groups.jsonl').write_text(json.dumps({'question_global_id': 'q', 'image_ids': ['w1_2_1', 'w3_4_1']}))
    S.main(['page', '--crops', str(made/'crops'), '--export', str(made/'export.json'), '--readings', str(made/'predictions.jsonl'),
            '--condition', 'erased_neutral', '--questions', str(made/'questions.jsonl'), '--answer-groups', str(made/'groups.jsonl'),
            '--output-dir', str(made/'page')])
    pack = json.loads((made/'page'/'pack'/'pack.json').read_text())
    assert [p['id'] for p in pack['pages']] == ['w1_2_1', 'w3_4_1']             # the page left without regions stays
    initial = json.loads((made/'page'/'initial.json').read_text())['pages']
    assert [r['text'] for r in initial['w1_2_1']['revisions']] == ['x', '', 'x'] and initial['w3_4_1']['revisions'] == []


def test_sheets_draft_only_untouched_texts_and_use_the_annotators_pixels_where_kept(tmp_path):
    import base64, io, json
    from argparse import Namespace
    import numpy as np
    from PIL import Image
    from revrender.data.pixel_masks import SCHEMA, encode
    from revrender.tools.annotate import struck_text
    from revrender.tools.annotate.pixel_offline import render_html
    grey = np.full((60, 120), 255, np.uint8); grey[20:30, 10:110] = 0
    buffer = io.BytesIO(); Image.fromarray(grey).save(buffer, format='PNG')
    page = {'id': 'p', 'question': '1', 'image': 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(), 'size': [120, 60], 'split': 'test'}
    box = [5, 15, 115, 35]; region = [[5, 15], [115, 15], [115, 35], [5, 35]]; full = encode(np.ones((20, 110), bool))
    rows = [{'id': f'r{k}', 'page': 0, 'box': box, 'region': region, 'runs': full, 'prefill': 'x', 'text': 'x'} for k in range(4)]
    payload = {'id': 'pk', 'schema': 'paper2-pixel-review-pack-v1', 'task': 'strike_text', 'pages': [page], 'rows': rows, 'image_revision': 'rev', 'standalone': True}
    (tmp_path/'page.html').write_text(render_html(payload, None))
    base = {'box': box, 'region': region, 'runs': encode(np.zeros((20, 110), bool)), 'text': 'x'}
    export = {'schema': SCHEMA, 'pack': 'pk', 'image_revision': 'rev', 'pages': {'p': {'revisions': [
        {**base, 'id': 'r0'}, {**base, 'id': 'r1', 'text': 'changed'}, {**base, 'id': 'r2', 'text': 'done', 'text_confirmed': True, 'confirmed': True},
        {**base, 'id': 'r3', 'ops': 2}]}}}
    (tmp_path/'export.json').write_text(json.dumps(export))
    struck_text.sheets(Namespace(source=tmp_path/'page.html', export=tmp_path/'export.json', output_dir=tmp_path/'out'))
    index = json.loads((tmp_path/'out'/'index.json').read_text()); refs = json.loads((tmp_path/'out'/'reference.json').read_text())
    assert {v['id']: v['pixels'] for v in index.values()} == {'r0': 'prefill', 'r3': 'annotator'}
    assert [v['text'] for v in refs.values()] == ['done'] and (tmp_path/'out'/'draft_01.png').exists()
