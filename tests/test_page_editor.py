"""The revision annotation page: its rules (run with node), where its questions start (page_prefill) and the pack that folds questions,
starting texts and editor into one page."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from revrender import pages as pages_module
from revrender.tools.annotate import page_pack as P
from revrender.tools.annotate import page_prefill as F

RULES = Path(P.__file__).with_name('page_rules.js')
NODE = shutil.which('node')


def run_rules(script):
    """The value of a node expression over page_rules.js's exports (R)."""
    code = f"const R = require({json.dumps(str(RULES))}); console.log(JSON.stringify((() => {{ {script} }})()));"
    return json.loads(subprocess.run([NODE, '-e', code], capture_output=True, text=True, check=True).stdout)


@pytest.mark.skipif(not NODE, reason='needs node')
def test_revisions_are_added_retyped_picked_and_removed():
    script = """
      const s = {revisions: [], next: 0};
      const a = R.addRevision(s, "line", [[0, 0], [100, 0], [100, 40], [0, 40]]);
      const b = R.addRevision(s, "blot", [[20.4, 10], [40, 10], [40, 30], [20, 30]]);
      const out = {ids: s.revisions.map(r => r.id), unreadable: s.revisions.map(r => r.unreadable), rounded: b.region[0]};
      out.inside = R.revisionAt(s, [30, 20], 2).id;          // the smaller region holding the point
      out.border = R.revisionAt(s, [99, 20], 2).id;          // near the big one's border
      out.none = R.revisionAt(s, [300, 300], 2);
      out.same = R.setKind(s, a.id, "line");
      a.text = "40a"; R.setKind(s, a.id, "blot"); out.withText = a.unreadable;
      R.setKind(s, b.id, "line"); b.text = ""; R.setKind(s, b.id, "blot"); out.empty = b.unreadable;
      out.removed = R.removeRevision(s, a.id); out.again = R.removeRevision(s, a.id);
      out.left = s.revisions.map(r => r.id); out.next = R.addRevision(s, "line", [[0, 0], [1, 0], [1, 1]]).id;
      return out;"""
    got = run_rules(script)
    assert got['ids'] == ['r1', 'r2'] and got['unreadable'] == [False, True] and got['rounded'] == [20, 10]
    assert got['inside'] == 'r2' and got['border'] == 'r1' and got['none'] is None
    assert got['same'] is False and got['withText'] is False and got['empty'] is True
    assert got['removed'] is True and got['again'] is False and got['left'] == ['r2'] and got['next'] == 'r3'


@pytest.mark.skipif(not NODE, reason='needs node')
def test_a_text_splits_into_words_and_formulas():
    got = run_rules(r"""return [R.mathParts("3个$\\frac{1}{2}$篮球"), R.mathParts("\\frac{5a^2}{60a+1}"), R.mathParts("AO="), R.mathParts("$$x^2$$"),
                               R.mathParts("x^2+1"), R.mathParts("")];""")
    assert got[0] == [{'math': False, 'text': '3个'}, {'math': True, 'text': '\\frac{1}{2}'}, {'math': False, 'text': '篮球'}]
    assert got[1] == [{'math': True, 'text': '\\frac{5a^2}{60a+1}'}] and got[2] == [{'math': False, 'text': 'AO='}]
    assert got[3] == [{'math': True, 'text': 'x^2'}] and got[4] == [{'math': True, 'text': 'x^2+1'}] and got[5] == []


@pytest.mark.skipif(not NODE, reason='needs node')
def test_time_on_a_question_counts_each_gap_up_to_a_minute():
    assert run_rules('return [R.activeSeconds([5, 30, 600, -2]), R.IDLE];') == [95, 60]


def test_a_thin_long_instance_is_a_struck_line_and_a_compact_one_a_blackout():
    line = np.zeros((40, 200), bool)
    line[18:22, 10:190] = True
    blob = np.zeros((40, 200), bool)
    blob[5:35, 50:90] = True
    assert F.kind_of(line, 40) == 'line' and F.kind_of(blob, 40) == 'blot'          # data.strike_bank's rule


def test_a_revisions_region_is_the_writing_its_ink_touches_and_not_the_next_line_or_word():
    ink = np.zeros((100, 140), bool)
    ink[10:30, 10:60] = True                  # a struck word
    ink[70:90, 10:100] = True                 # the next line, far below
    ink[10:30, 80:110] = True                 # the next word on the same line, not touched
    seeds = np.zeros_like(ink)
    seeds[19:21, 15:55] = True
    assert F.region_of(seeds, ink) == [7, 7, 63, 33]
    assert F.region_of(np.zeros_like(ink), ink) is None


def test_an_instance_on_the_lines_of_a_figure_is_left_out():
    grey = np.full((200, 300), 255, np.uint8)
    grey[20:30, 20:100] = 0
    grey[60:180, 150:152] = 0                 # a line of a figure
    record = {'lines': [{'box': [10, 10, 120, 40], 'source': 'ocr'}, {'box': [140, 50, 160, 190], 'source': 'ink'}]}
    word = np.zeros((30, 110), bool)
    word[10:20, 10:90] = True
    figure = np.zeros((140, 20), bool)
    figure[10:130, 10:12] = True
    got = F.from_revrender(grey, record, [{'line': 0, 'box': [10, 10, 120, 40], 'mask': word}, {'line': 1, 'box': [140, 50, 160, 190], 'mask': figure}])
    assert got == [{'kind': 'line', 'region': [[17, 17], [103, 17], [103, 33], [17, 33]]}]


def test_overlapping_regions_are_one_revision_in_reading_order():
    merged = F.merge([('line', [0, 0, 50, 20], 10), ('blot', [10, 0, 60, 20], 30), ('line', [200, 0, 220, 20], 5)])
    assert merged == [['blot', [0, 0, 60, 20], 40], ['line', [200, 0, 220, 20], 5]]
    revisions = [{'kind': 'line', 'region': F.corners(b)} for b in ([300, 105, 340, 135], [10, 100, 50, 130], [200, 10, 240, 40], [100, 98, 150, 128])]
    ordered, last = F.numbered(revisions)
    assert [r['id'] for r in ordered] == ['r1', 'r2', 'r3', 'r4'] and last == 4
    assert [F.box_of(r['region'])[0] for r in ordered] == [200, 10, 100, 300]


def test_a_geometry_export_gives_its_regions_and_kinds():
    grey = np.full((60, 120), 255, np.uint8)
    grey[20:30, 10:50] = 0
    state = {'strokes': [{'id': 's1', 'kind': 'line', 'path': [[12, 25], [48, 25]]}, {'id': 's3', 'kind': 'blot', 'path': [[60, 5], [70, 5], [70, 15], [60, 15]]},
                         {'id': 's5', 'kind': 'blot', 'path': [[90, 40], [100, 40], [100, 50], [90, 50]]}],
             'groups': [{'id': 'g2', 'strokes': ['s1'], 'region': None, 'region_auto': True},
                        {'id': 'g4', 'strokes': ['s3'], 'region': [[58, 3], [72, 3], [72, 17], [58, 17]], 'region_auto': False},
                        {'id': 'g6', 'strokes': ['s5'], 'region': None, 'region_auto': True}]}
    got = F.from_geometry(state, grey)
    assert got == [{'kind': 'line', 'region': [[7, 17], [53, 17], [53, 33], [7, 33]]}, {'kind': 'blot', 'region': [[58, 3], [72, 3], [72, 17], [58, 17]]}]


def write_question(tmp_path):
    """Two scanned questions, RevRender's output on them, a selection and a geometry export of the second."""
    scans = tmp_path/'scans'
    scans.mkdir()
    grey = np.full((100, 300), 255, np.uint8)
    grey[20:30, 20:200] = 0
    for name in ('w1_5_1', 'w2_6_1'):
        Image.fromarray(grey).convert('RGB').save(scans/f'{name}.jpg', quality=95)
    out_pages = tmp_path/'revrender'
    out_pages.mkdir()
    mask = np.zeros((20, 280), bool)
    mask[9:12, 10:180] = True
    for name in ('w1_5_1', 'w2_6_1'):
        pages_module.write_page(out_pages, name, (300, 100), [{'box': [10, 15, 290, 35], 'source': 'ocr', 'ocr_type': 'text', 'ocr_text': ''}],
                                [{'line': 0, 'score': .9, 'box': [10, 15, 290, 35], 'mask': mask}])
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps({'pages': [{'image_id': 'w1_5_1', 'split': 'test', 'question': 5, 'pilot': True},
                                               {'image_id': 'w2_6_1', 'split': 'test', 'question': 6, 'pilot': False},
                                               {'image_id': 'w3_7_1', 'split': 'val', 'question': 7, 'pilot': False}]}))
    previous = tmp_path/'geometry.json'
    previous.write_text(json.dumps({'schema': F.GEOMETRY_SCHEMA, 'pages': {'w2_6_1': {
        'strokes': [{'id': 's1', 'kind': 'blot', 'path': [[100, 15], [150, 15], [150, 35], [100, 35]]}],
        'groups': [{'id': 'g2', 'strokes': ['s1'], 'region': [[98, 13], [152, 13], [152, 37], [98, 37]], 'region_auto': False}]}}}))
    return scans, out_pages, selection, previous


def test_a_reading_starts_the_text_without_the_marks_it_drew_the_strike_with():
    cases = {r'$\overline{AD}$': 'AD', r'$58$': '58', r'$\frac{50}{60}a$': r'$\frac{50}{60}a$', r'是$\boxed{直}$角等': '是直角等',
             r'$\overline{\cancel{a+401}}$': 'a+401', '40□\n+6': '40? +6', r'$$x^2$$': r'$$x^2$$', r'$\textcircled{1}AB$': r'$\textcircled{1}AB$'}
    assert {k: P.starting_text(k) for k in cases} == cases


def test_the_prefill_and_the_pack_make_one_page_with_regions_and_starting_texts(tmp_path):
    scans, out_pages, selection, previous = write_question(tmp_path)
    prefill = tmp_path/'prefill'
    assert F.main(['--selection', str(selection), '--pages', str(scans), '--revrender', str(out_pages), '--which', 'test', '--previous', str(previous),
                   '--output-dir', str(prefill)]) == 0
    questions = json.loads((prefill/'prefill.json').read_text())['questions']
    first, second = questions
    assert [q['source'] for q in questions] == ['revrender', 'previous'] and first['next'] == 1
    assert first['revisions'] == [{'id': 'r1', 'kind': 'line', 'region': [[17, 17], [203, 17], [203, 33], [17, 33]]}]
    assert second['revisions'] == [{'id': 'r1', 'kind': 'blot', 'region': [[98, 13], [152, 13], [152, 37], [98, 37]]}]
    rows = json.loads((prefill/'pack.json').read_text())['rows']
    assert [r['id'] for r in rows] == ['w1_5_1#r1', 'w2_6_1#r1'] and Image.open(prefill/rows[0]['images']['raw']).size == (233, 40)
    readings = tmp_path/'predictions.jsonl'
    readings.write_text('\n'.join(json.dumps({'condition': c, 'id': 'w1_5_1#r1', 'text': t}) for c, t in (('raw_neutral', 'all'), ('raw_struck', '40□\n+6'))))
    out = tmp_path/'pack'
    assert P.main(['--prefill', str(prefill), '--pages', str(scans), '--readings', str(readings), '--output-dir', str(out)]) == 0
    html = (out/'标注.html').read_text(encoding='utf-8')
    assert '<!--PACK-->' not in html and '<!--RULES-->' not in html and 'function mathParts' in html
    pack = json.loads(html[html.index('window.PACK = ')+len('window.PACK = '):html.index(';</script>', html.index('window.PACK = '))].replace('<\\/', '</'))
    one, two = pack['pages']
    assert pack['id'] == 'paper2-revision-boxes-test' and one['image'].startswith('data:image/jpeg;base64,')
    assert one['revisions'][0]['text'] == '40? +6' and one['revisions'][0]['unreadable'] is False
    assert two['revisions'][0]['text'] == '' and two['revisions'][0]['unreadable'] is True
    with pytest.raises(FileExistsError):
        P.main(['--prefill', str(prefill), '--pages', str(scans), '--output-dir', str(out)])
    assert P.main(['--prefill', str(prefill), '--pages', str(scans), '--first', '1', '--output-dir', str(tmp_path/'one')]) == 0
    assert [q['id'] for q in json.loads((tmp_path/'one'/'questions.json').read_text())] == ['w1_5_1']
