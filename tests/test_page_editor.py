"""The revision annotation page: its rules (run with node) and the pack that folds questions, prefill and editor into one page."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from revrender import pages as pages_module
from revrender.tools.annotate import page_pack as P

RULES = Path(P.__file__).with_name('page_rules.js')
NODE = shutil.which('node')


def run_rules(script):
    """The value of a node expression over page_rules.js's exports (R)."""
    code = f"const R = require({json.dumps(str(RULES))}); console.log(JSON.stringify((() => {{ {script} }})()));"
    return json.loads(subprocess.run([NODE, '-e', code], capture_output=True, text=True, check=True).stdout)


@pytest.mark.skipif(not NODE, reason='needs node')
def test_two_struck_lines_join_at_their_closest_ends():
    got = run_rules("return [R.joinPaths([[0,0],[10,0]], [[12,0],[20,1]]), R.joinPaths([[0,0],[10,0]], [[20,1],[12,0]]), R.joinPaths([[10,0],[0,0]], [[12,0],[20,1]])];")
    assert got[0] == [[0, 0], [10, 0], [12, 0], [20, 1]] and got[1] == got[0] and got[2] == got[0]


@pytest.mark.skipif(not NODE, reason='needs node')
def test_a_revisions_region_is_the_writing_its_strokes_touch_and_not_the_next_line():
    script = """
      const w = 120, h = 100, ink = new Uint8Array(w * h);
      const block = (l, t, r, b) => { for (let y = t; y < b; y++) for (let x = l; x < r; x++) ink[y * w + x] = 1; };
      block(10, 10, 60, 30);                    // a struck word
      block(10, 70, 100, 90);                   // the next line, far below
      block(70, 10, 100, 30);                   // the next word on the same line, not touched
      const line = R.autoRegion([{kind: "line", path: [[15, 20], [55, 20]]}], ink, w, h);
      const blot = R.autoRegion([{kind: "blot", path: [[72, 12], [98, 12], [98, 28], [72, 28]]}], ink, w, h);
      const none = R.autoRegion([{kind: "line", path: [[0, 50], [5, 50]]}], ink, w, h);
      return {line, blot, none};"""
    got = run_rules(script)
    assert got['line'] == [[7, 7], [63, 7], [63, 33], [7, 33]]
    assert got['blot'] == [[67, 7], [103, 7], [103, 33], [67, 33]] and got['none'] is None


@pytest.mark.skipif(not NODE, reason='needs node')
def test_strokes_are_joined_grouped_split_and_removed_and_every_stroke_stays_in_one_group():
    script = """
      const s = {strokes: [], groups: [], next: 0};
      const a = R.addStroke(s, "line", [[0, 0], [10, 0]]), b = R.addStroke(s, "line", [[12, 0], [20, 0]]);
      const c = R.addStroke(s, "blot", [[30, 0], [40, 0], [40, 10], [30, 10]]);
      const out = {three: s.groups.length};
      out.blotJoin = R.join(s, a.id, c.id);
      R.join(s, a.id, b.id);
      out.joined = {strokes: s.strokes.length, groups: s.groups.length, path: s.strokes[0].path};
      R.group(s, [a.id, c.id]);
      out.grouped = s.groups.map(g => g.strokes);
      s.groups[0].region = [[0, 0], [1, 0], [1, 1]]; s.groups[0].region_auto = false;
      R.ungroup(s, [c.id]);
      out.ungrouped = s.groups.map(g => [g.strokes, g.region_auto]);
      R.remove(s, [a.id]);
      out.removed = s.groups.map(g => g.strokes);
      out.ids = new Set(s.strokes.map(x => x.id).concat(s.groups.map(g => g.id))).size === s.strokes.length + s.groups.length;
      return out;"""
    got = run_rules(script)
    assert got['three'] == 3 and got['blotJoin'] is False
    assert got['joined'] == {'strokes': 2, 'groups': 2, 'path': [[0, 0], [10, 0], [12, 0], [20, 0]]}
    assert got['grouped'] == [['s1', 's5']] and got['ungrouped'] == [[['s1'], False], [['s5'], True]]       # a region moved by hand stays
    assert got['removed'] == [['s5']] and got['ids']


@pytest.mark.skipif(not NODE, reason='needs node')
def test_regions_are_computed_again_only_where_they_were_computed():
    script = """
      const w = 50, h = 30, ink = new Uint8Array(w * h).fill(0);
      for (let y = 10; y < 20; y++) for (let x = 10; x < 30; x++) ink[y * w + x] = 1;
      const s = {strokes: [], groups: [], next: 0};
      R.addStroke(s, "line", [[12, 15], [28, 15]]); R.addStroke(s, "line", [[12, 14], [20, 14]]);
      s.groups[1].region = [[0, 0], [5, 0], [5, 5], [0, 5]]; s.groups[1].region_auto = false;
      R.refreshRegions(s, ink, w, h);
      return s.groups.map(g => g.region);"""
    auto, manual = run_rules(script)
    assert auto == [[7, 7], [33, 7], [33, 23], [7, 23]] and manual == [[0, 0], [5, 0], [5, 5], [0, 5]]


def test_a_thin_long_instance_is_a_struck_line_and_a_compact_one_a_blackout_hull():
    mask = np.zeros((40, 200), bool)
    mask[18:22, 10:190] = True
    kind, path = P.stroke_of(mask, [100, 50, 300, 90], 40)
    assert kind == 'line' and path[0] == [110, 70] and path[-1][0] == 289
    short = np.zeros((40, 200), bool)
    short[18:21, 30:60] = True
    assert P.stroke_of(short, [100, 50, 300, 90], 40) == ('line', [[130, 69], [159, 69]])
    blob = np.zeros((40, 200), bool)
    blob[5:35, 50:90] = True
    kind, corners = P.stroke_of(blob, [100, 50, 300, 90], 40)
    assert kind == 'blot' and sorted(map(tuple, corners)) == [(150, 55), (150, 85), (190, 55), (190, 85)]
    assert P.stroke_of(np.zeros((4, 4), bool), [0, 0, 4, 4], 4) is None
    assert P.simplify([[0, 0], [5, 0.5], [10, 0], [10, 10], [0, 10]]) == [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_an_earlier_export_starts_the_question_its_marks_become_strokes_and_its_texts_are_dropped():
    old = {'lines': [{'text': 'x=[[3]]5'}], 'marks': [{'shape': 'box', 'x': 1, 'y': 2, 'w': 3, 'h': 4}, {'shape': 'line', 'path': [[0, 0], [9, 1]]},
                                                        {'shape': 'line', 'path': [[5, 5]]}]}
    state = P.from_previous(old)
    assert [s['kind'] for s in state['strokes']] == ['blot', 'line'] and state['strokes'][0]['path'] == [[1, 2], [4, 2], [4, 6], [1, 6]]
    assert [g['strokes'] for g in state['groups']] == [['s1'], ['s3']] and state['next'] == 4
    new = {'strokes': [{'id': 's1', 'kind': 'line', 'path': [[0, 0], [1, 1]]}], 'groups': [{'id': 'g2', 'strokes': ['s1'], 'region': None, 'region_auto': True}], 'next': 2, 'done': True}
    assert P.from_previous(new) == {k: new[k] for k in ('strokes', 'groups', 'next')}


def test_the_pack_folds_scans_and_prefilled_strokes_into_one_page(tmp_path):
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
                                               {'image_id': 'w2_6_1', 'split': 'test', 'question': 6, 'pilot': False}]}))
    previous = tmp_path/'old.json'
    previous.write_text(json.dumps({'schema': P.OLD_SCHEMA, 'pages': {'w2_6_1': {'marks': [{'shape': 'box', 'x': 1, 'y': 2, 'w': 3, 'h': 4}]}}}))
    out = tmp_path/'pack'
    assert P.main(['--selection', str(selection), '--pages', str(scans), '--revrender', str(out_pages), '--which', 'test', '--previous', str(previous), '--output-dir', str(out)]) == 0
    html = (out/'标注.html').read_text(encoding='utf-8')
    assert '<!--PACK-->' not in html and '<!--RULES-->' not in html and 'function autoRegion' in html
    pack = json.loads(html[html.index('window.PACK = ')+len('window.PACK = '):html.index(';</script>', html.index('window.PACK = '))].replace('<\\/', '</'))
    first, second = pack['pages']
    assert pack['id'] == 'paper2-revisions-test' and first['image'].startswith('data:image/jpeg;base64,') and not first['from_previous']
    assert [s['kind'] for s in first['strokes']] == ['line'] and first['groups'][0]['strokes'] == [first['strokes'][0]['id']] and first['groups'][0]['region'] is None
    assert second['from_previous'] and second['strokes'][0]['kind'] == 'blot'
    with pytest.raises(FileExistsError):
        P.main(['--selection', str(selection), '--pages', str(scans), '--revrender', str(out_pages), '--output-dir', str(out)])
    assert P.main(['--selection', str(selection), '--pages', str(scans), '--revrender', str(out_pages), '--which', 'test', '--first', '1', '--output-dir', str(tmp_path/'one')]) == 0
    assert json.loads((tmp_path/'one'/'questions.json').read_text())[0]['id'] == 'w1_5_1' and len(json.loads((tmp_path/'one'/'questions.json').read_text())) == 1
