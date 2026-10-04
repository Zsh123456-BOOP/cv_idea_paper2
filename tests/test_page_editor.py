"""The page annotation editor: its rules (run with node), and the pack that folds questions, prefill and editor into one page."""
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
    code = f"const R = require({json.dumps(str(RULES))}); console.log(JSON.stringify(({script})));"
    return json.loads(subprocess.run([NODE, '-e', code], capture_output=True, text=True, check=True).stdout)


@pytest.mark.skipif(not NODE, reason='needs node')
def test_a_line_is_read_into_its_final_text_and_its_struck_stretches():
    got = run_rules("['x=[[3]]5', '[[2?]]+1', '[[?]]', '[[旧]]新', 'a+b', 'x=[[3', 'x=3]]', '[[a[[b]]]]', '[[ ]]'].map(R.parseLine)")
    assert got[0] == {'final': 'x=5', 'struck': [{'text': '3', 'legible': 'yes'}], 'error': None}
    assert got[1]['struck'][0]['legible'] == 'partly' and got[1]['final'] == '+1' and got[2]['struck'][0]['legible'] == 'no'
    assert got[3]['final'] == '新' and got[3]['struck'][0]['text'] == '旧' and got[4] == {'final': 'a+b', 'struck': [], 'error': None}
    assert got[5]['error'] and got[6]['error'] and got[7]['error'] and got[8]['error']


@pytest.mark.skipif(not NODE, reason='needs node')
def test_a_mark_belongs_to_the_line_that_holds_its_centre_and_the_editor_warns_when_counts_differ():
    lines = [{'poly': [[0, 0], [100, 0], [100, 20], [0, 20]], 'text': 'x=[[3]]5'}, {'poly': [[0, 25], [100, 25], [110, 50], [0, 45]], 'text': 'y=2'},
             {'poly': [[0, 60], [100, 60], [100, 80], [0, 80]], 'text': '页脚', 'not_writing': True}]
    marks = [{'type': 'blot', 'shape': 'box', 'x': 20, 'y': 5, 'w': 10, 'h': 10}, {'type': 'line', 'shape': 'line', 'path': [[10, 35], [60, 38]]},
             {'type': 'blot', 'shape': 'box', 'x': 10, 'y': 65, 'w': 5, 'h': 5}, {'type': 'blot', 'shape': 'box', 'x': 300, 'y': 300, 'w': 5, 'h': 5}]
    got = run_rules(f"R.checks({json.dumps(lines)}, {json.dumps(marks)})")
    assert got['assigned']['line'] == [[0], [1], []] and got['assigned']['outside'] == [2, 3]          # a "not writing" line takes no mark
    assert got['line'][0] == '' and '1' in got['line'][1] and got['line'][2] == '' and got['page']
    assert run_rules("R.pointInPolygon([5, 5], [[0,0],[10,0],[10,10],[0,10]]) && !R.pointInPolygon([15, 5], [[0,0],[10,0],[10,10],[0,10]])")


def test_the_strong_models_spaced_reading_closes_up_but_keeps_latex_apart():
    assert P.tidy('2 0 . ( 1 ) 解 : 设 一 个 球 x 元') == '20.(1)解:设一个球x元'
    assert P.tidy('5 0 \\geq a') == '50\\geq a' and P.tidy('\\frac { 6 0 0 0 } { x + 2 0 }') == '\\frac{6000}{x+20}' and P.tidy('') == ''


def test_a_thin_long_instance_is_a_struck_line_and_a_compact_one_a_box():
    mask = np.zeros((40, 200), bool)
    mask[18:22, 10:190] = True
    line = P.mark_of(mask, [100, 50, 300, 90], 40)
    assert line['shape'] == 'line' and line['path'][0] == [110, 70] and line['path'][-1][0] == 289 and len(line['path']) >= 2
    blob = np.zeros((40, 200), bool)
    blob[5:35, 50:90] = True
    box = P.mark_of(blob, [100, 50, 300, 90], 40)
    assert box == {'type': 'blot', 'shape': 'box', 'x': 150, 'y': 55, 'w': 40, 'h': 30, 'from': 'revrender'}
    assert P.mark_of(np.zeros((4, 4), bool), [0, 0, 4, 4], 4) is None
    short = np.zeros((40, 200), bool)
    short[18:21, 30:60] = True                                                  # shorter than the line is high: one step, still two ends
    assert P.mark_of(short, [100, 50, 300, 90], 40)['path'] == [[130, 69], [159, 69]]


def test_the_pack_folds_scans_lines_texts_and_marks_into_one_page(tmp_path):
    scans = tmp_path/'scans'
    scans.mkdir()
    grey = np.full((100, 300), 255, np.uint8)
    grey[20:30, 20:200] = 0                                                        # a line of writing and a stroke through it
    Image.fromarray(grey).convert('RGB').save(scans/'w1_5_1.jpg', quality=95)
    out_pages = tmp_path/'revrender'
    out_pages.mkdir()
    mask = np.zeros((20, 280), bool)
    mask[4:8, 10:180] = True
    pages_module.write_page(out_pages, 'w1_5_1', (300, 100), [{'box': [10, 15, 290, 35], 'source': 'ocr', 'ocr_type': 'text', 'ocr_text': 'x = 3'},
                                                               {'box': [10, 60, 290, 90], 'source': 'ocr', 'ocr_type': 'footer', 'ocr_text': '第1页'}],
                            [{'line': 0, 'score': .9, 'box': [10, 15, 290, 35], 'mask': mask}])
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps({'pages': [{'image_id': 'w1_5_1', 'split': 'test', 'question': 5, 'pilot': True, 'writer': 'w1', 'lines': 2, 'edited': 1}]}))
    catalogue = tmp_path/'catalogue.jsonl'
    catalogue.write_text(json.dumps({'image_id': 'w1_5_1', 'representative_line_id': 'w1_5_1:L0001', 'box_1000': [33.4, 150, 966.6, 350]})+'\n')
    root = tmp_path/'nowhere'
    out = tmp_path/'pack'
    assert P.main(['--selection', str(selection), '--pages', str(scans), '--revrender', str(out_pages), '--catalogue', str(catalogue), '--root', str(root), '--output-dir', str(out)]) == 0
    html = (out/'标注.html').read_text(encoding='utf-8')
    assert '<!--PACK-->' not in html and '<!--RULES-->' not in html and 'function parseLine' in html
    pack = json.loads(html[html.index('window.PACK = ')+len('window.PACK = '):html.index(';</script>', html.index('window.PACK = '))].replace('<\\/', '</'))
    page = pack['pages'][0]
    assert pack['id'] == 'paper2-pilot' and page['width'] == 300 and page['image'].startswith('data:image/jpeg;base64,')
    assert [l['text'] for l in page['lines']] == ['x=3', '第1页'] and [l['not_writing'] for l in page['lines']] == [False, True]
    assert page['lines'][0]['poly'] == [[10, 15], [290, 15], [290, 35], [10, 35]] and page['lines'][0]['line_id'] == 'w1_5_1:L0001'
    assert len(page['marks']) == 1 and page['marks'][0]['shape'] == 'line'
    with pytest.raises(FileExistsError):
        P.main(['--selection', str(selection), '--pages', str(scans), '--revrender', str(out_pages), '--catalogue', str(catalogue), '--root', str(root), '--output-dir', str(out)])


@pytest.mark.skipif(not NODE, reason='needs node')
def test_the_python_reader_reads_a_line_exactly_as_the_editor_does():
    from revrender.data.page_labels import parse_line
    cases = ['x=[[3]]5', '[[2?]]+1', '[[?]]', '[[??]]a', '[[旧]]新', 'a+b', '', 'x=[[3', 'x=3]]', '[[a[[b]]]]', '[[ ]]', '[[ ]]x[[y]]', 'a]]b[[c]]', '[[1]][[2]]3']
    js = run_rules(f"{json.dumps(cases)}.map(R.parseLine)")
    for case, j in zip(cases, js):
        final, struck, error = parse_line(case)
        assert (final, struck, error) == (j['final'], j['struck'], j['error']), case
