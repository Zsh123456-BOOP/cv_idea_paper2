"""The lines of struck regions (data.line_set) and their readings scored line by line (eval.line_reading)."""
import json

import numpy as np
import pytest
from PIL import Image

from revrender.data import line_set as LS
from revrender.eval import line_reading as LR
from revrender.recognize_lines import UNREADABLE


def box(l, t, r, b):
    return [[l, t], [r, t], [r, b], [l, b]]


def runs(mask):
    flat, out, i = mask.ravel(), [], 0
    while i < len(flat):
        if flat[i]:
            j = i
            while j < len(flat) and flat[j]:
                j += 1
            out.append([i, j-i]); i = j
        else:
            i += 1
    return out


def test_the_two_images_of_a_line():
    grey = np.full((60, 200), 255, np.uint8)
    grey[20:40, 20:60] = 50; grey[30, 15:65] = 0                 # struck writing and its strike
    grey[20:40, 80:100] = 60                                    # an unreadable region
    grey[20:40, 120:160] = 70                                   # writing nobody struck
    grey[5:10, 0:200] = 0                                       # the line above, outside the polygon
    strike = np.zeros((20, 50), bool); strike[10, :] = True     # the strike, in the box 15..65 x 20..40
    a = {'box': [15, 20, 65, 40], 'region': box(15, 20, 65, 40), 'runs': runs(strike), 'readable': True}
    u = {'box': [78, 18, 102, 42], 'region': box(78, 18, 102, 42), 'runs': [], 'readable': False}
    page = Image.fromarray(grey); line = {'polygon': box(10, 15, 170, 45), 'whole_line': False}
    before, after = LS.line_images(page, line, [a, u])
    assert before.shape == (42, 172)                            # the polygon's box and six pixels: 4..176 x 9..51
    at = lambda image, x, y: image[y-9, x-4]
    assert at(before, 30, 30) == 255 and at(before, 30, 25) == 50     # the strike white, the struck writing left
    assert at(before, 90, 30) == 255 and at(after, 90, 30) == 255     # the unreadable region white in both
    assert at(after, 30, 25) == 255 and at(before, 140, 30) == 70 == at(after, 140, 30)
    assert (before[:6] == 255).all() and (before[:, :6] == 255).all()          # outside the polygon (x < 10, y < 15) white
    assert LS.line_images(page, {**line, 'whole_line': True}, [a, u])[1] is None


def test_merge_orders_regions_and_keeps_the_fold(tmp_path):
    rows = [{'id': f'q#{k}', 'page': 'q', 'writer': 'w', 'split': 'train', 'paper1_split': 'train', 'image_source': '', 'text': t, 'readable': True,
             'region': box(*b), 'box': list(b), 'runs': [], 'mode': 'erase'} for k, t, b in (('a', 'x', (100, 10, 120, 30)), ('b', 'y', (10, 10, 30, 30)))]
    export = {'schema': LS.REVIEW, 'regions': [
        {'id': 'q#a', 'page': 'q', 'text': 'x2', 'readable': True, 'corrected': True, 'added': False, 'region_changed': False, 'pixels_changed': False, 'deleted': False,
         'line': 'q#L1', 'whole_line': False, 'mode': 'erase', 'region': box(100, 10, 120, 30), 'box': [100, 10, 120, 30], 'runs': [[0, 3]]},
        {'id': 'q#b', 'page': 'q', 'text': 'y', 'readable': True, 'corrected': False, 'added': False, 'region_changed': False, 'pixels_changed': False, 'deleted': False,
         'line': 'q#L1', 'whole_line': False, 'mode': 'erase', 'region': box(10, 10, 30, 30), 'box': [10, 10, 30, 30], 'runs': []},
        {'id': 'q#add1', 'page': 'q', 'text': '', 'readable': False, 'corrected': True, 'added': True, 'region_changed': True, 'pixels_changed': False, 'deleted': False,
         'line': 'q#N1', 'whole_line': True, 'mode': 'erase', 'region': box(10, 50, 60, 70), 'box': [10, 50, 60, 70], 'runs': []}],
        'lines': [{'id': 'q#L1', 'page': 'q', 'polygon': box(0, 5, 130, 35), 'regions': ['q#a', 'q#b'], 'whole_line': False, 'adjusted': True},
                  {'id': 'q#N1', 'page': 'q', 'polygon': box(10, 50, 60, 70), 'regions': ['q#add1'], 'whole_line': True, 'adjusted': False}]}
    (tmp_path/'regions.json').write_text(json.dumps(rows)); (tmp_path/'export.json').write_text(json.dumps(export))
    (tmp_path/'folds').mkdir(); (tmp_path/'folds'/'pack.json').write_text(json.dumps({'rows': [{'writer': 'w', 'fold': 3, 'val': False}]}))
    LS.main(['merge', '--dataset', str(tmp_path/'regions.json'), '--folds', str(tmp_path/'folds'), '--review', str(tmp_path/'export.json'), '--output-dir', str(tmp_path/'out')])
    regions = {r['id']: r for r in json.loads((tmp_path/'out'/'regions.json').read_text())}; lines = json.loads((tmp_path/'out'/'lines.json').read_text())
    assert regions['q#a']['text'] == 'x2' and regions['q#a']['line_review']['text'] == 'x' and regions['q#a']['strike_pixels'] == 3
    assert regions['q#add1']['writer'] == 'w' and regions['q#add1']['fold'] == 3 and regions['q#add1']['pixels_confirmed']
    assert lines[0]['regions'] == ['q#b', 'q#a']                # left to right
    chosen, skipped = LS.chosen_lines(lines, regions)
    assert [c['text'] for c in chosen] == ['yx2'] and skipped == {'nothing_readable': 1}
    fixes = {'schema': LS.FIXES, 'fixes': [{'region': 'q#b', 'original_text': 'y', 'original_readable': True, 'text': 'y3', 'readable': True,
                                            'text_changed': True, 'pixels_changed': True, 'box': [10, 10, 30, 30], 'runs': [[0, 4]]}]}
    (tmp_path/'fixes.json').write_text(json.dumps(fixes))
    LS.main(['merge', '--dataset', str(tmp_path/'regions.json'), '--folds', str(tmp_path/'folds'), '--review', str(tmp_path/'export.json'),
             '--fixes', str(tmp_path/'fixes.json'), '--output-dir', str(tmp_path/'fixed')])
    fixed = {r['id']: r for r in json.loads((tmp_path/'fixed'/'regions.json').read_text())}
    assert fixed['q#b']['text'] == 'y3' and fixed['q#b']['strike_pixels'] == 4 and fixed['q#b']['reading_fix']['text'] == 'y'
    stale = {**fixes, 'fixes': [{**fixes['fixes'][0], 'original_text': 'not y'}]}; (tmp_path/'stale.json').write_text(json.dumps(stale))
    with pytest.raises(Exception, match='another text'):
        LS.main(['merge', '--dataset', str(tmp_path/'regions.json'), '--folds', str(tmp_path/'folds'), '--review', str(tmp_path/'export.json'),
                 '--fixes', str(tmp_path/'stale.json'), '--output-dir', str(tmp_path/'stale')])


def test_a_line_read_three_ways():
    assert LR.difference('$x=12+3$', 'x=+3') == '12'
    assert LR.unmatched('x=12', 'y=') and not LR.unmatched('x=12', 'x=')
    rows = [{'id': 'p', 'kind': 'partial', 'readable_regions': ['p1', 'p2']}, {'id': 'w', 'kind': 'whole', 'readable_regions': ['w1']},
            {'id': 'gone', 'kind': 'partial', 'readable_regions': ['g1']}]
    read = LR.line_readings(rows, {'p1': 'a', 'p2': UNREADABLE, 'w1': 'b', 'g1': 'c'}, {'p': 'ab=1', 'w': 'z=1', 'gone': 'c'}, {'p': '=1'}, {'p': 'ab'})
    assert read == {'crops': {'p': 'a', 'w': 'b'}, 'difference': {'p': 'ab', 'w': 'z=1'}, 'pair': {'p': 'ab', 'w': 'z=1'}}


def test_the_spelling_of_division_and_dot():
    from revrender.eval.coverage import prepare
    assert prepare('$a\\div b$') == prepare('a÷b') and prepare('$a\\cdot b$') == prepare('a·b') == prepare('a⋅b')


def test_the_spelling_of_roots_parallels_and_escapes():
    from revrender.eval.coverage import prepare
    assert prepare('$\\sqrt{12}$') == prepare('√12') == prepare('√(12)') and prepare('$2\\sqrt{x}$') == prepare('2√x')
    assert prepare('$\\sqrt{x+1}$') == '√(x+1)' != prepare('√x+1')          # round more than one number or letter the parentheses stay
    assert prepare('$\\sqrt 3$') == '√3' and prepare('$\\sqrt[3]{8}$') == '3√8'
    assert prepare('FG//BC') == prepare('FG‖BC') == prepare('$FG\\parallel BC$') == 'FG∥BC' and prepare('1/2/3') == '1/2/3'
    assert prepare('a &gt; b') == 'a>b' and prepare('a &lt; b') == 'a<b'


def test_unchanged_regions(tmp_path):
    base = {'box': [0, 0, 4, 4], 'region': [[0, 0], [4, 0], [4, 4]], 'runs': [[0, 3]], 'text': 'ab', 'readable': True}
    old = [{'id': 'same', **base}, {'id': 'text', **base}, {'id': 'pixels', **base}, {'id': 'gone', **base}, {'id': 'unreadable', **base}]
    new = [{'id': 'same', **base}, {'id': 'text', **base, 'text': 'ac'}, {'id': 'pixels', **base, 'runs': [[1, 3]]}, {'id': 'gone', **base, 'deleted': True},
           {'id': 'unreadable', **base, 'readable': False}, {'id': 'added', **base}]
    (tmp_path/'old.json').write_text(json.dumps(old)); (tmp_path/'new.json').write_text(json.dumps(new))
    LS.main(['unchanged', '--old', str(tmp_path/'old.json'), '--new', str(tmp_path/'new.json'), '--output', str(tmp_path/'out.json')])
    out = json.loads((tmp_path/'out.json').read_text())
    assert out['pixels'] == ['same', 'text'] and out['all'] == ['same']
