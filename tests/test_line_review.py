"""The line page: the prefilled lines and whole-line guess (line_review.py) and its state rules (line_rules.js, run with node)."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from revrender.tools.annotate import line_review as LR

RULES = Path(LR.__file__).with_name('line_rules.js')
NODE = shutil.which('node')


def region(l, t, r, b, name):
    return {'id': name, 'box': [l, t, r, b], 'region': LR.rectangle(l, t, r, b)}


def test_regions_go_to_the_catalogue_line_or_an_ink_band():
    grey = np.full((200, 400), 255, np.uint8)
    grey[20:40, 10:300] = 0                                   # line 1, in the catalogue
    grey[100:120, 20:120] = 0; grey[100:120, 150:200] = 0      # line 2, not in it: a gap of 30 columns, under two heights
    catalogue = [{'box_1000': [0, 90, 800, 210]}]             # 0..320 x 18..42
    a, b = region(50, 22, 80, 38, 'a'), region(200, 20, 260, 40, 'b')
    c = region(30, 100, 60, 120, 'c')
    lines = LR.estimate_lines(grey, [a, b, c], catalogue)
    assert [[r['id'] for r in line['regions']] for line in lines] == [['a', 'b'], ['c']]
    assert [line['source'] for line in lines] == ['catalogue', 'ink']
    assert lines[0]['box'] == [0, 15, 323, 45]                # the catalogue line and its regions, three pixels round
    assert lines[1]['box'] == [17, 97, 203, 123]              # the band follows the ink across the gap


def test_a_region_holding_its_line_is_guessed_whole():
    grey = np.full((60, 200), 255, np.uint8); grey[20:40, 10:110] = 0; grey[20:40, 150:160] = 0
    whole, part = region(5, 18, 115, 42, 'w'), region(5, 18, 60, 42, 'p')
    assert LR.whole_share(grey, whole, [0, 15, 170, 45]) == pytest.approx(100/110)
    assert LR.whole_share(grey, part, [0, 15, 170, 45]) == pytest.approx(50/110)
    assert LR.whole_share(grey, whole, [0, 15, 170, 45]) >= LR.WHOLE > LR.whole_share(grey, part, [0, 15, 170, 45])


def run_rules(script):
    code = f"const L = require({json.dumps(str(RULES))}); console.log(JSON.stringify((() => {{ {script} }})()));"
    return json.loads(subprocess.run([NODE, '-e', code], capture_output=True, text=True, check=True).stdout)


PACK = """
  const box = (l, t, r, b) => [[l, t], [r, t], [r, b], [l, b]];
  const pack = {id: 'p1', pages: [{id: 'q', size: [400, 200],
    regions: [{id: 'q#a', region: box(10, 10, 50, 30), box: [10, 10, 50, 30], runs: [[0, 4]], mode: 'erase', text: 'x', readable: true},
              {id: 'q#b', region: box(60, 10, 90, 30), box: [60, 10, 90, 30], runs: [], mode: 'keep', text: '', readable: false},
              {id: 'q#c', region: box(10, 60, 300, 80), box: [10, 60, 300, 80], runs: [], mode: 'erase', text: '2', readable: true}],
    lines: [{id: 'q#L1', regions: ['q#a', 'q#b'], polygon: box(0, 5, 300, 35), estimate: box(0, 5, 300, 35), whole: false},
            {id: 'q#L2', regions: ['q#c'], polygon: box(10, 60, 300, 80), estimate: box(0, 55, 320, 85), whole: true}]}]};
"""


@pytest.mark.skipif(not NODE, reason='needs node')
def test_whole_line_mark_moves_a_region_to_its_own_line_and_back():
    out = run_rules(PACK + """
      const s = L.initial(pack), o = {};
      o.start = [s.regions['q#c'].whole_line, s.lines['q#L2'].before_whole];
      L.setWhole(pack, s, 'q#a', true);                       // shares a line: a line of its own, the region's polygon
      o.own = [s.regions['q#a'].line, s.lines[s.regions['q#a'].line].polygon, L.members(s, 'q#L1')];
      try { L.reshape(pack, s, s.regions['q#a'].line, box(0, 0, 9, 9)); } catch (e) { o.refused = e.message; }
      try { L.assign(s, 'q#b', s.regions['q#a'].line); } catch (e) { o.noJoin = e.message.length > 0; }
      L.setWhole(pack, s, 'q#a', false); o.back = s.lines[s.regions['q#a'].line].polygon;
      L.assign(s, 'q#a', 'q#L1'); o.empty = L.members(s, 'q#N1'); L.removeLine(s, 'q#N1'); o.removed = !s.lines['q#N1'];
      L.setWhole(pack, s, 'q#c', false); o.c = s.lines['q#L2'].polygon;
      L.reshape(pack, s, 'q#L2', box(0, 50, 330, 90)); o.adjusted = s.lines['q#L2'].adjusted;
      return o;
    """)
    assert out['start'] == [True, [[0, 55], [320, 55], [320, 85], [0, 85]]]
    assert out['own'] == ['q#N1', [[10, 10], [50, 10], [50, 30], [10, 30]], ['q#b']]
    assert '先取消' in out['refused'] and out['noJoin']
    assert out['back'] == [[0, 5], [300, 5], [300, 35], [0, 35]]      # the line it came from
    assert out['empty'] == [] and out['removed'] and out['c'] == [[0, 55], [320, 55], [320, 85], [0, 85]] and out['adjusted']


@pytest.mark.skipif(not NODE, reason='needs node')
def test_export_keeps_the_original_answer_and_imports_back():
    out = run_rules(PACK + """
      const s = L.initial(pack); s.regions['q#b'].readable = true; s.regions['q#b'].text = 'y'; s.regions['q#a'].seen = true;
      L.newLine(pack, s, 0, box(0, 150, 50, 190));              // an empty line is left out
      const e = L.exported(pack, s), back = L.imported(pack, JSON.parse(JSON.stringify(e)));
      let other; try { L.imported({...pack, id: 'p2'}, e); } catch (err) { other = err.message; }
      return {regions: e.regions, lines: e.lines.map(l => [l.id, l.regions, l.whole_line, l.adjusted]), same: JSON.stringify(back) === JSON.stringify(s), other};
    """)
    b = next(r for r in out['regions'] if r['id'] == 'q#b')
    assert b == {'id': 'q#b', 'page': 'q', 'text': 'y', 'readable': True, 'original_text': '', 'original_readable': False, 'corrected': True,
                 'line': 'q#L1', 'whole_line': False, 'seen': False, 'deleted': False, 'added': False, 'region_changed': False,
                 'pixels_changed': False, 'mode': 'keep', 'region': [[60, 10], [90, 10], [90, 30], [60, 30]],
                 'original_region': [[60, 10], [90, 10], [90, 30], [60, 30]], 'box': [60, 10, 90, 30], 'runs': []}
    assert out['lines'] == [['q#L1', ['q#a', 'q#b'], False, False], ['q#L2', ['q#c'], True, False]]
    assert out['same'] and '不属于' in out['other']


@pytest.mark.skipif(not NODE, reason='needs node')
def test_regions_are_reshaped_painted_added_and_taken_out():
    out = run_rules(PACK + """
      const s = L.initial(pack), o = {};
      L.reshapeRegion(pack, s, 'q#a', box(5, 10, 50, 30));        // grown to the left: the strike pixels kept where they were
      o.grown = [L.strike(pack, s, 'q#a', [5, 10, 50, 30]), L.changes(pack, s, 'q#a')];
      L.reshapeRegion(pack, s, 'q#a', box(20, 10, 50, 30)); L.reshapeRegion(pack, s, 'q#a', box(10, 10, 50, 30));
      o.back = L.strike(pack, s, 'q#a', [10, 10, 50, 30]);         // shrunk past them and grown again: back
      const w = L.window(pack, s, 'q#a'), px = L.maskIn(pack, s, 'q#a', w); px[(12 - w[1]) * (w[2] - w[0]) + 30 - w[0]] = 1;
      L.setMask(pack, s, 'q#a', w, px); o.painted = [L.strike(pack, s, 'q#a', [10, 10, 50, 30]), L.changes(pack, s, 'q#a').pixels];
      L.setWhole(pack, s, 'q#c', true); L.reshapeRegion(pack, s, 'q#c', box(10, 58, 310, 82)); o.wholeFollows = s.lines['q#L2'].polygon;
      const added = L.addRegion(pack, s, 0, box(100, 12, 120, 28)), alone = L.addRegion(pack, s, 0, box(100, 150, 120, 170));
      o.added = [added, s.regions[added].line, alone, s.regions[alone].line, L.ids(pack, s)];
      L.removeRegion(s, 'q#b'); o.removed = L.members(s, 'q#L1');
      L.removeRegion(s, alone); L.removeLine(s, s.regions[alone].line);
      L.restoreRegion(pack, s, alone); o.restored = s.lines[s.regions[alone].line].polygon;
      L.removeRegion(s, alone);
      const e = L.exported(pack, s); L.check(pack, JSON.parse(JSON.stringify(s)));
      o.exported = e.regions.map(r => [r.id, r.deleted, r.added, r.line]); o.addedRow = e.regions.find(r => r.id === added);
      return o;
    """)
    assert out['grown'][0] == [[5, 4]] and out['grown'][1] == {'added': False, 'text': False, 'region': True, 'pixels': False}
    assert out['back'] == [[0, 4]]
    assert out['painted'] == [[[0, 4], [100, 1]], True]                 # row 2, column 20 of a box 40 wide
    assert out['wholeFollows'] == [[10, 58], [310, 58], [310, 82], [10, 82]]
    assert out['added'] == ['q#add1', 'q#L1', 'q#add2', 'q#N1', ['q#a', 'q#b', 'q#c', 'q#add1', 'q#add2']]
    assert out['removed'] == ['q#a', 'q#add1']
    assert out['restored'] == [[96, 146], [124, 146], [124, 174], [96, 174]]
    assert out['exported'] == [['q#a', False, False, 'q#L1'], ['q#b', True, False, None], ['q#c', False, False, 'q#L2'], ['q#add1', False, True, 'q#L1']]
    assert out['addedRow']['original_text'] is None and out['addedRow']['runs'] == [] and out['addedRow']['region_changed']
