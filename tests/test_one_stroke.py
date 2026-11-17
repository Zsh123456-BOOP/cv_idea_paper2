import random

import numpy as np

from revrender.data import one_stroke as O
from revrender.data.strike_bank import line_measures


def letters(shape, xs, top=15, bottom=45, pen=3, bar=12):
    """Grey with letter-like strokes: an upright and a bar on top at each x."""
    grey = np.full(shape, 255, np.uint8)
    for x in xs:
        grey[top:bottom, x:x+pen] = 30; grey[top:top+pen, x:x+bar] = 30
    return grey


def erased(grey, strike_rows, x0, x1):
    """The strike drawn on ``grey`` and what an annotator erases of it: its pixels off the writing."""
    writing = grey < 200
    struck = grey.copy(); struck[strike_rows, x0:x1] = 40
    return struck, (struck < 200) & ~writing


def test_a_strike_broken_where_it_lay_on_writing_is_one_line_whole_and_without_writing():
    grey = letters((60, 200), range(30, 170, 22))
    struck, strike = erased(grey, slice(29, 32), 20, 180)
    units, why = O.units_of(struck, strike, 40)
    assert [u['kind'] for u in units] == ['line'] and not why
    repaired = units[0]['repaired']
    m = line_measures(repaired)
    assert m['cover'] == 1 and abs(m['fill']-1) < .05
    assert (repaired[29:32, 30:33] > .5).all() and not (repaired[15:26] > .3).any()       # across the crossings, not up the letters


def test_two_strokes_over_one_word_are_a_group_and_a_long_paper_gap_splits_a_line():
    grey = letters((60, 200), range(30, 170, 22))
    struck, strike = erased(grey, slice(24, 27), 20, 180)
    struck[34:37, 20:180] = 40; strike |= (struck < 200) & ~(grey < 200)
    units, _ = O.units_of(struck, strike, 40)
    assert [u['kind'] for u in units] == ['group'] and len(units[0]['segments']) == 2


def test_a_line_carried_to_a_host_strikes_its_writing_at_the_same_height_and_angle():
    origin = letters((60, 200), range(30, 110, 16), bar=8)
    struck, strike = erased(origin, slice(36, 39), 24, 110)              # low on its writing (body 15-45)
    unit = O.units_of(struck, strike, 40)[0][0]
    writing = O.writing_of(struck, strike)
    frame, _ = O.struck(writing, unit['repaired'] >= .3)
    unit.update(frame=frame, measures=O.measures(unit['repaired'], writing, frame))
    host_grey = letters((120, 400), range(40, 330, 32), top=40, bottom=100, pen=6, bar=16)   # twice as tall, twice the pen
    host = {'grey': host_grey, 'template': None, 'row': O.host_row(host_grey, (20, 30, 380, 110))}
    got, why = O.put(unit, unit['repaired'], host, random.Random(42))
    assert got is not None, why
    assert abs(got['measures']['height']-unit['measures']['height']) <= .15 and got['measures']['height'] > .6
    L, T, R, B = got['box']
    assert np.array_equal(got['target'], host_grey[T:B, L:R]) and (got['image'] <= got['target']).all()
    assert 1.6 < got['scale'] < 2.4 and not (got['remove'] & (got['target'] < 235)).any()
    ys, xs = np.nonzero(got['alpha'] >= .5)
    assert abs(np.polyfit(xs, ys, 1)[0]) < .05                            # level writing: a level line
