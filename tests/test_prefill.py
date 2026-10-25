import json

import numpy as np

from revrender.data.prefill_set import held_out
from revrender.eval.prefill import edits, summary_of, switch


def test_edits_count_the_strike_left_and_the_writing_erased_over_the_strike():
    image = np.full((20, 40), 1., np.float32); image[5, 2:38] = .1; image[10:15, 5:8] = .2
    gold = np.zeros(image.shape, bool); gold[5, 2:38] = True
    restored = image.copy(); restored[5, 2:20] = 1.; restored[10:12, 5:8] = 1.     # half the strike, six pixels of writing
    e = edits(image, gold, restored)
    assert (e['missed'], e['writing_erased'], e['strike']) == (18, 6, 36) and abs(e['to_fix']-24/36) < 1e-9 and e['blocks'] == 1
    s = summary_of([e, {**e, 'to_fix': 0., 'missed': 0, 'writing_erased': 0, 'blocks': 0}])
    assert s['no_edit_share'] == .5 and abs(s['to_fix']-12/36) < 1e-9


def test_the_switch_rule_wants_fewer_pixels_to_fix_and_no_more_writing_erased(tmp_path):
    region = lambda i, w, f, e: {'id': i, 'writer': w, 'to_fix': f, 'blocks': 10*f, 'missed': 0, 'writing_erased': e, 'strike': 10, 'writing': 100}
    cur = [region(f'r{k}', f'w{k%5}', .5, 2) for k in range(20)]; new = [region(f'r{k}', f'w{k%5}', .3, 3) for k in range(20)]
    (tmp_path/'s.json').write_text(json.dumps({m: {'sets': {'h': {'regions': r}}} for m, r in (('cur', cur), ('new', new))}))
    from types import SimpleNamespace
    switch(SimpleNamespace(scores=tmp_path/'s.json', set='h', current='cur', candidate='new', output=tmp_path/'a.json', measure='blocks'))
    assert json.loads((tmp_path/'a.json').read_text())['replace']
    worse = [dict(r, writing_erased=10) for r in new]
    (tmp_path/'t.json').write_text(json.dumps({m: {'sets': {'h': {'regions': r}}} for m, r in (('cur', cur), ('new', worse))}))
    switch(SimpleNamespace(scores=tmp_path/'t.json', set='h', current='cur', candidate='new', output=tmp_path/'b.json', measure='blocks'))
    assert not json.loads((tmp_path/'b.json').read_text())['replace']        # writing erased up 8 points


def test_the_held_out_writers_are_a_stable_seventh_or_so():
    writers = [f'{k:08x}-writer' for k in range(2000)]
    share = np.mean([held_out(w) for w in writers])
    assert .12 < share < .18 and held_out(writers[3]) == held_out(writers[3])
