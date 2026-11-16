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


def test_a_smaller_share_holds_out_a_subset_of_the_writers():
    writers = [f'{k:08x}-writer' for k in range(2000)]
    tenth = {w for w in writers if held_out(w, .1)}
    assert .08 < len(tenth)/len(writers) < .12 and tenth <= {w for w in writers if held_out(w)}


def test_a_dataset_gives_its_confirmed_certain_regions(tmp_path):
    from PIL import Image
    from revrender.data.pixel_masks import encode
    from revrender.data.prefill_set import dataset_regions
    page = np.full((40, 60), 255, np.uint8); page[20, 10:50] = 0
    (tmp_path/'pages').mkdir(); Image.fromarray(page).save(tmp_path/'pages'/'aaaaaaaa-0000-0000-0000-000000000000_1_1.png')
    mask = np.zeros((20, 50), bool); mask[10, :40] = True
    row = lambda i, **k: {'id': f'aaaaaaaa-0000-0000-0000-000000000000_1_1#{i}', 'page': 'aaaaaaaa-0000-0000-0000-000000000000_1_1', 'box': [10, 10, 60, 30],
                          'region': None, 'runs': encode(mask), 'pixels_confirmed': True, 'edits': 3, 'prefill_model': 'm', **k}
    (tmp_path/'r.json').write_text(json.dumps([row('a'), row('b', deleted=True), row('c', pixels_confirmed=False), row('d', pixels_uncertain=True)]))
    from unittest import mock
    with mock.patch('revrender.pages.page_files', lambda folders, ids: {i: tmp_path/'pages'/f'{i}.png' for i in ids}):
        found = dataset_regions(tmp_path/'r.json', [tmp_path/'pages'])
    assert [r['id'][-1] for _, _, r, _, _ in found] == ['a'] and found[0][2]['ops'] == 3 and found[0][4].sum() == 40


def test_tiers_split_the_blocks_by_how_heavy_the_strike_is(tmp_path):
    from types import SimpleNamespace
    from revrender.eval.prefill import tiers
    region = lambda i, share, b: {'id': f'r{i}', 'writer': 'w', 'blocks': b, 'strike': 100*(i+1), 'writing': 100, 'missed': b, 'writing_erased': 0, 'to_fix': .1,
                                  'strike_share': share, 'near_share': .3, 'missed_on_writing': b, 'erased_near_strike': 0}
    regions = [region(0, .1, 1), region(1, .1, 1), region(2, .3, 2), region(3, .5, 6)]
    (tmp_path/'s.json').write_text(json.dumps({'m': {'sets': {'h': {'regions': regions}}}}))
    (tmp_path/'r.jsonl').write_text(json.dumps({'region': 'r3', 'readable': True, 'label': 'ab', 'a': 'ab'}) + '\n' + json.dumps({'region': 'r0', 'readable': True, 'label': 'ab', 'a': 'xb'}))
    tiers(SimpleNamespace(scores=tmp_path/'s.json', set='h', train_set=None, readings=tmp_path/'r.jsonl', output=tmp_path/'t.json'))
    t = json.loads((tmp_path/'t.json').read_text())
    s = t['models']['m']['strike_share']
    assert (s['light']['blocks'], s['middle']['blocks'], s['heavy']['blocks']) == (1, 2, 6) and s['tiers_mean'] == 3
    assert t['readings']['a']['strike_share']['heavy']['exact'] == 1 and t['readings']['a']['strike_share']['light']['exact'] == 0
    assert t['models']['m']['where']['missed_on_writing'] == 1
