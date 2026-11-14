"""The transcription scores and the cross-validation pack of struck regions."""
import random

import numpy as np
from PIL import Image

from revrender.data import struck_pack
from revrender.data.pixel_masks import encode
from revrender.eval import transcription
from revrender.eval.revision_boxes import held
from revrender.recognize_lines import CONDITIONS, MAPPED, UNREADABLE, image_keys, prompt


def test_edit_counts_and_rates():
    assert transcription.edits('abc', 'abc') == (0, 0, 0)
    assert transcription.edits('abc', 'axc') == (1, 0, 0)
    assert transcription.edits('abc', 'ac') == (0, 1, 0)
    assert transcription.edits('abc', 'abcd') == (0, 0, 1)
    assert transcription.edits('', 'ab') == (0, 0, 2) and transcription.edits('ab', '') == (0, 2, 0)
    rows = [transcription.counts('x=5', 'x=5'), transcription.counts('2', '2x+3')]
    r = transcription.rates(rows)
    assert r['cr'] == 1. and abs(r['ar']-1/4) < 1e-9 and abs(r['cer']-3/4) < 1e-9          # the added characters cost AR, not CR
    assert r['exact'] == .5 and abs(r['ned']-(0+3/4)/2) < 1e-9


def test_correct_rate_is_the_earlier_read_back():
    rng = random.Random(42)
    for _ in range(200):
        label, reading = (''.join(rng.choice('abc12=+') for _ in range(rng.randint(1, 9))) for _ in range(2))
        c = transcription.counts(label, reading)
        assert abs((c['n']-c['s']-c['d'])/c['n']-held(label, reading)) < 1e-9


def test_refusal_and_pairing():
    readable = {'a_1': True, 'a_2': False, 'b_1': True, 'b_2': False}
    readings = {'a_1': 'x', 'a_2': UNREADABLE, 'b_1': UNREADABLE, 'b_2': 'y'}
    f = transcription.refusal(readings, readable)
    assert f == {'unreadable': 2, 'refusals': 2, 'precision': .5, 'recall': .5, 'f1': .5, 'readable_refused': .5}
    assert transcription.counts('abc', UNREADABLE) == {'n': 3, 's': 0, 'd': 3, 'i': 0, 'm': 0, 'exact': 0.}
    reference = {k: transcription.counts('abc', 'abd') for k in ('a_1', 'b_1', 'c_1')}
    version = {k: transcription.counts('abc', 'abc') for k in ('a_1', 'b_1', 'c_1')}
    p = transcription.paired(reference, version, {'a_1': 'a', 'b_1': 'b', 'c_1': 'c'}, np.random.default_rng(42), draws=50)
    assert p['regions'] == 3 and p['writers'] == 3 and abs(p['ar']['difference']-1/3) < 1e-9 and p['exact']['interval'] == [1., 1.]


def test_folds_are_whole_writers_of_about_equal_regions():
    counts = {f'w{i}': n for i, n in enumerate([38, 28, 23, 19, 17, 9, 9, 6, 6, 5, 4, 3, 2, 2, 1, 1])}
    fold = struck_pack.fold_of(counts)
    assert fold == struck_pack.fold_of(dict(reversed(list(counts.items())))) and set(fold.values()) == set(range(struck_pack.FOLDS))
    sizes = [sum(n for w, n in counts.items() if fold[w] == k) for k in range(struck_pack.FOLDS)]
    assert max(sizes)-min(sizes) <= max(counts.values())


def test_layered_images_of_a_region():
    grey = np.full((60, 80), 255, np.uint8); grey[20:30, 10:50] = 40            # writing
    grey[24:26, 5:60] = 90                                                        # the strike over it
    strike = np.zeros((20, 60), bool); strike[9:11, :55] = True                   # in the box (5, 15, 65, 35)
    row = {'box': [5, 15, 65, 35], 'runs': encode(strike), 'region': None}
    scan, erased, layer = struck_pack.layered(Image.fromarray(grey), row)
    assert scan.size == erased.size == layer.size == (71, 32)                     # the box and six pixels, inside the page
    s, e, m = np.asarray(scan), np.asarray(erased), np.asarray(layer)
    assert layer.mode == 'L' and set(np.unique(m)) == {0, 255} and (m > 0).sum() == 2*55
    assert (e[m > 0] == 255).all() and (e[m == 0] == s[m == 0]).all()


def test_revision_conditions():
    assert image_keys('pair_revision') == ('raw', 'erased') and image_keys('raw_revision') == ('raw',)
    assert 'mapped_revision' in MAPPED and CONDITIONS['mapped_revision'][0] == 'raw'
    assert UNREADABLE in prompt('raw_revision', {'boxes': []}) and UNREADABLE in prompt('erased_revision', {'boxes': []})


def test_separator_views_of_a_region():
    image = np.array([[1., .2, .2], [.9, .2, 1.]], np.float32); restored = np.array([[1., 1., .6], [.9, .2, 1.]], np.float32)
    share = struck_pack.taken_away(image, restored)
    assert np.allclose(share, [[0, 1, .5], [0, 0, 0]])                            # all of one ink pixel, half of the next, paper never
    canvas = struck_pack.pasted(np.zeros((4, 5), np.uint8), np.full((3, 4), 7, np.uint8), (8, 18, 12, 21), (10, 20))
    assert canvas.sum() == 7*2*1 and (canvas[0, :2] == 7).all()                   # the box overhangs the window's left and top
    assert struck_pack.pasted(np.zeros((2, 2), np.uint8), np.ones((2, 2), np.uint8), (50, 50, 52, 52), (0, 0)).sum() == 0


def test_views_of_a_separated_region():
    scan = np.array([[255, 60, 60, 60]], np.uint8); share = np.array([[0, 0, 64, 255]], np.uint8)
    mild, strong = struck_pack.lightened(scan, share, 2), struck_pack.lightened(scan, share, .5)
    assert (mild >= scan).all() and (strong >= mild).all()                       # an erasing only lightens; the strong one more
    assert mild[0, 1] == strong[0, 1] == 60 and mild[0, 3] == strong[0, 3] == 255 and mild[0, 2] < strong[0, 2]
    page = Image.fromarray(np.full((200, 600), 255, np.uint8)); erased = np.full((20, 40), 200, np.uint8)
    band = struck_pack.in_context(page, erased, (300, 90, 340, 110))
    assert band.mode == 'RGB' and band.size == (40+2*struck_pack.CONTEXT*20, 40)
    b = np.asarray(band); x, y = struck_pack.CONTEXT*20, 10                       # the window's corner in the band
    assert (b[y:y+20, x:x+40] == 200).all() and tuple(b[y-2, x]) == (0, 0, 255) and tuple(b[y+21, x+39]) == (0, 0, 255)
    assert struck_pack.in_context(page, erased, (5, 0, 45, 20)).size == (45+struck_pack.CONTEXT*20, 30)      # clipped at the page's edge


def test_conditions_of_several_images():
    assert image_keys('context_revision') == ('raw', 'erased', 'context')
    assert MAPPED == {'mapped_revision', 'pair_mapped_revision'}
    from revrender.models.vlm_tokens import layer_maps
    a, b, layer = Image.new('RGB', (30, 20)), Image.new('RGB', (50, 10)), Image.new('RGB', (30, 20), (255, 255, 255))
    maps = layer_maps([a, b], layer)
    assert maps[0] is layer and maps[1].size == (50, 10) and not np.asarray(maps[1]).any()


def test_agreement_of_two_arms():
    rows = {'a': {'text': 'x=5'}, 'b': {'text': 'y'}, 'c': {'text': '2', 'readable': True}, 'd': {'text': '', 'readable': False}}
    first = {'a': 'x=5', 'b': 'v', 'c': '2', 'd': UNREADABLE}; second = {'a': 'x=5', 'b': 'y', 'c': '7', 'd': 'q'}
    got = transcription.agreement(first, second, rows)
    assert got == {'regions': 4, 'differ': .75, 'exact_where_same': 1., 'exact_where_differ': .5, 'wrong_caught': 1., 'unreadable_differ': 1.}


def test_a_reading_caught_in_a_loop_is_cut_to_one_block():
    from revrender.models.vlm_tokens import looped
    assert looped([1, 2, 3, 4, 5, 6, 7, 8, 9], 4) == 0
    assert looped([9, 8] + [1, 2, 3]*4, 4) == 9 and looped([9, 8] + [1, 2, 3]*3, 4) == 0          # four writings of a block of three: keep one
    assert looped([7]*11, 4) == 0 and looped([7]*12, 4) == 9                                       # a short stutter is left alone
    assert looped([1, 2, 3, 4, 5, 6]*2, 4) == 0
