"""The HTML viewer of tightened masks: a card per mask with its layers, the counts, and the masks that have no ink left."""
import json
import re

import numpy as np
import pytest
from PIL import Image

from revrender import pages, tighten
from revrender.tools import tighten_view


def build(tmp_path, with_scores=False):
    """A page with a dark block, a region mask over it and a mask on the paper only; the same output stands for the gold."""
    before = tmp_path/'before'
    before.mkdir()
    grey = np.full((120, 300), 255, np.uint8)
    grey[50:70, 80:200] = 15
    lines = [{'box': [10, 30, 290, 90], 'source': 'ocr', 'ocr_type': 'text', 'ocr_text': 'x'}]
    elsewhere = np.zeros((60, 280), bool)
    elsewhere[:8, :40] = True
    pages.write_page(before, 'p1', (300, 120), lines, [{'line': 0, 'score': .9, 'box': [10, 30, 290, 90], 'mask': np.ones((60, 280), bool)},
                                                      {'line': 0, 'score': .4, 'box': [10, 30, 290, 90], 'mask': elsewhere}])
    (before/'run_manifest.json').write_text('{}')
    images = tmp_path/'images'
    images.mkdir()
    Image.fromarray(grey).save(images/'p1.jpg', quality=98, subsampling=0)
    after = tmp_path/'after'
    tighten.main(['--pages-output', str(before), '--images', str(images), '--output-dir', str(after)])
    site = tmp_path/'site'
    scores = []
    if with_scores:
        for name, found in (('before', .5), ('after', .75)):
            (tmp_path/f'{name}.json').write_text(json.dumps({'scores': {'gold_components': 4, 'found': found, 'drawn_right': found/2, 'false_positive': 3}}))
        scores = ['--scores', str(tmp_path/'before.json'), str(tmp_path/'after.json')]
    assert tighten_view.main(['--model', str(before), str(after), '--gold', str(before), str(after), '--images', str(images), *scores,
                              '--output-dir', str(site)]) == 0
    return site


def embedded(path):
    return json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', path.read_text(encoding='utf-8'), re.S).group(1)
                      .replace('<\\/', '</'))


def test_there_is_a_card_per_mask_with_its_layers_and_the_one_without_ink_is_marked(tmp_path):
    site = build(tmp_path)
    cards = embedded(site/'compare.html')['cards']
    assert len(cards) == 4 and {c['set'] for c in cards} == {'model', 'gold'}
    for c in cards:
        for suffix in 'cbaek':
            assert (site/'assets'/f'{c["file"]}_{suffix}.png').is_file()
    model = sorted((c for c in cards if c['set'] == 'model'), key=lambda c: c['n'])
    assert model[0]['shape'] != 'gone' and 0 < model[0]['after'] < model[0]['before'] and model[0]['score'] == .9
    assert model[1]['shape'] == 'gone' and model[1]['after'] == 0 and model[1]['before'] == 8*40
    assert model[0]['ghost'] <= model[0]['ghost_ink']                      # erasing the tightened mask leaves no more behind than erasing the ink only
    assert model[0]['specks'] == 0 and model[1]['specks'] == 0             # one piece, and nothing


def test_a_layer_is_red_where_the_ink_is_kept_blue_where_the_paper_is_given_up_and_white_where_it_is_erased(tmp_path):
    site = build(tmp_path)
    card = next(c for c in embedded(site/'compare.html')['cards'] if c['set'] == 'model' and c['n'] == 0)
    after = np.asarray(Image.open(site/'assets'/f'{card["file"]}_a.png'))
    erased = np.asarray(Image.open(site/'assets'/f'{card["file"]}_e.png'))
    assert after.shape[:2] == (card['h'], card['w']) and after.shape[2] == 4
    red, blue = (after[..., 0] > 200) & (after[..., 3] > 0), (after[..., 2] > 200) & (after[..., 0] < 100) & (after[..., 3] > 0)
    assert red.sum() == card['after'] and blue.sum() == card['before']-card['after'] and not (red & blue).any()
    assert ((erased[..., 3] == 255).sum() == card['after']) and (erased[..., :3][erased[..., 3] == 255] == 255).all()


def test_the_index_has_the_counts_and_the_scores_when_they_are_given(tmp_path):
    index = (build(tmp_path, with_scores=True)/'index.html').read_text(encoding='utf-8')
    assert '模型：掩膜个数' in index and '金标：掩膜个数' in index and '被丢弃的</td><td>1<' in index
    assert '50.0% → 75.0%' in index and 'compare.html' in index
    plain = tmp_path/'plain'
    plain.mkdir()
    assert '模型对金标' not in (build(plain)/'index.html').read_text(encoding='utf-8')


def test_the_site_is_create_only_and_a_shape_is_told_by_its_box(tmp_path):
    site = build(tmp_path)
    with pytest.raises(FileExistsError):
        tighten_view.main(['--model', str(tmp_path/'before'), str(tmp_path/'after'), '--images', str(tmp_path/'images'), '--output-dir', str(site)])
    thin = np.zeros((40, 200), bool)
    thin[18:21, 10:190] = True
    full = np.zeros((40, 40), bool)
    full[5:35, 5:35] = True
    ring = np.zeros((40, 40), bool)
    ring[5:35, 5:9] = ring[5:35, 31:35] = ring[5:9, 5:35] = ring[31:35, 5:35] = True
    assert (tighten_view.shape_of(thin), tighten_view.shape_of(full), tighten_view.shape_of(ring)) == ('line', 'block', 'other')
