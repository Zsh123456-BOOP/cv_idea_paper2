"""The HTML viewer of revrender.pages output: the answer-length split, the transparent masks, the pages and their data."""
import json
import re

import numpy as np
import pytest
from PIL import Image

from revrender import pages
from revrender.tools import pages_view


def build(tmp_path, min_chars='10', png=()):
    out = tmp_path/'pages_out'
    out.mkdir()
    lines = [{'box': [10, 50, 190, 90], 'source': 'ocr', 'ocr_type': 'text', 'ocr_text': 'x = 1', 'shape': None}]
    instance = {'line': 0, 'score': .91, 'box': [10, 20, 110, 60], 'mask': np.zeros((40, 100), bool)}
    instance['mask'][10:30, 20:80] = True
    pages.write_page(out, 'w1_3_1', (200, 100), lines, [instance])
    pages.write_page(out, 'w2_s4_1', (200, 100), lines, [])
    (out/'run_manifest.json').write_text('{}')
    images = tmp_path/'images'
    images.mkdir()
    for name in ('w1_3_1', 'w2_s4_1'):
        Image.new('L', (200, 100), 255).save(images/f'{name}.png' if name in png else images/f'{name}.jpg')
    texts = tmp_path/'qwen.jsonl'
    texts.write_text(json.dumps({'image_id': 'w1_3_1', 'text': r'\( x = 3 \) 所以 y = 2 ，因此 z = 5'})+'\n'+
                     json.dumps({'image_id': 'w2_s4_1', 'text': 'x'})+'\n')
    tags = tmp_path/'tags.json'
    tags.write_text(json.dumps({'w1_3_1': {'split': 'test', 'rows': {'test': 2}}}))
    site = tmp_path/'site'
    assert pages_view.main(['--pages-output', str(out), '--images', str(images), '--answer-text', str(texts), '--tags', str(tags),
                            '--min-chars', min_chars, '--output-dir', str(site)]) == 0
    return site


def embedded(path):
    return json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', path.read_text(encoding='utf-8'), re.S).group(1)
                      .replace('<\\/', '</'))


def test_a_page_with_too_little_writing_is_set_aside_and_the_rest_are_kept(tmp_path):
    site = build(tmp_path)
    assert (site/'kept_ids.txt').read_text() == 'w1_3_1\n' and (site/'filtered_ids.txt').read_text() == 'w2_s4_1\n'
    kept, filtered = embedded(site/'kept.html'), embedded(site/'filtered.html')
    assert kept['mode'] == 'kept' and filtered['mode'] == 'filtered' and kept['min_chars'] == 10
    card = kept['cards'][0]
    assert (card['q'], card['supp'], card['n'], card['mask'], card['split'], card['rows']) == (3, False, 1, True, 'test', {'test': 2})
    assert card['chars'] >= 10 and card['scores'] == [.91] and card['boxes'] == [[30, 30, 90, 50]] and card['lines'] == [[[10, 50, 190, 90], 'ocr']]
    other = filtered['cards'][0]
    assert (other['q'], other['supp'], other['n'], other['mask'], other['chars']) == (4, True, 0, False, 1)
    assert 'kept.html' in (site/'index.html').read_text(encoding='utf-8')


def test_the_threshold_decides_and_the_mask_is_red_only_on_the_recognised_revision(tmp_path):
    site = build(tmp_path, min_chars='1000')
    assert embedded(site/'kept.html')['cards'] == [] and len(embedded(site/'filtered.html')['cards']) == 2
    mask = np.asarray(Image.open(site/'assets'/'w1_3_1_mask.png'))
    assert mask.shape == (100, 200, 4) and mask[40, 60, 3] > 0 and mask[40, 60, 0] > 200 and mask[5, 5, 3] == 0
    assert not (site/'assets'/'w2_s4_1_mask.png').exists() and (site/'assets'/'w2_s4_1.jpg').is_file()


def test_the_site_is_create_only_and_refuses_a_page_without_a_transcription(tmp_path):
    site = build(tmp_path)
    with pytest.raises(FileExistsError):
        pages_view.main(['--pages-output', str(tmp_path/'pages_out'), '--images', str(tmp_path/'images'), '--answer-text', str(tmp_path/'qwen.jsonl'),
                         '--output-dir', str(site)])
    (tmp_path/'only_one.jsonl').write_text(json.dumps({'image_id': 'w1_3_1', 'text': 'x = 3 y = 2 z = 5'})+'\n')
    with pytest.raises(Exception, match='no transcription'):
        pages_view.main(['--pages-output', str(tmp_path/'pages_out'), '--images', str(tmp_path/'images'), '--answer-text', str(tmp_path/'only_one.jsonl'),
                         '--output-dir', str(tmp_path/'site2')])


def test_question_numbers_and_embedded_json_are_read_safely():
    assert pages_view.question_of('abc_7_1') == (7, False) and pages_view.question_of('abc_s9_2') == (9, True)
    assert '</script>' not in pages_view.embed({'text': '</script><b>'}) and json.loads(pages_view.embed({'a': '中文'}).replace('<\\/', '</')) == {'a': '中文'}


def test_a_cleaned_page_is_a_png_and_its_mask_does_not_take_its_name(tmp_path):
    site = build(tmp_path, png=('w1_3_1',))
    card = embedded(site/'kept.html')['cards'][0]
    assert card['ext'] == 'png' and (site/'assets'/'w1_3_1.png').is_file() and (site/'assets'/'w1_3_1_mask.png').is_file()
    assert np.asarray(Image.open(site/'assets'/'w1_3_1.png')).ndim == 2 and np.asarray(Image.open(site/'assets'/'w1_3_1_mask.png')).ndim == 3
    assert embedded(site/'filtered.html')['cards'][0]['ext'] == 'jpg'
