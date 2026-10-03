"""The pilot keeps existing cuts and applies only the declared image intervention."""
import json

import numpy as np
import pytest
from PIL import Image

from revrender import recognize
from revrender.data import recognition_pack as pack
from revrender.tools import recognition_view
from revrender.util import RefusedError


def test_selection_is_fixed_balanced_and_prefers_distinct_writers():
    candidates = [{'image_id': f'w{i}_{i % 7 + 2}_1', 'stratum': kind}
                  for i, kind in enumerate(['none', 'thin', 'block', 'mixed']*15)]
    chosen = pack.choose(candidates)
    assert chosen == pack.choose(list(reversed(candidates)))
    assert len({p['image_id'].split('_')[0] for p in chosen}) == 40
    assert all(sum(p['stratum'] == kind for p in chosen) == 10 for kind in ('none', 'thin', 'block', 'mixed'))
    with pytest.raises(RefusedError):
        pack.choose(candidates, 3)


def test_ordering_keeps_every_original_box_and_index():
    lines = [{'index': 4, 'box': [10, 70, 150, 95], 'source': 'ink'},
             {'index': 0, 'box': [10, 10, 150, 35], 'source': 'ocr'},
             {'index': 2, 'box': [10, 40, 150, 65], 'source': 'split'}]
    ordered = pack.ordered_lines({'size': [200, 100], 'lines': lines})
    assert [p['index'] for p in ordered] == [0, 2, 4]
    assert all(any(p is original for original in lines) for p in ordered)


def test_erasure_changes_exactly_the_mask_and_hint_keeps_unmarked_pixels():
    pixels = np.full((100, 200, 3), 255, np.uint8)
    pixels[20:70, 30:150] = 70
    mask = np.zeros((40, 60), bool)
    mask[10:15, 5:40] = True
    variants, union = pack.page_variants(Image.fromarray(pixels), [{'box': [20, 20, 80, 60], 'mask': mask}])
    assert union.sum() == mask.sum()
    assert np.array_equal(variants['raw'], pixels)
    erased, hint = np.asarray(variants['erased']), np.asarray(variants['hint'])
    assert (erased[union] == 255).all()
    assert np.array_equal(erased[~union], pixels[~union])
    assert np.array_equal(hint[~union], pixels[~union])
    assert (hint[union, 0] > hint[union, 1]).all()


def test_recognition_prompt_shares_the_task_and_only_d_adds_the_hint():
    for task in ('verbatim', 'deletion'):
        assert recognize.messages('B', task) == recognize.messages('C', task)
        assert recognize.messages('B', task)[1]['content'][0]['text'].startswith(recognize.INSTRUCTIONS[task])
        assert sum(x['type'] == 'image' for x in recognize.messages('B', task)[1]['content']) == 1
        assert sum(x['type'] == 'image' for x in recognize.messages('D', task)[1]['content']) == 2
    assert recognize.clean_text('[空]</transcription>') == ''
    assert recognize.clean_text('x=7\\n y=3</transcription>extra') == 'x=7\\n y=3'
    assert recognize.clean_text('<del>x=7</del> y=3</transcription>') == '<del>x=7</del> y=3'


def test_model_receives_only_the_exact_original_cut(tmp_path):
    colors = {'raw': 'black', 'erased': 'white', 'hint': 'red'}
    row = {'images': {k: f'{k}.png' for k in colors}}
    for k, color in colors.items():
        Image.new('RGB', (120, 40), color).save(tmp_path/row['images'][k])
    assert [im.size for im in recognize.input_images(tmp_path, row, 'D')] == [(120, 40), (120, 40)]
    assert recognize.input_images(tmp_path, row, 'B')[0].getpixel((0, 0)) == (0, 0, 0)
    assert recognize.input_images(tmp_path, row, 'C')[0].getpixel((0, 0)) == (255, 255, 255)


def test_revision_subset_is_fixed_positive_and_covers_every_positive_page(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    Image.new('RGB', (120, 40), 'white').save(source/'image.png')
    files = {k: 'image.png' for k in ('raw', 'erased', 'hint')}
    pages = [{'image_id': f'p{j}', 'split': 'dev', 'images': files,
              'rows': [{'id': f'p{j}:L{i}', 'masked_pixels': i if j < 3 else 0,
                        'box': [0, 0, 120, 40], 'images': files} for i in range(4)]} for j in range(4)]
    (source/'pack.json').write_text(json.dumps({'pages': pages, 'seed': 42}))
    pack.revision_subset(source, 7, tmp_path/'a')
    pack.revision_subset(source, 7, tmp_path/'b')
    result = json.loads((tmp_path/'a/pack.json').read_text())
    assert result == json.loads((tmp_path/'b/pack.json').read_text())
    assert len(result['pages']) == 3
    assert sum(len(p['rows']) for p in result['pages']) == 7
    assert all(r['masked_pixels'] > 0 for p in result['pages'] for r in p['rows'])
    assert result['selection']['candidate_rows'] == 9


def test_a_test_page_is_rejected_before_its_image_or_masks_are_read(tmp_path):
    (tmp_path/'ids.txt').write_text('writer_3_1\n')
    (tmp_path/'tags.json').write_text(json.dumps({'writer_3_1': {'split': 'test'}}))
    (tmp_path/'index.json').write_text(json.dumps({'pages': {'writer_3_1': {'kept': True}}}))
    with pytest.raises(RefusedError, match='development page'):
        pack.main(['--pages-output', str(tmp_path/'missing'), '--images', str(tmp_path/'missing'),
                   '--ids', str(tmp_path/'ids.txt'), '--tags', str(tmp_path/'tags.json'),
                   '--page-index', str(tmp_path/'index.json'), '--output-dir', str(tmp_path/'out')])
    assert not (tmp_path/'out').exists()


def test_comparison_refuses_duplicate_or_unrelated_predictions(tmp_path):
    path = tmp_path/'predictions.jsonl'
    item = {'row_id': 'writer:L001', 'test': 'deletion', 'mode': 'B', 'text': '', 'truncated': False}
    path.write_text(json.dumps(item)+'\n')
    assert recognition_view.predictions([path], {'writer:L001'})['writer:L001']['deletion']['B']['text'] == ''
    with pytest.raises(RefusedError, match='duplicate prediction'):
        recognition_view.predictions([path, path], {'writer:L001'})
    with pytest.raises(RefusedError, match='does not belong'):
        recognition_view.predictions([path], {'another:L001'})
