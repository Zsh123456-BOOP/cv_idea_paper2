"""The pilot keeps existing cuts and applies only the declared image intervention."""
import json

import numpy as np
import pytest
from PIL import Image

from revrender import recognize
from revrender.data import recognition_pack as pack
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
    assert recognize.messages('B') == recognize.messages('C')
    assert sum(x['type'] == 'image' for x in recognize.messages('B')[1]['content']) == 2
    assert sum(x['type'] == 'image' for x in recognize.messages('D')[1]['content']) == 3
    assert recognize.clean_text('[空]</transcription>') == ''
    assert recognize.clean_text('x=7\\n y=3</transcription>extra') == 'x=7\\n y=3'


def test_a_test_page_is_rejected_before_its_image_or_masks_are_read(tmp_path):
    (tmp_path/'ids.txt').write_text('writer_3_1\n')
    (tmp_path/'tags.json').write_text(json.dumps({'writer_3_1': {'split': 'test'}}))
    (tmp_path/'index.json').write_text(json.dumps({'pages': {'writer_3_1': {'kept': True}}}))
    with pytest.raises(RefusedError, match='development page'):
        pack.main(['--pages-output', str(tmp_path/'missing'), '--images', str(tmp_path/'missing'),
                   '--ids', str(tmp_path/'ids.txt'), '--tags', str(tmp_path/'tags.json'),
                   '--page-index', str(tmp_path/'index.json'), '--output-dir', str(tmp_path/'out')])
    assert not (tmp_path/'out').exists()
