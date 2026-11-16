import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from revrender.data import prefill_variants as V


def test_a_region_in_context_keeps_its_pixels_and_scores_only_inside(tmp_path):
    page = np.full((100, 120), 255, np.uint8); page[50, 30:90] = 0; page[40:60, 10:20] = 0
    (tmp_path/'pages').mkdir(); Image.fromarray(page).save(tmp_path/'pages'/'p_1_1.png')
    box = [30, 45, 90, 55]; crop = page[45:55, 30:90].copy(); mask = (crop < 128).astype(np.uint8)
    s = tmp_path/'set'; s.mkdir()
    (s/'training_native.json').write_text(json.dumps([{'id': 'p_1_1#det0000', 'page': 'p_1_1', 'box': box, 'region': None, 'group_uuid': 'w'}]))
    np.savez_compressed(s/'training_native.npz', image_0=crop, mask_0=mask, target_0=np.full(crop.shape, 255, np.uint8), weight_0=np.ones(crop.shape, np.uint8))
    (s/'records.json').write_text('[]')
    with mock.patch('revrender.pages.page_files', lambda folders, ids: {i: tmp_path/'pages'/f'{i}.png' for i in ids}):
        V.context(SimpleNamespace(set=s, pages=[tmp_path/'pages'], output_dir=tmp_path/'out'))
    a = np.load(tmp_path/'out'/'training_native.npz'); row = json.loads((tmp_path/'out'/'training_native.json').read_text())[0]
    L, T, R, B = row['window']
    assert (L, T, R, B) == (14, 29, 106, 71) and a['image_0'].shape == (42, 92)
    assert a['inner_0'].sum() == crop.size and a['weight_0'].sum() == crop.size and a['mask_0'].sum() == mask.sum()
    assert (a['image_0'] == page[T:B, L:R]).all() and a['target_0'][16:26, 16:76].min() == 255 and a['target_0'][11, 0] == 0


def test_subsets_are_nested_prefixes_of_writers(tmp_path):
    rows = [{'id': f'r{k}', 'group_uuid': f'w{k % 8}', 'page': 'p', 'box': [0, 0, 2, 2]} for k in range(40)]
    s = tmp_path/'set'; s.mkdir(); (s/'training_native.json').write_text(json.dumps(rows))
    np.savez_compressed(s/'training_native.npz', **{f'image_{k}': np.zeros((2, 2), np.uint8) for k in range(40)}, **{f'mask_{k}': np.zeros((2, 2), np.uint8) for k in range(40)})
    for name in ('validation_native.json', 'summary.json'): (s/name).write_text('[]' if name.endswith('native.json') else '{}')
    np.savez_compressed(s/'validation_native.npz')
    got = {}
    for share in (.25, .5):
        V.subset(SimpleNamespace(set=s, share=share, output_dir=tmp_path/f'o{share}'))
        got[share] = {r['group_uuid'] for r in json.loads((tmp_path/f'o{share}'/'training_native.json').read_text())}
    assert len(got[.25]) == 2 and len(got[.5]) == 4 and got[.25] < got[.5]


def test_augmentation_moves_tile_target_and_weight_together():
    from revrender.baselines.restoration import augmented, inside
    x = np.ones((3, 64, 64), np.float32); x[:, 30:34, 10:54] = .2; y = np.ones_like(x); w = np.ones_like(x)
    a, b, c = augmented(x, y, w, np.random.default_rng(42))
    assert a.shape == x.shape and b.min() == 1. and a.min() < .5 and set(np.unique(c)) <= {0., 1.}
    assert ((a < .5) <= (c > 0)).all()          # ink brought in only where the weight is kept
    im, pred, gold = inside(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)), np.eye(4, dtype=np.uint8))
    assert im.sum() == 12 and gold.sum() == 4
