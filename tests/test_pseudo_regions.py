import json

import numpy as np
import pytest
from PIL import Image

from revrender.data.pseudo_regions import combine, eligible, main
from revrender.data.restoration_pairs import tiles


def library(tmp_path):
    (tmp_path/'images').mkdir()
    page = np.full((40, 60), 255, np.uint8); page[10:12, 5:55] = 40; page[5:30, 20:22] = 30      # a strike across one writing stroke
    Image.fromarray(page).save(tmp_path/'images/p.png'); Image.fromarray(page).save(tmp_path/'images/q.png')
    pages = [{'id': 'w1_1_1', 'image': 'images/p.png', 'split': 'train', 'writer_group': 'w1'},
             {'id': 'w2_1_1', 'image': 'images/q.png', 'split': 'test', 'writer_group': 'w2'},
             {'id': 'w3_1_1', 'image': 'images/q.png', 'split': 'train', 'writer_group': 'w3'}]
    rows = [{'id': 'w1_1_1#det0001', 'page': 0, 'box': [0, 0, 60, 40], 'confirmed': False},
            {'id': 'w1_1_1#det0000', 'page': 0, 'box': [0, 0, 30, 40], 'confirmed': True},
            {'id': 'w2_1_1#det0000', 'page': 1, 'box': [0, 0, 60, 40], 'confirmed': False},
            {'id': 'w3_1_1#det0000', 'page': 2, 'box': [0, 0, 60, 40], 'confirmed': False}]
    (tmp_path/'pack.json').write_text(json.dumps({'pages': pages, 'rows': rows}))
    return page


def test_eligible_keeps_training_pages_of_other_writers_and_unconfirmed_regions(tmp_path):
    library(tmp_path)
    rows = eligible(tmp_path, {'w3'})
    assert [r['id'] for r in rows] == ['w1_1_1#det0001'] and rows[0]['group_uuid'] == 'w1' and rows[0]['box'] == [0, 0, 60, 40]
    assert len(eligible(tmp_path, set())) == 2 and len(eligible(tmp_path, set(), {'w3_1_1#det0000'})) == 1


def test_combine_takes_the_agreement_weights_out_the_doubt_and_appends_the_human_regions(tmp_path):
    page = library(tmp_path); rows = eligible(tmp_path, {'w3'})
    grey = page[0:40, 0:60]
    def teacher(name, erase):
        d = tmp_path/name; d.mkdir(); out = grey.copy(); out[erase] = 255
        (d/'regions.json').write_text(json.dumps(rows)); np.savez(d/'restored.npz', restored_0=out); (d/'run_manifest.json').write_text('{}')
        return d
    strike = np.zeros_like(grey, bool); strike[10:12, 5:55] = True
    a = teacher('a', strike); b = teacher('b', strike & (np.arange(60)[None] < 40)); c = teacher('c', np.zeros_like(strike))
    review = tmp_path/'review'; review.mkdir()
    human = np.full((20, 20), 255, np.uint8); human[5, :] = 50; hmask = np.zeros_like(human); hmask[5, :] = 1
    (review/'training_native.json').write_text(json.dumps([{'box': [0, 0, 20, 20], 'group_uuid': 'h', 'id': 'h_1_1#det0000', 'page': 'h_1_1', 'region': []}]))
    np.savez(review/'training_native.npz', image_0=human, mask_0=hmask)
    (review/'validation_native.json').write_text('[]'); np.savez(review/'validation_native.npz'); (review/'summary.json').write_text('{"validation_writers": ["v"]}')
    source = tmp_path/'pairs'; source.mkdir(); (source/'records.json').write_text(json.dumps([{'group_uuid': 'w9', 'split': 'val'}, {'group_uuid': 'w1', 'split': 'train'}]))
    out = tmp_path/'set'
    main(['combine', '--predictions', str(a), str(b), str(c), '--review-set', str(review), '--source', str(source), '--agree', '2', '--human-copies', '3', '--output-dir', str(out)])
    native = json.loads((out/'training_native.json').read_text()); arrays = np.load(out/'training_native.npz')
    assert len(native) == 4 and native[0]['id'] == 'w1_1_1#det0001' and native[1]['id'] == 'h_1_1#det0000' and native[3]['id'] == 'h_1_1#det0000~copy2'
    mask, weight = arrays['mask_0'].astype(bool), arrays['weight_0']
    assert mask[11, 10] and not mask[11, 50] and weight[11, 50] == 0 and weight[11, 10] == 1 and weight[20, 20] == 1   # two agree / one alone / paper
    assert not mask[20, 21] and weight[20, 21] == 1                                                                    # the writing nobody erased
    assert (arrays['weight_1'] == 1).all() and arrays['mask_1'][5, 3] == 1
    records = json.loads((out/'records.json').read_text()); assert [r['source'] for r in records] == ['pseudo_agreement'] + ['human_confirmed']*3
    assert (out/'validation_native.json').exists() and json.loads((out/'summary.json').read_text())['counts']['pseudo_regions'] == 1


def test_tiles_carry_the_weight_beside_the_target():
    image = np.full((30, 300), 200, np.uint8); mask = np.zeros_like(image); mask[:, :10] = 1; weight = np.ones_like(image); weight[:, 10:20] = 0
    parts = tiles(image, mask, weight)
    assert len(parts) == 2 and all(len(p) == 3 for p in parts)
    tile, target, w = parts[0]; oy = (256-30)//2
    assert target[oy, 5] == 255 and tile[oy, 5] == 200 and w[oy, 15] == 0 and w[oy, 25] == 1 and w[0, 0] == 1
    assert all(t[2].dtype == np.uint8 for t in parts) and all((t[2] == 1).all() for t in tiles(image, mask)[:1])


def test_soft_combine_targets_the_teachers_mean_with_full_weight(tmp_path):
    page = library(tmp_path); rows = eligible(tmp_path, {'w3'}); grey = page[0:40, 0:60]
    def teacher(name, erase):
        d = tmp_path/name; d.mkdir(); out = grey.copy(); out[erase] = 255
        (d/'regions.json').write_text(json.dumps(rows)); np.savez(d/'restored.npz', restored_0=out); (d/'run_manifest.json').write_text('{}')
        return d
    strike = np.zeros_like(grey, bool); strike[10:12, 5:55] = True
    a = teacher('a', strike); b = teacher('b', np.zeros_like(strike))
    review = tmp_path/'review'; review.mkdir()
    (review/'training_native.json').write_text('[]'); np.savez(review/'training_native.npz')
    (review/'validation_native.json').write_text('[]'); np.savez(review/'validation_native.npz'); (review/'summary.json').write_text('{}')
    out = tmp_path/'soft'
    main(['combine', '--predictions', str(a), str(b), '--review-set', str(review), '--soft', '--human-copies', '0', '--output-dir', str(out)])
    arrays = np.load(out/'training_native.npz')
    assert arrays['target_0'][11, 10] == round((255 + int(grey[11, 10]))/2) and arrays['target_0'][20, 21] == grey[20, 21] and (arrays['weight_0'] == 1).all()
    tile, target, weight = tiles(arrays['image_0'], arrays['mask_0'], arrays['weight_0'], arrays['target_0'])[0]
    oy, ox = (256-40)//2, (256-60)//2
    assert target[oy+11, ox+10] == arrays['target_0'][11, 10] and target[0, 0] == 255 and tile[oy+11, ox+10] == grey[11, 10]
