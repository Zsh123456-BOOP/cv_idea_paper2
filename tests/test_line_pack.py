"""The pack of human-labelled lines: who is in it, and the three images a line gets."""
import json

import numpy as np
import pytest
from PIL import Image

from revrender import infer
from revrender.data import line_pack as L

ROWS = [
    {'line_id': 'w1_3_1:L0002', 'image_id': 'w1_3_1', 'kind': 'pair', 'split': 'train', 'text_before': 'x=3', 'text_after': 'x=-3', 'ink_kind': 'struck'},
    {'line_id': 'w1_3_1:L0001', 'image_id': 'w1_3_1', 'kind': 'no_edit', 'split': 'dev'},
    {'line_id': 'w2_4_1:L0001', 'image_id': 'w2_4_1', 'kind': 'pair', 'split': 'test', 'text_before': 'y=2'},
    {'line_id': 'w1_3_1:L0003', 'image_id': 'w1_3_1', 'kind': 'pair', 'split': 'none', 'ink_kind': 'struck'},
    {'line_id': 'w1_3_1:L0004', 'image_id': 'w1_3_1', 'kind': 'edit_no_text', 'split': 'none'},
]


def test_a_test_writers_line_and_a_pair_without_text_are_left_out_and_the_rest_come_by_line_id():
    chosen = L.select(ROWS)
    assert [r['line_id'] for r in chosen] == ['w1_3_1:L0001', 'w1_3_1:L0002'] and L.select(ROWS, 1)[0]['line_id'] == 'w1_3_1:L0001'


def test_the_columns_under_a_mask_are_blanked_to_the_full_height():
    rgb = Image.fromarray(np.full((20, 40, 3), 120, np.uint8))
    mask = np.zeros((20, 40), bool)
    mask[5:10, 10:20] = True
    out = np.asarray(L.blank(rgb, [mask]))
    assert (out[:, 6:24] == 255).all() and (out[:, :6] == 120).all() and (out[:, 24:] == 120).all()
    assert (np.asarray(L.blank(rgb, [np.zeros((20, 40), bool)])) == 120).all()


def test_the_masked_ink_is_whited_out_in_one_image_and_tinted_red_in_another():
    rgb = Image.fromarray(np.full((20, 40, 3), 120, np.uint8))
    mask = np.zeros((20, 40), bool)
    mask[5:10, 10:20] = True
    erased, tinted = np.asarray(L.erase(rgb, mask)), np.asarray(L.tint(rgb, mask))
    assert (erased[5:10, 10:20] == 255).all() and (erased[0] == 120).all()
    assert (tinted[5:10, 10:20, 0] > tinted[5:10, 10:20, 1]).all() and (tinted[0] == 120).all()
    assert L.boxes_of([mask], (40, 20)) == [[250, 250, 500, 500]] and L.boxes_of([np.zeros((20, 40), bool)], (40, 20)) == []


def test_a_pack_has_the_three_images_the_labels_and_the_boxes_of_each_line(tmp_path, monkeypatch):
    pages_dir = tmp_path/'pages'
    pages_dir.mkdir()
    page = np.full((100, 400), 255, np.uint8)
    page[60:70, 100:300] = 20                                   # a dark bar in the second line
    Image.fromarray(page).save(pages_dir/'w1_3_1.png')
    catalogue = tmp_path/'catalogue.jsonl'
    catalogue.write_text(''.join(json.dumps({'image_id': 'w1_3_1', 'representative_line_id': f'w1_3_1:L000{k}', 'alias_line_ids': [], 'box_1000': box}) + '\n'
                                 for k, box in ((1, [0, 0, 1000, 400]), (2, [0, 500, 1000, 900]))))
    pairs = tmp_path/'pairs.jsonl'
    pairs.write_text(''.join(json.dumps(r)+'\n' for r in ROWS[:2]))
    freeze = tmp_path/'freeze.json'
    freeze.write_text('{}')
    monkeypatch.setattr(infer, 'load_checkpoint', lambda path, expected_sha=None: (object(), {'step': 1}))
    monkeypatch.setattr(L, 'frozen_point', lambda path, sha: {'mask_threshold': .4, 'nms': .5, 'score_threshold': .3})
    monkeypatch.setattr(infer, 'predict_lines', lambda model, strips, point, batch_size=8: [[(np.asarray(s) < 128, .9)] if (np.asarray(s) < 128).any() else [] for s in strips])
    (tmp_path/'checkpoint.pt').write_text('x')
    out = tmp_path/'pack'
    assert L.main(['--pairs', str(pairs), '--catalogue', str(catalogue), '--pages', str(pages_dir), '--checkpoint', str(tmp_path/'checkpoint.pt'),
                   '--freeze', str(freeze), '--output-dir', str(out)]) == 0
    pack = json.loads((out/'pack.json').read_text())
    by_id = {r['line_id']: r for r in pack['rows']}
    clean, revised = by_id['w1_3_1:L0001'], by_id['w1_3_1:L0002']
    assert clean['masked_pixels'] == 0 and clean['boxes'] == [] and revised['masked_pixels'] > 1000 and len(revised['boxes']) == 1
    assert revised['text_before'] == 'x=3' and revised['kind'] == 'pair' and clean['kind'] == 'no_edit'
    raw = np.asarray(Image.open(out/revised['images']['raw']))
    erased = np.asarray(Image.open(out/revised['images']['erased']))
    assert raw.shape[:2] == (40, 400) and (erased == 255).mean() > (raw == 255).mean() and (erased[..., 0] > 200).all()
    assert (np.asarray(Image.open(out/clean['images']['erased'])) == np.asarray(Image.open(out/clean['images']['raw']))).all()
    assert (np.asarray(Image.open(out/clean['images']['blanked'])) == np.asarray(Image.open(out/clean['images']['raw']))).all()
    assert (np.asarray(Image.open(out/revised['images']['blanked']))[:, 90:310] == 255).all()
    manifest = json.loads((out/'run_manifest.json').read_text())
    assert (manifest['rows'], manifest['pairs'], manifest['controls'], manifest['with_mask']) == (2, 1, 1, 1)
    with pytest.raises(FileExistsError):
        L.main(['--pairs', str(pairs), '--catalogue', str(catalogue), '--pages', str(pages_dir), '--checkpoint', str(tmp_path/'checkpoint.pt'),
                '--freeze', str(freeze), '--output-dir', str(out)])
