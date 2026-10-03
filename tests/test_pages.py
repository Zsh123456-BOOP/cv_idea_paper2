"""Whole-question images: a page cut into pieces, the pieces read as strips, the instances pasted back."""
import json

import numpy as np
import pytest
from PIL import Image

from revrender import pages
from revrender.data import handwriting
from revrender.data.gold import unletterbox, letterbox


def page_with_bar():
    image = np.full((200, 600), 255, np.uint8)
    image[110:122, 150:330] = 20                         # a dark stroke inside the second line
    return Image.fromarray(image)


def line(box, source='ocr'):
    return {'box': box, 'source': source, 'ocr_type': 'text', 'ocr_text': 'x = 1'}


LINES = [line([6, 10, 594, 50]), line([6, 100, 594, 140])]


def dark_strips(model, strips, point, batch_size=8):
    """A stand-in for RevRender: every dark pixel of a strip is an instance (so the paste-back can be checked)."""
    return [[(np.asarray(s) < 128, .9)] if (np.asarray(s) < 128).any() else [] for s in strips]


def test_a_box_beyond_the_page_is_clamped_and_an_empty_one_is_dropped():
    assert pages.clamped([-5, 10, 1003, 20]) == [0., 10., 1000., 20.]
    assert pages.clamped([500, 10, 500, 20]) is None and pages.clamped([1000, 10, 1200, 20]) is None
    assert pages.pixel_box([10., 500., 990., 700.], (600, 200)) == [6, 100, 594, 140]


def test_the_letterbox_and_its_inverse_agree_on_a_block():
    block = np.zeros((40, 300), bool)
    block[10:30, 60:200] = True
    canvas = np.asarray(letterbox(Image.fromarray(block.astype(np.uint8)*255), 768, 96, 0, mask=True)) > 127
    back = unletterbox(canvas, 300, 40)
    assert back.shape == block.shape and (back & block).sum()/(back | block).sum() > .95


def test_instances_are_pasted_back_where_the_stroke_is(monkeypatch):
    monkeypatch.setattr(pages.infer, 'predict_lines', dark_strips)
    image = page_with_bar()
    instances = pages.read_page(None, {}, image, LINES)
    assert [i['line'] for i in instances] == [1]                       # the first line holds no stroke
    placed = pages.paste(instances[0], image.size)
    truth = np.zeros((200, 600), bool)
    truth[110:122, 150:330] = True
    assert placed.shape == truth.shape and (placed & truth).sum()/(placed | truth).sum() > .8
    assert instances[0]['box'] == [6, 100, 594, 140]


def test_the_same_mark_seen_from_two_pieces_is_one_instance_and_apart_marks_are_two():
    mask = np.ones((10, 40), bool)
    a = {'line': 0, 'score': .9, 'box': [0, 0, 40, 10], 'mask': mask}
    b = {'line': 1, 'score': .6, 'box': [2, 0, 42, 10], 'mask': mask.copy()}           # the same stroke from a neighbouring piece
    c = {'line': 2, 'score': .8, 'box': [100, 0, 140, 10], 'mask': mask.copy()}
    kept = pages.merge_duplicates([b, c, a])
    assert [i['line'] for i in kept] == [0, 2] and pages.overlap(a, b) > .8 and pages.overlap(a, c) == 0.


def test_a_page_is_cut_into_ocr_lines_ink_split_lines_and_leftover_handwriting():
    text = ('<|det|>text [10, 50, 990, 250]<|/det|>first line x = 1\n'
            '<|det|>equation [100, 400, 900, 700]<|/det|>\\[\na = 1 \\\\\nb = 2\n\\]\n'
            '<|det|>image [800, 800, 990, 990]<|/det|>')
    size = (600, 200)
    ink = np.zeros((200, 600), bool)
    ink[20:40, 60:500] = True                            # the first line
    ink[84:96, 70:300] = True                            # the first line of the multi-line region
    ink[110:124, 70:300] = True                          # its second line
    ink[170:185, 40:200] = True                          # handwriting the OCR never boxed
    lines = pages.page_lines(text, size, ink)
    sources = [(l['source'], l['box']) for l in lines]
    assert [s for s, _ in sources] == ['ocr', 'split', 'split', 'ink']
    assert sources[1][1][1] < 100 < sources[2][1][1] and sources[1][1][3] < sources[2][1][1]      # the split falls in the gap, not at the middle
    leftover = lines[3]
    assert leftover['box'][1] <= 170 and leftover['box'][3] >= 185 and leftover['box'][0] <= 40 and leftover['box'][2] >= 200


def test_the_page_files_are_written_and_read_back(tmp_path, monkeypatch):
    monkeypatch.setattr(pages.infer, 'predict_lines', dark_strips)
    image = page_with_bar()
    instances = pages.read_page(None, {}, image, LINES)
    pages.write_page(tmp_path, 'page_a', image.size, LINES, instances)
    record = json.loads((tmp_path/'page_a.json').read_text())
    assert record['size'] == [600, 200] and [l['source'] for l in record['lines']] == ['ocr', 'ocr'] and record['instances'][0]['line'] == 1
    packed = np.load(tmp_path/'page_a.npz')['mask_0']
    box = record['instances'][0]['box']
    width = box[2]-box[0]
    assert np.array_equal(np.unpackbits(packed, axis=-1)[:, :width].astype(bool), instances[0]['mask'])
    tinted = np.asarray(pages.overlay(image.convert('L'), instances))
    assert tinted.shape == (200, 600, 3) and (tinted[..., 0] != tinted[..., 1]).any()


def test_pages_are_found_by_id_and_stored_ocr_text_is_read(tmp_path):
    for name in ('a', 'b'):
        page_with_bar().save(tmp_path/f'{name}.jpg')
    assert sorted(pages.page_files([tmp_path])) == ['a', 'b'] and list(pages.page_files([tmp_path], ['b'])) == ['b']
    with pytest.raises(Exception, match='not found'):
        pages.page_files([tmp_path], ['c'])
    (tmp_path/'ocr.jsonl').write_text(json.dumps({'image_id': 'a', 'text': 'x'})+'\n'+json.dumps({'image_id': 'b', 'text': None})+'\n')
    assert pages.read_ocr_text([tmp_path/'ocr.jsonl']) == {'a': 'x', 'b': ''}
