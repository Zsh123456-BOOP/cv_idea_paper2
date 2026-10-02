import json

import numpy as np
import pytest

from revrender.data import gold, lines


def test_a_thin_stroke_is_widened_by_one_pixel_and_a_blot_is_left_alone():
    mask = np.zeros((96, 768), np.uint8)
    mask[40, 100:200] = 255                       # a one-pixel line: pen width < 5
    mask[10:40, 400:440] = 255                    # a filled blot: wide
    out = gold.correct_mask(mask)
    assert out[39:42, 150].all() and not out[37, 150]                    # 1 -> 3 rows
    assert out[10:40, 400:440].all() and not out[9, 420] and not out[10, 399]
    assert gold.correct_mask(np.zeros((96, 768), np.uint8)).sum() == 0


def test_rasterise_draws_boxes_filled_and_polylines_stroked():
    raster = gold.rasterise([{'shape': 'box', 'x': 10, 'y': 5, 'w': 20, 'h': 10},
                             {'shape': 'line', 'path': [[50, 20], [90, 20]], 'stroke_width': 3}], 120, 40)
    a = np.asarray(raster) > 0
    assert a[5:15, 10:30].all() and a[19:22, 55:85].all() and not a[30:, :].any()
    segment = np.asarray(gold.rasterise([{'shape': 'line', 'x1': 0, 'y1': 5, 'x2': 50, 'y2': 5}], 60, 20)) > 0
    assert segment[4:7, 10:40].all()                                        # the default stroke is 3 px


def test_the_strip_is_a_nearest_letterbox_centred_on_the_canvas():
    from PIL import Image
    raster = Image.new('L', (384, 24), 0)
    raster.paste(255, (0, 0, 384, 24))
    strip = gold.strip_mask(raster)
    assert strip.shape == (96, 768) and strip.dtype == np.uint8
    assert strip[24:72, :].all() and not strip[:24].any() and not strip[72:].any()      # scale 2, centred vertically
    assert set(np.unique(strip)) <= {0, 255}                                              # NEAREST never makes grey


def test_gold_mask_composes_the_three_steps():
    marks = [{'shape': 'box', 'x': 20, 'y': 10, 'w': 60, 'h': 30}]
    mask = gold.gold_mask(marks, 192, 48)
    assert mask.dtype == bool and mask.shape == (96, 768) and mask.any()
    assert mask[:, 192:576].sum() == mask.sum()            # a 192x48 crop is scaled 2x and centred horizontally


def entry(line, image, box, aliases=()):
    return {'representative_line_id': line, 'alias_line_ids': list(aliases), 'image_id': image, 'box_1000': box,
            'group_uuid': 'w', 'question_no': '3', 'ocr_text': ''}


def test_the_line_catalogue_indexes_by_id_alias_and_image(tmp_path):
    entries = [entry('img_1_1:L0002', 'img_1_1', [0, 500, 1000, 700], ['old:L9']),
               entry('img_1_1:L0001', 'img_1_1', [0, 100, 1000, 300]), entry('other_2_1:L0001', 'other_2_1', [0, 0, 500, 500])]
    path = tmp_path/'c.jsonl'
    path.write_text('\n'.join(json.dumps(e) for e in entries)+'\n')
    read = lines.read_catalogue(path)
    index = lines.index_by_line(read)
    assert index['old:L9'] is index['img_1_1:L0002']
    grouped = lines.lines_by_image(read)
    assert [e['representative_line_id'] for e in grouped['img_1_1']] == ['img_1_1:L0001', 'img_1_1:L0002']
    with pytest.raises(Exception, match='changed'):
        lines.read_catalogue(path, expected_sha='0'*64)


def test_a_thousandths_box_maps_to_pixels_and_is_validated():
    assert lines.crop_box([100, 200, 600, 700], 1000, 500) == (100, 100, 600, 350)
    assert lines.crop_box([0.5, 0.5, 1, 1], 10, 10) == (0, 0, 1, 1)         # at least one pixel
    for bad in ([5, 5, 4, 9], [0, 0, 1001, 5], [1, 2, 3]):
        with pytest.raises(Exception):
            lines.crop_box(bad, 100, 100)
    from PIL import Image
    assert lines.crop_line(Image.new('L', (200, 100)), [0, 0, 500, 500]).size == (100, 50)


def test_the_default_catalogue_is_the_one_paths_toml_names(tmp_path, monkeypatch):
    catalogue = tmp_path/'c.jsonl'
    catalogue.write_text(json.dumps(entry('i_1_1:L0001', 'i_1_1', [0, 0, 1000, 500]))+'\n')
    paths = tmp_path/'paths.toml'
    paths.write_text(f'[raw_data]\nline_catalogue = "{catalogue}"\n')
    monkeypatch.setenv('REVRENDER_PATHS', str(paths))
    assert [e['image_id'] for e in lines.default_catalogue()] == ['i_1_1']
