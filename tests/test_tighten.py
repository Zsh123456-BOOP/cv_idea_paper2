"""Masks cut to the pixels that are not white paper: the ink and its soft edge stay, the paper around and between strokes goes."""
import json

import numpy as np
import pytest
from PIL import Image

from revrender import pages, tighten


def page_with_a_stroke():
    grey = np.full((60, 200), 255, np.uint8)
    grey[28:32, 40:160] = 20                           # a strike-through line, 4 px thick
    grey[26:28, 40:160] = 225                          # its soft edge, 2 px on each side
    grey[32:34, 40:160] = 225
    grey[24:26, 40:160] = 250                          # the JPEG halo beyond it: paper to the eye
    grey[10, 100] = 225                                # a speck of noise far from any ink
    grey[29, 161] = 225                                # a faint speck two pixels from the stroke's end, a white pixel between: not edge
    grey[34, 100] = 225                                # faint and joined to the edge, but a third step from the ink
    return grey


def test_the_ink_and_its_soft_edge_are_non_white_and_the_paper_and_its_stray_specks_are_not():
    white = tighten.non_white(page_with_a_stroke())
    assert white[28:32, 40:160].all()
    assert white[26:28, 40:160].all() and white[32:34, 40:160].all()
    assert not white[24:26, 40:160].any()                              # 250 is paper
    assert not white[10, 100] and not white[:20].any() and not white[40:].any()
    assert not white[29, 161] and not white[34, 100]                   # faint pixels that do not touch the stroke, or are too far along it
    assert white.sum() == 8*120


def test_a_mask_loses_the_paper_around_the_stroke_and_never_gains_a_pixel():
    white = tighten.non_white(page_with_a_stroke())
    mask = np.ones((40, 140), bool)                                    # the whole region over the stroke
    cut = tighten.tighten_instance({'line': 3, 'score': .9, 'box': [30, 10, 170, 50], 'mask': mask}, white)
    assert cut['box'] == [40, 26, 160, 34] and cut['mask'].shape == (8, 120) and cut['mask'].all()
    assert (cut['line'], cut['score']) == (3, .9)
    ring = np.zeros((40, 140), bool)                                   # a mask that is only paper
    ring[:5] = True
    assert tighten.tighten_instance({'line': 0, 'score': .5, 'box': [30, 10, 170, 50], 'mask': ring}, white) is None
    thin = np.zeros((40, 140), bool)
    thin[18:21, 20:60] = True                                          # a mask that is part ink (rows 28-30), part soft edge (27)
    cut = tighten.tighten_instance({'line': 0, 'score': .5, 'box': [30, 10, 170, 50], 'mask': thin}, white)
    placed = np.zeros((60, 200), bool)
    placed[cut['box'][1]:cut['box'][3], cut['box'][0]:cut['box'][2]] = cut['mask']
    old = np.zeros((60, 200), bool)
    old[10:50, 30:170] = thin
    assert (placed & ~old).sum() == 0 and placed.sum() == old.sum()    # every pixel of it was ink


def test_a_page_keeps_the_instances_with_ink_and_each_names_its_source():
    grey = page_with_a_stroke()
    over = {'line': 0, 'score': .8, 'box': [30, 10, 170, 50], 'mask': np.ones((40, 140), bool)}
    paper = {'line': 1, 'score': .7, 'box': [0, 0, 30, 8], 'mask': np.ones((8, 30), bool)}
    again = {'line': 2, 'score': .6, 'box': [60, 20, 100, 40], 'mask': np.ones((20, 40), bool)}
    out = tighten.tighten_page([over, paper, again], grey)
    assert [i['from_instance'] for i in out] == [0, 2] and [i['line'] for i in out] == [0, 2]
    assert tighten.tighten_page(out, grey)[0]['mask'].sum() == out[0]['mask'].sum()          # cutting twice is cutting once


def write_source(root, with_manifest=True):
    """Two pages: a dark block with a region mask over it (and a mask on the paper only), and a page with nothing."""
    out = root/'pages_out'
    out.mkdir()
    grey = np.full((120, 300), 255, np.uint8)
    grey[50:70, 80:200] = 15
    lines = [{'box': [10, 30, 290, 90], 'source': 'ocr', 'ocr_type': 'text', 'ocr_text': 'x'}]
    region = np.ones((60, 280), bool)
    elsewhere = np.zeros((60, 280), bool)
    elsewhere[:8, :40] = True
    pages.write_page(out, 'p1', (300, 120), lines, [{'line': 0, 'score': .9, 'box': [10, 30, 290, 90], 'mask': region},
                                                    {'line': 0, 'score': .4, 'box': [10, 30, 290, 90], 'mask': elsewhere}])
    pages.write_page(out, 'p2', (300, 120), lines, [])
    if with_manifest:
        (out/'run_manifest.json').write_text('{}')
    images = root/'images'
    images.mkdir()
    Image.fromarray(grey).save(images/'p1.jpg', quality=98, subsampling=0)
    Image.fromarray(np.full((120, 300), 255, np.uint8)).save(images/'p2.jpg')
    return out, images


def test_the_tool_writes_the_same_layout_with_tight_boxes_and_counts_what_it_dropped(tmp_path):
    out, images = write_source(tmp_path)
    new = tmp_path/'tight'
    assert tighten.main(['--pages-output', str(out), '--images', str(images), '--output-dir', str(new)]) == 0
    record, instances = pages.read_output(new, 'p1')
    assert record['size'] == [300, 120] and record['lines'][0]['source'] == 'ocr' and len(instances) == 1
    assert instances[0]['from_instance'] == 0 and instances[0]['score'] == .9
    left, top, right, bottom = instances[0]['box']
    assert abs(left-80) <= 3 and abs(top-50) <= 3 and abs(right-200) <= 3 and abs(bottom-70) <= 3
    assert instances[0]['mask'].shape == (bottom-top, right-left) and instances[0]['mask'].mean() > .9
    assert pages.read_output(new, 'p2')[1] == []
    manifest = json.loads((new/'run_manifest.json').read_text())
    assert (manifest['pages'], manifest['instances_in'], manifest['instances_out'], manifest['dropped']) == (2, 2, 1, 1)
    assert manifest['pixels_out'] < manifest['pixels_in'] and 0 < manifest['removed_share_of_pixels'] < 1
    assert manifest['ink_below'] == 200 and manifest['edge_below'] == 240 and manifest['edge_px'] == 2 and manifest['seed'] == 42


def test_the_tool_is_create_only_and_refuses_an_image_that_is_not_the_page(tmp_path):
    out, images = write_source(tmp_path)
    new = tmp_path/'tight'
    args = ['--pages-output', str(out), '--images', str(images), '--output-dir', str(new)]
    tighten.main(args)
    with pytest.raises(FileExistsError):
        tighten.main(args)
    Image.fromarray(np.full((100, 300), 255, np.uint8)).save(images/'p1.jpg')
    with pytest.raises(Exception, match='not the page'):
        tighten.main(['--pages-output', str(out), '--images', str(images), '--output-dir', str(tmp_path/'other')])


def test_only_the_named_pages_are_cut_and_a_missing_one_is_refused(tmp_path):
    out, images = write_source(tmp_path)
    (tmp_path/'ids.txt').write_text('p2\n')
    tighten.main(['--pages-output', str(out), '--images', str(images), '--ids', str(tmp_path/'ids.txt'), '--output-dir', str(tmp_path/'only')])
    assert sorted(p.stem for p in (tmp_path/'only').glob('*.json') if p.name != 'run_manifest.json') == ['p2']
    (tmp_path/'ids.txt').write_text('p9\n')
    with pytest.raises(Exception, match='not in'):
        tighten.main(['--pages-output', str(out), '--images', str(images), '--ids', str(tmp_path/'ids.txt'), '--output-dir', str(tmp_path/'none')])
