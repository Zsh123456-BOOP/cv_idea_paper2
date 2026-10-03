"""The HTML viewer of cleaned pages: a card per chosen page with the ink the cleaning took off, outside the frame and inside it."""
import json
import re

import numpy as np
import pytest
from PIL import Image

from revrender.data import clean_pages
from revrender.tools import clean_view

W, H = 600, 200


def scan(seed):
    """A page of a question: the frame (rows 22 and 170) and its neighbours' lines (10 and 190), writing, a mark outside, dust inside."""
    grey = np.full((H, W), 255, np.uint8)
    for y in (10, 22, 170, 190):
        for x in range(20, 580, 16):
            grey[y-1:y+2, x:x+10] = 60
    for x in (24, 576):
        for y in range(34, 158, 16):
            grey[y:y+10, x-1:x+2] = 60
    grey[60:76, 100:300] = 20
    grey[176:199, 0:18] = 0
    grey[110:112, 200+seed:202+seed] = 30
    return grey


def build(tmp_path, count=10, per_question=4):
    scans = tmp_path/'scans'
    scans.mkdir()
    for i in range(count):
        Image.fromarray(scan(i)).save(scans/f'w{i}_3_1.jpg', quality=98, subsampling=0)
        Image.fromarray(scan(i)).save(scans/f'w{i}_s5_1.jpg', quality=98, subsampling=0)
    clean = tmp_path/'clean'
    clean_pages.main(['--pages', str(scans), '--output-dir', str(clean)])
    site = tmp_path/'site'
    assert clean_view.main(['--before', str(scans), '--after', str(clean), '--per-question', str(per_question), '--output-dir', str(site)]) == 0
    return site


def embedded(path):
    return json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', path.read_text(encoding='utf-8'), re.S).group(1)
                      .replace('<\\/', '</'))


def test_each_question_shows_its_share_of_pages_with_the_ink_taken_off_inside_and_outside_the_frame(tmp_path):
    site = build(tmp_path)
    cards = embedded(site/'index.html')['cards']
    assert len(cards) == 8 and sorted({c['q'] for c in cards}) == [3, 5] and sum(c['supp'] for c in cards) == 4
    for c in cards:
        assert c['outside'] > 400 and 0 < c['inside'] <= 8                   # the neighbours' lines and the mark; the specks
        assert (site/'assets'/f'{c["id"]}_before.jpg').is_file() and (site/'assets'/f'{c["id"]}_after.png').is_file()
        left, top, right, bottom = c['box']
        assert top > 10 and bottom < 190 and left <= 24 and right >= 576


def test_the_site_is_create_only_and_the_choice_is_fixed(tmp_path):
    site = build(tmp_path)
    again = tmp_path/'site2'
    clean_view.main(['--before', str(tmp_path/'scans'), '--after', str(tmp_path/'clean'), '--per-question', '4', '--output-dir', str(again)])
    assert [c['id'] for c in embedded(site/'index.html')['cards']] == [c['id'] for c in embedded(again/'index.html')['cards']]
    with pytest.raises(FileExistsError):
        clean_view.main(['--before', str(tmp_path/'scans'), '--after', str(tmp_path/'clean'), '--output-dir', str(site)])


def test_ink_taken_off_is_told_apart_by_where_it_was():
    before = np.full((50, 100), 255, np.uint8)
    after = before.copy()
    before[2:4, 10:30] = 0                      # outside the box (rows 10..40)
    before[20:22, 50:52] = 0                    # inside it
    before[30:32, 60:80] = 0                    # inside it, and kept
    after[30:32, 60:80] = 0
    assert clean_view.removed_ink(before, after, (5, 10, 95, 40)) == (40, 4)
