"""A labmate's data never comes in: page lookup, the line catalogue and the folder scan (data.allowed)."""
import json

import pytest
from PIL import Image

from revrender import pages
from revrender.data import allowed as A
from revrender.data import lines
from revrender.util import RefusedError

OURS = sorted(A.allowed_writers())[0]
THEIRS = '00000000-0000-4000-8000-000000000000'         # a uuid outside the list


def test_the_list_holds_the_unlabeled_writers_only():
    assert len(A.allowed_writers()) == 499 and THEIRS not in A.allowed_writers()
    assert A.allowed(f'{OURS}_3_1') and A.allowed(f'{OURS}_3_1:L0002') and not A.allowed(f'{THEIRS}_s5_1')
    assert A.allowed('w1_2_1')                                                  # a test fixture's name is no student


def test_page_lookup_never_lists_or_returns_a_labmate_page(tmp_path):
    for writer in (OURS, THEIRS):
        Image.new('L', (4, 4)).save(tmp_path/f'{writer}_3_1.jpg')
    assert list(pages.page_files([tmp_path])) == [f'{OURS}_3_1']
    with pytest.raises(RefusedError, match="labmate"):
        pages.page_files([tmp_path], [f'{THEIRS}_3_1'])


def test_the_catalogue_drops_a_labmate_s_lines_and_the_scan_finds_them(tmp_path):
    path = tmp_path/'catalogue.jsonl'
    path.write_text(''.join(json.dumps({'image_id': f'{w}_3_1', 'representative_line_id': f'{w}_3_1:L0001'})+'\n' for w in (OURS, THEIRS)))
    assert [r['image_id'] for r in lines.read_catalogue(path)] == [f'{OURS}_3_1']
    found = A.scan([tmp_path])[str(tmp_path)]
    assert found == [f'{THEIRS}_3_1']
