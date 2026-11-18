import json
from types import SimpleNamespace

from revrender.data import struck_pack


def test_method_split_fills_validation_by_regions_then_readable(tmp_path):
    rows = [{'id': f'{w}#{k}', 'writer': w, 'readable': k % 3 != 0} for w in (f'w{i:02d}' for i in range(30)) for k in range(4)]
    (tmp_path/'pack').mkdir(); (tmp_path/'pack'/'pack.json').write_text(json.dumps({'rows': rows}))
    struck_pack.method_split(SimpleNamespace(pack=tmp_path/'pack', output=tmp_path/'split.json'))
    split = json.loads((tmp_path/'split.json').read_text())
    order = sorted({r['writer'] for r in rows}, key=struck_pack.split_order)
    val = set(split['validation']); held = [r for r in rows if r['writer'] in val]
    assert set(order[:len(val)]) == val and not val & set(split['training'])
    assert len(held) >= struck_pack.SPLIT_REGIONS and sum(r['readable'] for r in held) >= struck_pack.SPLIT_READABLE
    fewer = [r for r in rows if r['writer'] in order[:len(val)-1]]
    assert len(fewer) < struck_pack.SPLIT_REGIONS or sum(r['readable'] for r in fewer) < struck_pack.SPLIT_READABLE


def test_window_scores_count_inside_the_polygon_only():
    import numpy as np
    import pytest
    pytest.importorskip('skimage')
    image = np.ones((10, 10)); image[2:4, 1:9] = 0; image[6:8, 1:9] = 0          # a strike (rows 2-3) and writing (rows 6-7)
    mask = np.zeros((10, 10), np.uint8); mask[2:4, 1:9] = 1
    inner = np.ones((10, 10), np.uint8); inner[:, 8:] = 0
    restored = image.copy(); restored[2:4, 1:8] = 1; restored[6:8, 8:] = 1      # the strike erased; writing outside the polygon erased
    scores = struck_pack.window_scores([(image, restored, mask, inner)])
    assert scores['removal']['recall'] == 1 and scores['removal']['fp'] == 0
    assert scores['psnr'] > 40 and scores['regions'] == 1


def test_raw_context_words_differ_from_context_revision_only_where_agreed():
    from revrender import recognize_lines as r
    words = r.prompt('raw_context_revision', {'boxes': []})
    assert r.PAIR not in words and r.CONTEXT_RAW in words
    assert r.CONTEXT_RAW == r.CONTEXT.replace('第三张', '第二张').replace('（框里是去掉划线后的样子）', '（框里是原图）')
    assert words == r.prompt('context_revision', {'boxes': []}).replace(r.PAIR+'\n', '').replace(r.CONTEXT, r.CONTEXT_RAW)
