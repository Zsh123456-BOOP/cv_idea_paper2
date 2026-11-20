import json
import types

import numpy as np
import pytest

from revrender.data import synth_views
from revrender.data.restoration_pairs import canvas_tiles


@pytest.mark.parametrize('shape', [(40, 90), (256, 256), (300, 700), (31, 610)])
def test_untiled_undoes_canvas_tiles(shape):
    crop = np.random.default_rng(42).integers(0, 255, shape).astype(np.uint8)
    assert np.array_equal(synth_views.untiled(canvas_tiles(crop, 255), *shape), crop)


def test_untiled_refuses_a_wrong_count():
    with pytest.raises(Exception):
        synth_views.untiled(canvas_tiles(np.zeros((40, 300), np.uint8), 255)[:1], 40, 300)


def test_answered_drops_refusals_and_empties(tmp_path):
    pack = tmp_path/'pack'; pack.mkdir(); read = tmp_path/'read'; read.mkdir()
    rows = [{'id': f'S{k}', 'line_id': f'S{k}', 'source': 'H', 'images': {}, 'text': '', 'readable': True} for k in range(4)]
    (pack/'pack.json').write_text(json.dumps({'schema': 'revrender-line-pack', 'rows': rows}))
    texts = ['$x$', '[读不出]', '', '12']
    (read/'predictions.jsonl').write_text(''.join(json.dumps({'condition': 'c', 'id': f'S{k}', 'text': t})+'\n' for k, t in enumerate(texts)))
    synth_views.answered(types.SimpleNamespace(pack=pack, readings=read, take=4, output_dir=tmp_path/'out'))
    kept = json.loads((tmp_path/'out'/'pack.json').read_text())['rows']
    assert [r['text'] for r in kept] == ['$x$', '12'] and all(r['split'] == 'train' and r['readable'] for r in kept)
    manifest = json.loads((tmp_path/'out'/'run_manifest.json').read_text())
    assert manifest['left_out'] == {'refused': 1, 'empty': 1}


def test_validation_writers_hold_a_tenth():
    pytest.importorskip('torch')
    from revrender.train_clean_view import validation_writers
    regions = [(f'r{k}', f'w{k % 20}', []) for k in range(200)]
    held = validation_writers(regions)
    count = sum(w in held for _, w, _ in regions)
    assert 20 <= count < 30 and held == validation_writers(list(reversed(regions)))


def test_student_features_pick_image_by_image():
    torch = pytest.importorskip('torch')
    from contextlib import contextmanager
    from revrender.models.vlm_tokens import Reader
    state = {'on': True}

    class View:
        @contextmanager
        def disable_adapter(self):
            state['on'] = False
            try: yield
            finally: state['on'] = True
    sizes = [2, 3, 1]

    def features(pixel_values, grid=None):      # the student's tokens are 1, the teacher's 0
        value = float(state['on'])
        embeds = tuple(torch.full((n, 2), value) for n in sizes)
        return embeds, [torch.full((sum(sizes), 2), value+10*d) for d in range(3)]
    reader = Reader.__new__(Reader); reader.torch, reader.view, reader.flags = torch, View(), [False, True, False]
    embeds, deep = reader.student_features(features, False)(None)
    assert [float(e[0, 0]) for e in embeds] == [0., 1., 0.]
    assert deep[1][:, 0].tolist() == [10., 10., 11., 11., 11., 10.]
    off, _ = reader.student_features(features, True)(None)
    assert [float(e[0, 0]) for e in off] == [0., 0., 0.]
