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


def test_student_features_pick_image_by_image_and_adapter():
    torch = pytest.importorskip('torch')
    from contextlib import contextmanager
    from revrender.models.vlm_tokens import Students, student_features
    state = {'on': True, 'name': None}

    class View:
        @contextmanager
        def disable_adapter(self):
            state['on'] = False
            try: yield
            finally: state['on'] = True

        def set_adapter(self, name, inference_mode=False):
            assert inference_mode
            state['name'] = name
    sizes = [2, 3, 1, 2]

    def features(pixel_values, grid=None):      # the teacher's tokens are 0, adapter foldN's are N+1
        value = float(int(state['name'][-1])+1) if state['on'] else 0.
        embeds = tuple(torch.full((n, 2), value) for n in sizes)
        return embeds, [torch.full((sum(sizes), 2), value+10*d) for d in range(3)]
    students = Students(View(), ['fold1', 'fold3'])
    students.flags = [None, 'fold3', None, 'fold1']
    embeds, deep = student_features(features, students, False)(None)
    assert [float(e[0, 0]) for e in embeds] == [0., 4., 0., 2.]
    assert deep[1][:, 0].tolist() == [10., 10., 14., 14., 14., 10., 12., 12.]
    assert students.encoded == {'fold1': 1, 'fold3': 1}
    off, _ = student_features(features, students, True)(None)
    assert [float(e[0, 0]) for e in off] == [0., 0., 0., 0.]


def diff(exact_lo, ar, exact_hi=None):
    return {'exact': {'difference': exact_lo, 'interval': [exact_lo, exact_hi if exact_hi is not None else exact_lo+.05]}, 'ar': {'difference': ar, 'interval': [0, 0]}}


def test_grade_has_four_levels():
    from revrender.eval.privileged import grade
    assert grade(diff(.01, .03)) == 'holds'
    assert grade(diff(.01, .01)) == 'exact_up_ar_short'
    assert grade(diff(-.08, .03, -.01)) == 'worse'
    assert grade(diff(-.01, .05)) == 'cannot_tell'


def test_next_version_breaks_ties_for_no_nafnet_then_shared():
    from revrender.eval.privileged import next_version
    held = {n: 'holds' for n in ('VR-真+合', 'NV', 'NVR', 'MV')}
    assert next_version(held, {'VR-真+合': .550, 'NV': .551, 'NVR': .560, 'MV': .540}) == 'NVR'
    assert next_version(held, {'VR-真+合': .559, 'NV': .560, 'NVR': .560, 'MV': .540}) == 'VR-真+合'
    assert next_version({'NV': 'holds', 'NVR': 'holds'}, {'NV': .559, 'NVR': .560}) == 'NV'
    assert next_version({'NV': 'cannot_tell'}, {'NV': .9}) is None


def test_tint_marks_the_erased_ink_red_and_keeps_the_rest():
    from PIL import Image
    from revrender.data.struck_pack import tint
    scan = np.full((4, 5, 3), 255, np.uint8); scan[1, 1] = scan[2, 3] = 20; scan[0, 0] = (0, 0, 255)
    erased = scan.copy(); erased[1, 1] = 255                                                  # the restorer took one ink pixel
    got = np.asarray(tint(Image.fromarray(scan), Image.fromarray(erased)))
    assert got[1, 1].tolist() == [255, 20, 20] and got[2, 3].tolist() == [20, 20, 20] and got[0, 0].tolist() == [0, 0, 255]
    with pytest.raises(Exception):
        tint(Image.fromarray(scan), Image.fromarray(scan[:3]))

