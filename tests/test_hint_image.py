import pytest

from revrender.data import hint_image


def test_hint_text_rules():
    assert hint_image.hint_text('[读不出]') is None and hint_image.hint_text('  ') is None and hint_image.hint_text(None) is None
    assert hint_image.hint_text('$$x^2$$') == '$x^2$' and hint_image.hint_text('a$b') == 'ab' and hint_image.hint_text('3?4') == '34'


def test_wrong_answer_changes_the_middle_symbol_outside_commands():
    assert hint_image.wrong_answer('$\\frac{1}{2}$') == '$\\frac{1}{3}$'      # m = 2 (1, 2), counted from 0: index 1
    assert hint_image.wrong_answer('a+b=9') == 'a+c=9'                       # a + b = 9, m = 5: index 2
    assert hint_image.wrong_answer('z') == 'a' and hint_image.wrong_answer('Z9') == 'Z0'
    assert hint_image.wrong_answer('∠') is None and hint_image.wrong_answer('\\begin{cases}x\\end{cases}') is None


def test_render_fits_the_ink_box():
    pytest.importorskip('matplotlib')
    import numpy as np
    image = np.full((40, 100), 255, np.uint8); image[10:30, 20:80] = 0; inner = np.ones_like(image, bool)
    hint, plain = hint_image.render('$x=1$', image, inner)
    ys, xs = np.nonzero(hint < 200)
    assert plain is False and xs.min() >= 20 and xs.max() < 80 and ys.min() >= 10 and ys.max() < 30
    assert hint_image.render('[读不出]', image, inner)[1] is None


def test_windows_writes_a_review_set_per_fold(tmp_path):
    pytest.importorskip('matplotlib')
    import json
    import numpy as np
    fold = tmp_path/'sets'/'fold_0'; fold.mkdir(parents=True); (fold/'run_manifest.json').write_text('{}')
    image = np.full((40, 100), 255, np.uint8); image[10:30, 20:80] = 0; inner = np.ones_like(image)
    for name in ('training_native', 'validation_native', 'heldout_native'):
        (fold/f'{name}.json').write_text(json.dumps([{'id': 'r1', 'windowed': True}]))
        np.savez_compressed(fold/f'{name}.npz', image_0=image, inner_0=inner)
    (tmp_path/'a.json').write_text(json.dumps({'r1': '$x=1$'})); (tmp_path/'r.json').write_text(json.dumps({'r1': None}))
    hint_image.main(['windows', '--sets', str(tmp_path/'sets'), '--answers', str(tmp_path/'a.json'), '--readings', str(tmp_path/'r.json'),
                     '--output-dir', str(tmp_path/'out')])
    out = tmp_path/'out'/'fold_0'
    assert (out/'run_manifest.json').exists() and (tmp_path/'out'/'run_manifest.json').exists()
    training, validation = np.load(out/'training_native.npz'), np.load(out/'validation_native.npz')
    assert (training['hint_answer_0'] < 200).any() and (training['hint_reading_0'] == 255).all() and (validation['hint_0'] == 255).all()
