"""models.transplant: the strike pixels, the selections, the token rows and the hook on a toy tower; eval.transplant_closeness's counts."""
import numpy as np
import pytest
from PIL import Image


def test_strike_pixels_any_and_brighter():
    from revrender.models.transplant import strike_pixels
    scan = np.full((4, 5), 255, np.uint8); scan[1, 1] = 100; scan[2, 2] = 100; scan[3, 3] = 230
    places = scan.copy(); places[1, 1] = 255; places[2, 2] = 140; places[3, 3] = 255; places[0, 0] = 254
    a, b = Image.fromarray(scan), Image.fromarray(places)
    assert set(zip(*np.nonzero(strike_pixels(a, b, 'any')))) == {(0, 0), (1, 1), (2, 2), (3, 3)}
    # brighter by more than 0.2 (51 of 255) on ink (grey below 200): (1, 1) only -- (2, 2) by 40, (3, 3) not ink, (0, 0) not ink
    assert set(zip(*np.nonzero(strike_pixels(a, b, 'brighter')))) == {(1, 1)}
    with pytest.raises(ValueError):
        strike_pixels(a, Image.fromarray(places[:, :4]), 'any')


def test_selections():
    from revrender.models.transplant import selection
    labels = np.array([0, 1, 2, 1, 0])
    assert selection(labels, 'struck').tolist() == [True, False, False, False, True]
    assert selection(labels, 'next').tolist() == [True, True, False, True, True]
    assert selection(labels, 'all').all() and not selection(labels, 'none').any()


def test_token_rows_offsets():
    torch = pytest.importorskip('torch')
    from revrender.models.transplant import token_rows
    grid = torch.tensor([[1, 2, 2], [1, 2, 4], [1, 2, 2]])
    rows, stats = token_rows(grid, [False, True, True], [None, np.array([1, 0, 0, 0, 0, 0, 0, 1], bool), np.array([0, 1, 1, 0], bool)])
    assert rows.tolist() == [4, 11, 13, 14]
    assert stats == [None, {'tokens': 8, 'replaced': 2}, {'tokens': 4, 'replaced': 2}]


def test_hook_on_a_toy_tower():
    torch = pytest.importorskip('torch')
    from revrender.models.transplant import Transplant, run_with_transplant
    torch.manual_seed(42)

    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.blocks = torch.nn.ModuleList(torch.nn.Linear(3, 3) for _ in range(3))

        def forward(self, x, grid):
            for b in self.blocks:
                x = torch.tanh(b(x))
            return x
    tower = Tower()
    scan, clean = torch.randn(6, 3), torch.randn(6, 3)
    grid = torch.tensor([[1, 1, 2], [1, 2, 2]])
    plant = Transplant(2, 3); plant.companion = (clean, grid); plant.flags = [False, True]
    plant.selections = [None, np.ones(4, bool)]
    got = run_with_transplant(tower, tower, plant, scan, grid)
    assert torch.equal(got[2:], tower(clean, grid)[2:]) and torch.equal(got[:2], tower(scan, grid)[:2])
    plant.selections = [None, np.zeros(4, bool)]
    assert torch.equal(run_with_transplant(tower, tower, plant, scan, grid), tower(scan, grid))
    plant.selections = [None, np.array([0, 1, 0, 0], bool)]                 # one row from the companion after two blocks
    got, whole = run_with_transplant(tower, tower, plant, scan, grid), tower(scan, grid)
    after2 = torch.tanh(tower.blocks[1](torch.tanh(tower.blocks[0](clean))))
    assert torch.allclose(got[3], torch.tanh(tower.blocks[2](after2[3:4]))[0]) and torch.equal(got[[0, 1, 2, 4, 5]], whole[[0, 1, 2, 4, 5]])
    assert plant.stats == [None, {'tokens': 4, 'replaced': 1}]


def test_nafnet_places_counts():
    from revrender.eval.transplant_closeness import overlap
    naf, human = np.array([0, 1, 2, 0, 2]), np.array([0, 0, 1, 2, 2])
    assert overlap(naf, human) == {'struck': {'both': 1, 'nafnet': 2, 'human': 2}, 'next': {'both': 2, 'nafnet': 3, 'human': 3},
                                   'arm': {'chosen': 3, 'chosen_on_human_struck': 2, 'human_struck': 2, 'human_struck_and_next': 3}}


def test_install_sets_the_readers_mix():
    """Reader.read hands the flags and the companion images to ``reader.mix``: both installs must set it (T2's smoke run failed on
    'student images without a student' when the mix's install left it None)."""
    pytest.importorskip('torch')
    import types
    from revrender.models import feature_mix, transplant
    for install in (lambda r: feature_mix.install(r, alpha=.5), lambda r: transplant.install(r, 9)):
        inner = types.SimpleNamespace(get_image_features=lambda *a, **k: None, visual=types.SimpleNamespace(blocks=[None]*27))
        reader = types.SimpleNamespace(model=types.SimpleNamespace(model=inner), students=None, mix=None)
        got = install(reader)
        assert reader.mix is got and got is not None
