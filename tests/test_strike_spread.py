"""eval.strike_spread: the classes of the cells under a strike mask, and their merge-group order."""
import numpy as np
import pytest


def test_classes_struck_next_far():
    from revrender.eval.strike_spread import classes
    mask = np.zeros((40, 40), bool); mask[12, 12] = True                  # one source pixel, cells of 10 x 10 source pixels
    got = classes(mask, 4, 4)
    assert got[1, 1] == 0 and (got == 0).sum() == 1
    assert all(got[i, j] == 1 for i in range(3) for j in range(3) if (i, j) != (1, 1))
    assert got[3, 3] == 2 and got[0, 3] == 2


def test_a_pixel_strikes_the_cell_its_source_rectangle_holds():
    from revrender.eval.strike_spread import classes
    mask = np.zeros((10, 10), bool); mask[0, 4] = True
    got = classes(mask, 1, 3)                                              # 3 cells over 10 columns: the middle one from [3.33, 6.67)
    assert got[0, 1] == 0


def test_merge_ordered_matches_the_tokens_order():
    torch = pytest.importorskip('torch')
    from revrender.eval.strike_spread import merge_ordered
    from revrender.models.maa import raster
    grid = np.arange(4*6).reshape(4, 6)
    tokens = torch.as_tensor(merge_ordered(grid, 2)).reshape(-1, 1)
    assert np.array_equal(raster(tokens, 1, 4, 6, 2).reshape(4, 6).numpy(), grid)
