from PIL import Image

from revrender.models.internvl import closest_grid, tiles
from revrender.models.vlm_tokens import looped


def test_tiles_follow_the_card():
    assert closest_grid(10., 1000, 100) == (10, 1)
    cut = tiles(Image.new('RGB', (1000, 100)))
    assert len(cut) == 11 and all(t.size == (448, 448) for t in cut)          # ten tiles and the thumbnail
    assert len(tiles(Image.new('RGB', (300, 300)))) == 1                      # one tile, no thumbnail
    assert closest_grid(1., 2000, 2000) == (3, 3)                             # a tie on shape goes to the larger grid a big image fills


def test_looped_unchanged():
    assert looped([1, 2, 3]*4, 4) == 9 and looped([1, 2, 3, 4], 4) == 0
