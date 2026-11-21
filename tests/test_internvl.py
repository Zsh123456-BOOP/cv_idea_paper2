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


def test_accumulated_is_the_batch_loss():
    torch = __import__('pytest').importorskip('torch')
    from revrender.train_clean_view import accumulated, pair_loss
    g = torch.Generator().manual_seed(42)
    scale = torch.nn.Parameter(torch.ones(1))
    data = [(torch.randn(n, 6, generator=g), torch.randn(n, 6, generator=g)) for n in (5, 9, 3, 7)]     # (student input, teacher) per region
    student_of = lambda piece: [torch.cat([s for s, _ in piece])*scale]
    teacher_of = lambda piece: [torch.cat([t for _, t in piece])]
    whole = pair_loss(student_of(data), teacher_of(data)); whole.backward(); grad = scale.grad.clone(); scale.grad = None
    total = accumulated(student_of, teacher_of, [data[:2], data[2:]])
    assert abs(total-float(whole)) < 1e-6 and torch.allclose(scale.grad, grad, atol=1e-6)
