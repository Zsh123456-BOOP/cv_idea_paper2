"""The separator inside the reader's training: its hint and the cut of the hint into the reader's patches."""
import pytest

torch = pytest.importorskip('torch')

from revrender.models import hint_separator


def test_a_hint_only_takes_ink_away_and_keeps_the_crop():
    darker = lambda x: x*.5                                                      # a network that would darken everything
    lighter = lambda x: torch.ones_like(x)                                       # one that erases everything
    scan = torch.rand(40, 70)
    assert torch.equal(hint_separator.hint(darker, scan), scan)
    assert torch.equal(hint_separator.hint(lighter, scan), torch.ones(40, 70))
    assert hint_separator.hint(lighter, torch.rand(300, 20)) is None             # larger than the canvas: not made here
    seen = []
    hint_separator.hint(lambda x: (seen.append(x), x)[1], scan)
    canvas = seen[0]; top, left = (256-40)//2, (256-70)//2
    assert canvas.shape == (1, 3, 256, 256) and torch.equal(canvas[0, 0, top:top+40, left:left+70], scan) and float(canvas[0, 0, 0, 0]) == 1.


def test_patches_are_cut_in_the_order_of_the_merge_blocks():
    rows, cols, patch = 4, 4, 2
    image = torch.arange(rows*patch*cols*patch, dtype=torch.float32).reshape(rows*patch, cols*patch)/100       # already the grid's size: resizing leaves it
    got = hint_separator.patches_of(image, rows, cols, patch, 2, 2, [0., 0., 0.], [1., 1., 1.])
    assert got.shape == (16, 3*2*patch*patch)
    block = lambda r, c: image[r*patch:(r+1)*patch, c*patch:(c+1)*patch].reshape(-1)
    order = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
    for k, (r, c) in enumerate(order):
        assert torch.allclose(got[k].reshape(3, 2, -1), block(r, c).expand(3, 2, -1), atol=1e-5)
    image.requires_grad_(True)
    hint_separator.patches_of(image, rows, cols, patch, 2, 2, [.5]*3, [.5]*3).sum().backward()
    assert image.grad is not None and float(image.grad.abs().sum()) > 0          # the reading's loss reaches the hint's pixels
