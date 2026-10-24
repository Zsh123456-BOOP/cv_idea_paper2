import pytest

torch = pytest.importorskip('torch')

from revrender.models.separation import IGNORE, RevSep, compose, masked_bce, separation_loss


class Body(torch.nn.Module):
    """NAFNet's interface in miniature: an intro, an ending on 32 features, the input added back."""
    def __init__(self):
        super().__init__(); self.intro = torch.nn.Conv2d(3, 32, 3, padding=1); self.ending = torch.nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):
        return self.ending(self.intro(x)) + x


def test_revsep_reads_grey_and_hint_and_its_heads_see_the_endings_input():
    torch.manual_seed(42); net = RevSep(Body()); grey = torch.rand(2, 1, 16, 24); hint = (torch.rand(2, 1, 16, 24) > .5).float()
    restored, erase, cross = net(grey, hint)
    assert restored.shape == erase.shape == cross.shape == (2, 1, 16, 24) and net.features is None   # nothing held between calls
    stacked = torch.cat([grey, hint, torch.zeros_like(grey)], 1)
    assert torch.allclose(restored, net.body(stacked)[:, :1])
    logits = net.heads(net.body.intro(stacked))
    assert torch.allclose(erase, logits[:, :1]) and torch.allclose(cross, logits[:, 1:])


def test_the_composition_redraws_crossings_whitens_the_erased_and_keeps_the_rest():
    grey = torch.full((1, 1, 1, 3), .3); restored = torch.full_like(grey, .6)
    erase = torch.tensor([[[[5., 5., -5.]]]]); cross = torch.tensor([[[[5., -5., -5.]]]])
    assert compose(grey, restored, erase, cross, hard=True).flatten().tolist() == pytest.approx([.6, 1., .3])
    assert compose(grey, restored, erase*10, cross*10).flatten().tolist() == pytest.approx([.6, 1., .3], abs=1e-6)


def test_ignored_labels_carry_no_loss():
    labels = torch.tensor([1, 0, IGNORE], dtype=torch.uint8); a = torch.tensor([2., -1., 0.]); b = torch.tensor([2., -1., 9.])
    assert masked_bce(a, labels) == masked_bce(b, labels)
    assert masked_bce(a, torch.full((3,), IGNORE, dtype=torch.uint8)) == 0


def test_the_loss_trains_the_heads_and_without_heads_only_h1():
    torch.manual_seed(42); net = RevSep(Body()); grey = torch.rand(2, 1, 8, 8); target = torch.rand(2, 1, 8, 8)
    hint = torch.zeros_like(grey); erase = (torch.rand(2, 1, 8, 8) > .5).to(torch.uint8); cross = torch.full_like(erase, IGNORE)
    loss, terms = separation_loss(net, grey, target, hint, erase, cross)
    assert torch.isfinite(loss) and set(terms) == {'restored', 'composed', 'erase', 'cross'} and terms['cross'] == 0
    loss.backward(); assert net.heads.weight.grad.abs().sum() > 0
    net.zero_grad(set_to_none=True)
    loss, terms = separation_loss(net, grey, target, hint, erase, cross, heads=False); loss.backward()
    assert set(terms) == {'restored'} and net.heads.weight.grad is None
