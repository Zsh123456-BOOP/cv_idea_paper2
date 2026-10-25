import pytest

torch = pytest.importorskip('torch')

from revrender.models.separation import IGNORE, RADIUS, RevSep, compose, masked_bce, render, run, separation_loss


class Body(torch.nn.Module):
    """NAFNet's interface in miniature: an intro, an ending on 32 features, the input added back."""
    def __init__(self):
        super().__init__(); self.intro = torch.nn.Conv2d(3, 32, 3, padding=1); self.ending = torch.nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):
        return self.ending(self.intro(x)) + x


def test_revsep_reads_grey_and_hint_and_its_heads_see_the_endings_input():
    torch.manual_seed(42); net = RevSep(Body()); grey = torch.rand(2, 1, 16, 24); hint = (torch.rand(2, 1, 16, 24) > .5).float()
    restored, erase, cross, stroke, writing = net(grey, hint)
    assert restored.shape == erase.shape == cross.shape == (2, 1, 16, 24) and stroke is None and writing is None and net.features is None   # nothing held
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


def test_a_reading_term_joins_the_loss_when_it_gives_one():
    torch.manual_seed(42); net = RevSep(Body()); grey = torch.rand(1, 1, 8, 8); erase = torch.zeros(1, 1, 8, 8, dtype=torch.uint8)
    seen = []
    def reading(out): seen.append(out.shape); return out.mean()
    _, terms = separation_loss(net, grey, grey, grey*0, erase, erase, reading=reading)
    assert seen == [(1, 1, 8, 8)] and 'reading' in terms
    _, terms = separation_loss(net, grey, grey, grey*0, erase, erase, reading=lambda out: None)
    assert 'reading' not in terms


def test_the_drawn_mask_covers_the_centreline_within_its_half_width():
    centre = torch.zeros(1, 1, 21, 21); centre[0, 0, 10, 4:17] = 1; half = torch.full_like(centre, 2.)
    drawn = render(centre, half)
    assert drawn[0, 0, 10, 10] > .99 and drawn[0, 0, 12, 10] > .5 and drawn[0, 0, 15, 10] < .01 and drawn[0, 0, 10, 0] < .01
    assert render(torch.zeros_like(centre), half).abs().max() < 1e-6


def test_the_stroke_version_draws_its_mask_and_feeds_it_to_a_second_pass():
    torch.manual_seed(42); net = RevSep(Body(), stroke=True); grey = torch.rand(1, 1, 16, 16); hint = torch.zeros_like(grey)
    restored, erase, cross, stroke, _ = net(grey, hint)
    assert stroke[0].shape == (1, 1, 16, 16) and 0 < stroke[1].min() and stroke[1].max() < RADIUS
    calls = []
    original = net.forward
    net.forward = lambda g, h, extra=None: calls.append((h is hint, extra is None)) or original(g, h, extra)
    for mode, expected in (('detector', [(True, True)]), ('stroke', [(True, True), (False, True)]), ('both', [(True, True), (True, False)])):
        calls.clear(); out = run(net, grey, hint, mode); assert calls == expected and out[4].shape == grey.shape


def test_the_stroke_terms_join_the_loss_and_reach_head_a():
    torch.manual_seed(42); net = RevSep(Body(), stroke=True); grey = torch.rand(1, 1, 16, 16); zero = torch.zeros(1, 1, 16, 16, dtype=torch.uint8)
    mask = zero.clone(); mask[..., 7:9, 2:14] = 1; centre = zero.clone(); centre[..., 8, 2:14] = 1; half = centre.float()*1.5
    loss, terms = separation_loss(net, grey, grey, grey*0, zero, zero, strokes=(mask, centre, half), hint_from='both')
    assert {'centre', 'width', 'stroke'} <= set(terms); loss.backward(); assert net.heads.weight.grad[2:].abs().sum() > 0


def test_the_writing_gate_keeps_what_it_calls_writing_and_never_redraws():
    from revrender.models.separation import gated
    grey = torch.full((1, 1, 1, 3), .3); erase = torch.tensor([[[[5., 5., -5.]]]]); writing = torch.tensor([[[[-5., 5., -5.]]]])
    assert gated(grey, erase, writing).flatten().tolist() == pytest.approx([1., .3, .3])
    restored = torch.full_like(grey, .9); cross = torch.full_like(grey, -9.)
    assert compose(grey, restored, erase*10, cross, writing=writing*10).flatten().tolist() == pytest.approx([1., .3, .3], abs=1e-4)


def test_the_writing_term_weighs_the_struck_writing_more():
    torch.manual_seed(42); net = RevSep(Body(), gate=True); grey = torch.rand(1, 1, 8, 8); zero = torch.zeros(1, 1, 8, 8, dtype=torch.uint8)
    labels = zero.clone(); labels[..., 2:4, :] = 1; near = torch.zeros(1, 1, 8, 8, dtype=torch.bool); near[..., 2:4, :4] = True
    _, light = separation_loss(net, grey, grey, grey*0, zero, zero, writing=(labels, near), gate_weights=(1., 1.))
    _, heavy = separation_loss(net, grey, grey, grey*0, zero, zero, writing=(labels, near), gate_weights=(1., 8.))
    assert light['writing'] != heavy['writing'] and net(grey, grey)[4].shape == grey.shape


def test_pixel_and_sample_weights_reach_the_psnr_terms():
    torch.manual_seed(42); net = RevSep(Body()); grey = torch.rand(2, 1, 8, 8); target = torch.rand(2, 1, 8, 8); zero = torch.zeros(2, 1, 8, 8, dtype=torch.uint8)
    _, plain = separation_loss(net, grey, target, grey*0, zero, zero, heads=False)
    _, same = separation_loss(net, grey, target, grey*0, zero, zero, heads=False, weights=(torch.ones_like(grey), torch.ones(2)))
    _, tilted = separation_loss(net, grey, target, grey*0, zero, zero, heads=False, weights=(torch.ones_like(grey), torch.tensor([1., .1])))
    assert abs(plain['restored']-same['restored']) < 1e-5 and abs(plain['restored']-tilted['restored']) > 1e-4
