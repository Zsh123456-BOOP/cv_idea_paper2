"""models.feature_mix and eval.feature_scale (三个不训练的检验): the chord's ends bit for bit, the matched alpha, the pooled variance."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')


def tokens(seed, n=64, w=32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, w, generator=g)


def test_the_chord_ends_are_x_and_c_bit_for_bit():
    from revrender.models.feature_mix import mixed_level
    x, c = tokens(1).bfloat16(), tokens(2).bfloat16()
    assert torch.equal(mixed_level(x, c, None, 0., None)[0], x)
    assert torch.equal(mixed_level(x, c, None, 1., None)[0], c)


def test_matched_alpha_gives_the_students_cosine():
    from revrender.models.feature_mix import matched_alpha
    x, c = tokens(1), tokens(2)
    s = .3*x+.7*c+.2*tokens(3)                                            # mostly closer than x, some maybe not
    alpha, farther = matched_alpha(x, c, s)
    cos = lambda u, v: torch.nn.functional.cosine_similarity(u.double(), v.double(), dim=-1)
    z = (1-alpha[:, None])*x.double()+alpha[:, None]*c.double()
    assert torch.equal(farther, cos(s, c) < cos(x, c))
    assert (cos(z, c)-cos(s, c))[~farther].abs().max() < 1e-10
    assert bool((alpha[farther] == 0).all()) and bool(((alpha >= 0) & (alpha <= 1)).all())


def test_matched_levels_keep_s_or_x_where_s_is_farther():
    from revrender.models.feature_mix import mixed_level
    x, c = tokens(1).bfloat16(), tokens(2).bfloat16()
    s = (x.float()*1.0+.5*tokens(4)).bfloat16()                           # farther than x on many tokens
    zx, stx = mixed_level(x, c, s, None, 'x'); zs, sts = mixed_level(x, c, s, None, 's')
    assert stx['taken_back'] == sts['taken_back'] > 0 and stx['largest_cos_gap'] < 1e-4 and sts['kept_s_bit_for_bit']
    cos = lambda u, v: torch.nn.functional.cosine_similarity(u.double(), v.double(), dim=-1)
    farther = cos(s, c) < cos(x, c)
    assert torch.equal(zx[farther], x[farther]) and torch.equal(zs[farther], s[farther]) and torch.equal(zx[~farther], zs[~farther])


def test_pooled_and_within_variance_from_the_sums():
    from revrender.eval.feature_scale import QUANTITIES, _ratios
    rng = np.random.default_rng(0)
    regions = [[rng.normal(size=(n, 5))*(1+q) for q in range(len(QUANTITIES))] for n in rng.integers(3, 9, size=6)]
    wn = np.array([float(len(r[0])) for r in regions])
    ws = np.array([[[t.sum(0) for t in r]] for r in regions]); wq = np.array([[[float((t*t).sum()) for t in r]] for r in regions])
    win = np.array([[[float(((t-t.mean(0))**2).sum()) for t in r]] for r in regions])
    got = _ratios(np.ones(len(regions)), wn, ws, wq, win)
    pooled = [np.concatenate([r[q] for r in regions]).var(0).sum() for q in range(len(QUANTITIES))]
    within = [sum(((r[q]-r[q].mean(0))**2).sum() for r in regions)/wn.sum() for q in range(len(QUANTITIES))]
    c = QUANTITIES.index('c')
    for v in ('s', 'x', 'M'):
        assert got[f'R_{v}'][0] == pytest.approx(pooled[QUANTITIES.index(v)]/pooled[c])
        assert got[f'R_within_{v}'][0] == pytest.approx(within[QUANTITIES.index(v)]/within[c])
