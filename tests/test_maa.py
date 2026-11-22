import math

import pytest

from revrender.train_clean_view import maa_schedule


def test_maa_schedule_is_transformers_cosine():
    assert maa_schedule(0, 100, 2000) == 0. and maa_schedule(50, 100, 2000) == .5 and maa_schedule(100, 100, 2000) == 1.
    assert abs(maa_schedule(1050, 100, 2000)-.5) < 1e-12 and maa_schedule(2000, 100, 2000) == 0.
    transformers = pytest.importorskip('transformers'); torch = pytest.importorskip('torch')
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.)
    hf = transformers.get_scheduler('cosine', optimizer, num_warmup_steps=100, num_training_steps=2000)
    for step in range(2000):
        assert math.isclose(optimizer.param_groups[0]['lr'], maa_schedule(step, 100, 2000), abs_tol=1e-12)
        optimizer.step(); hf.step()


def test_raster_round_trip_and_order():
    torch = pytest.importorskip('torch')
    from revrender.models.maa import merge_order, raster
    t, h, w, m = 1, 6, 10, 2
    # tokens in Qwen's merge-group order, each carrying its own row and column
    rows = torch.arange(h).view(h, 1).expand(h, w); cols = torch.arange(w).view(1, w).expand(h, w)
    grid = torch.stack([rows, cols], -1).view(t, h, w, 2)
    qwen = grid.view(t, h//m, m, w//m, m, 2).permute(0, 1, 3, 2, 4, 5).reshape(-1, 2)      # the processor's patch order
    back = raster(qwen, t, h, w, m)
    assert torch.equal(back, grid) and torch.equal(merge_order(back, m), qwen)


def test_adapter_starts_as_nothing_and_keeps_small_images():
    torch = pytest.importorskip('torch')
    from revrender.models.maa import MLP_HIDDEN, adapter_set
    torch.manual_seed(42)
    one = adapter_set(32, 1)[0]
    x = torch.randn(1, 6, 10, 32).to(torch.bfloat16)                     # a side under 7: the window shrinks, no position bias
    assert torch.equal(one(x), x)
    with torch.no_grad():
        one.alpha_mlp.fill_(.5); one.alpha_pool.fill_(.1)
    y = one(x)
    assert y.shape == x.shape and y.dtype == x.dtype and not torch.equal(y, x)
    width = 32
    per = 2*width + (width*MLP_HIDDEN+MLP_HIDDEN)+(MLP_HIDDEN*width+width) + (width*384+384)+(128*width+width)+8*13*13+1 + 2
    assert sum(q.numel() for q in one.parameters()) == per
