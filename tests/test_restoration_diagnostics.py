import numpy as np

from revrender.eval.restoration_diagnostics import edge_gradient, paired


def test_paired_differences_give_writer_and_strike_intervals_on_the_shared_strikes():
    x = {f'w{k}_q#1': .5 for k in range(6)}; y = {**{f'w{k}_q#1': .7 for k in range(6)}, 'w9_q#1': 1.}
    r = paired(x, y, np.random.default_rng(42))
    assert r['strikes'] == 6 and abs(r['difference']-.2) < 1e-9 and r['writer_interval'][0]-1e-9 <= .2 <= r['writer_interval'][1]+1e-9
    assert abs(r['strike_interval'][0]-.2) < 1e-9


def test_the_edge_gradient_drops_when_writing_is_lightened():
    scan = np.ones((20, 20), np.float32); scan[5:15, 8:12] = .1
    edge = (scan < .5) & ~((np.roll(scan, 1, 1) < .5) & (np.roll(scan, -1, 1) < .5))
    assert edge_gradient((scan+1)/2, edge) < edge_gradient(scan, edge) and edge_gradient(scan, edge*False) is None
