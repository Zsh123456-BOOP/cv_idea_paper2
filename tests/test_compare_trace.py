from revrender.tools import compare_trace as C


def step(n, loss, state='s', pieces=None):
    return {'step': n, 'loss': loss, 'input_sha256': 'i', 'state_sha256': state, 'pieces': pieces or {'stage_0': 1.0}}


def test_equal_traces_are_identical_and_any_difference_is_named():
    a = [step(1, 1.5), step(2, 1.25)]
    assert all(r[2] for r in C.compare(a, [dict(s) for s in a]))
    b = [step(1, 1.5), step(2, 1.25, state='x', pieces={'stage_0': 1.0000001})]
    different = {(r[0], r[1]) for r in C.compare(a, b) if not r[2]}
    assert different == {(2, 'state_sha256'), (2, 'pieces.stage_0')}
    assert not C.compare(a, a[:1])[0][2]
