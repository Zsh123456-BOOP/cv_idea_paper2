import pytest

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


def test_curves_compare_at_the_steps_both_reached():
    from revrender.tools import compare_curve as K
    entry = lambda step, f1, loss: {'step': step, 'validation': {'f1': f1}, 'loss_terms': {'stage_0': loss}}
    old = {2000: entry(2000, .1, 1.), 4000: entry(4000, .2, 2.)}
    new = {2000: entry(2000, .1, 1.)}
    assert all(r[2] for r in K.compare(old, new)) and len(K.compare(old, new)) == 2
    new[2000] = entry(2000, .1, 1.5)
    assert [r[:2] for r in K.compare(old, new) if not r[2]] == [(2000, 'loss_terms')]
    assert K.compare(old, {}) == []


def test_curves_say_whether_they_have_the_same_steps(tmp_path, capsys):
    import json
    from revrender.tools import compare_curve as K
    entry = lambda step: {'step': step, 'validation': {'f1': .1}, 'loss_terms': {'stage_0': 1.}}
    for name, steps in (('old', (2000, 4000)), ('short', (2000,)), ('same', (2000, 4000))):
        (tmp_path/f'{name}.jsonl').write_text(''.join(json.dumps(entry(s))+'\n' for s in steps))
    summary = lambda other: (K.main([str(tmp_path/'old.jsonl'), str(tmp_path/f'{other}.jsonl')]), json.loads(capsys.readouterr().out.splitlines()[-1]))[1]
    assert summary('short')['identical'] and not summary('short')['same_steps']
    assert summary('same')['identical'] and summary('same')['same_steps']


def test_compare_checkpoint_reads_tensors_not_bytes(tmp_path):
    torch = pytest.importorskip('torch')
    from revrender.tools import compare_checkpoint
    state = {'a.weight': torch.arange(6.).reshape(2, 3), 'b.bias': torch.ones(3)}
    first = {'state_dict': state, 'step': 19000}
    again = {'state_dict': {k: v.clone() for k, v in state.items()}, 'step': 19000}
    assert compare_checkpoint.compare(first, again)['identical']
    changed = {'state_dict': {**again['state_dict'], 'b.bias': torch.ones(3) + 1e-7}, 'step': 19000}
    result = compare_checkpoint.compare(first, changed)
    assert not result['identical'] and list(result['different']) == ['b.bias']
    assert not compare_checkpoint.compare(first, {**again, 'step': 18500})['identical']
    reordered = {'state_dict': dict(reversed(list(again['state_dict'].items()))), 'step': 19000}
    assert compare_checkpoint.compare(first, reordered)['same_names'] and not compare_checkpoint.compare(first, reordered)['same_order']
    torch.save(first, tmp_path/'a.pt')
    torch.save(again, tmp_path/'b.pt')
    assert compare_checkpoint.main([str(tmp_path/'a.pt'), str(tmp_path/'b.pt')]) == 0
