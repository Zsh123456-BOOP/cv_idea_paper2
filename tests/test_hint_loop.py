import json
from types import SimpleNamespace

from revrender.eval import hint_loop


def folder(tmp_path, name, readings):
    d = tmp_path/name; d.mkdir()
    (d/'predictions.jsonl').write_text(''.join(json.dumps({'condition': 'c', 'id': k, 'line_id': k, 'text': t, 'raw': t, 'finish_reason': 'stop'},
                                                          ensure_ascii=False)+'\n' for k, t in readings.items()))
    return str(d)


def test_the_loop_stops_where_a_reading_meets_its_hint(tmp_path):
    initial = folder(tmp_path, 'r0', {'a': 'x', 'b': 'x', 'c': '[读不出]', 'd': 'p'})
    rounds = [folder(tmp_path, f'round{n}', r) for n, r in enumerate([{'a': 'x', 'b': 'y', 'c': '[读不出]', 'd': 'q'},
                                                                      {'a': 'z', 'b': 'y', 'c': 'w', 'd': 'r'},
                                                                      {'a': 'z', 'b': 'z', 'c': 'w', 'd': 's'}], 1)]
    hint_loop.compose(SimpleNamespace(initial=initial, round=rounds, output_dir=tmp_path/'loop'))
    loop = {r['id']: r for r in map(json.loads, (tmp_path/'loop'/'loop.jsonl').read_text().splitlines())}
    final = {r['id']: r['text'] for r in map(json.loads, (tmp_path/'loop'/'predictions.jsonl').read_text().splitlines())}
    assert {k: (v['stopped_at'], v['converged']) for k, v in loop.items()} == {'a': (1, True), 'b': (2, True), 'c': (1, True), 'd': (3, False)}
    assert final == {'a': 'x', 'b': 'y', 'c': '[读不出]', 'd': 's'}


def test_same_reads_refusals_and_spelling():
    assert hint_loop.same('[读不出]', '[读不出]') and not hint_loop.same('[读不出]', 'x') and hint_loop.same('$x$', 'x')
