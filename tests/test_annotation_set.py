"""The questions chosen for annotation: whole questions with enough writing and revisions, spread over question numbers and writers."""
import json
from collections import Counter

import pytest

from revrender.data import annotation_set as A


def catalogue(pages):
    """pages: [(image_id, lines, edited)] -> catalogue records."""
    out = []
    for image_id, n, edited in pages:
        for k in range(n):
            out.append({'image_id': image_id, 'group_uuid': image_id.split('_')[0], 'representative_line_id': f'{image_id}:L{k:04d}', 'has_edit': k < edited, 'box_1000': [0, 0, 1, 1]})
    return out


def test_a_candidate_has_enough_lines_and_revised_lines_and_a_writer_in_a_split():
    cat = catalogue([('a_3_1', 12, 3), ('a_4_1', 11, 5), ('b_3_1', 15, 2), ('c_s5_1', 20, 4), ('d_3_1', 30, 9)])
    pools = A.candidates(cat, {'a': 'test', 'b': 'test', 'c': 'dev', 'd': 'none'}, 12, 3)
    assert [p['image_id'] for p in pools['test']] == ['a_3_1'] and [p['image_id'] for p in pools['val']] == ['c_s5_1'] and pools['val'][0]['question'] == 5
    assert 'train' not in pools                                            # writers paper 1 did not use belong to no split


def test_questions_are_spread_over_question_numbers_and_writers_before_any_repeats():
    pool = [{'image_id': f'w{w}_{q}_1', 'writer': f'w{w}', 'question': q, 'lines': 10, 'edited': 3} for w in range(6) for q in (3, 4, 5)]
    chosen = A.choose(pool, 60)
    assert sum(p['lines'] for p in chosen) == 60 and len(chosen) == 6
    assert max(Counter(p['writer'] for p in chosen).values()) == 1 and sorted(Counter(p['question'] for p in chosen).values()) == [2, 2, 2]
    assert A.choose(pool, 60) == chosen                                     # the same choice every time
    assert len(A.choose(pool, 1000)) == len(pool)                           # a target above what there is takes everything


def test_the_tool_writes_the_selection_with_a_pilot_and_one_split_per_writer(tmp_path):
    pages = [(f't{w}_{q}_1', 13, 3) for w in range(4) for q in (3, 4)] + [(f'v{w}_{q}_1', 13, 3) for w in range(3) for q in (3, 4)] + [(f'r{w}_5_1', 13, 4) for w in range(3)]
    path = tmp_path/'catalogue.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in catalogue(pages)))
    mapping = tmp_path/'index_manifest.json'
    mapping.write_text(json.dumps({'rows': [{'group_uuid': f't{w}', 'split': 'test'} for w in range(4)] + [{'group_uuid': f'v{w}', 'split': 'dev'} for w in range(3)]
                                   + [{'group_uuid': f'r{w}', 'split': 'train'} for w in range(3)]}))
    allowed = tmp_path/'allowed.txt'
    allowed.write_text('\n'.join([f't{w}' for w in range(4)]+[f'v{w}' for w in range(3)]+[f'r{w}' for w in range(3)]))
    out = tmp_path/'out'
    assert A.main(['--catalogue', str(path), '--mapping', str(mapping), '--allowed-writers', str(allowed), '--lines', 'test=50', 'val=26', 'train=26', '--pilot', '2', '--output-dir', str(out)]) == 0
    selection = json.loads((out/'selection.json').read_text())
    assert selection['summary']['test']['lines'] == 52 and selection['summary']['val']['questions'] == 2 and selection['summary']['train']['questions'] == 2
    assert sum(p['pilot'] for p in selection['pages']) == 2 and all(p['split'] == 'test' for p in selection['pages'] if p['pilot'])
    with pytest.raises(Exception, match='not enough'):
        A.main(['--catalogue', str(path), '--mapping', str(mapping), '--allowed-writers', str(allowed), '--lines', 'test=5000', '--output-dir', str(tmp_path/'too_many')])
    # a labmate's writer (not in the allowed set) is never drawn; earlier selections are left out; all takes the rest
    allowed.write_text('\n'.join(['t0', 't1', 't2']))
    assert A.main(['--catalogue', str(path), '--mapping', str(mapping), '--allowed-writers', str(allowed), '--exclude', str(out/'selection.json'),
                   '--lines', 'test=all', '--output-dir', str(tmp_path/'rest')]) == 0
    rest = json.loads((tmp_path/'rest'/'selection.json').read_text())['pages']
    first = {p['image_id'] for p in selection['pages']}
    assert rest and all(p['writer'] != 't3' and p['image_id'] not in first for p in rest)
    assert {p['image_id'] for p in rest} | (first & {f't{w}_{q}_1' for w in range(3) for q in (3, 4)}) == {f't{w}_{q}_1' for w in range(3) for q in (3, 4)}
