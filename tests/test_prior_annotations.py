"""The earlier project's human before/after pairs, no-edit judgements and line readings, joined on the catalogue's line id."""
import json

import pytest

from revrender.data import prior_annotations as P


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), encoding='utf-8')


def jsonl(rows):
    return ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)


def earlier_project(tmp_path):
    """A tiny copy of the earlier project's layout: three revised lines, two blind judgements, readings of every line."""
    root = tmp_path/'earlier'
    write(root/P.FILES['pair_key'], jsonl([{'uid': 'R001', 'line_id': 'w1_3_1:L0002', 'image_id': 'w1_3_1'},
                                           {'uid': 'R002', 'line_id': 'w1_3_1:L0003', 'image_id': 'w1_3_1'},
                                           {'uid': 'R003', 'line_id': 'w2_4_1:L0001', 'image_id': 'w2_4_1'}]))
    write(root/P.FILES['pair_returned'], {'records': {
        'R001': {'text_before': 'x=3', 'text_after': 'x=-3', 'ink_kind': 'struck'},
        'R002': {'no_revision': True, 'illegible': False},
        'R003': {'text_before': 'y=2', 'no_rewrite': True, 'ink_kind': 'struck'}}})
    write(root/P.FILES['combined_export'], {'records': {
        'A_A0001': {'has_edit': 'false'}, 'A_A0002': {'has_edit': 'true'}, 'B_b1': {'edit_kind': 'strike_rewrite', 'text_before': '5', 'text_after': '6'}}})
    write(root/P.FILES['combined_b_key'], jsonl([{'uid': 'B_b1', 'line_id': 'w3_5_1:L0004', 'image_id': 'w3_5_1'}]))
    write(root/P.FILES['audit_sample'], jsonl([{'audit_id': 'A0001', 'representative_line_id': 'w1_3_1:L0001', 'image_id': 'w1_3_1'},
                                               {'audit_id': 'A0002', 'representative_line_id': 'w2_4_1:L0002', 'image_id': 'w2_4_1'}]))
    reading = {'w1_3_1:L0002': ('x = - 3', 'x=-3'), 'w1_3_1:L0001': ('a = b', 'a=b'), 'w2_4_1:L0001': ('y = 2', 'y=4'), 'w3_5_1:L0004': ('6', '5')}
    for model, relative in P.LINE_READINGS.items():
        write(root/relative, jsonl([{'representative_line_id': k, 'alias_line_ids': [k], 'model_text': v[0 if model == 'qwen35_35b' else 1],
                                     'ulocr_text': 'u '+k} for k, v in reading.items()]))
    return root


def test_the_human_sources_are_joined_on_the_line_id_and_told_apart_by_kind(tmp_path):
    pairs = P.read_pairs(earlier_project(tmp_path))
    assert len(pairs) == 6
    assert pairs['w1_3_1:L0002']['text_before'] == 'x=3' and pairs['w1_3_1:L0002']['sources'] == ['pair_v1']
    assert P.kind_of(pairs['w1_3_1:L0002']) == 'pair' and P.kind_of(pairs['w2_4_1:L0001']) == 'pair' and pairs['w2_4_1:L0001']['no_rewrite']
    assert P.kind_of(pairs['w1_3_1:L0003']) == 'no_revision'
    assert P.kind_of(pairs['w1_3_1:L0001']) == 'no_edit' and P.kind_of(pairs['w2_4_1:L0002']) == 'edit_no_text'
    assert P.kind_of(pairs['w3_5_1:L0004']) == 'pair' and pairs['w3_5_1:L0004']['ink_kind'] == 'strike_rewrite'


def test_the_readings_of_the_wanted_lines_are_read_for_every_model(tmp_path):
    root = earlier_project(tmp_path)
    texts = P.read_line_text(root, {'w1_3_1:L0002', 'w2_4_1:L0001'})
    assert set(texts) == {'ulocr', 'qwen35_35b', 'infinity'}
    assert texts['qwen35_35b'] == {'w1_3_1:L0002': 'x = - 3', 'w2_4_1:L0001': 'y = 2'} and texts['infinity']['w2_4_1:L0001'] == 'y=4'
    assert texts['ulocr']['w1_3_1:L0002'] == 'u w1_3_1:L0002' and len(P.read_line_text(root)['qwen35_35b']) == 4


def test_the_tool_writes_the_table_and_how_much_of_the_labelled_text_the_models_read(tmp_path):
    root = earlier_project(tmp_path)
    tags = tmp_path/'tags.json'
    tags.write_text(json.dumps({'w1_3_1': {'split': 'train'}, 'w2_4_1': {'split': 'test'}, 'w3_5_1': {'split': 'dev'}}))
    out = tmp_path/'out'
    assert P.main(['--root', str(root), '--tags', str(tags), '--output-dir', str(out)]) == 0
    rows = {r['line_id']: r for r in map(json.loads, (out/'pairs.jsonl').read_text(encoding='utf-8').splitlines())}
    assert rows['w2_4_1:L0001']['split'] == 'test' and rows['w1_3_1:L0002']['kind'] == 'pair' and rows['w1_3_1:L0002']['qwen35_35b'] == 'x = - 3'
    summary = json.loads((out/'summary.json').read_text())
    assert summary['revised_non_test_lines'] == 2 and summary['no_edit_non_test_lines'] == 1           # the test writer's line is left out
    q = summary['readings_by_coverage']['qwen35_35b']
    assert q['leak'] == {'lines': 1, 'mean': 0.} and q['rewrite'] == {'lines': 2, 'mean': 1.} and q['valid'] == {'lines': 1, 'mean': 1.}   # 'x=3' -> 'x=-3' strikes nothing, '5' -> '6' keeps nothing
    assert summary['qwen35_35b_vs_infinity']['revised']['lines'] == 2
    manifest = json.loads((out/'run_manifest.json').read_text())
    assert manifest['lines'] == 6 and manifest['seed'] == 42
    with pytest.raises(FileExistsError):
        P.main(['--root', str(root), '--output-dir', str(out)])


def test_the_split_comes_from_the_writer_mapping_and_a_test_writers_other_id_is_a_test_writer_too(tmp_path):
    root = earlier_project(tmp_path)
    mapping = tmp_path/'index_manifest.json'
    mapping.write_text(json.dumps({'rows': [{'group_uuid': 'w1', 'split': 'train'}, {'group_uuid': 'w2', 'split': 'dev'}, {'group_uuid': 'w9', 'split': 'test'}]}))
    twin = tmp_path/'same.json'
    twin.write_text(json.dumps({'pairs': [{'a': 'w2', 'b': 'w9'}]}))
    assert P.writer_splits(mapping) == {'w1': 'train', 'w2': 'dev', 'w9': 'test'}
    writers = P.writer_splits(mapping, twin)
    assert writers['w2'] == 'test' and writers['w1'] == 'train'
    pairs = P.read_pairs(root)
    assert P.split_of(pairs['w1_3_1:L0002'], {}, writers) == 'train' and P.split_of(pairs['w2_4_1:L0001'], {}, writers) == 'test'
    assert P.split_of(pairs['w3_5_1:L0004'], {}, writers) == 'none' and P.split_of(pairs['w3_5_1:L0004'], {'w3_5_1': {'split': 'dev'}}) == 'dev'
    out = tmp_path/'out'
    P.main(['--root', str(root), '--mapping', str(mapping), '--same-student', str(twin), '--output-dir', str(out)])
    assert json.loads((out/'summary.json').read_text())['revised_non_test_lines'] == 2        # w1's line and the writer without a split; w2 is a test writer now
