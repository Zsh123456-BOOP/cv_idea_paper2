"""The recipes of synthetic revisions: which lines may be hosts, what a recipe says, and that no test or evaluation writer gets in."""
import json

import pytest

from revrender.data import reader_corpus as C


def record(line, writer, **fields):
    return {'representative_line_id': line, 'image_id': f'{writer}_3_1', 'group_uuid': writer, 'has_edit': False, 'ocr_type': 'text', 'box_1000': [10, 10, 900, 100], **fields}


CATALOGUE = [record('w1_3_1:L1', 'w1'), record('w1_3_1:L2', 'w1'), record('w1_3_1:L3', 'w1', has_edit=True), record('w1_3_1:L4', 'w1', ocr_type='footer'),
             record('w1_3_1:L5', 'w1'), record('w1_3_1:L6', 'w1'), record('t1_3_1:L1', 't1'), record('e1_3_1:L1', 'e1'), record('d1_3_1:L1', 'd1')]
QWEN = {'w1_3_1:L1': 'x = 3 + 4', 'w1_3_1:L2': 'y = 2 x', 'w1_3_1:L3': 'z = 1 + 1', 'w1_3_1:L4': 'page 1 of 2', 'w1_3_1:L5': 'a', 'w1_3_1:L6': 'b = 5 c = 6',
        't1_3_1:L1': 'x = 3 + 4', 'e1_3_1:L1': 'x = 3 + 4', 'd1_3_1:L1': 'x = 3 + 4'}
INFINITY = {**QWEN, 'w1_3_1:L2': 'completely different text', 'w1_3_1:L6': 'b = 5 c = 6'}
WRITERS = {'w1': 'train', 't1': 'test', 'e1': 'train', 'd1': 'dev'}


def test_a_host_is_a_clean_line_of_a_training_writer_that_two_models_read_alike():
    chosen = C.hosts(CATALOGUE, {'qwen35_35b': QWEN, 'infinity': INFINITY}, WRITERS, {'e1'})
    assert [h['line_id'] for h in chosen] == ['w1_3_1:L1', 'w1_3_1:L6']              # edited, footer, disagreeing, too short, test, evaluation and dev lines are out
    assert chosen[0]['text'] == 'x = 3 + 4' and chosen[0]['writer'] == 'w1'


def test_each_host_becomes_one_recipe_per_variant_and_a_paste_takes_its_donors_from_the_same_writer():
    chosen = C.hosts(CATALOGUE, {'qwen35_35b': QWEN, 'infinity': INFINITY}, WRITERS, {'e1'})
    rows = C.recipes(chosen)
    assert len(rows) == len(chosen)*len(C.VARIANTS) and len({r['id'] for r in rows}) == len(rows)
    kinds = [r['kind'] for r in rows if r['line_id'] == 'w1_3_1:L1']
    assert kinds == list(C.VARIANTS)
    for r in rows:
        assert (r['text'] == '') == (r['kind'] == 'whole') and r['host_text'] and (bool(r['donors']) == (r['kind'] in ('paste', 'stack')))
        assert r['line_id'] not in r['donors'] and all(d.startswith(r['writer']) for d in r['donors'])
    assert C.recipes(chosen) == rows and C.recipes(chosen, seed=7) != rows        # the same corpus every time, another seed another draw
    alone = C.recipes(C.hosts(CATALOGUE, {'qwen35_35b': QWEN, 'infinity': INFINITY}, WRITERS, set()))
    assert [r['kind'] for r in alone if r['writer'] == 'e1'] == ['clean', 'whole']      # a writer with one host line has nothing to paste


def test_the_validation_writers_are_chosen_by_hash_not_by_position():
    assert C.val_writer('abc') == C.val_writer('abc')
    share = sum(C.val_writer(f'writer-{i}') for i in range(4000))/4000
    assert .03 < share < .07


def test_the_tool_writes_the_corpus_and_refuses_an_existing_directory(tmp_path, monkeypatch):
    catalogue = tmp_path/'catalogue.jsonl'
    catalogue.write_text(''.join(json.dumps(r)+'\n' for r in CATALOGUE))
    mapping = tmp_path/'index_manifest.json'
    mapping.write_text(json.dumps({'rows': [{'group_uuid': w, 'split': s} for w, s in WRITERS.items()]}))
    pack = tmp_path/'pack'
    pack.mkdir()
    (pack/'pack.json').write_text(json.dumps({'rows': [{'image_id': 'e1_3_1'}]}))
    monkeypatch.setattr(C, 'read_line_text', lambda root, wanted=None: {'qwen35_35b': QWEN, 'infinity': INFINITY})
    twin = tmp_path/'same.json'
    twin.write_text(json.dumps({'pairs': [{'a': 'e1', 'b': 'w9'}]}))
    out = tmp_path/'out'
    assert C.main(['--catalogue', str(catalogue), '--root', str(tmp_path), '--mapping', str(mapping), '--exclude-pack', str(pack), '--output-dir', str(out)]) == 0
    rows = [json.loads(line) for line in (out/'corpus.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {r['writer'] for r in rows} == {'w1'} and len(rows) == 2*len(C.VARIANTS)
    summary = json.loads((out/'summary.json').read_text())
    assert summary['hosts'] == 2 and summary['excluded_writers'] == 1 and json.loads((out/'run_manifest.json').read_text())['seed'] == 42
    with pytest.raises(FileExistsError):
        C.main(['--catalogue', str(catalogue), '--root', str(tmp_path), '--mapping', str(mapping), '--output-dir', str(out)])


def test_the_other_id_of_an_evaluation_writer_is_left_out_too(tmp_path, monkeypatch):
    catalogue = tmp_path/'catalogue.jsonl'
    more = [record('w2_3_1:L1', 'w2'), record('w2_3_1:L2', 'w2')]
    catalogue.write_text(''.join(json.dumps(r)+'\n' for r in CATALOGUE+more))
    mapping = tmp_path/'index_manifest.json'
    mapping.write_text(json.dumps({'rows': [{'group_uuid': w, 'split': s} for w, s in {**WRITERS, 'w2': 'train'}.items()]}))
    twin = tmp_path/'same.json'
    twin.write_text(json.dumps({'pairs': [{'a': 'e1', 'b': 'w2'}]}))
    pack = tmp_path/'pack'
    pack.mkdir()
    (pack/'pack.json').write_text(json.dumps({'rows': [{'image_id': 'e1_3_1'}]}))
    monkeypatch.setattr(C, 'read_line_text', lambda root, wanted=None: {'qwen35_35b': {**QWEN, 'w2_3_1:L1': 'x = 3 + 4', 'w2_3_1:L2': 'x = 3 + 4'}, 'infinity': {**INFINITY, 'w2_3_1:L1': 'x = 3 + 4', 'w2_3_1:L2': 'x = 3 + 4'}})
    out = tmp_path/'out'
    assert C.main(['--catalogue', str(catalogue), '--root', str(tmp_path), '--mapping', str(mapping), '--same-student', str(twin), '--exclude-pack', str(pack), '--output-dir', str(out)]) == 0
    assert {json.loads(line)['writer'] for line in (out/'corpus.jsonl').read_text(encoding='utf-8').splitlines()} == {'w1'}
