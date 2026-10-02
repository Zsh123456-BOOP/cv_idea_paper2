import json

import numpy as np
import pytest

from conftest import make_bundle
from revrender.data import bundle as B
from revrender.util import RefusedError, digest


def test_a_bundle_loads_with_its_splits_counts_and_instances(bundle):
    path, sha = bundle
    data = B.load_bundle(path, sha)
    assert len(data['train']) == 16 and len(data['dev']) == 6
    assert all(r['split'] == 'train' for r in data['train']) and all(r['split'] == 'dev' for r in data['dev'])
    assert all(len(r['instances']) == r['instance_count'] for r in data['train']+data['dev'])
    assert not data['train'][0]['instances'].flags.writeable
    assert data['train'][0]['image'].shape == (3, 96, 768)


def test_a_changed_manifest_or_file_is_refused(bundle, tmp_path):
    path, sha = bundle
    with pytest.raises(RefusedError, match='manifest changed'):
        B.load_bundle(path, 'bad')
    masks = np.load(path/'masks.npy')
    masks[0, 0, 0] = not masks[0, 0, 0]
    np.save(path/'masks.npy', masks)
    with pytest.raises(RefusedError, match='bundle file changed'):
        B.load_bundle(path, sha)


def test_a_test_row_in_the_records_is_refused(tmp_path):
    path, sha = make_bundle(tmp_path/'b', test_in_records=True)
    with pytest.raises(RefusedError, match='test sample'):
        B.load_bundle(path, sha)


def test_a_writer_in_two_splits_is_refused(tmp_path):
    path, sha = make_bundle(tmp_path/'b', extra_test_writer_in_train=True)
    with pytest.raises(RefusedError, match='leakage'):
        B.load_bundle(path, sha)


def test_a_path_that_names_a_frozen_location_is_refused(tmp_path):
    path, sha = make_bundle(tmp_path/'frozen_test_bundle')
    with pytest.raises(RefusedError, match='frozen'):
        B.load_bundle(path, sha)


def test_records_must_agree_with_their_pool_and_split(bundle):
    path, sha = bundle
    mapping = B.read_mapping(path/'index_manifest.json')
    records = json.loads((path/'records.json').read_text())
    B.validate_records(records, mapping)
    records[1]['legacy_split'] = 'validation_positive'
    with pytest.raises(RefusedError, match='sampler pool'):
        B.validate_records(records, mapping)
    records = json.loads((path/'records.json').read_text())
    records[2]['split'] = 'dev'
    with pytest.raises(RefusedError, match='identity/split'):
        B.validate_records(records, mapping)
    with pytest.raises(RefusedError, match='missing/duplicate'):
        B.validate_records(records[:-1], mapping)


def test_the_excluded_writers_are_the_validation_and_test_writers(bundle):
    path, _ = bundle
    assert B.excluded_writers(path) == {'d0', 'd1', 't0'}


def test_training_pools_hold_only_training_writers_and_refuse_a_held_out_one(bundle):
    path, sha = bundle
    data = B.load_bundle(path, sha)
    pools = B.train_pools(data, B.excluded_writers(path))
    assert {r['group_uuid'] for r in pools['fit_positive']+pools['fit_clean']} == {'w0', 'w1', 'w2', 'w3'}
    assert {r['group_uuid'] for r in pools['validation_positive']+pools['validation_clean']} == {'d0', 'd1'}
    assert all(r['image'].shape == (96, 768) and r['mask'].dtype == np.uint8 for r in pools['fit_positive'])
    with pytest.raises(RefusedError, match='training pool'):
        B.train_pools(data, {'w2'})


def test_extra_rows_are_added_sorted_and_checked(bundle):
    path, sha = bundle
    data = B.load_bundle(path, sha)

    def row(line, writer, positive):
        mask = np.zeros((96, 768), np.uint8)
        if positive:
            mask[20:30, 20:90] = 255
        return {'sample_id': 'extra:'+line, 'line_id': line, 'group_uuid': writer, 'split': 'train',
                'is_positive': positive, 'image': np.full((96, 768), 255, np.uint8), 'mask': mask,
                'line_skeleton': np.zeros((96, 768), np.uint8)}
    extra = {'fit_positive': [row('zz', 'x1', True), row('aa', 'x1', True)], 'fit_clean': [row('mm', 'x2', False)]}
    pools = B.train_pools(data, B.excluded_writers(path), extra)
    assert [r['line_id'] for r in pools['fit_positive'][-2:]] == ['aa', 'zz']
    assert len(pools['fit_clean']) == 9
    bad = {'fit_positive': [row('w0_1_1:L0000', 'x1', True)], 'fit_clean': []}
    with pytest.raises(RefusedError, match='collide'):
        B.train_pools(data, B.excluded_writers(path), bad)
    held = {'fit_positive': [row('qq', 'd0', True)], 'fit_clean': []}
    with pytest.raises(RefusedError, match='held-out|validation or test writer'):
        B.train_pools(data, B.excluded_writers(path), held)
