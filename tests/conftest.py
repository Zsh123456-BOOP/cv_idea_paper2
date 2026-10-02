"""Shared fixtures: a tiny synthetic train/validation bundle on disk (no student data), and helpers."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revrender.data import bundle as bundle_module  # noqa: E402
from revrender.util import digest  # noqa: E402


def line_mask(positive, seed):
    mask = np.zeros((96, 768), bool)
    if positive:
        rng = np.random.default_rng(seed)
        for k in range(1 + seed % 2):
            top, left = int(rng.integers(10, 60)), int(rng.integers(20, 600))
            mask[top:top+12, left:left+60+30*k] = True
            if k:
                left += 200
                mask[top:top+10, left:left+50] = True
    return mask


def make_bundle(root, test_in_records=False, extra_test_writer_in_train=False, name='synthetic_split'):
    """train: writers w0..w3 (4 lines each: 2 revised, 2 clean); dev: writers d0, d1 (3 lines each); test writer t0 only in
    the mapping.  Returns (directory, sha of run_manifest.json)."""
    root = Path(root)
    root.mkdir(parents=True)
    rows, records, mapping_rows = [], [], []
    index = 0
    spec = [('train', f'w{i}', 4) for i in range(4)] + [('dev', f'd{i}', 3) for i in range(2)]
    for split, writer, count in spec:
        for j in range(count):
            positive = j % 2 == 0 if split == 'train' else j != 2
            sid = f'{writer}_1_1:{writer}_1_1:L{j:04d}'
            rows.append((split, writer, positive, sid))
            mapping_rows.append({'sample_id': sid, 'group_uuid': writer, 'split': split, 'origin': 'src', 'index': index})
            index += 1
    n_bundle = len(rows)
    for j in range(2):
        mapping_rows.append({'sample_id': f't0_1_1:t0_1_1:L{j:04d}', 'group_uuid': 't0', 'split': 'test', 'origin': 'src', 'index': 100+j})
    images = np.full((n_bundle, 3, 96, 768), 250, np.uint8)
    masks = np.zeros((n_bundle, 96, 768), bool)
    for i, (split, writer, positive, sid) in enumerate(rows):
        masks[i] = line_mask(positive, i)
        images[i, :, masks[i]] = 40
        images[i, :, 5:9, 10:200] = 90                 # some "ink" on every line
        legacy = ('fit' if split == 'train' else 'validation') + ('_positive' if positive else '_clean')
        count = len(bundle_module.instances_from_mask(masks[i]))
        records.append({'index': i, 'split': split, 'legacy_split': legacy, 'sample_id': sid, 'group_uuid': writer,
                        'image_id': sid.split(':')[0], 'line_id': sid.split(':', 1)[1], 'is_positive': positive,
                        'instance_count': count, 'source_origin': 'src', 'source_index': i})
    if test_in_records:                          # the last validation line is replaced by a test line
        records[-1]['split'] = 'test'
        records[-1]['sample_id'] = 't0_1_1:t0_1_1:L0000'
    mapping = {'name': name, 'rows': mapping_rows}
    ids = sorted(r['sample_id'] for r in mapping_rows if r['split'] == 'test')
    mapping['common_test_sample_ids_sha256'] = hashlib.sha256(
        json.dumps(ids, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if extra_test_writer_in_train:               # a validation row of a training writer
        mapping['rows'][16]['group_uuid'] = 'w0'
    np.save(root/'images.npy', images)
    np.save(root/'masks.npy', masks)
    (root/'records.json').write_text(json.dumps(records))
    (root/'index_manifest.json').write_text(json.dumps(mapping))
    from collections import Counter
    manifest = {'schema': 'hme-revision-c1-train-dev-v1', 'candidate': name, 'new_test_rows_in_bundle': 0,
                'counts': dict(Counter(r['legacy_split'] for r in records)),
                'corrected_dev_components': sum(r['instance_count'] for r in records if r['split'] == 'dev'),
                'common_test_sample_ids_sha256': mapping['common_test_sample_ids_sha256'],
                'files': {f: {'sha256': digest(root/f), 'bytes': (root/f).stat().st_size}
                          for f in ('images.npy', 'masks.npy', 'records.json', 'index_manifest.json')}}
    (root/'run_manifest.json').write_text(json.dumps(manifest))
    return root, digest(root/'run_manifest.json')


@pytest.fixture()
def bundle(tmp_path):
    return make_bundle(tmp_path/'bundle')
