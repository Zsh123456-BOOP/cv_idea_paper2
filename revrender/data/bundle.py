"""Read a train/validation bundle (images.npy, masks.npy, records.json, index_manifest.json) and build the sampling pools.

A bundle is private student handwriting and lives outside Git.  It is read-only here: the loader checks the manifest
digest, every file's size and digest, the records against the split mapping, and that no test row or test writer is
anywhere in it.  Training pools are then built from the ``train`` rows only; ``train_pools`` refuses a pool that
contains a validation or test writer, which is the guarantee tests/test_no_test_rows.py checks.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..util import digest, json_digest, require, safe_path, schema_is

SCHEMA_PATTERN = r'hme-revision-c\d+-train-dev-v1'      # the schema tag of the bundles this reads
POOLS = ('fit_positive', 'fit_clean', 'validation_positive', 'validation_clean')
CANVAS = (96, 768)


def instances_from_mask(mask):
    """The line's gold components: 8-connected components of its mask, as an N x H x W bool array."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if not count:
        return np.zeros((0, *mask.shape), dtype=bool)
    return np.stack([labels == index for index in range(1, count + 1)])


def manifest_at(path, expected_sha=None):
    root = Path(path).resolve()
    if expected_sha is not None:
        require(digest(root/'run_manifest.json') == expected_sha, 'bundle manifest changed')
    manifest = json.loads((root/'run_manifest.json').read_text())
    require(schema_is(manifest['schema'], SCHEMA_PATTERN) and manifest['new_test_rows_in_bundle'] == 0, 'not a train/dev bundle')
    return manifest


def gold_source_sha(path, expected_sha=None):
    """The bundle a gold-v2 bundle was corrected from (earlier freezes were made on it), or None."""
    return (manifest_at(path, expected_sha).get('gold_v2') or {}).get('from_bundle_sha256')


def test_ids_sha256(mapping):
    ids = sorted(r['sample_id'] for r in mapping['rows'] if r['split'] == 'test')
    return json_digest(ids)


def read_mapping(path):
    """The split mapping: one row per line with its writer and split; no writer may sit in two splits."""
    mapping = json.loads(Path(path).read_text())
    rows = mapping['rows']
    require(rows and len({r['sample_id'] for r in rows}) == len(rows), 'sample identity mismatch')
    require({r['split'] for r in rows} <= {'train', 'dev', 'test'}, 'undeclared split in the mapping')
    writers = {s: {r['group_uuid'] for r in rows if r['split'] == s} for s in ('train', 'dev', 'test')}
    require(not (writers['train'] & writers['dev'] or writers['train'] & writers['test'] or writers['dev'] & writers['test']),
            'writer leakage between splits')
    if 'common_test_sample_ids_sha256' in mapping:
        require(test_ids_sha256(mapping) == mapping['common_test_sample_ids_sha256'], 'common test changed')
    return mapping


def validate_records(records, mapping):
    allowed = {r['sample_id']: r for r in mapping['rows'] if r['split'] in ('train', 'dev')}
    require(len(records) == len(allowed) and {r['sample_id'] for r in records} == set(allowed),
            'missing/duplicate/test sample in train/dev bundle')
    suffixes = [r['sample_id'].split(':', 1)[-1] for r in records]
    require(len(set(suffixes)) == len(suffixes), 'archive suffix IDs collide')
    for i, row in enumerate(records):
        source = allowed[row['sample_id']]
        require(row['index'] == i and row['split'] == source['split'] and row['group_uuid'] == source['group_uuid'],
                'row identity/split changed')
        if 'origin' in source:
            require(row['source_origin'] == source['origin'] and row['source_index'] == source['index'],
                    'source pointer changed')
        require(type(row['is_positive']) is bool, 'invalid positive flag')
        expected = ('fit' if row['split'] == 'train' else 'validation') + ('_positive' if row['is_positive'] else '_clean')
        require(row['legacy_split'] == expected, 'sampler pool mismatch')


def load_bundle(path, expected_sha=None):
    """{'train': [...], 'dev': [...], 'manifest': ...}; each row has image (3,96,768 uint8 view), mask, instances."""
    root = safe_path(path)
    manifest = manifest_at(root, expected_sha)
    for name, entry in manifest['files'].items():
        item = root/name
        require(not item.is_symlink() and item.stat().st_size == entry['bytes'] and digest(item) == entry['sha256'],
                f'bundle file changed: {name}')
    mapping = read_mapping(root/'index_manifest.json')
    require(mapping['name'] == manifest['candidate'], 'manifest/mapping mismatch')
    records = json.loads((root/'records.json').read_text())
    validate_records(records, mapping)
    images = np.load(root/'images.npy', mmap_mode='r', allow_pickle=False)
    masks = np.load(root/'masks.npy', mmap_mode='r', allow_pickle=False)
    require(images.shape == (len(records), 3, *CANVAS) and images.dtype == np.uint8, 'image shape/dtype changed')
    require(masks.shape == (len(records), *CANVAS) and masks.dtype == np.bool_, 'mask shape/dtype changed')
    result = {'manifest': manifest, 'mapping_name': mapping['name'], 'train': [], 'dev': []}
    for i, row in enumerate(records):
        instances = instances_from_mask(masks[i])
        require(len(instances) == row['instance_count'] and bool(len(instances)) == row['is_positive'], 'label mismatch')
        instances.flags.writeable = False
        result[row['split']].append({**row, 'image': images[i], 'mask': masks[i], 'instances': instances})
    require(dict(Counter(r['legacy_split'] for r in records)) == manifest['counts'], 'sampler counts changed')
    require(sum(len(r['instances']) for r in result['dev']) == manifest['corrected_dev_components'],
            'dev components changed')
    return result


def excluded_writers(bundle_dir):
    """Validation and test writers of the bundle's mapping: no training pool may contain one."""
    mapping = json.loads((Path(bundle_dir)/'index_manifest.json').read_text())
    return {r['group_uuid'] for r in mapping['rows'] if r['split'] in ('dev', 'test')}


def make_rows(bundle):
    """The four sampling pools, in bundle order; a row carries its grey strip, its 0/255 mask and a zero skeleton."""
    rows = {name: [] for name in POOLS}
    for split in ('train', 'dev'):
        for row in bundle[split]:
            rows[row['legacy_split']].append({**row, 'image': row['image'][0],
                                              'mask': row['mask'].astype(np.uint8)*255,
                                              'line_skeleton': np.zeros_like(row['mask'], dtype=np.uint8)})
    return rows


def train_pools(bundle, excluded, extra=None):
    """The pools training may draw from: the bundle's train rows plus any extra rows, with no held-out writer.

    ``extra`` is {'fit_positive': [...], 'fit_clean': [...]} (see data.extra); its rows must not collide with the
    bundle's lines.  Validation pools are returned untouched for in-loop selection.
    """
    data = make_rows(bundle)
    for pool in ('fit_positive', 'fit_clean'):
        require(all(r['split'] == 'train' for r in data[pool]), 'a non-training row is in a training pool')
    require(not ({r['group_uuid'] for r in data['fit_positive']+data['fit_clean']} & set(excluded)),
            'a validation or test writer is in the training pool')
    if extra is not None:
        train_lines = {r['line_id'] for r in data['fit_positive']+data['fit_clean']}
        for pool in ('fit_positive', 'fit_clean'):
            require(not ({r['line_id'] for r in extra[pool]} & train_lines), 'extra rows collide with training rows')
            data[pool] = data[pool]+sorted(extra[pool], key=lambda r: r['line_id'])
        require(not ({r['group_uuid'] for r in data['fit_positive']+data['fit_clean']} & set(excluded)),
                'a validation or test writer is in the extra rows')
    return data
