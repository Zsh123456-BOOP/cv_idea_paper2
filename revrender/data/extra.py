"""Extra training rows beside the bundle's: the user-corrected rows (paper 1's ``corrected`` data).

The corrected folder holds images.npy / masks.npy / records.json and a run_manifest.json whose ``files`` digests are
checked here.  Rows of any validation or test writer are refused, so what this returns can be added to a training
pool without opening the test split.
"""
import json
from pathlib import Path

import numpy as np

from ..util import digest, require, safe_path, schema_is

CORRECTED_SCHEMA = r'hme-revision-corrected\d+-v1'     # the schema tag of the corrected-rows folders this reads


def extra_rows(path, expected_sha, excluded, schema=CORRECTED_SCHEMA):
    """The folder's rows as sampling-pool rows, and a record of what was read."""
    root = safe_path(path)
    require(digest(root/'run_manifest.json') == expected_sha, 'extra-row manifest changed')
    manifest = json.loads((root/'run_manifest.json').read_text())
    require(schema_is(manifest['schema'], schema) and manifest['test_touched'] is False, 'not a corrected-rows folder')
    for name, sha in manifest['files'].items():
        require(Path(name).name == name and digest(root/name) == sha, 'extra-row file changed: '+name)
    records = json.loads((root/'records.json').read_text())
    images = np.load(root/'images.npy', mmap_mode='r', allow_pickle=False)
    masks = np.load(root/'masks.npy', mmap_mode='r', allow_pickle=False)
    require(images.shape == masks.shape == (len(records), 96, 768), 'extra-row arrays do not match records')
    require(not ({r['group_uuid'] for r in records} & set(excluded)), 'a validation or test writer is in the extra rows')
    rows = {'fit_positive': [], 'fit_clean': []}
    for i, record in enumerate(records):
        mask = np.asarray(masks[i], bool)
        require(bool(mask.any()) == (record['pool'] == 'fit_positive'), 'extra-row pool disagrees with its mask')
        rows[record['pool']].append({'sample_id': 'extra:'+record['line_id'], 'line_id': record['line_id'],
                                     'group_uuid': record['group_uuid'], 'split': 'train',
                                     'legacy_split': record['pool'], 'is_positive': record['pool'] == 'fit_positive',
                                     'source': record['source'], 'image': np.asarray(images[i]),
                                     'mask': mask.astype(np.uint8)*255, 'line_skeleton': np.zeros(mask.shape, np.uint8)})
    return rows, {'manifest_sha256': expected_sha, 'positive': len(rows['fit_positive']), 'clean': len(rows['fit_clean']),
                  'writers': len({r['group_uuid'] for r in records}), 'policy': manifest.get('policy')}
