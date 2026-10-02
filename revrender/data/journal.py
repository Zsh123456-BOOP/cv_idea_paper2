"""Training-data options the native baselines of paper 1's journal runs use: a writer fraction and synthetic positives.

  kept_writers(writers, f)     the first ceil(f x n) writers by SHA-256 of SALT + writer: nested for every fraction,
                               so 1/8 is inside 1/4 inside 1/2
  synthetic_positives(...)     public synthetic revisions (Heil et al.'s seven strike styles, HandWritten Blots)
                               drawn on real clean training lines, as sampling-pool rows; validation and test
                               writers and lines are dropped, masks widened by WIDEN px, and lines of more than
                               MAX_COMPONENTS pieces skipped (the 32-query budget cannot hold more once augmentation
                               splits thin strokes)
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..util import digest, require

SALT = 'journal911:'
FRACTIONS = (1., .5, .25, .125)
MAX_COMPONENTS = 8
WIDEN = 1
SYNTHETIC = {'heil7_same_background': 'f74acf26c038d9a5713b904d7b8ce4c8c9155efa30a8665f0a7b23af3a991306',
             'handwritten_blot_same_background': '965c61f1e5102240cfae15170a40707dedfc10fb5a6c3978f3df32dd6372afea'}


def kept_writers(writers, fraction):
    ranked = sorted(set(writers), key=lambda w: hashlib.sha256((SALT+str(w)).encode()).hexdigest())
    return set(ranked[:math.ceil(fraction*len(ranked))])


def synthetic_positives(root, excluded_writers, excluded_lines):
    rows, dropped, crowded = [], {}, {}
    for arm, sha in SYNTHETIC.items():
        source = Path(root)/arm
        require(digest(source/'dataset.npz') == sha, f'{arm}: the prepared arrays changed')
        records = [json.loads(line) for line in (source/'rows.jsonl').read_text().splitlines() if line]
        arrays = np.load(source/'dataset.npz', allow_pickle=False)
        images, masks = arrays['images'], arrays['masks']
        require(images.shape == masks.shape == (len(records), 96, 768), f'{arm}: arrays do not match rows')
        dropped[arm] = crowded[arm] = 0
        for i, r in enumerate(records):
            if str(r['group_uuid']) in excluded_writers or str(r['line_id']) in excluded_lines:
                dropped[arm] += 1
                continue
            mask = np.asarray(masks[i]) > 0
            if not mask.any():
                continue
            mask = ndimage.binary_dilation(mask, np.ones((3, 3), bool), iterations=WIDEN)
            if ndimage.label(mask, np.ones((3, 3), bool))[1] > MAX_COMPONENTS:
                crowded[arm] += 1
                continue
            rows.append({'sample_id': f'synthetic:{arm}:{i}', 'line_id': f'synthetic:{arm}:{i}',
                         'group_uuid': str(r['group_uuid']), 'split': 'train', 'legacy_split': 'fit_positive',
                         'is_positive': True, 'source': arm, 'image': np.asarray(images[i], np.uint8),
                         'mask': mask.astype(np.uint8)*255, 'line_skeleton': np.zeros(mask.shape, np.uint8)})
    return rows, {'rows': len(rows), 'dropped_validation_or_test': dropped, 'dropped_over_components': crowded,
                  'max_components': MAX_COMPONENTS, 'widened_px': WIDEN, 'sha256': SYNTHETIC}
