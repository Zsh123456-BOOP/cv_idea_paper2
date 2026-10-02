"""What the revision-specific baselines (component classifier, strike-removal GAN, shape-feature methods, MathScrub)
share: reading a model's instances back as an archive, the operating point read from it, and its freeze record.

A baseline delivers, per validation line, candidate masks with scores.  They are archived one npz per line (the format of
the native baselines: sample_id, shape, packed masks, scores) so every evaluation reads them as it reads any model;
the model's operating point (mask NMS and score cut) is then chosen on validation by the frontier of eval.matching,
best F1 with ties to fewer false positives, as for every model.
"""
import json
from pathlib import Path

import numpy as np

from ..data.bundle import excluded_writers, load_bundle
from ..data.extra import extra_rows
from ..eval import matching
from ..eval.freeze import BUDGETS, SCHEMA, budget_table
from ..util import require

CANVAS = (96, 768)


def write_line_archive(directory, index, sample_id, masks, scores, canvas=CANVAS):
    """One line's candidates: ``masks`` a list of bool H x W arrays, ``scores`` their scores."""
    stack = np.asarray(masks, bool).reshape(len(masks), *canvas)
    np.savez_compressed(Path(directory)/f'{index:04d}.npz', sample_id=np.asarray(sample_id),
                        scores=np.asarray(scores, np.float64), shape=np.asarray(stack.shape, np.int64),
                        masks_packed=np.packbits(stack.reshape(len(stack), -1), axis=1) if len(stack)
                        else np.zeros((0, canvas[0]*canvas[1]//8), np.uint8))


def read_archive(archive, order, truths):
    """(frontier points, the best point with its pixel metrics) of an archive on the validation lines."""
    provide, available = matching.native_provider(archive)
    require(available == set(order), 'validation archive coverage mismatch')
    gold = sum(map(len, truths))
    points = matching.frontier(provide, order, truths, gold)
    point = matching.best_point(points)
    point.update(matching.pixel_iou(provide, order, truths, point), mask_threshold=.5)
    return points, point


def freeze_record(model, bundle_sha, archive, point, truths, points, **fields):
    """A validation freeze (schema revrender-freeze-v2) of a model that delivers whole components: its
    ``mask_threshold`` is kept at .5 for the readers that expect one."""
    return {'schema': SCHEMA, 'split': 'validation', 'model': model, 'test_touched': False, 'bundle_sha256': bundle_sha,
            'validation_archive': str(Path(archive).resolve()), 'mask_threshold': .5,
            'mask_threshold_rule': 'not applicable: a candidate is delivered whole; the field is kept for the readers',
            'score_threshold': point['score_threshold'], 'nms': point['nms'],
            'validation': {'gold_components': sum(map(len, truths)), 'own_best': point,
                           'tp_within_fp_budget': budget_table(points, BUDGETS)}, **fields}


def training_and_validation(bundle_dir, bundle_sha, corrected_dir, corrected_sha, with_corrected=True):
    """The bundle, its validation rows in evaluation order (revised first), and the training rows: the bundle's training
    split plus the user-corrected rows (a list of {'sample_id','image'(96x768 uint8),'mask'(bool),'group_uuid'})."""
    bundle = load_bundle(bundle_dir, bundle_sha)
    require(not any(r.get('split') == 'test' for r in bundle['train']+bundle['dev']), 'a test row reached the data')
    train = [{'sample_id': r['sample_id'], 'group_uuid': str(r['group_uuid']), 'image': np.asarray(r['image'][0]),
              'mask': np.asarray(r['mask'], bool)} for r in bundle['train']]
    info = None
    if with_corrected:
        rows, info = extra_rows(corrected_dir, corrected_sha, excluded_writers(bundle_dir))
        train += [{'sample_id': r['sample_id'], 'group_uuid': str(r['group_uuid']), 'image': np.asarray(r['image'], np.uint8),
                   'mask': np.asarray(r['mask']) > 0} for r in rows['fit_positive']+rows['fit_clean']]
    dev, order, truths = matching.validation_order(bundle['dev'])
    validation = [{'sample_id': r['sample_id'], 'group_uuid': str(r['group_uuid']), 'image': np.asarray(r['image'][0]),
                   'mask': np.asarray(r['mask'], bool)} for r in dev]
    return bundle, train, validation, order, truths, info


def write_text_json(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, indent=1, sort_keys=True)+'\n')
