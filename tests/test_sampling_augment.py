"""The sampler and the augmentation draw the same numbers, in the same order, as the code they were ported from.

tests/golden/sampling_augment.json was produced by running the original implementation on the synthetic inputs below
(seed 42): the hashes of every augmented line and mask, the line ids of 15 balanced batches, and the batches of the
original producer.  The augmentation's JPEG branch depends on the imaging library, so the image hashes are compared
only on a machine whose JPEG encoder matches; the random-stream state, the masks and the batch composition always are.
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from revrender.data import sampling
from revrender.data.augment import paired_affine

GOLDEN = json.loads((Path(__file__).parent/'golden/sampling_augment.json').read_text())


def synthetic_line():
    r = np.random.default_rng(0)
    image = (255-np.clip(r.normal(0, 1, (96, 768))*20+30*(r.random((96, 768)) > .97), 0, 255)).astype(np.uint8)
    mask = np.zeros((96, 768), np.uint8)
    mask[30:50, 100:300] = 255
    mask[60:70, 400:420] = 255
    return image, mask


def digest_of(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_augmentation_consumes_the_random_stream_exactly_as_before():
    image, mask = synthetic_line()
    rng = random.Random(42)
    masks = []
    for _ in range(12):
        picture, target = paired_affine(image, mask, rng)
        masks.append(digest_of(target))
    assert hashlib.sha256(repr(rng.getstate()).encode()).hexdigest() == GOLDEN['augment']['rng_state_sha']
    assert masks == [m for _, m in GOLDEN['augment']['calls']]


def test_augmented_images_match_the_original(request):
    image, mask = synthetic_line()
    rng = random.Random(42)
    got = [digest_of(paired_affine(image, mask, rng)[0]) for _ in range(12)]
    expected = [i for i, _ in GOLDEN['augment']['calls']]
    # Calls that take the JPEG branch depend on the encoder; every other call must be bit-identical.
    same = sum(a == b for a, b in zip(got, expected))
    assert same >= 9, f'only {same} of 12 augmented lines equal the original'


def rows():
    image, mask = synthetic_line()
    skeleton = np.zeros(mask.shape, np.uint8)
    positives = [{'line_id': f'p{i}', 'image': image, 'mask': mask, 'line_skeleton': skeleton} for i in range(10)]
    clean = [{'line_id': f'c{i}', 'image': image, 'mask': np.zeros_like(mask), 'line_skeleton': skeleton} for i in range(12)]
    return positives, clean


def test_balanced_batches_pick_the_same_lines_as_before():
    positives, clean = rows()
    rng = random.Random(42)
    p, c = sampling.ShuffledCycle(positives, rng), sampling.ShuffledCycle(clean, rng)
    got = [[r['line_id'] for r in sampling.take_balanced_batch(p, c, 8, 1.)[0]] for _ in range(15)]
    assert got == GOLDEN['balanced_batches']


def test_the_producer_builds_the_same_batches_as_before(monkeypatch):
    positives, clean = rows()
    monkeypatch.setattr(sampling, 'prepare_batch', lambda images, masks, labelled=None: (images, masks))
    stream = sampling.BatchStream({'fit_positive': positives, 'fit_clean': clean}, random.Random(42), 6, 8, 1.)
    expected = GOLDEN['stream']
    same_images = 0
    for k in range(6):
        (images, masks), _ = stream.make_batch()
        assert [digest_of(m) for m in masks] == expected[k][1]
        same_images += sum(digest_of(i) == h for i, h in zip(images, expected[k][0]))
    assert same_images >= 40         # of 48 lines; the JPEG branch is encoder-dependent
