"""Behaviour test: training data never yields a row of a validation or test writer."""
import random

import numpy as np
import pytest

from revrender.data import bundle as B
from revrender.data.sampling import BatchStream
from revrender.util import RefusedError


def extra_row(line, writer, positive):
    mask = np.zeros((96, 768), np.uint8)
    if positive:
        mask[20:30, 20:90] = 255
    return {'sample_id': 'extra:'+line, 'line_id': line, 'group_uuid': writer, 'split': 'train', 'is_positive': positive,
            'image': np.full((96, 768), 255, np.uint8), 'mask': mask, 'line_skeleton': np.zeros((96, 768), np.uint8)}


def test_every_batch_comes_from_training_writers_only(bundle):
    path, sha = bundle
    data = B.load_bundle(path, sha)
    held = B.excluded_writers(path)
    extra = {'fit_positive': [extra_row('x:L1', 'x1', True), extra_row('x:L2', 'x1', True)],
             'fit_clean': [extra_row('x:L3', 'x2', False), extra_row('x:L4', 'x2', False)]}
    pools = B.train_pools(data, held, extra)
    allowed = {r['line_id'] for r in data['train']} | {'x:L1', 'x:L2', 'x:L3', 'x:L4'}
    writers = {r['line_id']: r['group_uuid'] for r in pools['fit_positive']+pools['fit_clean']}
    stream = BatchStream(pools, random.Random(42), steps=60, batch_size=8, negative_ratio=1.)
    stream.track = True
    for _ in range(60):
        stream.make_batch()
    drawn = {line for batch in stream.lines for line in batch}
    assert drawn <= allowed
    assert not ({writers[line] for line in drawn} & held)
    assert len(stream.lines) == 60 and all(len(batch) == 8 for batch in stream.lines)
    assert drawn & {'x:L1', 'x:L2', 'x:L3', 'x:L4'}, 'the extra rows are used'


def test_the_threaded_stream_yields_the_same_batches_as_direct_calls(bundle):
    path, sha = bundle
    data = B.load_bundle(path, sha)
    pools = B.train_pools(data, B.excluded_writers(path))
    direct = BatchStream(pools, random.Random(42), steps=5, batch_size=8, negative_ratio=1.)
    direct.track = True
    expected = [direct.make_batch()[0] for _ in range(5)]
    threaded = BatchStream(pools, random.Random(42), steps=5, batch_size=8, negative_ratio=1.)
    threaded.start()
    try:
        got = [threaded.next() for _ in range(5)]
    finally:
        threaded.close()
    for (images, blocks), (e_images, e_blocks) in zip(got, expected):
        assert np.array_equal(images, e_images)
        assert all(np.array_equal(a, b) for a, b in zip(blocks, e_blocks))


@pytest.mark.parametrize('writer', ['d0', 'd1', 't0'])
def test_extra_rows_of_a_validation_or_test_writer_are_refused(bundle, writer):
    path, sha = bundle
    data = B.load_bundle(path, sha)
    extra = {'fit_positive': [extra_row('x:L1', writer, True)], 'fit_clean': []}
    with pytest.raises(RefusedError):
        B.train_pools(data, B.excluded_writers(path), extra)


def test_the_validation_pools_are_the_validation_writers_lines(bundle):
    path, sha = bundle
    data = B.load_bundle(path, sha)
    pools = B.train_pools(data, B.excluded_writers(path))
    dev_ids = {r['line_id'] for r in data['dev']}
    assert {r['line_id'] for r in pools['validation_positive']+pools['validation_clean']} == dev_ids
    assert not (dev_ids & {r['line_id'] for r in pools['fit_positive']+pools['fit_clean']})
