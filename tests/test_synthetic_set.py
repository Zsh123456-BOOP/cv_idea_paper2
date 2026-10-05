"""The stroke remover's training set: synthetic revisions drawn as paper 1's strips, one instance per revision, writers kept apart."""
import json

import numpy as np
import pytest
from PIL import Image

from revrender import train
from revrender.data import sampling
from revrender.data import synthetic_set as T


def write_inputs(tmp_path):
    """Two writers' pages (three word lines each), a corpus over them (writer b the corpus's validation writer) and a bank."""
    scans = tmp_path/'scans'
    scans.mkdir()
    page = np.full((300, 600, 3), 255, np.uint8)
    for row in range(3):
        for x in (40, 160, 300):
            page[30+row*90:60+row*90, x:x+80] = 20
    recipes = []
    for writer, split in (('a', 'train'), ('b', 'val')):
        Image.fromarray(page).save(scans/f'{writer}_5_1.jpg', quality=95)
        lines = [f'{writer}_5_1:{k}' for k in range(3)]
        for k, line_id in enumerate(lines):
            box = [0, round(1000*(15+k*90)/300), 1000, round(1000*(75+k*90)/300)]
            for n, kind in enumerate(('clean', 'whole', 'paste', 'stack')):
                recipes.append({'id': f'{line_id}#{n}', 'split': split, 'kind': kind, 'line_id': line_id, 'image_id': f'{writer}_5_1',
                                'writer': writer, 'box_1000': box, 'donors': [d for d in lines if d != line_id] if kind in ('paste', 'stack') else [],
                                'seed': 1000*k+n})
    corpus = tmp_path/'corpus.jsonl'
    corpus.write_text('\n'.join(json.dumps(r) for r in recipes))
    bank = tmp_path/'bank'
    bank.mkdir()
    stroke = np.full((6, 90), 255, np.uint8)
    stroke[2:4] = 40
    np.savez_compressed(bank/'bank.npz', piece_0=stroke, piece_1=np.full((20, 20), 30, np.uint8))
    (bank/'pieces.json').write_text(json.dumps([{'id': 'piece_0', 'kind': 'line', 'size': [90, 6], 'line_height': 40, 'ink_level': 40},
                                                {'id': 'piece_1', 'kind': 'blot', 'size': [20, 20], 'line_height': 40, 'ink_level': 30}]))
    (bank/'run_manifest.json').write_text('{}')
    return scans, corpus, bank


def test_the_set_is_strips_with_numbered_revisions_and_its_pools_keep_writers_apart(tmp_path):
    scans, corpus, bank = write_inputs(tmp_path)
    out = tmp_path/'set'
    assert T.main(['--corpus', str(corpus), '--bank', str(bank), '--pages', str(scans), '--copies', '2', '--second', '1', '--workers', '2',
                   '--output-dir', str(out)]) == 0
    records, images, labels = T.read_set(out)
    assert len(records) == 2*3*(1+3*2) and images.shape == labels.shape == (len(records), 96, 768)
    assert {r['split'] for r in records} == {'train', 'val'} and any(r['revisions'] == 2 for r in records)
    clean = [i for i, r in enumerate(records) if r['kinds'] == ['clean']]
    assert clean and not labels[clean].any()
    data = T.pools(out, None, excluded={'z'})
    assert all(r['labelled'] and r['image'].shape == (96, 768) for p in T.POOLS for r in data[p])
    two = next(r for r in data['fit_positive'] if len(r['instances']) == 2)
    assert (two['instances'][0] == (two['mask'] == 1)).all() and (two['instances'][1] == (two['mask'] == 2)).all()
    with pytest.raises(Exception, match='validation or test writer'):
        T.pools(out, None, excluded={'b'})
    with pytest.raises(Exception, match='changed'):
        T.read_set(out, expected_sha='0'*64)
    more = tmp_path/'over'
    assert T.main(['--corpus', str(corpus), '--bank', str(bank), '--pages', str(scans), '--copies', '1', '--over', '2', '--second', '0', '--workers', '1',
                   '--output-dir', str(more)]) == 0
    records, _, _ = T.read_set(more)
    assert sum(r['kinds'][0] == 'over' for r in records) == 2*6 and len(records) == 6*(1+3+2)


def test_a_labelled_line_trains_one_instance_per_revision_however_many_pieces_it_has():
    numbers = np.zeros((96, 768), np.uint8)
    numbers[40:44, 10:30] = 1
    numbers[40:44, 50:70] = 1                 # one struck line broken by the letter it crosses
    numbers[60:70, 200:220] = 2
    gold = (numbers > 0).astype(np.uint8)*255
    _, blocks = sampling.prepare_batch([np.zeros((96, 768), np.uint8)]*2, [numbers, gold], [True, False])
    assert len(blocks[0]) == 2 and blocks[0][0].sum() == 160 and len(blocks[1]) == 3


def test_training_takes_a_bundle_or_a_synthetic_set_with_the_mapping():
    base = ['--stage', 'smoke', '--backbone-weights', 'w.pkl', '--output-dir', 'out']
    assert train.parse(base+['--synthetic', 's', '--synthetic-sha', 'x', '--mapping', 'm.json']).synthetic is not None
    for wrong in (['--synthetic', 's', '--synthetic-sha', 'x'], ['--bundle', 'b', '--bundle-sha', 'y', '--synthetic', 's', '--synthetic-sha', 'x', '--mapping', 'm'],
                  ['--synthetic', 's', '--mapping', 'm'], []):
        with pytest.raises(SystemExit):
            train.parse(base+wrong)
    assert train.grey({'image': np.zeros((3, 96, 768))}).shape == (96, 768) and train.grey({'image': np.zeros((96, 768))}).shape == (96, 768)


def test_a_mixed_batch_takes_its_share_of_revised_and_clean_lines_from_the_real_pools():
    import random
    row = lambda name, k: {'line_id': f'{name}{k}', 'image': np.zeros((96, 768), np.uint8), 'mask': np.zeros((96, 768), np.uint8)}
    cycles = [sampling.ShuffledCycle([row(n, k) for k in range(10)], random.Random(1)) for n in ('sp', 'sc', 'rp', 'rc')]
    rows = sampling.take_mixed_batch(cycles, 8, 1., .5)
    assert [r['line_id'][:2] for r in rows] == ['sp', 'sp', 'rp', 'rp', 'sc', 'sc', 'rc', 'rc']
    rows = sampling.take_mixed_batch(cycles, 8, 1., .25)
    assert sum(r['line_id'].startswith('r') for r in rows) == 2


def test_mixing_needs_both_sources_and_a_share():
    base = ['--stage', 'smoke', '--backbone-weights', 'w.pkl', '--output-dir', 'out', '--synthetic', 's', '--synthetic-sha', 'x', '--mapping', 'm']
    mixed = train.parse(base+['--bundle', 'b', '--bundle-sha', 'y', '--real-share', '.5', '--corrected-bundle', 'c', '--corrected-sha', 'z', '--init', 'p.pt'])
    assert mixed.real_share == .5 and mixed.init is not None
    for wrong in (['--bundle', 'b', '--bundle-sha', 'y'], ['--real-share', '.5'], ['--bundle', 'b', '--bundle-sha', 'y', '--real-share', '1.5'],
                  ['--corrected-bundle', 'c', '--corrected-sha', 'z']):
        with pytest.raises(SystemExit):
            train.parse(base+wrong)
