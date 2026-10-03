"""The paper-1 gold put back on the whole-question page: where each strip component lands, and which rows are read."""
import json

import numpy as np
import pytest
from PIL import Image

from revrender import pages
from revrender.data import bundle as B
from revrender.data import lines as L
from revrender.data import page_gold
from revrender.data.gold import unletterbox

BOXES = ([50, 50, 950, 300], [50, 350, 950, 600], [50, 650, 950, 900])
SIZE = (400, 300)


def catalogue_and_pages(tmp_path, path, sha):
    """A catalogue that puts the three lines of every writer one under the other, and a blank page for each writer."""
    data = B.load_bundle(path, sha)
    entries, ids = [], set()
    for row in data['train']+data['dev']:
        entries.append({'image_id': row['image_id'], 'representative_line_id': row['line_id'], 'alias_line_ids': [],
                        'box_1000': BOXES[int(row['line_id'][-1])%3]})
        ids.add(row['image_id'])
    catalogue = tmp_path/'catalogue.jsonl'
    catalogue.write_text(''.join(json.dumps(e)+'\n' for e in entries))
    images = tmp_path/'images'
    images.mkdir()
    for image_id in ids:
        Image.new('L', SIZE, 255).save(images/f'{image_id}.jpg')
    return data, catalogue, images


def run(tmp_path, bundle, extra=(), name='gold'):
    path, sha = bundle
    data, catalogue, images = catalogue_and_pages(tmp_path, path, sha) if not (tmp_path/'catalogue.jsonl').exists() else (
        B.load_bundle(path, sha), tmp_path/'catalogue.jsonl', tmp_path/'images')
    out = tmp_path/name
    assert page_gold.main(['--bundle', str(path), '--bundle-sha', sha, '--catalogue', str(catalogue), '--pages', str(images), *extra,
                           '--output-dir', str(out)]) == 0
    return data, out


def test_a_gold_component_lands_on_its_line_box_as_the_strip_resized_back(tmp_path, bundle):
    data, out = run(tmp_path, bundle)
    ids = sorted(p.stem for p in out.glob('*.json') if p.name != 'run_manifest.json')
    assert ids == ['d0_1_1', 'd1_1_1']                                      # the default is the validation split
    record, instances = pages.read_output(out, 'd0_1_1')
    assert record['size'] == list(SIZE) and [l['source'] for l in record['lines']] == ['gold']*3
    box = list(L.crop_box(BOXES[0], *SIZE))
    assert record['lines'][0]['box'] == box
    rows = sorted((r for r in data['dev'] if r['image_id'] == 'd0_1_1'), key=lambda r: r['line_id'])
    first = [i for i in instances if i['line'] == 0]
    assert len(first) == len(rows[0]['instances']) > 0 and all(i['box'] == box and i['score'] == 1. for i in first)
    for instance, component in zip(first, rows[0]['instances']):
        assert np.array_equal(instance['mask'], unletterbox(component, box[2]-box[0], box[3]-box[1]))
    assert [i['line'] for i in instances if i['line'] == 2] == []             # the third line of a validation writer is clean


def test_the_training_rows_are_read_only_when_asked_for(tmp_path, bundle):
    _, out = run(tmp_path, bundle, ['--split', 'train', '--split', 'dev'], name='both')
    ids = sorted(p.stem for p in out.glob('*.json') if p.name != 'run_manifest.json')
    assert ids == ['d0_1_1', 'd1_1_1', 'w0_1_1', 'w1_1_1', 'w2_1_1', 'w3_1_1']
    manifest = json.loads((out/'run_manifest.json').read_text())
    assert manifest['splits'] == ['train', 'dev'] and manifest['pages'] == 6 and manifest['lines'] == 4*4+2*3


def test_a_line_that_is_not_in_the_catalogue_is_refused(tmp_path, bundle):
    path, sha = bundle
    data, catalogue, images = catalogue_and_pages(tmp_path, path, sha)
    kept = catalogue.read_text().splitlines()[:-1]
    catalogue.write_text('\n'.join(kept)+'\n')
    with pytest.raises(Exception, match='not in the catalogue'):
        page_gold.main(['--bundle', str(path), '--bundle-sha', sha, '--catalogue', str(catalogue), '--pages', str(images),
                        '--split', 'train', '--split', 'dev', '--output-dir', str(tmp_path/'gold')])
