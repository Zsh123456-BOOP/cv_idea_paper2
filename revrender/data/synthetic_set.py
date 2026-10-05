"""The training set of the stroke remover: synthetic revisions on clean lines, drawn once as paper 1's strips, and the pools training
draws from it.

    python -m revrender.data.synthetic_set --corpus corpus.jsonl --bank STRIKE_BANK --pages DIR [--pages DIR ...]
        [--copies 4] [--over 0] [--gold-share .7] [--second .3] [--workers 16] --output-dir OUT

Every recipe of the reader's corpus (``data.reader_corpus``: clean lines of paper 1's training writers -- never a test writer's, nor one
of a writer of the line pack -- a small share of the writers its own validation split) is drawn ``--copies`` times with seeds of its own
(a clean line once): ``data.paste_strikes.compose`` with the bank of real marks (``data.strike_bank``) for ``--gold-share`` of the
strikes, and a paste or a stack gets a second revision with probability ``--second``.  ``--over N`` adds N samples per host line of
one to three of its own characters struck or blacked out where they stand (``paste_strikes``'s ``over``), and an ``over`` may then
be a sample's second revision too.  A sample is letterboxed onto the 96 x 768 strip
the way a line is read (``pages.read_page``: grey, BILINEAR on white) and its revisions' numbers with it (NEAREST): RevRender trains on
one instance per revision, the pixels removing it takes (a struck line's that are not writing, a blackout's all).

OUT/images.npy (N, 96, 768) uint8 grey, OUT/labels.npy (N, 96, 768) uint8 (0 none, k the k-th revision), OUT/records.json (one per
sample: id, host line, writer, split, kinds, styles, revisions), OUT/summary.json and OUT/sheet.png (a sample to look at).  The
strips are student handwriting: OUT stays out of Git.
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from .. import pages as pages_module
from ..util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from . import lines as lines_module
from .gold import CANVAS, letterbox
from .paste_strikes import Bank, compose

SECOND_KINDS = ('paste', 'stack')
OVER_KINDS = ('paste', 'stack', 'over')     # the kinds that take a second revision when --over is given
POOLS = ('fit_positive', 'fit_clean', 'validation_positive', 'validation_clean')
_BANK = None                # each worker's bank of real marks


def sample_seed(recipe_seed, copy):
    return int(hashlib.sha256(f'strokes:{recipe_seed}:{copy}'.encode()).hexdigest()[:8], 16)


def strip(image, labels):
    """(grey strip, numbers strip) of a sample on paper 1's canvas."""
    height, width = CANVAS
    grey = np.asarray(letterbox(image.convert('L'), width, height, 255))
    return grey, np.asarray(letterbox(Image.fromarray(labels), width, height, 0, mask=True))


def draw(recipe, copy, crops, bank, gold_share, second, second_kinds=SECOND_KINDS):
    """One sample of a recipe: (grey strip, numbers strip, record)."""
    rng = random.Random(sample_seed(recipe['seed'], copy))
    host, donors = crops[recipe['line_id']], [crops[d] for d in recipe['donors']]
    made = compose(host, donors, rng, recipe['kind'], bank=bank, gold_share=gold_share)
    kinds, styles = [made['kind']], [made['style']]
    if made['kind'] in second_kinds and (donors or 'over' in second_kinds) and rng.random() < second:
        kind = rng.choice([k for k in second_kinds if donors or k == 'over'])
        again = compose(np.asarray(made['image']), donors, rng, kind, bank=bank, gold_share=gold_share, labels=made['labels'])
        if again['kind'] != 'clean':
            made = again
            kinds.append(again['kind'])
            styles.append(again['style'])
    grey, numbers = strip(made['image'], made['labels'])
    record = {'id': f'{recipe["id"]}~{copy}', 'line_id': recipe['line_id'], 'group_uuid': recipe['writer'], 'split': recipe['split'],
              'kinds': kinds, 'styles': styles, 'revisions': int(len(np.unique(numbers[numbers > 0])))}
    return grey, numbers, record


def _start(bank_dir):
    global _BANK
    _BANK = Bank(bank_dir)


def _writer(task):
    """The samples of one writer's recipes (the writer's pages opened once)."""
    recipes, boxes, folders, copies, over, gold_share, second = task
    needed = sorted({r['line_id'] for r in recipes} | {d for r in recipes for d in r['donors']})
    files = pages_module.page_files(folders, sorted({boxes[i][0] for i in needed}))
    crops, opened = {}, {}
    for line_id in needed:
        image_id, box = boxes[line_id]
        if image_id not in opened:
            with Image.open(files[image_id]) as page:
                opened[image_id] = page.convert('RGB')
        page = opened[image_id]
        crops[line_id] = np.asarray(page.crop(lines_module.crop_box(box, *page.size)))
    out, second_kinds = [], OVER_KINDS if over else SECOND_KINDS
    for recipe in recipes:
        for copy in range(1 if recipe['kind'] == 'clean' else copies):
            out.append(draw(recipe, copy, crops, _BANK, gold_share, second, second_kinds))
        if recipe['kind'] == 'clean':
            for copy in range(over):
                in_place = {**recipe, 'id': recipe['id']+'+over', 'kind': 'over'}
                out.append(draw(in_place, copy, crops, _BANK, gold_share, second, second_kinds))
    return out


def read_set(path, expected_sha=None):
    """(records, images, labels) of a synthetic set, its files checked against its manifest."""
    root = Path(path)
    manifest = json.loads((root/'run_manifest.json').read_text())
    if expected_sha is not None:
        require(digest(root/'run_manifest.json') == expected_sha, 'synthetic set manifest changed')
    for name, sha in manifest['files'].items():
        require(digest(root/name) == sha, f'synthetic set file changed: {name}')
    records = json.loads((root/'records.json').read_text())
    images = np.load(root/'images.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(root/'labels.npy', mmap_mode='r', allow_pickle=False)
    require(images.shape == labels.shape == (len(records), *CANVAS), 'synthetic set arrays do not match its records')
    return records, images, labels


def pools(path, expected_sha, excluded):
    """The four sampling pools of a synthetic set, as ``data.bundle.train_pools`` gives them: a row carries its grey strip, its
    revisions' numbers as the mask (``labelled``: one instance per number, not per connected piece) and those instances.  No row may be
    a writer of ``excluded`` (paper 1's validation and test writers); the set's training and validation writers are apart."""
    records, images, labels = read_set(path, expected_sha)
    data = {name: [] for name in POOLS}
    for i, record in enumerate(records):
        numbers = np.asarray(labels[i])
        instances = np.stack([numbers == k for k in range(1, int(numbers.max())+1)]) if numbers.any() else np.zeros((0, *CANVAS), bool)
        pool = ('fit' if record['split'] == 'train' else 'validation')+('_positive' if len(instances) else '_clean')
        data[pool].append({'sample_id': record['id'], 'line_id': record['id'], 'group_uuid': record['group_uuid'], 'split': record['split'],
                           'legacy_split': pool, 'is_positive': bool(len(instances)), 'image': np.asarray(images[i]), 'mask': numbers,
                           'labelled': True, 'instances': instances, 'line_skeleton': np.zeros(CANVAS, np.uint8)})
    train = {r['group_uuid'] for p in ('fit_positive', 'fit_clean') for r in data[p]}
    held = {r['group_uuid'] for p in ('validation_positive', 'validation_clean') for r in data[p]}
    require(not (train & held), 'a writer is in both the training and the validation split of the synthetic set')
    require(not ((train | held) & set(excluded)), 'a validation or test writer of paper 1 is in the synthetic set')
    require(all(data[p] for p in POOLS), 'a sampling pool of the synthetic set is empty')
    return data


def sheet(images, labels, records, path, count=24):
    """A sample of strips with their revisions tinted, to look at."""
    rng = random.Random(0)
    chosen = rng.sample(range(len(records)), min(count, len(records)))
    tiles = []
    for i in chosen:
        rgb = np.repeat(np.asarray(images[i])[..., None], 3, 2).copy()
        numbers = np.asarray(labels[i])
        for k, colour in ((1, (220, 0, 0)), (2, (0, 90, 220))):
            rgb[numbers == k] = colour
        tiles.append(rgb)
    Image.fromarray(np.concatenate([np.concatenate([t, np.full((4, CANVAS[1], 3), 160, np.uint8)]) for t in tiles])).save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--corpus', type=Path, required=True, help="the reader's corpus (data.reader_corpus's corpus.jsonl)")
    parser.add_argument('--bank', type=Path, required=True, help='the bank of real marks (data.strike_bank)')
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    parser.add_argument('--copies', type=int, default=4, help='samples drawn from each recipe with a revision (a clean one: one)')
    parser.add_argument('--over', type=int, default=0, help='samples per host line of a few of its own characters struck where they stand')
    parser.add_argument('--gold-share', type=float, default=.7, help='share of the strikes that are real marks')
    parser.add_argument('--second', type=float, default=.3, help='chance that a paste or stack gets a second revision')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    recipes = [json.loads(line) for line in args.corpus.read_text(encoding='utf-8').splitlines() if line.strip()]
    require(recipes, 'empty corpus')
    boxes = {r['line_id']: (r['image_id'], r['box_1000']) for r in recipes}
    by_writer = defaultdict(list)
    for r in recipes:
        by_writer[r['writer']].append(r)
    tasks = [(by_writer[w], boxes, args.pages, args.copies, args.over, args.gold_share, args.second) for w in sorted(by_writer)]
    with Pool(args.workers, initializer=_start, initargs=(args.bank,)) as pool:
        drawn = [s for chunk in pool.imap(_writer, tasks) for s in chunk]
    out = make_output_dir(args.output_dir)
    np.save(out/'images.npy', np.stack([g for g, _, _ in drawn]))
    np.save(out/'labels.npy', np.stack([n for _, n, _ in drawn]))
    records = [r for _, _, r in drawn]
    write_json(out/'records.json', records)
    images, labels = np.load(out/'images.npy', mmap_mode='r'), np.load(out/'labels.npy', mmap_mode='r')
    sheet(images, labels, records, out/'sheet.png')
    summary = {'samples': len(records), 'by_split': dict(Counter(r['split'] for r in records)),
               'positive': sum(r['revisions'] > 0 for r in records), 'two_revisions': sum(r['revisions'] >= 2 for r in records),
               'by_kind': dict(Counter(r['kinds'][0] for r in records)), 'by_style': dict(Counter(s for r in records for s in r['styles'] if s)),
               'writers': len({r['group_uuid'] for r in records}), 'lost_in_strip': sum(len(r['kinds']) > r['revisions'] and r['kinds'][0] != 'clean' for r in records)}
    write_json(out/'summary.json', summary)
    files = {name: digest(out/name) for name in ('images.npy', 'labels.npy', 'records.json')}
    write_manifest(out, config=to_jsonable(args), inputs={str(args.corpus): digest(args.corpus), str(args.bank/'run_manifest.json'): digest(args.bank/'run_manifest.json')},
                   files=files, **summary)
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
