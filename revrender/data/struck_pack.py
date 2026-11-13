"""Line packs of real struck regions and their struck text, for teaching a VLM to read what was crossed out (pilot of 2026-11-08).

    python -m revrender.data.struck_pack train --export E [--export E ...] --pages DIR [...] --same-student P --exclude-pack EVAL_PACK --output-dir OUT
    python -m revrender.data.struck_pack eval --evaluation EVAL --restorer NAME --output-dir OUT
    python -m revrender.data.struck_pack folds --dataset regions.json --pages DIR [...] --hold-out NAME [...] --output-dir OUT

``train``: every region of the exports whose struck text is confirmed and readable (``eval.revision_boxes.prepare`` leaves something
besides ``?``), cut as the unified evaluation cuts its OCR crops (the region's polygon box and six pixels, on the page the export names
as its ``image_source``, else the first of ``--pages`` holding it).  No writer of ``--exclude-pack`` (a student's two ids one writer)
may train; a tenth of the remaining writers, by hash, are the pack's own validation (``split`` val), the rest ``train``.  ``eval``:
the OCR strikes of a unified evaluation (``eval.unified_restoration``) as a pack whose every row has the scan (``images.raw``) and the
given restorer's erased crop (``images.erased``), so one reading can compare reading the scan with reading after erasing.

``folds``: the regions of the merged dataset (``data.struck_dataset``) whose pixels and struck text are both confirmed, as one pack for
a cross-validation over writers.  The regions of the exports named by ``--hold-out`` (the test page) and every other region of their
writers stay out.  The writers go to ``FOLDS`` folds of about equal regions (``fold``); a tenth of the writers, by hash, are marked
``val`` (a reader taught on the other folds stops by them).  A row is one region: ``readable`` as the annotator said, ``target`` what a
reader is taught to give (the text in the spelling of ``eval.coverage.prepare``, or ``recognize_lines.UNREADABLE``), and three images of
its OCR window from the annotator's own pixels: the scan (``raw``), the scan with the strike pixels white (``erased``) and the layer map
(``map``: white the strike pixels, black the rest).  A region is cut from the page its export showed the annotator (the dataset's
``image_source`` when that folder is here, else the first of ``--pages`` holding the page): the pixels were drawn on that image.  The
text is student work: OUT stays out of Git.
"""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .. import pages as pages_module
from ..util import SEED, digest, make_output_dir, require, write_json, write_manifest
from .pixel_masks import reviewed_crop
from .prior_annotations import twins

VALIDATION = .1
FOLDS = 5


def validation_writer(writer):
    return int(hashlib.sha256(f'struck-reader-val:{SEED}:{writer}'.encode()).hexdigest()[:8], 16)/16**8 < VALIDATION


def window(region, size):
    """The OCR crop of a region: its polygon's box and six pixels, inside the page."""
    points = np.asarray(region); w, h = size
    l, t = np.maximum(0, points.min(0)-6).astype(int); r, b = np.minimum([w, h], points.max(0)+6).astype(int)
    return int(l), int(t), int(r), int(b)


def train_pack(args):
    from ..eval.revision_boxes import prepare
    other = twins(args.same_student); canon = lambda w: min(w, other.get(w, w))
    excluded = {canon(r['line_id'].split('_')[0]) for r in json.loads((args.exclude_pack/'pack.json').read_text())['rows']}
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); rows = []; skipped = {'excluded_writer': 0, 'not_readable': 0}
    for export in args.export:
        data = json.loads(export.read_text()); own = Path(data.get('image_source') or '')
        files = pages_module.page_files([own] if data.get('image_source') and own.is_dir() else args.pages, sorted(data['pages']))
        for page_id, page in sorted(data['pages'].items()):
            writer = canon(page_id.split('_')[0])
            chosen = [r for r in page.get('revisions', []) if r.get('text_confirmed') and not r.get('deleted') and r.get('region')]
            if not chosen: continue
            if writer in excluded: skipped['excluded_writer'] += len(chosen); continue
            with Image.open(files[page_id]) as im:
                grey = im.convert('L')
                for r in chosen:
                    if r.get('unreadable') or not prepare(r.get('text', '')).replace('?', ''): skipped['not_readable'] += 1; continue
                    l, t, rr, b = window(r['region'], grey.size); path = f'crops/{len(rows):05d}.png'; grey.crop((l, t, rr, b)).save(out/path)
                    key = page_id+'#'+r['id'].split('#')[-1]
                    rows.append({'id': key, 'line_id': key, 'split': 'val' if validation_writer(writer) else 'train', 'kind': 'line', 'text': r['text'],
                                 'images': {'raw': path}, 'size': [rr-l, b-t], 'boxes': [], 'writer': writer, 'export': str(export)})
    require(rows, 'no readable struck text'); require(len({r['id'] for r in rows}) == len(rows), 'a region in two exports')
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    summary = {'regions': len(rows), 'train': sum(r['split'] == 'train' for r in rows), 'val': sum(r['split'] == 'val' for r in rows),
               'writers': len({r['writer'] for r in rows}), 'excluded_writers': len(excluded), **skipped}
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items() if k != 'command'},
                   inputs={str(p): digest(p) for p in [*args.export, args.same_student, args.exclude_pack/'pack.json']}, **summary)
    print(json.dumps(summary), flush=True)


def eval_pack(args):
    root = args.evaluation; scans = json.loads((root/'inputs/pack.json').read_text())['rows']
    erased = {r['line_id']: r for r in json.loads((root/'runs'/args.restorer/'pack/pack.json').read_text())['rows']}
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); rows = []
    for k, r in enumerate(scans):
        raw, cleaned = f'crops/{k:05d}_raw.png', f'crops/{k:05d}_erased.png'
        shutil.copyfile(root/'inputs'/r['images']['erased'], out/raw); shutil.copyfile(root/'runs'/args.restorer/'pack'/erased[r['line_id']]['images']['erased'], out/cleaned)
        rows.append({**r, 'id': r['split']+'|'+r['line_id'], 'model': r['split'], 'images': {'raw': raw, 'erased': cleaned}})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    write_manifest(out, config={'evaluation': str(root), 'restorer': args.restorer}, inputs={str(p): digest(p) for p in
                   [root/'inputs/pack.json', root/'runs'/args.restorer/'pack/pack.json']}, rows=len(rows))
    print(json.dumps({'rows': len(rows)}), flush=True)


def fold_of(counts, folds=FOLDS):
    """Writers to folds of about equal regions: in a seeded order, the larger first, each to the fold with the fewest regions so far."""
    order = sorted(counts); random.Random(SEED).shuffle(order); order.sort(key=lambda w: -counts[w])
    filled, out = [0]*folds, {}
    for writer in order:
        k = min(range(folds), key=lambda f: (filled[f], f)); out[writer] = k; filled[k] += counts[writer]
    return out


def layered(page, row):
    """(scan, erased, layer map) of a region's OCR window, from the annotator's pixels: the strike pixels white in ``erased`` and in the
    map, which is black elsewhere."""
    grey = np.asarray(page.convert('L')); bl, bt, br, bb = row['box']
    l, t, r, b = window(row['region'] or [[bl, bt], [br, bt], [br, bb], [bl, bb]], page.size)
    _, mask = reviewed_crop(page, {'box': row['box'], 'runs': row['runs'], 'region': row['region']})
    strike = np.zeros(grey.shape, bool); strike[bt:bb, bl:br] = mask.astype(bool)
    scan = grey[t:b, l:r]; erased = scan.copy(); erased[strike[t:b, l:r]] = 255
    return Image.fromarray(scan), Image.fromarray(erased), Image.fromarray((255*strike[t:b, l:r]).astype(np.uint8))


def folds_pack(args):
    from ..eval.coverage import prepare
    from ..recognize_lines import UNREADABLE
    regions = json.loads(args.dataset.read_text())
    held = {r['writer'] for r in regions if any(name in r['export'] for name in args.hold_out)}
    skipped = {'held_out_writer': 0, 'pixels_not_confirmed': 0, 'text_not_confirmed': 0, 'no_text': 0}; chosen = []
    for r in regions:
        target = prepare(r['text'] or '').replace('?', '') if r['readable'] else ''
        reason = ('held_out_writer' if r['writer'] in held else 'pixels_not_confirmed' if not r['pixels_confirmed'] or r['pixels_uncertain'] or r['not_revision'] or not r['runs']
                  else 'text_not_confirmed' if not r['text_confirmed'] or r['readable'] is None else 'no_text' if r['readable'] and not target else None)
        if reason: skipped[reason] += 1
        else: chosen.append({**r, 'target': target if r['readable'] else UNREADABLE})
    require(chosen, 'no region to teach or read')
    fold = fold_of({w: sum(r['writer'] == w for r in chosen) for w in {r['writer'] for r in chosen}})
    source = lambda r: r['image_source'] if r['image_source'] and Path(r['image_source']).is_dir() else ''
    files = {folder: pages_module.page_files([folder] if folder else args.pages, sorted({r['page'] for r in chosen if source(r) == folder}))
             for folder in {source(r) for r in chosen}}
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); rows = []
    for folder, page_id in sorted({(source(r), r['page']) for r in chosen}):
        with Image.open(files[folder][page_id]) as page:
            for r in sorted((r for r in chosen if (source(r), r['page']) == (folder, page_id)), key=lambda r: r['id']):
                images = {name: f'crops/{len(rows):05d}_{name}.png' for name in ('raw', 'erased', 'map')}
                for name, image in zip(images, layered(page, r)): image.save(out/images[name])
                rows.append({'id': r['id'], 'line_id': r['id'], 'kind': 'line', 'writer': r['writer'], 'fold': fold[r['writer']], 'val': validation_writer(r['writer']),
                             'readable': bool(r['readable']), 'text': r['text'] if r['readable'] else '', 'target': r['target'], 'images': images,
                             'size': list(image.size), 'boxes': [], 'strike_pixels': r['strike_pixels'], 'mode': r['mode']})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    side = lambda sel: {'regions': len(sel), 'readable': sum(r['readable'] for r in sel), 'unreadable': sum(not r['readable'] for r in sel), 'writers': len({r['writer'] for r in sel})}
    summary = {**side(rows), 'folds': [side([r for r in rows if r['fold'] == k]) for k in range(FOLDS)], 'val': side([r for r in rows if r['val']]),
               'held_out_writers': len(held), **skipped}
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items() if k != 'command'},
                   inputs={str(args.dataset): digest(args.dataset)}, **summary)
    print(json.dumps(summary), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--export', type=Path, action='append', required=True); t.add_argument('--pages', type=Path, action='append', required=True)
    for key in ('same-student', 'exclude-pack', 'output-dir'): t.add_argument('--'+key, type=Path, required=True)
    e = sub.add_parser('eval')
    for key in ('evaluation', 'output-dir'): e.add_argument('--'+key, type=Path, required=True)
    e.add_argument('--restorer', required=True, help='the version of the evaluation whose erased crops sit beside the scan')
    f = sub.add_parser('folds')
    f.add_argument('--dataset', type=Path, required=True, help='regions.json of data.struck_dataset'); f.add_argument('--pages', type=Path, action='append', required=True)
    f.add_argument('--hold-out', action='append', default=[], help='part of the name of an export whose regions, and their writers, stay out (the test page)')
    f.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv); {'train': train_pack, 'eval': eval_pack, 'folds': folds_pack}[args.command](args)


if __name__ == '__main__':
    main()
