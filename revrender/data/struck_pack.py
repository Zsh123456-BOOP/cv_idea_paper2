"""Line packs of real struck regions and their struck text, for teaching a VLM to read what was crossed out (pilot of 2026-11-08).

    python -m revrender.data.struck_pack train --export E [--export E ...] --pages DIR [...] --same-student P --exclude-pack EVAL_PACK --output-dir OUT
    python -m revrender.data.struck_pack eval --evaluation EVAL --restorer NAME --output-dir OUT

``train``: every region of the exports whose struck text is confirmed and readable (``eval.revision_boxes.prepare`` leaves something
besides ``?``), cut as the unified evaluation cuts its OCR crops (the region's polygon box and six pixels, on the page the export names
as its ``image_source``, else the first of ``--pages`` holding it).  No writer of ``--exclude-pack`` (a student's two ids one writer)
may train; a tenth of the remaining writers, by hash, are the pack's own validation (``split`` val), the rest ``train``.  ``eval``:
the OCR strikes of a unified evaluation (``eval.unified_restoration``) as a pack whose every row has the scan (``images.raw``) and the
given restorer's erased crop (``images.erased``), so one reading can compare reading the scan with reading after erasing.  The text
is student work: OUT stays out of Git.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .. import pages as pages_module
from ..util import SEED, digest, make_output_dir, require, write_json, write_manifest
from .prior_annotations import twins

VALIDATION = .1


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--export', type=Path, action='append', required=True); t.add_argument('--pages', type=Path, action='append', required=True)
    for key in ('same-student', 'exclude-pack', 'output-dir'): t.add_argument('--'+key, type=Path, required=True)
    e = sub.add_parser('eval')
    for key in ('evaluation', 'output-dir'): e.add_argument('--'+key, type=Path, required=True)
    e.add_argument('--restorer', required=True, help='the version of the evaluation whose erased crops sit beside the scan')
    args = p.parse_args(argv); {'train': train_pack, 'eval': eval_pack}[args.command](args)


if __name__ == '__main__':
    main()
