"""The prefill model's human data (plan 预填模型 of 2026-10-25): every confirmed region of the annotator's exports, real only.

    python -m revrender.data.prefill_set --export E [--export E ...] --pages DIR [--pages DIR ...] --same-student P \
        --fixed-validation FIXED [--new-export E] --output-dir OUT

A region counts when its pixels are confirmed and certain (``pixel_masks.reviewed_crop``, on the page the export names as its
``image_source``, else the first of ``--pages`` holding it); a student's two ids are one writer.  The
held-out writers are decided by a hash of the writer (``HOLDOUT`` of them, seed 42), so the split never has to be drawn again and a
writer first seen in a later batch falls on the same side forever; the fixed validation's writers never train (their 16 regions are
reported apart).  This set feeds the prefill tool only: by the user's word of 2026-10-23 every annotated region may train it,
the test questions' too, so unlike ``data.review_set`` no split of paper 1 is refused here (a labmate's pages still are,
``pages.page_files``).  OUT holds, as a review set does, ``training_native.*`` (training writers), ``validation_native.*`` (held-out
writers), ``records.json`` and ``summary.json`` (the held-out and fixed writers, every labelled page); ``--new-export``: that
export's regions also as ``new_native.*``, whatever their side (the forward test of the plan's section 4), each row with the
annotator's edit count and seconds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .. import pages as pages_module
from ..util import SEED, digest, make_output_dir, require, write_json, write_manifest
from .pixel_masks import has_uncertainty, reviewed_crop
from .prior_annotations import twins

HOLDOUT = .15


def held_out(writer):
    return int(hashlib.sha256(f'prefill-holdout:{SEED}:{writer}'.encode()).hexdigest()[:8], 16)/16**8 < HOLDOUT


def regions(exports, folders):
    """Every confirmed, certain region of the exports: (export, page id, revision, grey crop, mask)."""
    seen = set(); out = []
    for export in exports:
        data = json.loads(export.read_text()); own = Path(data.get('image_source') or '')
        files = pages_module.page_files([own] if data.get('image_source') and own.is_dir() else folders, sorted(data['pages']))
        for page_id, page in sorted(data['pages'].items()):
            chosen = [r for r in page.get('revisions', []) if r.get('confirmed') and not r.get('deleted') and not has_uncertainty(r)]
            if not chosen: continue
            with Image.open(files[page_id]) as im:
                for r in chosen:
                    key = page_id+'#'+r['id'].split('#')[-1]; require(key not in seen, f'{key}: in two exports'); seen.add(key)
                    image, mask = reviewed_crop(im, r); out.append((export, page_id, r, image, mask))
    return out


def save(out, name, rows):
    arrays = {}
    for k, (_, _, _, image, mask, _) in enumerate(rows): arrays[f'image_{k}'] = image; arrays[f'mask_{k}'] = mask.astype(np.uint8)
    np.savez_compressed(out/f'{name}.npz', **arrays); write_json(out/f'{name}.json', [row for *_, row in rows])


def build(args):
    other = twins(args.same_student); canon = lambda w: min(w, other.get(w, w))
    fixed = {r['group_uuid'] for r in json.loads((args.fixed_validation/'validation_native.json').read_text())}
    found = regions([*args.export, *([args.new_export] if args.new_export else [])], args.pages)
    sides = {'train': [], 'holdout': [], 'new': []}
    for export, page_id, r, image, mask in found:
        writer = canon(page_id.split('_')[0])
        row = {'id': page_id+'#'+r['id'].split('#')[-1], 'group_uuid': writer, 'page': page_id, 'box': r['box'], 'region': r.get('region'),
               'export': str(export), 'ops': r.get('ops'), 'seconds': r.get('seconds'), 'model': r.get('model')}
        item = (export, page_id, r, image, mask, row)
        if args.new_export and export == args.new_export: sides['new'].append(item)
        if writer in fixed: continue
        sides['holdout' if held_out(writer) else 'train'].append(item)
    require(sides['train'] and sides['holdout'], 'a side is empty')
    out = make_output_dir(args.output_dir)
    save(out, 'training_native', sides['train']); save(out, 'validation_native', sides['holdout'])
    if args.new_export: save(out, 'new_native', sides['new'])
    write_json(out/'records.json', [{'id': row['id'], 'group_uuid': row['group_uuid'], 'split': 'train', 'source': 'human_confirmed'}
                                    for *_, row in sides['train']])
    summary = {'training_regions': len(sides['train']), 'held_out_regions': len(sides['holdout']), 'new_regions': len(sides['new']),
               'training_writers': len({row['group_uuid'] for *_, row in sides['train']}),
               'validation_writers': sorted({row['group_uuid'] for *_, row in sides['holdout']}), 'fixed_writers': sorted(fixed),
               'pages': sorted({page for _, page, *_ in found}), 'holdout_share': HOLDOUT, 'skipped_fixed_writers': sum(
                   canon(page.split('_')[0]) in fixed for _, page, *_ in found)}
    write_json(out/'summary.json', summary)
    inputs = {str(p): digest(p) for p in [*args.export, *([args.new_export] if args.new_export else []), args.same_student,
                                          args.fixed_validation/'validation_native.json']}
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items()}, inputs=inputs,
                   **{k: v for k, v in summary.items() if k != 'pages'})
    print(json.dumps({k: v for k, v in summary.items() if k not in ('pages', 'validation_writers', 'fixed_writers')}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export', type=Path, action='append', required=True); p.add_argument('--pages', type=Path, action='append', required=True)
    for name in ('same-student', 'fixed-validation', 'output-dir'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--new-export', type=Path, help="the latest batch: its regions also apart, for the forward test")
    build(p.parse_args(argv))


if __name__ == '__main__':
    main()
