"""Variants of a prefill set's native regions (``data.prefill_set``, ``data.pseudo_regions combine``) for the trials after round 2
(plan of 2026-11-16).

    python -m revrender.data.prefill_variants context --set DIR --pages DIR [--pages DIR ...] --output-dir OUT
    python -m revrender.data.prefill_variants subset --set DIR --share 0.25 --output-dir OUT

``context`` (trial C): each region cut again from its page with ``CONTEXT`` of its box's height on every side (at least ``LEAST`` pixels,
up to the page's edge), the page as it is round the box; the pixels to remove, the teachers' target and the loss weight placed in that
window, the weight 0 outside the region (the box, and inside it the polygon), and ``inner_<k>`` the region, inside which alone a model is
scored (``baselines.restoration.inside``).  Every native file of the set (training, validation, new) is cut so; a region whose crop
does not match its page inside the region is refused.  ``subset`` (trial A): the set's training regions of the first writers in the
order of the held-out hash (``prefill_set.held_out``'s), as many as bring the regions nearest ``--share`` of them; a smaller share's
writers are a prefix of a larger one's.  The validation files are copied as they are.
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
from .pixel_masks import region_pixels

CONTEXT = .5
LEAST = 16
NATIVE = ('training_native', 'validation_native', 'new_native')


def window(box, size):
    l, t, r, b = box; w, h = size; m = max(LEAST, round(CONTEXT*(b-t)))
    return max(0, l-m), max(0, t-m), min(w, r+m), min(h, b+m)


def with_context(grey, row, arrays, k):
    """The region's arrays in its window: image, mask, weight, inner (and target, when the set has one)."""
    l, t, r, b = row['box']; L, T, R, B = window(row['box'], grey.shape[::-1]); y, x = t-T, l-L
    inner_box = region_pixels(row.get('region'), row['box']) if row.get('region') else np.ones((b-t, r-l), bool)
    image = grey[T:B, L:R].copy(); crop = arrays[f'image_{k}']
    require(crop.shape == (b-t, r-l) and np.abs(crop.astype(int)-grey[t:b, l:r])[inner_box].mean() < 1, f"{row['id']}: its crop is not its page's")
    def place(a, fill, dtype):
        z = np.full(image.shape, fill, dtype); z[y:y+b-t, x:x+r-l] = a; return z
    weight = arrays[f'weight_{k}'] if f'weight_{k}' in arrays.files else np.ones(crop.shape, np.uint8)
    out = {'image': image, 'mask': place(arrays[f'mask_{k}'].astype(np.uint8) * inner_box, 0, np.uint8),
           'weight': place(weight.astype(np.uint8) * inner_box, 0, np.uint8), 'inner': place(inner_box.astype(np.uint8), 0, np.uint8)}
    if f'target_{k}' in arrays.files:
        target = image.copy(); target[y:y+b-t, x:x+r-l] = np.where(inner_box, arrays[f'target_{k}'], grey[t:b, l:r]); out['target'] = target
    return out, [L, T, R, B]


def context(args):
    out = make_output_dir(args.output_dir); counts = {}; inputs = {}
    for name in NATIVE:
        if not (args.set/f'{name}.json').exists(): continue
        rows = json.loads((args.set/f'{name}.json').read_text()); arrays = np.load(args.set/f'{name}.npz')
        files = pages_module.page_files(args.pages, sorted({r['page'] for r in rows})); made = {}; placed = []
        for page in sorted({r['page'] for r in rows}):
            with Image.open(files[page]) as im: grey = np.asarray(im.convert('L'))
            for k, r in ((k, r) for k, r in enumerate(rows) if r['page'] == page):
                value, box = with_context(grey, r, arrays, k)
                made.update({f'{key}_{k}': v for key, v in value.items()}); placed.append((k, {**r, 'window': box}))
        np.savez_compressed(out/f'{name}.npz', **made); write_json(out/f'{name}.json', [r for _, r in sorted(placed, key=lambda p: p[0])])
        counts[name] = len(rows); inputs[str(args.set/f'{name}.npz')] = digest(args.set/f'{name}.npz')
    for name in ('records.json', 'summary.json'):
        if (args.set/name).exists(): shutil.copy(args.set/name, out/name)
    write_manifest(out, config={'set': str(args.set), 'pages': [str(p) for p in args.pages], 'context': CONTEXT, 'least': LEAST}, inputs=inputs, regions=counts)
    print(json.dumps(counts), flush=True)


def order(writer):
    return int(hashlib.sha256(f'prefill-holdout:{SEED}:{writer}'.encode()).hexdigest()[:8], 16)


def subset(args):
    rows = json.loads((args.set/'training_native.json').read_text()); arrays = np.load(args.set/'training_native.npz')
    writers = sorted({r['group_uuid'] for r in rows}, key=order); want = args.share*len(rows); best, total = (abs(want), 0), 0
    for n, w in enumerate(writers, 1):
        total += sum(r['group_uuid'] == w for r in rows)
        if abs(total-want) < best[0]: best = (abs(total-want), n)
    chosen = set(writers[:best[1]]); keep = [k for k, r in enumerate(rows) if r['group_uuid'] in chosen]
    out = make_output_dir(args.output_dir)
    np.savez_compressed(out/'training_native.npz', **{f'{key}_{i}': arrays[f'{key}_{k}'] for i, k in enumerate(keep)
                                                      for key in ('image', 'mask', 'weight', 'target') if f'{key}_{k}' in arrays.files})
    write_json(out/'training_native.json', [rows[k] for k in keep])
    for name in ('validation_native.json', 'validation_native.npz', 'summary.json'): shutil.copy(args.set/name, out/name)
    write_json(out/'records.json', [{'id': rows[k]['id'], 'group_uuid': rows[k]['group_uuid'], 'split': 'train', 'source': 'human_confirmed'} for k in keep])
    summary = {'share': args.share, 'regions': len(keep), 'of': len(rows), 'writers': len(chosen), 'of_writers': len(writers)}
    write_manifest(out, config={'set': str(args.set), 'share': args.share}, inputs={str(args.set/'training_native.npz'): digest(args.set/'training_native.npz')}, **summary)
    print(json.dumps(summary), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('context')
    for key in ('set', 'output-dir'): c.add_argument('--'+key, type=Path, required=True)
    c.add_argument('--pages', type=Path, action='append', required=True)
    s = sub.add_parser('subset')
    for key in ('set', 'output-dir'): s.add_argument('--'+key, type=Path, required=True)
    s.add_argument('--share', type=float, required=True)
    args = p.parse_args(argv); {'context': context, 'subset': subset}[args.command](args)


if __name__ == '__main__':
    main()
