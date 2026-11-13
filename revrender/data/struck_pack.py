"""Line packs of real struck regions and their struck text, for teaching a VLM to read what was crossed out (pilot of 2026-11-08).

    python -m revrender.data.struck_pack train --export E [--export E ...] --pages DIR [...] --same-student P --exclude-pack EVAL_PACK --output-dir OUT
    python -m revrender.data.struck_pack eval --evaluation EVAL --restorer NAME --output-dir OUT
    python -m revrender.data.struck_pack folds --dataset regions.json --pages DIR [...] --hold-out NAME [...] --output-dir OUT
    python -m revrender.data.struck_pack fold-sets --dataset regions.json --pages DIR [...] --hold-out NAME [...] --output-dir OUT
    python -m revrender.data.struck_pack separated --pack FOLDS_PACK --sets SETS --separator RUN x5 --upstream UP --output-dir OUT
    python -m revrender.data.struck_pack views --pack SEPARATED_PACK --sets SETS --dataset regions.json --pages DIR [...] --hold-out NAME [...] --output-dir OUT

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
``image_source`` when that folder is here, else the first of ``--pages`` holding the page): the pixels were drawn on that image.

``fold-sets``: the same regions and folds as the review sets a separator (a restorer of ``baselines.restoration``) trains on, one per
fold in OUT/fold_K: ``training_native.*`` the other folds' regions (the ``val`` writers apart, as ``validation_native.*``, by which its
versions are chosen) and ``heldout_native.*`` fold K's own, each row with its OCR window; ``summary.json`` names what a student of
teachers (``data.pseudo_regions``) must keep apart for this fold: fold K's writers, the ``val`` writers and the held-out ones, and every
page the dataset labels.  ``separated``: the folds pack again, but
``erased`` and ``map`` now come from those separators, each restoring only the regions of the fold it never trained on: ``erased`` the
restored crop, ``map`` the share of each ink pixel's darkness it took away (0 none, white all).  ``views``: a separated pack again
with three more images of each region, all from what it already holds and the page: ``mild`` and ``strong``, the scan lightened by
the square and by the square root of the map (an erasing that takes little, one that takes much; neither can darken a pixel), and
``context``, the line the region stands in (``CONTEXT`` region heights to each side, half a height above and below) with the erased
crop laid back in and a blue box round the region.  The text is student work: OUT stays out of Git.
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
INK = 200      # grey below which a pixel is ink, as the removal metrics read it
CONTEXT = 6    # the line shown round a region: this many region heights to its left and to its right


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


def chosen_regions(args):
    """(the regions to teach and read, each with its target, fold and val mark; {(source, page): file}; the held-out writers; the counts
    of the regions left out and why)."""
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
    listed = {folder: pages_module.page_files([folder] if folder else args.pages, sorted({r['page'] for r in chosen if source(r) == folder}))
              for folder in {source(r) for r in chosen}}
    chosen = sorted(({**r, 'fold': fold[r['writer']], 'val': validation_writer(r['writer']), 'source': source(r)} for r in chosen), key=lambda r: (r['source'], r['page'], r['id']))
    return chosen, {(folder, page): path for folder, files in listed.items() for page, path in files.items()}, held, skipped


def folds_pack(args):
    chosen, files, held, skipped = chosen_regions(args)
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); rows = []
    for key in sorted({(r['source'], r['page']) for r in chosen}):
        with Image.open(files[key]) as page:
            for r in (r for r in chosen if (r['source'], r['page']) == key):
                images = {name: f'crops/{len(rows):05d}_{name}.png' for name in ('raw', 'erased', 'map')}
                for name, image in zip(images, layered(page, r)): image.save(out/images[name])
                rows.append({'id': r['id'], 'line_id': r['id'], 'kind': 'line', 'writer': r['writer'], 'fold': r['fold'], 'val': r['val'],
                             'readable': bool(r['readable']), 'text': r['text'] if r['readable'] else '', 'target': r['target'], 'images': images,
                             'size': list(image.size), 'boxes': [], 'strike_pixels': r['strike_pixels'], 'mode': r['mode']})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    side = lambda sel: {'regions': len(sel), 'readable': sum(r['readable'] for r in sel), 'unreadable': sum(not r['readable'] for r in sel), 'writers': len({r['writer'] for r in sel})}
    summary = {**side(rows), 'folds': [side([r for r in rows if r['fold'] == k]) for k in range(FOLDS)], 'val': side([r for r in rows if r['val']]),
               'held_out_writers': len(held), **skipped}
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items() if k != 'command'},
                   inputs={str(args.dataset): digest(args.dataset)}, **summary)
    print(json.dumps(summary), flush=True)


def fold_sets(args):
    chosen, files, held, _ = chosen_regions(args); cut = {}
    labelled = sorted({r['page'] for r in json.loads(args.dataset.read_text())})
    for key in sorted({(r['source'], r['page']) for r in chosen}):
        with Image.open(files[key]) as page:
            for r in (r for r in chosen if (r['source'], r['page']) == key):
                bl, bt, br, bb = r['box']
                cut[r['id']] = (*reviewed_crop(page, {'box': r['box'], 'runs': r['runs'], 'region': r['region']}),
                                list(window(r['region'] or [[bl, bt], [br, bt], [br, bb], [bl, bb]], page.size)))
    out = make_output_dir(args.output_dir); summary = []
    for k in range(FOLDS):
        sides = {'training_native': [r for r in chosen if r['fold'] != k and not r['val']], 'validation_native': [r for r in chosen if r['fold'] != k and r['val']],
                 'heldout_native': [r for r in chosen if r['fold'] == k]}
        require(all(sides.values()), f'fold {k}: a side is empty'); folder = out/f'fold_{k}'; folder.mkdir()
        for name, rows in sides.items():
            np.savez_compressed(folder/f'{name}.npz', **{f'{part}_{i}': cut[r['id']][j] for i, r in enumerate(rows) for j, part in enumerate(('image', 'mask'))})
            write_json(folder/f'{name}.json', [{'id': r['id'], 'group_uuid': r['writer'], 'page': r['page'], 'box': r['box'], 'region': r['region'], 'window': cut[r['id']][2]} for r in rows])
        write_json(folder/'records.json', [{'id': r['id'], 'group_uuid': r['writer'], 'split': 'train', 'source': 'human_confirmed'} for r in sides['training_native']])
        summary.append({name: len(rows) for name, rows in sides.items()})
        write_json(folder/'summary.json', {'validation_writers': sorted({r['writer'] for r in chosen if r['fold'] == k or r['val']} | held), 'fixed_writers': [], 'pages': labelled})
        write_manifest(folder, config={'dataset': str(args.dataset), 'fold': k, 'hold_out': args.hold_out}, inputs={str(args.dataset): digest(args.dataset)}, **summary[-1])
    print(json.dumps(summary), flush=True)


def taken_away(image, restored):
    """The share of each ink pixel's darkness a restorer took away (0 where there is no ink), of float images in [0, 1]."""
    return np.where(image < INK/255, np.clip((restored-image)/np.maximum(1-image, 1/255), 0, 1), 0)


def pasted(canvas, part, box, origin):
    """The canvas (a window whose corner on the page is ``origin``) with the part of the page box ``box`` laid on it where they meet."""
    (bl, bt, br, bb), (l, t) = box, origin; h, w = canvas.shape
    x0, y0, x1, y1 = max(bl, l), max(bt, t), min(br, l+w), min(bb, t+h)
    if x0 < x1 and y0 < y1:
        canvas[y0-t:y1-t, x0-l:x1-l] = part[y0-bt:y1-bt, x0-bl:x1-bl]
    return canvas


def separated_pack(args):
    from ..baselines.restoration import restore_native
    from ..baselines.restoration_export import selected_predictor
    from .pixel_masks import region_pixels
    require(len(args.separator) == FOLDS, f'{FOLDS} separators, in the order of the folds')
    source = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); rows = []; inputs = {str(args.pack/'pack.json'): digest(args.pack/'pack.json')}; shares = []
    for k, run in enumerate(args.separator):
        folder = args.sets/f'fold_{k}'; native = json.loads((folder/'heldout_native.json').read_text()); arrays = np.load(folder/'heldout_native.npz')
        predict, checkpoint, _ = selected_predictor(run, args.upstream, None); inputs[str(checkpoint)] = digest(checkpoint)
        for i, n in enumerate(native):
            row = source[n['id']]; require(row['fold'] == k, f"{n['id']}: not of fold {k} in the pack")
            image = arrays[f'image_{i}'].astype(np.float32)/255; restored = np.clip(restore_native(predict, image, None), 0, 1)
            inside = region_pixels(n['region'], n['box']); share = taken_away(image, restored)*inside
            with Image.open(args.pack/row['images']['raw']) as opened: scan = np.asarray(opened.convert('L')).copy()
            origin = n['window'][:2]
            erased = np.where(pasted(np.zeros(scan.shape, bool), inside, n['box'], origin),                     # outside the polygon the scan stays
                              pasted(scan.copy(), np.round(255*restored).astype(np.uint8), n['box'], origin), scan)
            layer = pasted(np.zeros(scan.shape, np.uint8), np.round(255*share).astype(np.uint8), n['box'], origin)
            images = {name: f'crops/{len(rows):05d}_{name}.png' for name in ('raw', 'erased', 'map')}
            for name, array in zip(images, (scan, erased, layer)): Image.fromarray(array).save(out/images[name])
            rows.append({**row, 'images': images}); shares.append(float(share[arrays[f'mask_{i}'] > 0].mean()) if arrays[f'mask_{i}'].any() else None)
    require(len(rows) == len(source), 'the sets and the pack hold different regions')
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    known = [s for s in shares if s is not None]
    write_manifest(out, config={'pack': str(args.pack), 'sets': str(args.sets), 'separators': [str(r) for r in args.separator]}, inputs=inputs, regions=len(rows),
                   strike_taken_away_mean=float(np.mean(known)))
    print(json.dumps({'regions': len(rows), 'strike_taken_away_mean': float(np.mean(known))}), flush=True)


def lightened(scan, share, power):
    """The scan with each pixel's darkness taken away by ``share ** power``: lighter where the map says strike, never darker."""
    x, a = np.asarray(scan, np.float32)/255, (np.asarray(share, np.float32)/255)**power
    return np.round(255*(x+a*(1-x))).astype(np.uint8)


def in_context(page, erased, window):
    """The line a region stands in, as an RGB image: the page round the region's window with the erased crop laid back in and a blue box
    drawn just outside the window."""
    from PIL import ImageDraw
    l, t, r, b = window; h = b-t; width, height = page.size
    left, top, right, bottom = max(0, l-CONTEXT*h), max(0, t-h//2), min(width, r+CONTEXT*h), min(height, b+h//2)
    band = page.convert('L').crop((left, top, right, bottom)); band.paste(Image.fromarray(erased), (l-left, t-top))
    band = band.convert('RGB'); ImageDraw.Draw(band).rectangle((l-left-2, t-top-2, r-left+1, b-top+1), outline=(0, 0, 255), width=2)
    return band


def views_pack(args):
    chosen, files, _, _ = chosen_regions(args); place = {r['id']: (r['source'], r['page']) for r in chosen}
    windows = {n['id']: n['window'] for k in range(FOLDS) for n in json.loads((args.sets/f'fold_{k}/heldout_native.json').read_text())}
    source = json.loads((args.pack/'pack.json').read_text())['rows']
    require({r['id'] for r in source} == set(windows) == set(place), 'the pack, the sets and the dataset hold different regions')
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); rows = []
    for key in sorted(set(place.values())):
        with Image.open(files[key]) as page:
            for row in (r for r in source if place[r['id']] == key):
                opened = {name: np.asarray(Image.open(args.pack/row['images'][name]).convert('L')) for name in ('raw', 'erased', 'map')}
                made = {**{name: Image.fromarray(array) for name, array in opened.items()},
                        'mild': Image.fromarray(lightened(opened['raw'], opened['map'], 2)), 'strong': Image.fromarray(lightened(opened['raw'], opened['map'], .5)),
                        'context': in_context(page, opened['erased'], windows[row['id']])}
                images = {name: f'crops/{len(rows):05d}_{name}.png' for name in made}
                for name, image in made.items(): image.save(out/images[name])
                rows.append({**row, 'images': images})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    write_manifest(out, config={'pack': str(args.pack), 'sets': str(args.sets), 'dataset': str(args.dataset), 'context': CONTEXT},
                   inputs={str(args.pack/'pack.json'): digest(args.pack/'pack.json'), str(args.dataset): digest(args.dataset)}, regions=len(rows))
    print(json.dumps({'regions': len(rows)}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--export', type=Path, action='append', required=True); t.add_argument('--pages', type=Path, action='append', required=True)
    for key in ('same-student', 'exclude-pack', 'output-dir'): t.add_argument('--'+key, type=Path, required=True)
    e = sub.add_parser('eval')
    for key in ('evaluation', 'output-dir'): e.add_argument('--'+key, type=Path, required=True)
    e.add_argument('--restorer', required=True, help='the version of the evaluation whose erased crops sit beside the scan')
    for name in ('folds', 'fold-sets', 'views'):
        f = sub.add_parser(name)
        if name == 'views':
            for key in ('pack', 'sets'): f.add_argument('--'+key, type=Path, required=True)
        f.add_argument('--dataset', type=Path, required=True, help='regions.json of data.struck_dataset'); f.add_argument('--pages', type=Path, action='append', required=True)
        f.add_argument('--hold-out', action='append', default=[], help='part of the name of an export whose regions, and their writers, stay out (the test page)')
        f.add_argument('--output-dir', type=Path, required=True)
    d = sub.add_parser('separated')
    for key in ('pack', 'sets', 'upstream', 'output-dir'): d.add_argument('--'+key, type=Path, required=True)
    d.add_argument('--separator', type=Path, action='append', required=True, help="a fold's separator run (baselines.restoration), given once per fold in order")
    args = p.parse_args(argv)
    {'train': train_pack, 'eval': eval_pack, 'folds': folds_pack, 'fold-sets': fold_sets, 'separated': separated_pack, 'views': views_pack}[args.command](args)


if __name__ == '__main__':
    main()
