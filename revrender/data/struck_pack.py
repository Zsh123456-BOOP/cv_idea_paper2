"""Packs of real struck regions and their struck text, for teaching and testing readers of what was crossed out.

    python -m revrender.data.struck_pack folds --dataset regions.json --pages DIR [...] --hold-out NAME [...] --output-dir OUT
    python -m revrender.data.struck_pack fold-sets --dataset regions.json --pages DIR [...] --hold-out NAME [...] --output-dir OUT
    python -m revrender.data.struck_pack separated --pack FOLDS_PACK --sets SETS --separator RUN x5 --upstream UP --output-dir OUT
    python -m revrender.data.struck_pack views --pack PACK --dataset regions.json --pages DIR [...] --hold-out NAME [...] [--questions Q --answer-groups G] --output-dir OUT
    python -m revrender.data.struck_pack answers --pack PACK --output OUT.json
    python -m revrender.data.struck_pack counterfactual --pack SEPARATED_PACK --human FOLDS_PACK --output-dir OUT
    python -m revrender.data.struck_pack split --pack PACK --output configs/method_split.json
    python -m revrender.data.struck_pack scores --pack SEPARATED_PACK --sets SETS --fold 0 --output OUT.json
    python -m revrender.data.struck_pack tinted --raw RAW_VIEWS --erased ERASED_VIEWS --output-dir OUT

``tinted`` (plan of 2026-11-20, 第五轮_X与V结合, section 4.3): a views pack whose ``erased`` window and ``context`` line show a
restorer's erasure in red: red the gray of the erased views' image, green and blue the gray of the raw views' (the restorer's erased
ink red, the ink it left black, all of the scan's ink still there); where the two images agree pixel for pixel (the line outside the
region, its blue box) the raw views' image as it is.  ``raw`` stays the scan.  Every image is read back and checked
(OUT/summary.json); one that is not as made fails the pack.

``split`` (``method_split``): one 9:1 split of the writers, kept for the method stage; ``--split FILE`` makes ``folds``, ``fold-sets`` and
``views`` take its two folds (0 its validation writers, 1 its training ones), and ``fold-sets`` then holds the readers' windows
themselves (white outside the polygon, the polygon as ``inner_<k>``): fold_0 and fold_1 both train on fold 1 and stop by fold 0, each
restoring its own fold.  ``separated --separator none`` leaves a fold unerased (the scan as its erased crop); ``scores`` gives the pixel
scores of a separated pack's windows of one fold (``window_scores``).  ``fold-sets --windowed``: the readers' windows for the pack's
own folds, each trained on all the other folds and stopped by itself.  ``fixed`` (``fixed_pack``): a separated pack with one kind of
its pixel errors put right from the annotator's pixels, for reading what that kind of error costs.

``folds``: the regions of the merged dataset (``data.struck_dataset``) whose pixels and struck text are both confirmed, as one pack for
a cross-validation over writers.  The regions of the exports named by ``--hold-out`` (the test page) and every other region of their
writers stay out.  The writers go to ``FOLDS`` folds of about equal regions (``fold``); a tenth of the writers, by hash, are marked
``val`` (a reader taught on the other folds stops by them); ``--folds-from PACK`` keeps instead each writer's fold and val mark in an
earlier folds pack (the folds drawn anew would move with every region added).  A row is one region: ``readable`` as the annotator said, ``target`` what a
reader is taught to give (the text in the spelling of ``eval.coverage.prepare``, or ``recognize_lines.UNREADABLE``), and three images of
its OCR window (the region's polygon box and six pixels, white outside the polygon) from the annotator's own pixels: the scan (``raw``), the scan with the strike
pixels white (``erased``) and the layer map (``map``: white the strike pixels, black the rest); ``--erased-by RUN``: a restorer's strike on the
annotator's crop in place of the annotator's pixels, so a reader taught on it sees what the prefill gives it.  A region is cut from the page its
export showed the annotator (the dataset's ``image_source`` when that folder is here, else the first of ``--pages`` holding the page):
the pixels were drawn on that image.

``fold-sets``: the same regions and folds as the review sets a separator (a restorer of ``baselines.restoration``) trains on, one per
fold in OUT/fold_K: ``training_native.*`` the other folds' regions (the ``val`` writers apart, as ``validation_native.*``, by which its
versions are chosen) and ``heldout_native.*`` fold K's own, each row with its OCR window; ``summary.json`` names what a student of
teachers (``data.pseudo_regions``) must keep apart for this fold: fold K's writers, the ``val`` writers and the held-out ones, and every
page the dataset labels.  ``separated``: the folds pack again, but ``erased`` and ``map`` now come from those separators, each
restoring only the regions of the fold it never trained on: ``erased`` the restored crop, ``map`` the share of each ink pixel's
darkness it took away (0 none, white all).  ``views``: a pack again (folds or separated) with ``context``, the line the region stands in
(``CONTEXT`` region heights to each side, half a height above and below) with the erased crop laid back in and a blue box round the
region; ``--questions`` and ``--answer-groups`` (the question table and the pages of each question) put on each row the
``problem`` its page answers (``eval.revision_judgment.plain`` of the question's statement), for a reader asked with the question.
``answers``: whether ``latex_answer`` of every readable row's text comes back to its ``target`` in the spelling of
``eval.coverage.prepare`` (a reader taught on the LaTeX answers, ``train_reader --answer latex``, is scored as before).

``counterfactual``: a separated pack again with each region's evidence taken apart by the annotator's own strike pixels (the folds
pack's ``map``), for a reader taught what a change of the evidence should change (``train_reader --counterfactual``) and tested on it
(``recognize_lines --intervention``): ``raw_strike_off``, the scan without the strike (the struck writing still there);
``raw_writing_off``, the strike pixels alone on white (nothing left to read), and ``erased_writing_off``, what the separator's
map leaves of that.  The text is student work: OUT stays out of Git.
"""
import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .. import pages as pages_module
from ..util import SEED, digest, make_output_dir, require, write_json, write_manifest
from .pixel_masks import decode, reviewed_crop

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


def fold_of(counts, folds=FOLDS):
    """Writers to folds of about equal regions: in a seeded order, the larger first, each to the fold with the fewest regions so far."""
    order = sorted(counts); random.Random(SEED).shuffle(order); order.sort(key=lambda w: -counts[w])
    filled, out = [0]*folds, {}
    for writer in order:
        k = min(range(folds), key=lambda f: (filled[f], f)); out[writer] = k; filled[k] += counts[writer]
    return out


def layered(page, row):
    """(scan, erased, layer map) of a region's OCR window, from the annotator's pixels: everything outside the annotator's polygon white
    (the window is the polygon's box and six pixels, and the ink of the neighbours it holds is not the region's); the strike pixels white
    in ``erased`` and in the map, which is black elsewhere."""
    from .pixel_masks import region_pixels
    grey = np.asarray(page.convert('L')); bl, bt, br, bb = row['box']
    l, t, r, b = window(row['region'] or [[bl, bt], [br, bt], [br, bb], [bl, bb]], page.size)
    _, mask = reviewed_crop(page, {'box': row['box'], 'runs': row['runs'], 'region': row['region']})
    inside = np.zeros(grey.shape, bool); inside[bt:bb, bl:br] = region_pixels(row['region'], row['box'])
    strike = np.zeros(grey.shape, bool); strike[bt:bb, bl:br] = mask.astype(bool); strike &= inside
    scan = np.where(inside[t:b, l:r], grey[t:b, l:r], 255).astype(np.uint8); erased = scan.copy(); erased[strike[t:b, l:r]] = 255
    return Image.fromarray(scan), Image.fromarray(erased), Image.fromarray((255*strike[t:b, l:r]).astype(np.uint8))


SPLIT_REGIONS, SPLIT_READABLE = 61, 50   # plan of 2026-11-18 (方法阶段第一轮): about a tenth of the 610 regions, at least 50 readable


def split_order(writer):
    return hashlib.sha256(f'{writer}:{SEED}'.encode()).hexdigest()


def method_split(args):
    """One split of a pack's writers into training and validation, kept for the rounds after it (plan of 2026-11-18, section one): the
    writers in the order of ``split_order`` go to validation until it holds ``SPLIT_REGIONS`` regions, and on until ``SPLIT_READABLE``
    of them are readable; the rest train.  OUT holds writer UUIDs only (configs/method_split.json)."""
    rows = json.loads((args.pack/'pack.json').read_text())['rows']; writers = sorted({r['writer'] for r in rows}, key=split_order)
    side = lambda chosen: [r for r in rows if r['writer'] in chosen]; val = []
    for writer in writers:
        held = side(val)
        if len(held) >= SPLIT_REGIONS and sum(r['readable'] for r in held) >= SPLIT_READABLE: break
        val.append(writer)
    count = lambda sel: {'writers': len({r['writer'] for r in sel}), 'regions': len(sel), 'readable': sum(r['readable'] for r in sel)}
    result = {'rule': f'writers ordered by sha256("<writer>:{SEED}"), to validation until it holds {SPLIT_REGIONS} regions and {SPLIT_READABLE} readable; the rest train',
              'pack': str(args.pack), 'pack_sha256': digest(args.pack/'pack.json'), 'validation': sorted(val), 'training': sorted(set(writers)-set(val)),
              'counts': {'validation': count(side(set(val))), 'training': count(side(set(writers)-set(val)))}}
    require(not args.output.exists(), f'{args.output} exists'); write_json(args.output, result)
    print(json.dumps(result['counts']), flush=True)


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
    if args.split:   # two folds: 0 the validation writers of a method split (configs/method_split.json), 1 the training ones
        split = json.loads(args.split.read_text()); known = set(split['validation']) | set(split['training'])
        require({r['writer'] for r in chosen} <= known, f'writers not in {args.split}')
        fold, val = {w: 0 if w in split['validation'] else 1 for w in known}, lambda w: False
    elif args.holdout_share:   # two folds: 0 the prefill set's held-out writers (data.prefill_set.held_out), 1 the rest
        from .prefill_set import held_out
        fold, val = {w: 0 if held_out(w, args.holdout_share) else 1 for w in {r['writer'] for r in chosen}}, validation_writer
    elif args.folds_from:   # the folds and val marks of an earlier pack, writer by writer (a region added keeps its writer's)
        kept = {r['writer']: (r['fold'], r['val']) for r in json.loads((args.folds_from/'pack.json').read_text())['rows']}
        require({r['writer'] for r in chosen} <= set(kept), f'writers not in {args.folds_from}')
        fold, val = {w: f for w, (f, _) in kept.items()}, lambda w: kept[w][1]
    else:
        fold, val = fold_of({w: sum(r['writer'] == w for r in chosen) for w in {r['writer'] for r in chosen}}), validation_writer
    source = lambda r: r['image_source'] if r['image_source'] and Path(r['image_source']).is_dir() else ''
    listed = {folder: pages_module.page_files([folder] if folder else args.pages, sorted({r['page'] for r in chosen if source(r) == folder}))
              for folder in {source(r) for r in chosen}}
    chosen = sorted(({**r, 'fold': fold[r['writer']], 'val': val(r['writer']), 'source': source(r)} for r in chosen), key=lambda r: (r['source'], r['page'], r['id']))
    return chosen, {(folder, page): path for folder, files in listed.items() for page, path in files.items()}, held, skipped


def model_strike(predict, page, r):
    """The region's strike as a restorer erases it (``tools.annotate.pixel_library``'s rule) on the annotator's crop, inside the polygon."""
    from ..baselines.restoration import restore_native
    from ..tools.annotate.pixel_library import REMOVED
    from .pixel_masks import encode, region_pixels
    image = reviewed_crop(page, {**r, 'runs': []})[0].astype(np.float32)/255
    mask = ((restore_native(predict, image)-image) > REMOVED) & (image < 245/255)
    if r.get('region'): mask &= region_pixels(r['region'], r['box'])
    return encode(mask)


def folds_pack(args):
    chosen, files, held, skipped = chosen_regions(args)
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); rows = []
    predict = None
    if args.erased_by:   # the erased crop as the prefill will see it: a restorer's strike in place of the annotator's
        from ..baselines.restoration_export import selected_predictor
        predict = selected_predictor(args.erased_by, args.upstream, args.erased_step)[0]
    for key in sorted({(r['source'], r['page']) for r in chosen}):
        with Image.open(files[key]) as page:
            for r in (r for r in chosen if (r['source'], r['page']) == key):
                if predict: r = {**r, 'runs': model_strike(predict, page, r)}
                images = {name: f'crops/{len(rows):05d}_{name}.png' for name in ('raw', 'erased', 'map')}
                for name, image in zip(images, layered(page, r)): image.save(out/images[name])
                rows.append({'id': r['id'], 'line_id': r['id'], 'kind': 'line', 'writer': r['writer'], 'fold': r['fold'], 'val': r['val'],
                             'readable': bool(r['readable']), 'text': r['text'] if r['readable'] else '', 'target': r['target'], 'images': images,
                             'size': list(image.size), 'boxes': [], 'strike_pixels': r['strike_pixels'], 'mode': r['mode']})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    side = lambda sel: {'regions': len(sel), 'readable': sum(r['readable'] for r in sel), 'unreadable': sum(not r['readable'] for r in sel), 'writers': len({r['writer'] for r in sel})}
    summary = {**side(rows), 'folds': [side([r for r in rows if r['fold'] == k]) for k in range(max(r['fold'] for r in rows)+1)], 'val': side([r for r in rows if r['val']]),
               'held_out_writers': len(held), **skipped}
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items() if k != 'command'},
                   inputs={str(args.dataset): digest(args.dataset)}, **summary)
    print(json.dumps(summary), flush=True)


def windowed(page, r):
    """(scan, strike mask, window, polygon) of a region's OCR window as the readers see it (``layered``: white outside the polygon), the
    polygon in the window's coordinates."""
    from .pixel_masks import region_pixels
    bl, bt, br, bb = r['box']; scan, _, strike = layered(page, r)
    at = window(r['region'] or [[bl, bt], [br, bt], [br, bb], [bl, bb]], page.size)
    inner = pasted(np.zeros((at[3]-at[1], at[2]-at[0]), bool), region_pixels(r['region'], r['box']), r['box'], at[:2])
    return np.asarray(scan), (np.asarray(strike) > 0).astype(np.uint8), list(at), inner.astype(np.uint8)


def fold_sets(args):
    chosen, files, held, _ = chosen_regions(args); cut = {}; wide = args.split or args.windowed
    labelled = sorted({r['page'] for r in json.loads(args.dataset.read_text())})
    for key in sorted({(r['source'], r['page']) for r in chosen}):
        with Image.open(files[key]) as page:
            for r in (r for r in chosen if (r['source'], r['page']) == key):
                bl, bt, br, bb = r['box']
                cut[r['id']] = windowed(page, r) if wide else (*reviewed_crop(page, {'box': r['box'], 'runs': r['runs'], 'region': r['region']}),
                                list(window(r['region'] or [[bl, bt], [br, bt], [br, bb], [bl, bb]], page.size)))
    out = make_output_dir(args.output_dir); summary = []; parts = ('image', 'mask', None, 'inner') if wide else ('image', 'mask')
    for k in range(2 if args.split else FOLDS):
        # a method split: always train on its training writers (fold 1) and stop by its validation writers (fold 0); fold K's regions restored.
        # --windowed folds: train on all the other folds (the val marks ignored), stop by fold K and restore it
        sides = ({'training_native': [r for r in chosen if r['fold'] == 1], 'validation_native': [r for r in chosen if r['fold'] == 0],
                  'heldout_native': [r for r in chosen if r['fold'] == k]} if args.split else
                 {'training_native': [r for r in chosen if r['fold'] != k], 'validation_native': [r for r in chosen if r['fold'] == k],
                  'heldout_native': [r for r in chosen if r['fold'] == k]} if wide else
                 {'training_native': [r for r in chosen if r['fold'] != k and not r['val']], 'validation_native': [r for r in chosen if r['fold'] != k and r['val']],
                  'heldout_native': [r for r in chosen if r['fold'] == k]})
        require(all(sides.values()), f'fold {k}: a side is empty'); folder = out/f'fold_{k}'; folder.mkdir()
        for name, rows in sides.items():
            np.savez_compressed(folder/f'{name}.npz', **{f'{part}_{i}': cut[r['id']][j] for i, r in enumerate(rows) for j, part in enumerate(parts) if part})
            write_json(folder/f'{name}.json', [{'id': r['id'], 'group_uuid': r['writer'], 'page': r['page'], 'box': r['box'], 'region': r['region'], 'window': cut[r['id']][2],
                                                'mode': r['mode'], **({'windowed': True} if wide else {})} for r in rows])
        write_json(folder/'records.json', [{'id': r['id'], 'group_uuid': r['writer'], 'split': 'train', 'source': 'human_confirmed', 'mode': r['mode']} for r in sides['training_native']])
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


def window_scores(pieces):
    """The pixel scores of restored windows (image, restored, human strike mask, polygon; float images in [0, 1]): the removal counts
    inside the polygon (``baselines.restoration.removal_metrics``), and against the annotator's erased window (the scan with its strike
    white) the PSNR of all their pixels together and the mean SSIM of the windows (each composed as the readers see it: the scan outside
    the polygon)."""
    from skimage.metrics import structural_similarity
    from ..baselines.restoration import inside, removal_counts, removal_metrics
    counts, squared, pixels, ssim = np.zeros(5, np.int64), 0., 0, []
    for image, restored, mask, inner in pieces:
        counts += removal_counts(*inside(image, restored, mask, inner))
        shown = np.where(inner.astype(bool), restored, image); human = np.where(mask.astype(bool) & inner.astype(bool), 1., image)
        squared += float(((shown-human)**2).sum()); pixels += shown.size
        side = min(shown.shape)
        if side >= 3: ssim.append(float(structural_similarity(human, shown, data_range=1., win_size=min(7, side-1+side % 2))))
    return {'removal': removal_metrics(counts), 'psnr': float(-10*np.log10(max(squared/max(1, pixels), 1e-12))), 'ssim': float(np.mean(ssim)),
            'regions': len(pieces), 'ssim_regions': len(ssim)}


def separated_pack(args):
    from ..baselines.restoration import restore_native
    from ..baselines.restoration_export import selected_predictor
    from .pixel_masks import region_pixels
    folds = len(list(args.sets.glob('fold_*'))); require(len(args.separator) == folds, f'{folds} separators, in the order of the folds')
    hints = json.loads(args.hints.read_text()) if args.hints else None   # a hinted restorer's hint of each region (data.hint_image hints)
    if hints: from .hint_image import render
    require(len(args.checkpoint or []) in (0, 1, folds), 'one --checkpoint for all the separators, or one each')
    source = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); rows = []; inputs = {str(args.pack/'pack.json'): digest(args.pack/'pack.json')}; shares = []
    for k, run in enumerate(args.separator):
        folder = args.sets/f'fold_{k}'; native = json.loads((folder/'heldout_native.json').read_text()); arrays = np.load(folder/'heldout_native.npz')
        if str(run) == 'none':   # nothing erased: the scan as its own erased crop
            predict = lambda x: x
        else:
            step = (args.checkpoint or [None])[k if len(args.checkpoint or []) > 1 else 0]
            predict, checkpoint, _ = selected_predictor(run, args.upstream, step); inputs[str(checkpoint)] = digest(checkpoint)
        for i, n in enumerate(native):
            row = source[n['id']]; require(row['fold'] == k, f"{n['id']}: not of fold {k} in the pack")
            image = arrays[f'image_{i}'].astype(np.float32)/255; hint = None
            if hints is not None:
                require(n.get('windowed'), 'hints need windowed sets'); hint = render(hints[n['id']], arrays[f'image_{i}'], arrays[f'inner_{i}'])[0]
            restored = np.clip(restore_native(predict, image, None if hint is None else hint.astype(np.float32)/255), 0, 1)
            with Image.open(args.pack/row['images']['raw']) as opened: scan = np.asarray(opened.convert('L')).copy()
            if n.get('windowed'):   # the set holds the readers' window itself (white outside the polygon)
                inner = arrays[f'inner_{i}'].astype(bool)
                require(scan.shape == image.shape and (scan == arrays[f'image_{i}']).all(), f"{n['id']}: the set and the pack cut it differently")
                share = taken_away(image, restored)*inner
                erased = np.where(inner, np.round(255*restored), scan).astype(np.uint8); layer = np.round(255*share).astype(np.uint8)
            else:
                inside = region_pixels(n['region'], n['box']); share = taken_away(image, restored)*inside
                origin = n['window'][:2]
                erased = np.where(pasted(np.zeros(scan.shape, bool), inside, n['box'], origin),                     # outside the polygon the scan stays
                                  pasted(scan.copy(), np.round(255*restored).astype(np.uint8), n['box'], origin), scan)
                layer = pasted(np.zeros(scan.shape, np.uint8), np.round(255*share).astype(np.uint8), n['box'], origin)
            images = {name: f'crops/{len(rows):05d}_{name}.png' for name in ('raw', 'erased', 'map', *(('hint',) if hint is not None else ()))}
            for name, array in zip(images, (scan, erased, layer, hint)): Image.fromarray(array).save(out/images[name])
            rows.append({**row, 'images': images, **({'hint': hints[n['id']]} if hints is not None else {})}); shares.append(float(share[arrays[f'mask_{i}'] > 0].mean()) if arrays[f'mask_{i}'].any() else None)
    require(len(rows) == len(source), 'the sets and the pack hold different regions')
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    known = [s for s in shares if s is not None]
    if hints is not None: inputs[str(args.hints)] = digest(args.hints)
    write_manifest(out, config={'pack': str(args.pack), 'sets': str(args.sets), 'separators': [str(r) for r in args.separator], 'checkpoint': args.checkpoint,
                                'hints': str(args.hints) if args.hints else None},
                   inputs=inputs, regions=len(rows), strike_taken_away_mean=float(np.mean(known)))
    print(json.dumps({'regions': len(rows), 'strike_taken_away_mean': float(np.mean(known))}), flush=True)


FIXES = ('residual', 'damage', 'both')


def fixed_pack(args):
    """A separated pack with one kind of the restorer's pixel errors put right from the annotator's erased window (plan of 2026-11-19,
    a diagnosis, not a method): ``residual`` the strike pixels it left (erased by the annotator, not by it), ``damage`` the writing it
    erased (by it, not by the annotator), ``both``.  Erased as the F1 counts it: brighter than the scan by more than 0.2, on ink, inside
    the polygon (the windowed sets' ``inner``).  The map is the share taken away of the new erased crop."""
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); made, changed = [], {'residual': 0, 'damage': 0}
    for folder in sorted(args.sets.glob('fold_*')):
        native = json.loads((folder/'heldout_native.json').read_text()); arrays = np.load(folder/'heldout_native.npz')
        require(all(n.get('windowed') for n in native), 'not windowed sets')
        for i, n in enumerate(native):
            row = rows[n['id']]; image = arrays[f'image_{i}'].astype(np.float32)/255
            restored = np.asarray(Image.open(args.pack/row['images']['erased']).convert('L'), np.float32)/255
            inner = arrays[f'inner_{i}'].astype(bool); ink = (image < INK/255) & inner
            truth = arrays[f'mask_{i}'].astype(bool) & ink; removed = ((restored-image) > .2) & ink
            human = np.where(truth, 1., image); fix = np.zeros(image.shape, bool)
            if args.fix in ('residual', 'both'): fix |= truth & ~removed
            if args.fix in ('damage', 'both'): fix |= removed & ~truth
            changed['residual'] += int((truth & ~removed & fix).sum()); changed['damage'] += int((removed & ~truth & fix).sum())
            erased = np.where(fix, human, restored)
            images = {name: f'crops/{len(made):05d}_{name}.png' for name in ('raw', 'erased', 'map')}
            shutil.copyfile(args.pack/row['images']['raw'], out/images['raw'])
            Image.fromarray(np.round(255*erased).astype(np.uint8)).save(out/images['erased'])
            Image.fromarray(np.round(255*taken_away(image, erased)*inner).astype(np.uint8)).save(out/images['map'])
            made.append({**row, 'images': images})
    require(len(made) == len(rows), 'the sets and the pack hold different regions')
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': made})
    write_manifest(out, config={'pack': str(args.pack), 'sets': str(args.sets), 'fix': args.fix}, inputs={str(args.pack/'pack.json'): digest(args.pack/'pack.json')},
                   regions=len(made), pixels_changed=changed)
    print(json.dumps({'regions': len(made), 'pixels_changed': changed}), flush=True)


def pixel_scores(args):
    """``window_scores`` of a separated pack's erased windows of one fold of windowed sets (the windows as saved, 8 bits), to OUT.json."""
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    folder = args.sets/f'fold_{args.fold}'; native = json.loads((folder/'heldout_native.json').read_text()); arrays = np.load(folder/'heldout_native.npz')
    require(all(n.get('windowed') for n in native), 'not windowed sets')
    pieces = [(arrays[f'image_{i}'].astype(np.float32)/255, np.asarray(Image.open(args.pack/rows[n['id']]['images']['erased']).convert('L'), np.float32)/255,
               arrays[f'mask_{i}'], arrays[f'inner_{i}']) for i, n in enumerate(native)]
    result = {**window_scores(pieces), 'pack': str(args.pack), 'sets': str(folder)}
    require(not args.output.exists(), f'{args.output} exists'); write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('psnr', 'ssim', 'regions')} | {'f1': result['removal']['f1']}), flush=True)


def intervened(scan, strike, share):
    """(strike off, writing off, what the separator leaves of that) of a scan (grey uint8), by the annotator's strike pixels (bool)
    and the separator's map (uint8): the scan with the strike white; with everything but the strike white (not the ink alone: the pale
    rim of the writing would still show what was written); and that one lightened by the map."""
    strike_off = scan.copy(); strike_off[strike] = 255
    writing_off = np.where(strike, scan, 255).astype(np.uint8)
    return strike_off, writing_off, lightened(writing_off, share, 1)


def counterfactual_pack(args):
    human = {r['id']: r for r in json.loads((args.human/'pack.json').read_text())['rows']}
    source = json.loads((args.pack/'pack.json').read_text())['rows']; require({r['id'] for r in source} == set(human), 'the two packs hold different regions')
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); rows = []; grey = lambda root, path: np.asarray(Image.open(root/path).convert('L'))
    for row in source:
        scan, share = grey(args.pack, row['images']['raw']), grey(args.pack, row['images']['map']); strike = grey(args.human, human[row['id']]['images']['map']) > 0
        require(strike.shape == scan.shape, f"{row['id']}: the two packs cut it differently")
        made = dict(zip(('raw_strike_off', 'raw_writing_off', 'erased_writing_off'), intervened(scan, strike, share)))
        images = {name: f'crops/{len(rows):05d}_{name}.png' for name in (*row['images'], *made)}
        for name, path in row['images'].items(): shutil.copyfile(args.pack/path, out/images[name])
        for name, array in made.items(): Image.fromarray(array).save(out/images[name])
        rows.append({**row, 'images': images})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    write_manifest(out, config={'pack': str(args.pack), 'human': str(args.human)},
                   inputs={str(p/'pack.json'): digest(p/'pack.json') for p in (args.pack, args.human)}, regions=len(rows))
    print(json.dumps({'regions': len(rows)}), flush=True)


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


def questions_of(args):
    """{page: the plain statement of the question it answers} from the question table and the pages of each question."""
    from ..eval.revision_judgment import plain
    statement = {q['question_global_id']: q['problem_statement_markdown'] for q in map(json.loads, args.questions.read_text(encoding='utf-8').splitlines())}
    return {page: plain(statement.get(g['question_global_id'])) for g in map(json.loads, args.answer_groups.read_text(encoding='utf-8').splitlines()) for page in g['image_ids']}


def views_pack(args):
    chosen, files, _, _ = chosen_regions(args); place = {r['id']: (r['source'], r['page']) for r in chosen}; by_id = {r['id']: r for r in chosen}
    source = json.loads((args.pack/'pack.json').read_text())['rows']
    require({r['id'] for r in source} == set(place), 'the pack and the dataset hold different regions')
    require(bool(args.questions) == bool(args.answer_groups), '--questions and --answer-groups go together')
    problems = questions_of(args) if args.questions else None
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); rows = []
    for key in sorted(set(place.values())):
        with Image.open(files[key]) as page:
            for row in (r for r in source if place[r['id']] == key):
                r = by_id[row['id']]; bl, bt, br, bb = r['box']
                opened = {name: np.asarray(Image.open(args.pack/row['images'][name]).convert('L')) for name in ('raw', 'erased', 'map')}
                at = window(r['region'] or [[bl, bt], [br, bt], [br, bb], [bl, bb]], page.size)
                made = {**{name: Image.fromarray(array) for name, array in opened.items()}, 'context': in_context(page, opened['erased'], at)}
                images = {name: f'crops/{len(rows):05d}_{name}.png' for name in made}
                for name, image in made.items(): image.save(out/images[name])
                rows.append({**row, 'images': images, **({'problem': problems.get(r['page'], '')} if problems is not None else {})})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    inputs = {str(p): digest(p) for p in (args.pack/'pack.json', args.dataset, args.questions, args.answer_groups) if p}
    summary = {'regions': len(rows), **({'with_problem': sum(bool(r['problem']) for r in rows)} if problems is not None else {})}
    write_manifest(out, config={'pack': str(args.pack), 'dataset': str(args.dataset), 'context': CONTEXT, 'questions': str(args.questions)}, inputs=inputs, **summary)
    print(json.dumps(summary), flush=True)


def tint(scan, erased):
    """The RGB image of ``tinted`` (see the module's words) from a scan and the restorer's erased image of one size."""
    s, e = np.asarray(scan.convert('RGB')), np.asarray(erased.convert('RGB'))
    require(s.shape == e.shape, f'the scan is {s.shape[:2]} and the erased image {e.shape[:2]}')
    gray = lambda image: np.asarray(image.convert('L'))
    out = np.stack([gray(erased), gray(scan), gray(scan)], -1)
    same = (s == e).all(-1)
    out[same] = s[same]
    return Image.fromarray(out)


def tinted_pack(args):
    raw = json.loads((args.raw/'pack.json').read_text())['rows']; erased = {r['id']: r for r in json.loads((args.erased/'pack.json').read_text())['rows']}
    require(set(erased) == {r['id'] for r in raw}, 'the two packs hold different regions')
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); rows, checked = [], {'windows': 0, 'lines': 0, 'line_pixels_kept_as_scan': 0}
    for n, r in enumerate(raw):
        q = erased[r['id']]; images = {name: f'crops/{n:05d}_{name}.png' for name in ('raw', 'erased', 'context')}
        shutil.copyfile(args.raw/r['images']['raw'], out/images['raw'])
        pairs = {'erased': (args.raw/r['images']['erased'], args.erased/q['images']['erased']), 'context': (args.raw/r['images']['context'], args.erased/q['images']['context'])}
        for name, (a, b) in pairs.items():
            with Image.open(a) as scan, Image.open(b) as gone:
                tint(scan, gone).save(out/images[name])
                with Image.open(out/images[name]) as back:
                    got, s, e = np.asarray(back.convert('RGB')), np.asarray(scan.convert('RGB')), np.asarray(gone.convert('RGB'))
                    differ = ~(s == e).all(-1); gs, ge = np.asarray(scan.convert('L')), np.asarray(gone.convert('L'))
                    good = (np.array_equal(got[..., 0][differ], ge[differ]) and np.array_equal(got[..., 1][differ], gs[differ]) and np.array_equal(got[..., 2][differ], gs[differ])
                            and np.array_equal(got[~differ], s[~differ]))
                    require(good, f'{r["id"]} {name}: the tinted image is not as made')
                    if name == 'erased':
                        require(np.array_equal(got[..., 0], ge) and np.array_equal(got[..., 1], gs) and np.array_equal(got[..., 2], gs), f'{r["id"]}: the window is not red/scan')
                        checked['windows'] += 1
                    else:
                        checked['lines'] += 1; checked['line_pixels_kept_as_scan'] += int(((got != np.stack([ge, gs, gs], -1)).any(-1)).sum())
        rows.append({**r, 'images': images})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    summary = {'regions': len(rows), **checked}
    write_json(out/'summary.json', summary)
    write_manifest(out, config={'raw': str(args.raw), 'erased': str(args.erased)}, inputs={str(p/'pack.json'): digest(p/'pack.json') for p in (args.raw, args.erased)}, **summary)
    print(json.dumps(summary), flush=True)


def tidy(line):
    """A line of an annotated answer with its whitespace dropped, but for one space where it parts a command from a letter
    (``\\times x``) and after a LaTeX line break (``\\\\``, whose following ``{`` must not read as ``\\{``)."""
    parts, out = re.split(r'(\s+)', line.strip()), ''
    for k, part in enumerate(parts):
        if not part.isspace():
            out += part
        elif (re.search(r'\\[a-zA-Z]+$', out) and re.match(r'[a-zA-Z]', parts[k+1] if k+1 < len(parts) else '')) or out.endswith('\\\\'):
            out += ' '
    return out


def latex_answer(text):
    """The annotated text as a reader is taught to give it in LaTeX (``train_reader --answer latex``): ``?`` dropped (the scores drop it
    from the label), a display ``$$...$$`` made inline and two inline formulas side by side made one, whitespace tidied line by line
    (``tidy``), the line breaks kept, a formula left open closed at the end."""
    s = text.replace('?', '').strip()
    if s.startswith('$$') and s.endswith('$$') and len(s) > 4 and '$' not in s[2:-2]:
        s = '$'+s[2:-2]+'$'
    s = '\n'.join(tidy(line) for line in s.replace('$$', '').split('\n'))
    return s+'$' if s.count('$') % 2 else s


def answers_check(args):
    from ..eval.coverage import prepare
    rows = [r for r in json.loads((args.pack/'pack.json').read_text())['rows'] if r['readable']]
    differ = [{'id': r['id'], 'target': r['target'], 'latex': latex_answer(r['text'])} for r in rows if prepare(latex_answer(r['text'])) != r['target']]
    result = {'readable': len(rows), 'same': len(rows)-len(differ), 'differ': len(differ), 'changed_by_tidying': sum(latex_answer(r['text']) != r['text'].strip() for r in rows),
              'differing': differ, 'pack': str(args.pack/'pack.json'), 'pack_sha256': digest(args.pack/'pack.json')}
    require(not args.output.exists(), f'{args.output} exists'); write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'differing'}), flush=True)
    require(not differ, f'{len(differ)} LaTeX answers do not come back to their targets: {args.output}')


def library_pack(args):
    """The regions a model prefilled in a review library (``tools.annotate.pixel_library infer``), as a pack to read: each region not
    the annotator's (a group of near-identical crops once, as the library shows it), its OCR window from the model's pixels and polygon,
    all fold 0.  ``--dataset``: a region the merged dataset holds (confirmed pixels and text) carries its target, the rest are unlabelled
    (``labelled`` false, readable, target empty)."""
    from ..eval.coverage import prepare
    from ..recognize_lines import UNREADABLE
    from ..tools.annotate.pixel_library import duplicate_groups
    from .prefill_set import held_out
    source = json.loads((args.library/'pack.json').read_text()); pages = source['pages']
    groups = duplicate_groups(source['rows']); alias = {i for g in groups for i in g['ids'][1:]}; merged = {g['ids'][0]: g for g in groups}
    labels = {d['id']: d for d in json.loads(args.dataset.read_text()) if d.get('pixels_confirmed') and d.get('text_confirmed') and not d.get('deleted')} if args.dataset else {}
    chosen = [{**r, **({k: merged[r['id']][k] for k in ('box', 'region', 'runs')} if r['id'] in merged else {})}
              for r in source['rows'] if not r.get('preserved_manual') and not r.get('deleted') and r['id'] not in alias]
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); rows = []
    for pi in sorted({r['page'] for r in chosen}):
        page_id = pages[pi]['id']
        with Image.open(args.library/pages[pi]['image']) as page:
            for r in (r for r in chosen if r['page'] == pi):
                key = page_id+'#'+r['id'].split('#')[-1]; d = labels.get(key); writer = d['writer'] if d else pages[pi]['writer_group']
                target = (prepare(d['text'] or '').replace('?', '') if d['readable'] else UNREADABLE) if d else ''
                images = {name: f'crops/{len(rows):05d}_{name}.png' for name in ('raw', 'erased', 'map')}
                for name, image in zip(images, layered(page, r)): image.save(out/images[name])
                rows.append({'id': key, 'line_id': key, 'kind': 'line', 'writer': writer, 'fold': 0, 'val': False, 'labelled': bool(d),
                             'held_out': bool(d) and held_out(writer, args.holdout_share) if args.holdout_share else None,
                             'readable': bool(d['readable']) if d else True, 'text': (d['text'] or '') if d and d['readable'] else '', 'target': target,
                             'images': images, 'size': list(image.size), 'boxes': [], 'strike_pixels': int(decode(r['runs'], (r['box'][3]-r['box'][1], r['box'][2]-r['box'][0])).sum()),
                             'mode': 'erase'})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    summary = {'regions': len(rows), 'labelled': sum(r['labelled'] for r in rows), 'held_out': sum(bool(r['held_out']) for r in rows), 'aliases': len(alias)}
    write_json(out/'held_out.json', {'ids': [r['id'] for r in rows if r['held_out']]})
    inputs = {str(args.library/'pack.json'): digest(args.library/'pack.json'), **({str(args.dataset): digest(args.dataset)} if args.dataset else {})}
    write_manifest(out, config={k: str(v) for k, v in vars(args).items() if k != 'command'}, inputs=inputs, **summary)
    print(json.dumps(summary), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    for name in ('folds', 'fold-sets', 'views'):
        f = sub.add_parser(name)
        if name == 'views':
            f.add_argument('--pack', type=Path, required=True)
            f.add_argument('--questions', type=Path, help='the question table (JSON lines with question_global_id, problem_statement_markdown)')
            f.add_argument('--answer-groups', type=Path, help='the pages of each question (JSON lines with question_global_id, image_ids)')
        f.add_argument('--dataset', type=Path, required=True, help='regions.json of data.struck_dataset'); f.add_argument('--pages', type=Path, action='append', required=True)
        f.add_argument('--hold-out', action='append', default=[], help='part of the name of an export whose regions, and their writers, stay out (the test page)')
        f.add_argument('--folds-from', type=Path, help='a folds pack whose folds and val marks the writers keep (else they are drawn anew)')
        f.add_argument('--holdout-share', type=float, help="two folds: 0 the writers data.prefill_set holds out at this share, 1 the rest")
        f.add_argument('--split', type=Path, help='two folds: 0 the validation writers of this method split (``split``), 1 the training ones')
        f.add_argument('--windowed', action='store_true', help="fold-sets: the readers' windows, each fold trained on all the others and stopped by itself")
        f.add_argument('--erased-by', type=Path, help="folds: a restorer run whose strike makes the erased crop and the map (the annotator's pixels otherwise)")
        f.add_argument('--erased-step', help='the step directory of --erased-by'); f.add_argument('--upstream', type=Path, help='the restorers\' sources, with --erased-by')
        f.add_argument('--output-dir', type=Path, required=True)
    d = sub.add_parser('separated')
    for key in ('pack', 'sets', 'upstream', 'output-dir'): d.add_argument('--'+key, type=Path, required=True)
    d.add_argument('--separator', type=Path, action='append', required=True,
                   help="a fold's separator run (baselines.restoration), given once per fold in order; 'none': nothing erased")
    d.add_argument('--hints', type=Path, help="data.hint_image hints: each region's hint for a hinted restorer (NAFNet-hint)")
    d.add_argument('--checkpoint', action='append', help="a step directory in place of the selected one: once for every separator, or once each")
    b = sub.add_parser('library')
    for key in ('library', 'output-dir'): b.add_argument('--'+key, type=Path, required=True)
    b.add_argument('--dataset', type=Path, help='regions.json of a merged dataset: the regions it labels carry their targets')
    b.add_argument('--holdout-share', type=float, help="mark the labelled regions of the writers data.prefill_set holds out at this share")
    x = sub.add_parser('fixed')
    for key in ('pack', 'sets', 'output-dir'): x.add_argument('--'+key, type=Path, required=True)
    x.add_argument('--fix', choices=FIXES, required=True)
    q = sub.add_parser('scores')
    for key in ('pack', 'sets', 'output'): q.add_argument('--'+key, type=Path, required=True)
    q.add_argument('--fold', type=int, default=0)
    m = sub.add_parser('split')
    m.add_argument('--pack', type=Path, required=True); m.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('answers')
    a.add_argument('--pack', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('counterfactual')
    for key in ('pack', 'human', 'output-dir'): c.add_argument('--'+key, type=Path, required=True)
    t = sub.add_parser('tinted')
    for key in ('raw', 'erased', 'output-dir'): t.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args(argv)
    {'folds': folds_pack, 'fold-sets': fold_sets, 'separated': separated_pack, 'views': views_pack, 'counterfactual': counterfactual_pack,
     'library': library_pack, 'answers': answers_check, 'split': method_split, 'scores': pixel_scores, 'fixed': fixed_pack, 'tinted': tinted_pack}[args.command](args)


if __name__ == '__main__':
    main()
