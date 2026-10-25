"""Pseudo-labelled strike regions from the unlabelled library (``pixel_review``'s full-review pack) for a student restorer.

Noisy Student (Xie et al., CVPR 2020): teachers label unlabelled data and a student trains on those labels together with the human
ones.  U2PL (Wang et al., CVPR 2022): pixels the teachers are unsure about supervise nothing instead of being forced either way.
Here the teachers are the restorers' (b) versions chosen on the wider validation; a pixel of ink they agree on (``--agree`` of them
lighten it by the evaluation's threshold) is the strike, a pixel only fewer erase gets weight 0, the rest is writing.  Only pages of
the library's training split, writers apart from the wider validation's and the fixed validation's, regions outside the review set's
human ones (the library's own ``confirmed`` flag is older than the exports the review set was built from).  ``combine`` writes a
review set (``training_native.*`` with ``weight_<k>``) that ``data.restoration_pairs human`` tiles.

For the prefill model (plan 预填模型, ``--prefill-set``: a ``data.prefill_set``) the regions apart are its held-out and fixed
validation writers' and every page it labels; ``combine`` then repeats its human regions to ``--human-share`` of the regions and
gives each record a sample weight (``--pseudo-weight`` the pseudo-labelled, 1 the human)."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ..util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from ..baselines.restoration import restore_native

INK = 200      # grey below which a pixel is ink, as the removal metrics read it
ERASE = 51     # lightening (grey levels) that counts as erased: the metrics' .2


def excluded_writers(evaluation, review_set):
    """Writers of the wider validation (its pixel regions and read-back lines) and of the review set's fixed validation."""
    writers = {r['id'].split('_')[0] for r in json.loads((evaluation/'inputs/pixels.json').read_text())}
    writers |= {r['line_id'].split('_')[0] for r in json.loads((evaluation/'pack/pack.json').read_text())['rows']}
    writers |= {r['group_uuid'] for r in json.loads((review_set/'validation_native.json').read_text())}
    writers |= set(json.loads((review_set/'summary.json').read_text()).get('validation_writers', []))
    return writers


def human_ids(review_set):
    """The regions the review set already labels (its human training and fixed validation regions)."""
    return {r['id'] for name in ('training_native.json', 'validation_native.json') for r in json.loads((review_set/name).read_text())}


def eligible(library, excluded, labelled=frozenset(), labelled_pages=frozenset()):
    """The library's regions a student may learn from, page by page: training pages of writers not excluded and not labelled, regions
    neither confirmed in the library nor among the ``labelled`` ids."""
    pack = json.loads((library/'pack.json').read_text()); pages = pack['pages']; rows = []
    for r in pack['rows']:
        page = pages[r['page']]
        if (page['split'] != 'train' or page['writer_group'] in excluded or r.get('confirmed') or r['id'] in labelled
                or page['id'] in labelled_pages): continue
        rows.append({'id': r['id'], 'page': page['id'], 'image': str(library/page['image']), 'box': [int(v) for v in r['box']],
                     'group_uuid': page['writer_group']})
    return sorted(rows, key=lambda r: (r['image'], r['id']))


def crops(rows):
    """Each row's grey crop, one page image open at a time (rows sorted by page)."""
    image = None; path = None
    for r in rows:
        if r['image'] != path:
            with Image.open(r['image']) as im: image = np.asarray(im.convert('L'))
            path = r['image']
        l, t, rr, b = r['box']; yield r, image[t:b, l:rr]


def apart(args):
    """(writers, region ids, pages) a student may not learn from."""
    if args.prefill_set:
        s = json.loads((args.prefill_set/'summary.json').read_text())
        return set(s['validation_writers']) | set(s['fixed_writers']), set(), set(s['pages'])
    return excluded_writers(args.evaluation, args.review_set), human_ids(args.review_set), set()


def predict(args):
    """One teacher's restoration of every eligible region."""
    out = make_output_dir(args.output_dir); excluded, labelled, labelled_pages = apart(args)
    rows = eligible(args.library, excluded, labelled, labelled_pages)
    rows = rows[:args.limit] if args.limit else rows; require(rows, 'no eligible region')
    from ..baselines.restoration import HINTED
    from ..baselines.restoration_export import selected_predictor
    predictor, checkpoint, manifest = selected_predictor(args.run, args.upstream, args.checkpoint)
    hinted = manifest['config']['model'] in HINTED; require(args.detector or not hinted, 'a hinted teacher needs --detector')
    if hinted:
        from .. import pages
        from .separation import detector_hint
        detected = {}
    restored = {}
    for k, (r, grey) in enumerate(crops(rows)):
        hint = None
        if hinted:
            if r['page'] not in detected: detected = {r['page']: pages.read_output(args.detector, r['page'])[1]}
            hint = detector_hint(detected[r['page']], r['box'], grey).astype(np.float32)
        value = restore_native(predictor, grey.astype(np.float32)/255, hint); restored[f'restored_{k}'] = np.rint(value*255).astype(np.uint8)
    np.savez_compressed(out/'restored.npz', **restored); write_json(out/'regions.json', rows)
    summary = {'model': manifest['config']['model'], 'checkpoint': str(checkpoint), 'regions': len(rows), 'writers': len({r['group_uuid'] for r in rows}),
               'excluded_writers': len(excluded), 'excluded_pages': len(labelled_pages)}
    inputs = [args.library/'pack.json', checkpoint, args.run/'run_manifest.json', *([args.prefill_set/'summary.json'] if args.prefill_set else [])]
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in inputs}, **summary)
    print(json.dumps(summary), flush=True)


def combine(args):
    """The teachers' agreement as a review set: ``mask`` where at least ``agree`` teachers erased, ``weight`` 0 where fewer but some did;
    the review set's human training regions appended ``human_copies`` times with weight 1; its fixed validation copied.  Regions of the
    synthetic pairs' validation writers (``--source``) are left out so the student keeps that validation apart.  ``--soft``: the target
    is the teachers' mean image itself (``target_<k>``, weight 1 everywhere), knowledge distillation instead of a hard label."""
    out = make_output_dir(args.output_dir)
    teachers = [(json.loads((p/'regions.json').read_text()), np.load(p/'restored.npz')) for p in args.predictions]
    rows = teachers[0][0]; require(all(r == rows for r, _ in teachers), 'the teachers labelled different regions')
    apart = {r['group_uuid'] for r in json.loads((args.source/'records.json').read_text()) if r['split'] == 'val'} if args.source else set()
    native, arrays, records, counts = [], {}, [], Counter()
    for k, (r, grey) in enumerate(crops(rows)):
        if r['group_uuid'] in apart: counts['left_out_synthetic_validation_writer'] += 1; continue
        ink = grey < INK
        votes = sum(((a[f'restored_{k}'].astype(np.int16) - grey) > ERASE) & ink for _, a in teachers)
        mask = votes >= args.agree; unreliable = (votes > 0) & ~mask
        l, t, rr, b = r['box']; n = len(native)
        native.append({'box': r['box'], 'group_uuid': r['group_uuid'], 'id': r['id'], 'page': r['page'], 'region': [[l, t], [rr, t], [rr, b], [l, b]]})
        arrays[f'image_{n}'] = grey; arrays[f'mask_{n}'] = mask.astype(np.uint8)
        if args.soft:
            arrays[f'target_{n}'] = np.rint(np.mean([a[f'restored_{k}'].astype(np.float32) for _, a in teachers], 0)).astype(np.uint8)
            arrays[f'weight_{n}'] = np.ones_like(mask, dtype=np.uint8)
        else: arrays[f'weight_{n}'] = (~unreliable).astype(np.uint8)
        records.append({'group_uuid': r['group_uuid'], 'id': r['id'], 'line_id': r['id'], 'revisions': 1, 'source': 'pseudo_agreement', 'split': 'train',
                        'weight': args.pseudo_weight})
        counts['pseudo_regions'] += 1; counts['strike_pixels'] += int(mask.sum()); counts['unreliable_pixels'] += int(unreliable.sum()); counts['ink_pixels'] += int(ink.sum())
    human = json.loads((args.review_set/'training_native.json').read_text()); harrays = np.load(args.review_set/'training_native.npz')
    if args.human_share: args.human_copies = max(1, round(args.human_share/(1-args.human_share)*counts['pseudo_regions']/len(human)))
    for copy in range(args.human_copies):
        for k, r in enumerate(human):
            n = len(native); native.append({**r, 'id': r['id'] + (f'~copy{copy}' if copy else '')})
            arrays[f'image_{n}'] = harrays[f'image_{k}']; arrays[f'mask_{n}'] = harrays[f'mask_{k}']; arrays[f'weight_{n}'] = np.ones_like(harrays[f'mask_{k}'], dtype=np.uint8)
            records.append({'group_uuid': r['group_uuid'], 'id': native[-1]['id'], 'line_id': r['id'], 'revisions': 1, 'source': 'human_confirmed', 'split': 'train',
                            'weight': 1.})
    counts['human_regions'] = len(human); counts['human_copies'] = args.human_copies
    write_json(out/'training_native.json', native); np.savez_compressed(out/'training_native.npz', **arrays); write_json(out/'records.json', records)
    for name in ('validation_native.json', 'validation_native.npz'): (out/name).write_bytes((args.review_set/name).read_bytes())
    summary = {'counts': dict(counts), 'agree': args.agree, 'soft': args.soft, 'teachers': [str(p) for p in args.predictions], 'review_set': str(args.review_set),
               'pseudo_weight': args.pseudo_weight,
               'validation_writers': json.loads((args.review_set/'summary.json').read_text()).get('validation_writers', [])}
    write_json(out/'summary.json', summary)
    inputs = {str(p/'run_manifest.json'): digest(p/'run_manifest.json') for p in args.predictions}
    inputs[str(args.review_set/'training_native.npz')] = digest(args.review_set/'training_native.npz')
    write_manifest(out, config=to_jsonable(args), inputs=inputs, **summary)
    print(json.dumps(summary), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('predict', help=predict.__doc__)
    for name in ('library', 'run', 'upstream', 'output-dir'): a.add_argument('--'+name, type=Path, required=True)
    for name in ('evaluation', 'review-set', 'prefill-set'): a.add_argument('--'+name, type=Path, help='the regions apart: an evaluation and a review set, or a prefill set')
    a.add_argument('--checkpoint', required=True, help="the teacher's step directory within --run")
    a.add_argument('--limit', type=int, default=0, help='only the first regions (a smoke test)')
    a.add_argument('--detector', type=Path, help="a detection run: a hinted teacher's (RevSep) hints on each region")
    c = sub.add_parser('combine', help=combine.__doc__)
    c.add_argument('--predictions', type=Path, nargs='+', required=True); c.add_argument('--review-set', type=Path, required=True)
    c.add_argument('--source', type=Path, help="the synthetic pairs whose validation writers' regions are left out")
    c.add_argument('--agree', type=int, default=2, help='teachers that must erase a pixel for it to be the strike')
    c.add_argument('--human-copies', type=int, default=10, help='times the human training regions are repeated beside the pseudo-labelled ones')
    c.add_argument('--soft', action='store_true', help="the teachers' mean image as the target (distillation) instead of the agreement mask")
    c.add_argument('--human-share', type=float, help='repeat the human regions to this share of all regions (in place of --human-copies)')
    c.add_argument('--pseudo-weight', type=float, default=1., help="a pseudo-labelled region's sample weight (a human one's is 1)")
    c.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv); (predict if args.command == 'predict' else combine)(args)


if __name__ == '__main__': main()
