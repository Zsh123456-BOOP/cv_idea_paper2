"""Synthetic regions as reader packs, fold by fold (plan of 2026-11-20, 第四轮_特权视图三样试验, section five): the regions of
``data.synthesis paste`` (its cleaned arm S) a fold may learn from, cut as the real regions are cut.

    python -m revrender.data.synth_views build --synth SYNTH --pack VIEWS --set PREFILL_SET --bank BANK --same-student PAIRS
        --dataset regions.json --test-halves configs/test_halves.json --pages DIR --take 3000 --output-dir OUT
    python -m revrender.data.synth_views answered --pack OUT/fold_K_raw --readings TEACHER_READ --take 1500 --output-dir ANSWERED

``build``: for each fold K of the pack (``VIEWS``, a ``data.struck_pack folds``/``views`` pack: its regions' writers and folds) the
synthetic regions whose host writer and whose strike's source writer are neither in fold K nor among the test page's writers, a
writer and the same student's other id counting as one (``--same-student``); a region whose source cannot be traced is left out.
The source of an H unit is the prefill set's training region it was cut from (``--set``: its writer), of a P unit the bank piece
(``--bank``: its writer); a D unit is drawn by the program from no one's ink.  The test page's writers: those of the dataset's
regions from a test-page export and of the questions in ``--test-halves``.  The regions left are ordered by sha256("<id>:42") and the
first ``--take`` kept.  Each region kept, once for all the folds: the synthetic box and its clean host box come back from S's tiles
(``restoration_pairs.canvas_tiles`` undone); its window is the box and six pixels, white outside the box (the real regions' windows
are white outside their polygon); its line is the host page (``clean_pages.review_clean``, as the host was cut) round the window
with the window laid back in and a blue box (``struck_pack.in_context``).  OUT/fold_K_clean/pack.json: raw the synthetic window,
erased the clean one, context the line with the clean window (what a shared reader reads, ``context_revision``); OUT/fold_K_raw: raw
and erased the synthetic window, context the line with it (``raw_context_revision``).  Image paths are absolute (OUT/crops).
OUT/fold_K.json the fold's kept regions in order with their sources and writers; OUT/summary.json the counts.  Student work: never in Git.

``answered``: the first ``--take`` regions of a fold's raw pack with the answers a teacher read (``recognize_lines`` on the clean
pack): a refusal or an empty reading is left out (and counted); the reading is the row's text, readable, split train (what
``train_reader --pack`` teaches without ``--fold``).
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ..util import SEED, digest, make_output_dir, require, write_json, write_manifest
from .restoration_pairs import SIZE

PREFIX = 'synthetic'


def order_key(region_id):
    return hashlib.sha256(f'{region_id}:{SEED}'.encode()).hexdigest()


def untiled(tiles, h, w):
    """The (h, w) crop ``restoration_pairs.canvas_tiles`` cut into these tiles (in its order)."""
    ph, pw = max(SIZE, h), max(SIZE, w); oy, ox = (ph-h)//2, (pw-w)//2
    at = lambda n: sorted(set(list(range(0, n-SIZE+1, 192))+[n-SIZE]))
    places = [(y, x) for y in at(ph) for x in at(pw)]
    require(len(places) == len(tiles), f'{len(tiles)} tiles for a {h}x{w} crop that makes {len(places)}')
    canvas = np.full((ph, pw), 255, np.uint8)
    for (y, x), tile in zip(places, tiles): canvas[y:y+SIZE, x:x+SIZE] = tile
    return canvas[oy:oy+h, ox:ox+w]


def sources(args, other):
    """{region id: (source, material writer or None, why left out or None)} of the synthetic regions, host writers aside."""
    canonical = lambda w: min(w, other.get(w, w))
    regions = json.loads((args.synth/'regions.json').read_text())
    prefill = {r['id']: r['group_uuid'] for r in json.loads((args.set/'training_native.json').read_text())}
    pieces = {p['id']: p['writer'] for p in json.loads((args.bank/'pieces.json').read_text())}
    out = {}
    for r in regions:
        kind, _, rest = r['unit'].partition(':')
        if r['source'] == 'D':
            out[r['id']] = ('D', None, None)
        elif r['source'] == 'H':
            writer = prefill.get(rest.rsplit('#', 1)[0])
            out[r['id']] = ('H', canonical(writer) if writer else None, None if writer else 'material_not_traced')
        elif r['source'] == 'P':
            writer = pieces.get(rest.rsplit('#', 1)[0])
            out[r['id']] = ('P', canonical(writer) if writer else None, None if writer else 'material_not_traced')
        else:
            out[r['id']] = (r['source'], None, 'unknown_source')
    return regions, out


def test_writers(args, canonical):
    regions = json.loads(args.dataset.read_text()); halves = json.loads(args.test_halves.read_text())
    pages = set(halves['development']) | set(halves['sealed'])
    found = {r['writer'] for r in regions if 'test-page' in r['export'] or r['page'] in pages}
    found |= {p.split('_')[0] for p in pages}
    return {canonical(w) for w in found}


def build(args):
    from .. import pages as pages_module
    from .clean_pages import review_clean
    from .prior_annotations import twins
    from .struck_pack import in_context, window
    other = twins(args.same_student); canonical = lambda w: min(w, other.get(w, w))
    rows = json.loads((args.pack/'pack.json').read_text())['rows']
    folds = sorted({r['fold'] for r in rows}); fold_writers = {k: {canonical(r['writer']) for r in rows if r['fold'] == k} for k in folds}
    tests = test_writers(args, canonical)
    regions, traced = sources(args, other); by_id = {r['id']: r for r in regions}
    kept, why = {}, {}
    for k in folds:
        banned = fold_writers[k] | tests; why[k] = Counter(); fine = []
        for r in regions:
            source, material, bad = traced[r['id']]
            host = canonical(r['host_writer'])
            reason = (bad or ('host_in_fold' if host in fold_writers[k] else 'host_test_writer' if host in tests else None)
                      or ('material_in_fold' if material in fold_writers[k] else 'material_test_writer' if material in tests else None))
            if reason: why[k][reason] += 1
            else: fine.append(r['id'])
        fine.sort(key=order_key); kept[k] = {'eligible': len(fine), 'eligible_by_source': dict(Counter(traced[i][0] for i in fine)), 'ids': fine[:args.take]}
        require(not {canonical(by_id[i]['host_writer']) for i in kept[k]['ids']} & banned, 'a banned host kept')
    union = sorted({i for k in folds for i in kept[k]['ids']})
    out = make_output_dir(args.output_dir); (out/'crops').mkdir()
    # S's tiles, region by region
    records = json.loads((args.synth/'S'/'records.json').read_text()); tiles_of = {}
    for n, r in enumerate(records): tiles_of.setdefault(r['id'].rsplit('~', 1)[0], []).append(n)
    data = np.load(args.synth/'S'/'gold_pixels.npy', mmap_mode='r'); targets = np.load(args.synth/'S'/'targets.npy', mmap_mode='r')
    files = pages_module.page_files(args.pages, sorted({by_id[i]['host_page'] for i in union}))
    made, host_check = {}, []
    for page_id in sorted({by_id[i]['host_page'] for i in union}):
        with Image.open(files[page_id]) as opened:
            clean_page = Image.fromarray(review_clean(np.asarray(opened.convert('L'))))
        grey = np.asarray(clean_page)
        for i in (i for i in union if by_id[i]['host_page'] == page_id):
            L, T, R, B = by_id[i]['box_on_page']; n = tiles_of[i]
            synth, clean = untiled([data[j] for j in n], B-T, R-L), untiled([targets[j] for j in n], B-T, R-L)
            host_check.append(int(np.abs(clean.astype(int)-grey[T:B, L:R].astype(int)).max()))
            at = window([[L, T], [R, T], [R, B], [L, B]], clean_page.size); l, t, r, b = at
            pieces = {}
            for name, box in (('synth', synth), ('clean', clean)):
                w = np.full((b-t, r-l), 255, np.uint8); w[T-t:B-t, L-l:R-l] = box; pieces[name] = w
            images = {'synth': pieces['synth'], 'clean': pieces['clean'], 'line_synth': in_context(clean_page, pieces['synth'], at),
                      'line_clean': in_context(clean_page, pieces['clean'], at)}
            made[i] = {}
            for name, image in images.items():
                path = out/'crops'/f'{i}_{name}.png'; (image if isinstance(image, Image.Image) else Image.fromarray(image)).save(path); made[i][name] = str(path)
    summary = {'regions': len(regions), 'test_writers': len(tests), 'kept_union': len(union),
               'host_box_equals_target': {'max_abs_difference_max': max(host_check), 'regions_differing': sum(d > 0 for d in host_check)}, 'folds': {}}
    for k in folds:
        ids = kept[k]['ids']
        row = lambda i, keys: {'id': i, 'line_id': i, 'writer': canonical(by_id[i]['host_writer']), 'fold': -1, 'val': False, 'kind': 'line', 'mode': 'erase',
                               'boxes': [], 'readable': True, 'text': '', 'target': '', 'source': traced[i][0], 'tier': by_id[i]['tier'],
                               'images': {key: made[i][name] for key, name in keys.items()}}
        for name, keys in (('clean', {'raw': 'synth', 'erased': 'clean', 'context': 'line_clean'}), ('raw', {'raw': 'synth', 'erased': 'synth', 'context': 'line_synth'})):
            (out/f'fold_{k}_{name}').mkdir(); write_json(out/f'fold_{k}_{name}'/'pack.json', {'schema': 'revrender-line-pack', 'rows': [row(i, keys) for i in ids]})
        listing = [{'rank': n, 'id': i, 'source': traced[i][0], 'host_writer': canonical(by_id[i]['host_writer']), 'material_writer': traced[i][1],
                    'host_page': by_id[i]['host_page'], 'unit': by_id[i]['unit'], 'tier': by_id[i]['tier']} for n, i in enumerate(ids)]
        write_json(out/f'fold_{k}.json', listing)
        summary['folds'][k] = {'eligible': kept[k]['eligible'], 'eligible_by_source': kept[k]['eligible_by_source'], 'kept': len(ids),
                               'kept_by_source': dict(Counter(traced[i][0] for i in ids)), 'first_half_by_source': dict(Counter(traced[i][0] for i in ids[:len(ids)//2])),
                               'left_out': dict(why[k])}
    write_json(out/'summary.json', summary)
    write_manifest(out, config={k: str(v) for k, v in vars(args).items() if k != 'command'},
                   inputs={str(p): digest(p) for p in (args.synth/'regions.json', args.synth/'S'/'records.json', args.pack/'pack.json', args.set/'training_native.json',
                                                       args.bank/'pieces.json', args.same_student, args.dataset, args.test_halves)}, **summary)
    print(json.dumps(summary), flush=True)


def answered(args):
    from ..eval.transcription import read_arm, refused
    rows = json.loads((args.pack/'pack.json').read_text())['rows'][:args.take]
    _, readings, files = read_arm('teacher='+str(args.readings))
    require({r['id'] for r in rows} <= set(readings), 'the teacher did not read every region')
    left = Counter(); kept = []
    for r in rows:
        text = readings[r['id']]
        if refused(text): left['refused'] += 1; continue
        if not text.strip(): left['empty'] += 1; continue
        kept.append({**r, 'text': text, 'target': text, 'readable': True, 'split': 'train'})
    out = make_output_dir(args.output_dir); write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': kept})
    lengths = Counter('1-2' if len(t) <= 2 else '3-9' if len(t) <= 9 else '10+' for t in (r['text'] for r in kept))
    summary = {'read': len(rows), 'kept': len(kept), 'left_out': dict(left), 'refused_share': left['refused']/max(1, len(rows)),
               'lengths': dict(lengths), 'kept_by_source': dict(Counter(r['source'] for r in kept))}
    write_manifest(out, config={k: str(v) for k, v in vars(args).items() if k != 'command'},
                   inputs={str(p): digest(p) for p in (args.pack/'pack.json', *files)}, **summary)
    print(json.dumps(summary), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    for key in ('synth', 'pack', 'set', 'bank', 'same-student', 'dataset', 'test-halves', 'output-dir'): b.add_argument('--'+key, type=Path, required=True)
    b.add_argument('--pages', type=Path, action='append', required=True); b.add_argument('--take', type=int, default=3000)
    a = sub.add_parser('answered')
    for key in ('pack', 'readings', 'output-dir'): a.add_argument('--'+key, type=Path, required=True)
    a.add_argument('--take', type=int, default=1500)
    args = p.parse_args(argv); {'build': build, 'answered': answered}[args.command](args)


if __name__ == '__main__':
    main()
