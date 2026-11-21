"""Synthetic regions as reader packs, fold by fold (plan of 2026-11-20, 第四轮_特权视图三样试验, section five): the regions of
``data.synthesis paste`` (its cleaned arm S) a fold may learn from, cut as the real regions are cut.

    python -m revrender.data.synth_views build --synth SYNTH --pack VIEWS --set PREFILL_SET --bank BANK --same-student PAIRS
        --dataset regions.json --test-halves configs/test_halves.json --pages DIR --take 3000 --output-dir OUT
    python -m revrender.data.synth_views answered --pack OUT/fold_K_raw --readings TEACHER_READ --take 1500 --output-dir ANSWERED
    python -m revrender.data.synth_views covered --pack VIEWS --synth SYNTH --same-student PAIRS --dataset regions.json
        --test-halves configs/test_halves.json --pages DIR --take 1000 --output-dir OUT

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

``covered`` (plan of 2026-11-21, 第六轮, section 4.2): regions whose writing is covered by its strike, fold by fold.  The material:
the whole strike layer (the annotator's erased ink) of the pack's readable regions that keep their writing (mode ``keep``) or whose
strike holds at least ``HEAVY`` of their ink (``window_stats``), as an opacity in its own ink (``strike_bank.opacity``).  The host: a
short stretch of clean writing (``SHORT`` ink pixels at most, ``LEAST`` at least; of a box's stretches the one nearest in shape to
the writing the material struck, ``matching_stretch``) cut from the host boxes of ``data.synthesis paste``
(``--synth``, its regions' ``host_page`` and ``box_on_page``; ``HOSTS`` of them in sha256 order, no test-page writer), the page as
``clean_pages.review_clean`` leaves it.  The layer is scaled by the ratio of the host's writing height to the height of the writing
it struck (times 1 +- ``SCALE``), its writing's centre put on the host's and moved to where it covers the host's writing most
(``covered_try``: a grid of moves of up to ``REACH`` heights), the opacity brought
back to its own (``one_stroke.renormed``) and drawn in the host's ink a little varied (``paste_strikes.host_level``,
``restoration_pairs.draw_pair``: the target is the host as it was).  The box: the host's writing and the strike, ``BOX_MARGIN``
heights round.  A region is kept when at least ``COVER`` of its writing lies within two pixels of its strike.  ``TRIES`` tries per
material, each with a host and moves of its own (seeded by its id); a fold takes the kept tries whose material is of another fold
and whose host and material writers are neither in the fold nor test-page writers (a student's two ids one), in sha256("<id>:42")
order, the first ``--take``.  Cut and packed as ``build`` (fold_K_clean, fold_K_raw, fold_K.json); OUT/summary.json each fold's
acceptance numbers (``window_stats``: cover, strike share, writing ink, the darkness AUC folded) and the host-equals-target check;
OUT/contrast.html the first 40 of fold 0 beside real keep regions of the pack.

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
HEAVY = .40            # a material region's strike holds at least this share of its ink (or it keeps its writing)
SHORT, LEAST = 300, 40  # a host stretch's ink pixels
COVER = .55            # a covered region has at least this share of its writing within two pixels of the strike
SCALE = .15            # the layer's scale drawn within 1 +- this
REACH, GRID = .25, 5   # the moves tried: a GRID x GRID grid of up to REACH writing heights each way
TRIES = 80
HOSTS = 1500


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


def auc(pos, neg):
    """The area under the ROC of ``pos`` above ``neg`` (ties shared), None for fewer than 20 of either."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 20 or len(neg) < 20:
        return None
    values = np.concatenate([pos, neg]); _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    order = values.argsort(kind='mergesort'); ranks = np.empty(len(values)); ranks[order] = np.arange(1, len(values)+1)
    sums = np.zeros(len(counts)); np.add.at(sums, inverse, ranks); ranks = (sums/counts)[inverse]
    return float((ranks[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))


def window_stats(scan, clean):
    """A struck window against its clean one (uint8): ink under grey 200, the strike the ink the clean one is more than 0.2 lighter
    at, the writing the rest; the strike's share of the ink, the writing within two pixels of the strike (``covered``), the writing's
    pixels, and the darkness AUC of strike against writing folded to its larger side (None for fewer than 20 of either)."""
    from scipy import ndimage
    s, c = np.asarray(scan, np.float32)/255, np.asarray(clean, np.float32)/255
    ink = s < 200/255; strike = (c-s > .2) & ink; kept = ink & ~strike
    near = ndimage.binary_dilation(strike, iterations=2); a = auc((1-s)[strike], (1-s)[kept])
    return {'strike_share': float(strike.sum()/max(1, ink.sum())), 'covered': float((kept & near).sum()/max(1, kept.sum())), 'kept': int(kept.sum()),
            'strike': int(strike.sum()), 'dark_auc_folded': None if a is None else max(a, 1-a)}


def host_stretches(clean):
    """Short stretches of writing in a clean host box: runs of inked columns (characters), consecutive ones taken while their ink
    stays within ``SHORT``; each (column from, column to) with at least ``LEAST`` ink pixels."""
    ink = clean < 200; cols = ink.sum(0); runs, start = [], None
    for x, n in enumerate(list(cols)+[0]):
        if n and start is None: start = x
        if not n and start is not None: runs.append((start, x)); start = None
    out = []
    for i in range(len(runs)):
        total = 0
        for j in range(i, len(runs)):
            total += int(cols[runs[j][0]:runs[j][1]].sum())
            if total > SHORT: break
            if total >= LEAST: out.append((runs[i][0], runs[j][1]))
    return out


def matching_stretch(crop, stretches, writing):
    """Of a host box's stretches, the one whose writing is nearest in shape (width over height) to the writing the material struck:
    scaled to the host's height, the strike then lies over the host's writing as it lay over its own."""
    wt, wl, wb, wr = writing; want = (wr-wl)/max(1, wb-wt)
    def shape(stretch):
        ys, xs = np.nonzero(crop[:, stretch[0]:stretch[1]] < 200)
        return abs(np.log(max(1, xs.max()-xs.min()+1)/max(1, ys.max()-ys.min()+1)/want))
    return min(stretches, key=shape)


def covered_try(material, page, box, stretch, rng):
    """One try: (box on page, opacity in the box, ink level) or None.  ``material``: (opacity, writing box (t, l, b, r)).  The layer,
    scaled once (the try's own draw), is put at each of ``GRID`` x ``GRID`` moves of up to ``REACH`` heights round the host's
    writing's centre and drawn; the move that leaves the most of the writing within two pixels of the strike is taken (the first of
    equals): a real strike was drawn over its own writing, and over another's it has to be put where it lies over it."""
    from scipy import ndimage
    from .one_stroke import BOX_MARGIN, renormed
    from .paste_strikes import host_level
    from .restoration_pairs import draw_pair
    alpha, (wt, wl, wb, wr) = material; L, T, R, B = box
    crop = page[T:B, L:R]; c0, c1 = stretch; ys, xs = np.nonzero(crop[:, c0:c1] < 200)
    ht, hb, hl, hr = T+ys.min(), T+ys.max()+1, L+c0+xs.min(), L+c0+xs.max()+1; h = hb-ht
    if h < 8:
        return None
    k = h/max(1, wb-wt)*(1+rng.uniform(-SCALE, SCALE))
    scaled = renormed(np.clip(ndimage.zoom(alpha, k, order=1), 0, 1).astype(np.float32), alpha)
    level = host_level(page[ht:hb, hl:hr], rng)
    cy, cx = (wt+wb)/2*k, (wl+wr)/2*k
    ys, xs = np.nonzero(scaled >= .3)
    if not len(ys):
        return None
    m = max(2, int(round(BOX_MARGIN*h))); best = None
    for dy in np.linspace(-REACH, REACH, GRID):
        for dx in np.linspace(-REACH, REACH, GRID):
            y0, x0 = int(round((ht+hb)/2+dy*h-cy)), int(round((hl+hr)/2+dx*h-cx))
            t, l = min(ht, y0+ys.min())-m, min(hl, x0+xs.min())-m; b, r = max(hb, y0+ys.max()+1)+m, max(hr, x0+xs.max()+1)+m
            if t < 0 or l < 0 or b > page.shape[0] or r > page.shape[1]:
                continue
            canvas = np.zeros((b-t, r-l), np.float32); sh, sw = scaled.shape
            ty, tx = y0-t, x0-l; a0, b0 = max(0, -ty), max(0, -tx); a1, b1 = min(sh, canvas.shape[0]-ty), min(sw, canvas.shape[1]-tx)
            canvas[ty+a0:ty+a1, tx+b0:tx+b1] = scaled[a0:a1, b0:b1]
            image = draw_pair(page[t:b, l:r], canvas, level)[0]
            ink = image < 200; strike = (page[t:b, l:r].astype(np.int16)-image > 51) & ink; kept = ink & ~strike
            cover = (kept & ndimage.binary_dilation(strike, iterations=2)).sum()/max(1, kept.sum())
            if best is None or cover > best[0]:
                best = (cover, [int(l), int(t), int(r), int(b)], canvas)
    return None if best is None else (best[1], best[2], level)


def covered(args):
    import random
    from .. import pages as pages_module
    from .clean_pages import review_clean
    from .prior_annotations import twins
    from .restoration_pairs import draw_pair
    from .strike_bank import ink_level, opacity
    from .struck_pack import in_context, window
    other = twins(args.same_student); canonical = lambda w: min(w, other.get(w, w))
    rows = [r for r in json.loads((args.pack/'pack.json').read_text())['rows'] if r['readable']]
    folds = sorted({r['fold'] for r in rows}); fold_writers = {k: {canonical(r['writer']) for r in rows if r['fold'] == k} for k in folds}
    tests = test_writers(args, canonical)
    materials = {}
    for r in rows:
        scan, clean = (np.asarray(Image.open(args.pack/r['images'][k]).convert('L')) for k in ('raw', 'erased'))
        st = window_stats(scan, clean)
        if r['mode'] != 'keep' and st['strike_share'] < HEAVY:
            continue
        ink = scan < 200; strike = ((clean.astype(np.float32)-scan)/255 > .2) & ink; kept = ink & ~strike
        if strike.sum() < 20 or kept.sum() < 20:
            continue
        ys, xs = np.nonzero(kept)
        materials[r['id']] = {'row': r, 'opacity': opacity(scan, ink_level(scan, strike), strike), 'writing': (ys.min(), xs.min(), ys.max()+1, xs.max()+1), 'stats': st}
    hosts = [h for h in json.loads((args.synth/'regions.json').read_text()) if canonical(h['host_writer']) not in tests]
    hosts = sorted(hosts, key=lambda h: order_key(h['id']))[:HOSTS]
    tries = [(f'C{n:04d}-{t:02d}', m) for n, m in enumerate(sorted(materials)) for t in range(TRIES)]
    rng_of = lambda i: random.Random(int(order_key(i)[:16], 16))
    plan = {i: (m, hosts[rng_of(i).randrange(len(hosts))]) for i, m in tries}
    files = pages_module.page_files(args.pages, sorted({h['host_page'] for _, h in plan.values()}))
    made, why = {}, Counter()
    by_page = {}
    for i, (m, h) in plan.items(): by_page.setdefault(h['host_page'], []).append(i)
    for page_id in sorted(by_page):
        with Image.open(files[page_id]) as opened:
            grey = review_clean(np.asarray(opened.convert('L')))
        for i in by_page[page_id]:
            m, h = plan[i]; rng = rng_of(i)                                   # a fresh stream of the try's own (the host was drawn apart)
            L, T, R, B = h['box_on_page']; stretches = host_stretches(grey[T:B, L:R])
            if not stretches: why['no_short_stretch'] += 1; continue
            got = covered_try((materials[m]['opacity'], materials[m]['writing']), grey, (L, T, R, B), matching_stretch(grey[T:B, L:R], stretches, materials[m]['writing']), rng)
            if got is None: why['off_the_page_or_small'] += 1; continue
            box, alpha, level = got; l, t, r, b = box; clean = grey[t:b, l:r]
            image, target = draw_pair(clean, alpha, level)
            at = window([[l, t], [r, t], [r, b], [l, b]], (grey.shape[1], grey.shape[0])); wl, wt, wr, wb = at
            pieces = {}
            for name, part in (('synth', image), ('clean', target)):
                w = np.full((wb-wt, wr-wl), 255, np.uint8); w[t-wt:b-wt, l-wl:r-wl] = part; pieces[name] = w
            st = window_stats(pieces['synth'], pieces['clean'])
            if st['strike'] < 20 or st['kept'] < 20 or st['covered'] < COVER: why['not_covered'] += 1; continue
            made[i] = {'material': m, 'host': h, 'box': box, 'at': at, 'level': level, 'stats': st, 'equal': bool(np.array_equal(target, clean))}
    kept, left = {}, {}
    for k in folds:
        banned = fold_writers[k] | tests; left[k] = Counter(); fine = []
        for i, v in made.items():
            mrow = materials[v['material']]['row']; host = canonical(v['host']['host_writer']); mat = canonical(mrow['writer'])
            reason = ('material_in_fold' if mrow['fold'] == k or mat in banned else 'host_in_fold' if host in fold_writers[k] else 'host_test_writer' if host in tests else None)
            if reason: left[k][reason] += 1
            else: fine.append(i)
        fine.sort(key=order_key); kept[k] = fine[:args.take]
    union = sorted({i for k in folds for i in kept[k]})
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); paths = {}
    for page_id in sorted({made[i]['host']['host_page'] for i in union}):
        with Image.open(files[page_id]) as opened:
            clean_page = Image.fromarray(review_clean(np.asarray(opened.convert('L'))))
        grey = np.asarray(clean_page)
        for i in (i for i in union if made[i]['host']['host_page'] == page_id):
            v = made[i]; m, _ = plan[i]; rng = rng_of(i); L, T, R, B = v['host']['box_on_page']
            stretch = matching_stretch(grey[T:B, L:R], host_stretches(grey[T:B, L:R]), materials[m]['writing'])
            box, alpha, level = covered_try((materials[m]['opacity'], materials[m]['writing']), grey, (L, T, R, B), stretch, rng)
            require(box == v['box'], f'{i}: the try does not come out the same')
            l, t, r, b = box; image, target = draw_pair(grey[t:b, l:r], alpha, level); wl, wt, wr, wb = v['at']
            pieces = {}
            for name, part in (('synth', image), ('clean', target)):
                w = np.full((wb-wt, wr-wl), 255, np.uint8); w[t-wt:b-wt, l-wl:r-wl] = part; pieces[name] = w
            images = {'synth': pieces['synth'], 'clean': pieces['clean'], 'line_synth': in_context(clean_page, pieces['synth'], v['at']), 'line_clean': in_context(clean_page, pieces['clean'], v['at'])}
            paths[i] = {}
            for name, image in images.items():
                path = out/'crops'/f'{i}_{name}.png'; (image if isinstance(image, Image.Image) else Image.fromarray(image)).save(path); paths[i][name] = str(path)
    median = lambda values: float(np.median([x for x in values if x is not None])) if any(x is not None for x in values) else None
    summary = {'materials': len(materials), 'materials_by_fold': dict(Counter(materials[m]['row']['fold'] for m in materials)), 'tries': len(tries), 'kept_tries': len(made),
               'left_out_tries': dict(why), 'hosts': len(hosts), 'test_writers': len(tests), 'host_equals_target': all(made[i]['equal'] for i in union), 'folds': {}}
    for k in folds:
        ids = kept[k]; st = [made[i]['stats'] for i in ids]
        row = lambda i, keys: {'id': i, 'line_id': i, 'writer': canonical(made[i]['host']['host_writer']), 'fold': -1, 'val': False, 'kind': 'line', 'mode': 'erase',
                               'boxes': [], 'readable': True, 'text': '', 'target': '', 'source': 'covered', 'images': {key: paths[i][name] for key, name in keys.items()}}
        for name, keys in (('clean', {'raw': 'synth', 'erased': 'clean', 'context': 'line_clean'}), ('raw', {'raw': 'synth', 'erased': 'synth', 'context': 'line_synth'})):
            (out/f'fold_{k}_{name}').mkdir(); write_json(out/f'fold_{k}_{name}'/'pack.json', {'schema': 'revrender-line-pack', 'rows': [row(i, keys) for i in ids]})
        write_json(out/f'fold_{k}.json', [{'rank': n, 'id': i, 'material': made[i]['material'], 'material_writer': canonical(materials[made[i]['material']]['row']['writer']),
                                           'material_fold': materials[made[i]['material']]['row']['fold'], 'host_writer': canonical(made[i]['host']['host_writer']),
                                           'host_page': made[i]['host']['host_page'], 'host_region': made[i]['host']['id'], 'box_on_page': made[i]['box'], **made[i]['stats']}
                                          for n, i in enumerate(ids)])
        banned = fold_writers[k] | tests
        summary['folds'][k] = {'eligible': len(made)-sum(left[k].values()), 'kept': len(ids), 'left_out': dict(left[k]),
                               'materials_used': len({made[i]['material'] for i in ids}),
                               'uses_per_material': {'max': max(Counter(made[i]['material'] for i in ids).values()), 'median': float(np.median(list(Counter(made[i]['material'] for i in ids).values())))},
                               'covered_quantiles': {str(q): float(np.percentile([x['covered'] for x in st], q)) for q in (10, 25, 50, 75, 90)},
                               'covered_at_least_0.7': float(np.mean([x['covered'] >= .7 for x in st])),
                               'covered_min': min(x['covered'] for x in st), 'covered_median': median([x['covered'] for x in st]),
                               'strike_share_median': median([x['strike_share'] for x in st]), 'writing_ink_median': median([x['kept'] for x in st]),
                               'dark_auc_folded_median': median([x['dark_auc_folded'] for x in st]),
                               'sources_clean': not any(canonical(made[i]['host']['host_writer']) in banned or materials[made[i]['material']]['row']['fold'] == k
                                                        or canonical(materials[made[i]['material']]['row']['writer']) in banned for i in ids)}
        f = summary['folds'][k]
        f['accepted'] = (f['kept'] == args.take and f['covered_min'] >= COVER and .60 <= f['covered_median'] <= .90 and f['strike_share_median'] >= .40
                         and f['writing_ink_median'] <= 300 and (f['dark_auc_folded_median'] or 1) <= .70 and f['sources_clean'] and summary['host_equals_target'])
    contrast_page(out, [(i, paths[i]['synth'], paths[i]['clean'], made[i]['stats']) for i in kept[folds[0]][:40]],
                  [(r['id'], args.pack/r['images']['raw'], args.pack/r['images']['erased'], window_stats(*(np.asarray(Image.open(args.pack/r['images'][k]).convert('L')) for k in ('raw', 'erased'))))
                   for r in sorted((r for r in rows if r['mode'] == 'keep'), key=lambda r: order_key(r['id']))[:12]])
    write_json(out/'summary.json', summary)
    write_manifest(out, config={k: str(v) for k, v in vars(args).items() if k != 'command'}, constants={'HEAVY': HEAVY, 'SHORT': SHORT, 'LEAST': LEAST, 'COVER': COVER,
                   'SCALE': SCALE, 'REACH': REACH, 'GRID': GRID, 'TRIES': TRIES, 'HOSTS': HOSTS},
                   inputs={str(p): digest(p) for p in (args.pack/'pack.json', args.synth/'regions.json', args.same_student, args.dataset, args.test_halves)}, **summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'folds'} | {'folds': {k: {x: v[x] for x in ('kept', 'covered_median', 'strike_share_median', 'writing_ink_median', 'dark_auc_folded_median', 'accepted')} for k, v in summary['folds'].items()}}), flush=True)


def contrast_page(out, synthetic, real):
    """OUT/contrast.html: synthetic covered windows and their clean targets, and real keep regions (scan, annotator's erased) beside."""
    import base64
    import io
    def png(path):
        with Image.open(path) as image:
            image = image.convert('L'); image = image.resize((image.width*2, image.height*2), Image.NEAREST); buf = io.BytesIO(); image.save(buf, 'PNG')
        return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
    cell = lambda i, a, b, st, kind: (f'<div class="c"><div class="t">{kind} {i} · cover {st["covered"]:.2f} · strike {st["strike_share"]:.2f} · writing {st["kept"]} px</div>'
                                      f'<img src="{png(a)}"><img src="{png(b)}"></div>')
    body = ''.join(cell(*x, 'synthetic') for x in synthetic) + '<h2>real regions that keep their writing (scan, annotator erased)</h2>' + ''.join(cell(*x, 'real') for x in real)
    (out/'contrast.html').write_text('<!doctype html><meta charset="utf-8"><title>Covered synthetic regions</title><style>body{font:13px sans-serif;margin:16px}'
                                     '.c{display:inline-block;margin:6px;padding:6px;border:1px solid #ccc;vertical-align:top}.t{margin-bottom:4px}img{display:block;margin:2px 0;'
                                     'image-rendering:pixelated}</style><h1>Covered synthetic regions, fold 0 (first 40): synthetic, clean target</h1>'+body, encoding='utf-8')


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
    c = sub.add_parser('covered')
    for key in ('pack', 'synth', 'same-student', 'dataset', 'test-halves', 'output-dir'): c.add_argument('--'+key, type=Path, required=True)
    c.add_argument('--pages', type=Path, action='append', required=True); c.add_argument('--take', type=int, default=1000)
    a = sub.add_parser('answered')
    for key in ('pack', 'readings', 'output-dir'): a.add_argument('--'+key, type=Path, required=True)
    a.add_argument('--take', type=int, default=1500)
    args = p.parse_args(argv); {'build': build, 'answered': answered, 'covered': covered}[args.command](args)


if __name__ == '__main__':
    main()
