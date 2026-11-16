"""How much the annotator must fix on a prefill (plan 预填模型 of 2026-10-25, section 1), and whether a new prefill should replace it.

    python -m revrender.eval.prefill score --model NAME=RUN[:STEP] [...] --set NAME=NATIVE.json [...] --upstream UP --output-dir OUT
    python -m revrender.eval.prefill switch --scores OUT/scores.json --set NAME --current A --candidate B --output OUT.json

``score`` restores every region of each native set (``data.prefill_set``'s ``*_native``) with each model (a restorer run, at its
selected version or a given step directory) and counts on each region, as the editor shows it:
the strike pixels left (missed), the writing pixels erased (ink, grey < 200, the model brightened by more than 0.2 and the annotator
kept), the pixels to fix -- missed plus writing erased, over the strike's pixels --, the blocks to fix (8-connected components of at
least ``BLOCK`` pixels of either kind: about one stroke of the eraser or of the brush each), whether the region needs no edit (pixels
to fix at most ``FREE``), and pooled, the writing erased over the writing.  A region carrying the annotator's own edit count and
seconds (``new_native``) also correlates them with its pixels to fix.  ``switch`` applies the rule written before any reading: the
candidate replaces the current prefill when its pixels to fix, paired region by region, fall (mean difference below 0, the upper end
of its writer-resampled 95% interval below +1 point) and its writing erased rises by at most 2 points.  Revised before any candidate
was trained (2026-10-25): on training batch 1 the share of pixels to fix ran against the annotator's seconds and edits (Spearman
-0.2 to -0.4, a small region's share swings widest) while the blocks to fix followed them (+0.67 to +0.70), so ``--measure blocks``
(the default) pairs blocks, the candidate replacing when their mean difference is below 0 and the interval's upper end below
+``BLOCK_MARGIN`` of a block; the share is also reported pooled over the regions (``pooled_to_fix``).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..util import SEED, digest, make_output_dir, require, write_json, write_manifest

INK = 200/255
ERASED = .2
BLOCK = 8
FREE = .02
BLOCK_MARGIN = .25
NEAR = 2                         # pixels: a 3x3 dilation twice
STRIKE_CUTS = (.23, .35)         # the strike's share of the ink: light, middle, heavy (plan of 2026-11-16, fixed from now on)
NEAR_CUTS = (.28, .46)           # the writing's share near the strike


def blocks(mask):
    labels, n = ndimage.label(mask, np.ones((3, 3)))
    return int((np.bincount(labels.ravel())[1:] >= BLOCK).sum()) if n else 0


def edits(image, gold, restored):
    """The counts of one region: image and restored grey on [0, 1], gold the annotator's erased pixels.  With them the region's
    difficulty from the image and the annotator alone (plan of 2026-11-16): the strike's share of the ink (``strike_share``), the share
    of the writing left within ``NEAR`` pixels of the strike (``near_share``), and where the errors lie: strike missed where it presses
    on writing (within ``NEAR`` pixels of it), writing erased within ``NEAR`` pixels of the strike."""
    ink = image < INK; removed = ((restored-image) > ERASED) & ink; truth = gold.astype(bool) & ink; writing = ink & ~truth
    missed = truth & ~removed; wrong = removed & ~truth
    near_strike = ndimage.binary_dilation(truth, np.ones((3, 3)), iterations=NEAR); near_writing = ndimage.binary_dilation(writing, np.ones((3, 3)), iterations=NEAR)
    return {'missed': int(missed.sum()), 'writing_erased': int(wrong.sum()), 'strike': int(truth.sum()), 'writing': int(writing.sum()),
            'to_fix': float((missed.sum()+wrong.sum())/max(1, truth.sum())), 'blocks': blocks(missed)+blocks(wrong),
            'strike_share': float(truth.sum()/max(1, ink.sum())), 'near_share': float((writing & near_strike).sum()/max(1, writing.sum())),
            'missed_on_writing': int((missed & near_writing).sum()), 'erased_near_strike': int((wrong & near_strike).sum())}


def summary_of(regions):
    to_fix = [r['to_fix'] for r in regions]
    return {'regions': len(regions), 'to_fix': float(np.mean(to_fix)), 'blocks': float(np.mean([r['blocks'] for r in regions])),
            'no_edit_share': float(np.mean([v <= FREE for v in to_fix])),
            'writing_erased': sum(r['writing_erased'] for r in regions)/max(1, sum(r['writing'] for r in regions)),
            'missed_share': sum(r['missed'] for r in regions)/max(1, sum(r['strike'] for r in regions)),
            'pooled_to_fix': sum(r['missed']+r['writing_erased'] for r in regions)/max(1, sum(r['strike'] for r in regions))}


def spearman(a, b):
    from scipy.stats import spearmanr
    keep = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return float(spearmanr(*zip(*keep)).statistic) if len(keep) > 2 else None


def score(args):
    from ..baselines.restoration import restore_native
    from ..baselines.restoration_export import selected_predictor
    out = make_output_dir(args.output_dir); result = {}; inputs = {}
    sets = {}
    for spec in args.set:
        name, path = spec.split('=', 1); path = Path(path)
        sets[name] = (json.loads(path.read_text()), np.load(path.with_suffix('.npz')))
        inputs[str(path.with_suffix('.npz'))] = digest(path.with_suffix('.npz'))
    for spec in args.model:
        name, rest = spec.split('=', 1); run, _, step = rest.partition(':')
        predict, checkpoint, manifest = selected_predictor(Path(run), args.upstream, step or None); inputs[str(checkpoint)] = digest(checkpoint)
        result[name] = {'run': run, 'checkpoint': str(checkpoint), 'sets': {}}
        for set_name, (rows, arrays) in sets.items():
            regions = []
            for k, r in enumerate(rows):
                image = arrays[f'image_{k}'].astype(np.float32)/255
                restored = restore_native(predict, image)
                regions.append({'id': r['id'], 'writer': r['group_uuid'], **edits(image, arrays[f'mask_{k}'], restored),
                                **{k_: r[k_] for k_ in ('ops', 'seconds', 'model') if k_ in r}})
            s = summary_of(regions)
            if any('seconds' in r for r in regions):
                s['spearman_to_fix_seconds'] = spearman([r['to_fix'] for r in regions], [r.get('seconds') for r in regions])
                s['spearman_to_fix_ops'] = spearman([r['to_fix'] for r in regions], [r.get('ops') for r in regions])
                s['spearman_blocks_ops'] = spearman([r['blocks'] for r in regions], [r.get('ops') for r in regions])
            result[name]['sets'][set_name] = {'summary': s, 'regions': regions}
            print(name, set_name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}), flush=True)
    write_json(out/'scores.json', result)
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items() if k != 'func'}, inputs=inputs)


def switch(args):
    require(not args.output.exists(), 'output exists')
    scores = json.loads(args.scores.read_text())
    a = {r['id']: r for r in scores[args.current]['sets'][args.set]['regions']}; b = {r['id']: r for r in scores[args.candidate]['sets'][args.set]['regions']}
    ids = sorted(set(a) & set(b)); writers = sorted({a[i]['writer'] for i in ids}); by = {w: [i for i in ids if a[i]['writer'] == w] for w in writers}
    key = args.measure; margin = BLOCK_MARGIN if key == 'blocks' else .01
    diff = lambda chosen: float(np.mean([b[i][key]-a[i][key] for i in chosen]))
    rng = np.random.default_rng(SEED); draws = [diff([i for w in rng.choice(writers, len(writers)) for i in by[w]]) for _ in range(1000)]
    interval = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    erased = lambda s: sum(s[i]['writing_erased'] for i in ids)/max(1, sum(s[i]['writing'] for i in ids))
    rise = erased(b)-erased(a); d = diff(ids)
    result = {'set': args.set, 'current': args.current, 'candidate': args.candidate, 'regions': len(ids), 'writers': len(writers), 'measure': key,
              'difference': d, 'interval': interval, 'writing_erased_rise': rise,
              'replace': bool(d < 0 and interval[1] < margin and rise <= .02),
              'rule': f'replace when the paired {key} fall (mean < 0, writer-resampled 95% upper end < {margin}) and writing erased rises by at most 2 points'}
    write_json(args.output, {**result, 'inputs': {str(args.scores): digest(args.scores)}}); print(json.dumps(result), flush=True)


def tier(value, cuts):
    return 'light' if value <= cuts[0] else 'heavy' if value > cuts[1] else 'middle'


def by_tier(regions, key):
    """The regions' blocks by tier: count, blocks per region, blocks per 1,000 strike pixels; and the tiers' mean, each tier one weight."""
    out = {}
    for name in ('light', 'middle', 'heavy'):
        chosen = [r for r in regions if key(r) == name]
        out[name] = {'regions': len(chosen), 'blocks': float(np.mean([r['blocks'] for r in chosen])) if chosen else None,
                     'blocks_per_1000_strike': 1000*sum(r['blocks'] for r in chosen)/max(1, sum(r['strike'] for r in chosen)) if chosen else None}
    present = [v['blocks'] for v in out.values() if v['blocks'] is not None]
    return {**out, 'tiers_mean': float(np.mean(present)) if present else None}


def tiers(args):
    """Step 1 of the plan of 2026-11-16 (tables only): each model's blocks on a set by the strike's share of the ink, the writing's share
    near the strike and the strike's amount (thirds of the set), the tiers' mean; where its errors lie; on ``--train-set`` the same, to
    tell a model too small from data too few; ``--readings`` (eval.transcription --readings): each arm's AR and exact by the same tiers."""
    from .transcription import counts, rates
    require(not args.output.exists(), 'output exists')
    scores = json.loads(args.scores.read_text()); first = scores[next(iter(scores))]['sets'][args.set]['regions']
    cuts = [float(v) for v in np.percentile([r['strike'] for r in first], [100/3, 200/3])]
    keys = {'strike_share': lambda r: tier(r['strike_share'], STRIKE_CUTS), 'near_share': lambda r: tier(r['near_share'], NEAR_CUTS),
            'strike_amount': lambda r: tier(r['strike'], cuts)}
    def where(regions):
        total = lambda k: sum(r[k] for r in regions)
        return {'missed_share': total('missed')/max(1, total('strike')), 'writing_erased_share': total('writing_erased')/max(1, total('writing')),
                'errors_missed': total('missed')/max(1, total('missed')+total('writing_erased')),
                'missed_on_writing': total('missed_on_writing')/max(1, total('missed')), 'erased_near_strike': total('erased_near_strike')/max(1, total('writing_erased'))}
    result = {'set': args.set, 'cuts': {'strike_share': STRIKE_CUTS, 'near_share': NEAR_CUTS, 'strike_amount': cuts}, 'models': {}}
    for name, m in scores.items():
        regions = m['sets'][args.set]['regions']
        result['models'][name] = {'summary': summary_of(regions), **{k: by_tier(regions, f) for k, f in keys.items()}, 'where': where(regions)}
        if args.train_set and args.train_set in m['sets']:
            t = m['sets'][args.train_set]['regions']; result['models'][name]['train'] = {'summary': summary_of(t), 'where': where(t),
                                                                                         'strike_share': by_tier(t, keys['strike_share'])}
    if args.readings:
        feature = {r['id']: r for r in first}; rows = [json.loads(line) for line in args.readings.read_text().splitlines()]
        arms = [k for k in rows[0] if k not in ('region', 'line', 'writer', 'fold', 'readable', 'label')]; result['readings'] = {}
        for arm in arms:
            out = {}
            for k, f in keys.items():
                out[k] = {}
                for name in ('light', 'middle', 'heavy'):
                    chosen = [counts(r['label'], r[arm]) for r in rows if r['readable'] and r[arm] is not None and r['region'] in feature and f(feature[r['region']]) == name]
                    out[k][name] = {'regions': len(chosen), **({m: rates(chosen)[m] for m in ('ar', 'exact')} if chosen else {})}
                present = [v for v in out[k].values() if v['regions']]
                out[k]['tiers_mean'] = {m: float(np.mean([v[m] for v in present])) for m in ('ar', 'exact')} if present else None
            out['regions_with_features'] = sum(r['region'] in feature for r in rows if r['readable'])
            result['readings'][arm] = out
    write_json(args.output, {**result, 'inputs': {str(p): digest(p) for p in (args.scores, *([args.readings] if args.readings else []))}})
    print(json.dumps({n: {'blocks': m['summary']['blocks'], 'heavy': m['strike_share']['heavy']['blocks'], 'tiers_mean': m['strike_share']['tiers_mean']}
                      for n, m in result['models'].items()}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score')
    s.add_argument('--model', action='append', required=True, help='NAME=RUN[:STEP]'); s.add_argument('--set', action='append', required=True, help='NAME=NATIVE.json')
    s.add_argument('--upstream', type=Path, required=True); s.add_argument('--output-dir', type=Path, required=True)
    w = sub.add_parser('switch')
    for key in ('scores', 'output'): w.add_argument('--'+key, type=Path, required=True)
    for key in ('set', 'current', 'candidate'): w.add_argument('--'+key, required=True)
    w.add_argument('--measure', choices=('blocks', 'to_fix'), default='blocks')
    t = sub.add_parser('tiers')
    for key in ('scores', 'output'): t.add_argument('--'+key, type=Path, required=True)
    t.add_argument('--set', required=True); t.add_argument('--train-set'); t.add_argument('--readings', type=Path)
    args = p.parse_args(argv); {'score': score, 'switch': switch, 'tiers': tiers}[args.command](args)


if __name__ == '__main__':
    main()
