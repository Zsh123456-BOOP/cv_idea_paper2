"""How much the annotator must fix on a prefill (plan 预填模型 of 2026-10-25, section 1), and whether a new prefill should replace it.

    python -m revrender.eval.prefill score --model NAME=RUN[:STEP] [...] --set NAME=NATIVE.json[:HINTS.npz] [...] --upstream UP --output-dir OUT
    python -m revrender.eval.prefill switch --scores OUT/scores.json --set NAME --current A --candidate B --output OUT.json

``score`` restores every region of each native set (``data.prefill_set``'s ``*_native``) with each model (a restorer run, at its
selected version or a given step directory; a hinted one with the set's hints) and counts on each region, as the editor shows it:
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


def blocks(mask):
    labels, n = ndimage.label(mask, np.ones((3, 3)))
    return int((np.bincount(labels.ravel())[1:] >= BLOCK).sum()) if n else 0


def edits(image, gold, restored):
    """The counts of one region: image and restored grey on [0, 1], gold the annotator's erased pixels."""
    ink = image < INK; removed = ((restored-image) > ERASED) & ink; truth = gold.astype(bool) & ink
    missed = truth & ~removed; wrong = removed & ~truth
    return {'missed': int(missed.sum()), 'writing_erased': int(wrong.sum()), 'strike': int(truth.sum()), 'writing': int((ink & ~truth).sum()),
            'to_fix': float((missed.sum()+wrong.sum())/max(1, truth.sum())), 'blocks': blocks(missed)+blocks(wrong)}


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
    from ..baselines.restoration import HINTED, restore_native
    from ..baselines.restoration_export import selected_predictor
    out = make_output_dir(args.output_dir); result = {}; inputs = {}
    sets = {}
    for spec in args.set:
        name, rest = spec.split('=', 1); path, _, hints = rest.partition(':'); path = Path(path)
        sets[name] = (json.loads(path.read_text()), np.load(path.with_suffix('.npz')), np.load(hints) if hints else None)
        inputs[str(path.with_suffix('.npz'))] = digest(path.with_suffix('.npz'))
    for spec in args.model:
        name, rest = spec.split('=', 1); run, _, step = rest.partition(':')
        predict, checkpoint, manifest = selected_predictor(Path(run), args.upstream, step or None); inputs[str(checkpoint)] = digest(checkpoint)
        hinted = manifest['config']['model'] in HINTED; result[name] = {'run': run, 'checkpoint': str(checkpoint), 'sets': {}}
        for set_name, (rows, arrays, hints) in sets.items():
            require(hints is not None or not hinted, f'{set_name}: a hinted model needs the hints')
            regions = []
            for k, r in enumerate(rows):
                image = arrays[f'image_{k}'].astype(np.float32)/255
                restored = restore_native(predict, image, hints[f'hint_{k}'].astype(np.float32) if hinted else None)
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score')
    s.add_argument('--model', action='append', required=True, help='NAME=RUN[:STEP]'); s.add_argument('--set', action='append', required=True, help='NAME=NATIVE.json[:HINTS.npz]')
    s.add_argument('--upstream', type=Path, required=True); s.add_argument('--output-dir', type=Path, required=True)
    w = sub.add_parser('switch')
    for key in ('scores', 'output'): w.add_argument('--'+key, type=Path, required=True)
    for key in ('set', 'current', 'candidate'): w.add_argument('--'+key, required=True)
    w.add_argument('--measure', choices=('blocks', 'to_fix'), default='blocks')
    args = p.parse_args(argv); {'score': score, 'switch': switch}[args.command](args)


if __name__ == '__main__':
    main()
