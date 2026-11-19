"""Why a restorer's erased crop reads worse than the annotator's (plan of 2026-11-19, NAFNet五折与差距分析): one table and the analyses.

    python -m revrender.eval.erase_gap analyse --pack VIEWS --sets SETS --erased SEPARATED --arm NAME=DIR[,DIR...] [...]
        --scores REF=SCORES.json [...] --output-dir OUT
    python -m revrender.eval.erase_gap sheets --table OUT/regions.jsonl --human HUMAN_PACK --erased SEPARATED --output-dir SHEETS

``analyse`` reads the arms (``recognize_lines`` folders, every region read by a reader that never saw its writer), the windowed sets
(``data.struck_pack fold-sets --windowed``: each region's scan, the annotator's strike and polygon) and the restorer's erased windows,
and writes OUT/regions.jsonl, a row per region: writer, fold, readable, annotation mode, the label's spelt length, the groups of E, the
pixel measures of D and every arm's reading, raw output and why it stopped; and OUT/analysis.json, the plan's analyses:

  A  the cross table of H, E-NAFNet and R0 right or wrong; the regions H reads right and E-NAFNet not, by outcome (refused, 1, 2, 3 or
     more edits) and their characters substituted, deleted, inserted; every arm's characters substituted / deleted / inserted.
  D  on the regions H reads right: per region, inside the polygon, the strike left (residual share: missed over the annotator's strike),
     the writing erased (damage share: over the writing, ``eval.prefill.edits``), the pixels darkened by more than 0.2, and the residual
     and the damage near the writing / the strike or away from it (``eval.restoration_errors.zone_counts``, ``NEAR``); each in thirds of
     the regions (by value, ties by region id), E-NAFNet's exact share in each third, low minus high with a writer-resampled interval,
     and the Spearman correlation with E-NAFNet's NED.
  E  groups: annotation mode; the label's length (1-2, 3-9, 10 and more), single characters, fractions; the strike's share of the ink
     (``eval.prefill`` light / middle / heavy) and its amount (thirds of the set); the share of the strike pressing on writing (thirds).
     Per group: readable regions, exact and AR of H, E-NAFNet and R0, H minus E-NAFNet; groups of fewer than 30 readable regions are
     flagged not to be read.

B, C and F come from ``eval.transcription`` (``--scores REF=FILE``, one run per reference); analysis.json carries their differences,
intervals, the plan's verdicts (section four) and the share of the gap each F closes, (F - E) / (H - E).

``sheets``: the plan's G, three kinds of region H reads right and E-NAFNet not -- refused; missing characters (some deleted, none
substituted); wrong characters (the rest) -- the first ``SHEET_TAKE`` of each in the order of sha256("<region>:42"): the scan, the
annotator's erased crop, the restorer's, and where they differ (red the strike it left, blue the writing it erased), with the label and
the two readings.  Student work: never in Git.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..util import SEED, digest, make_output_dir, require, write_json, write_manifest
from .coverage import prepare
from .prefill import STRIKE_CUTS, edits as pixel_edits, spearman, tier
from .transcription import counts, read_arm, refused

ARMS = ('H', 'E-NAFNet', 'R0', 'F-residual', 'F-damage', 'F-both', 'E-later1', 'E-later2')
SMALL = 30          # a group of fewer readable regions is not read (rule 4)
SHEET_TAKE = 12
MEASURES = ('residual_share', 'damage_share', 'darkened', 'residual_near', 'residual_away', 'damage_near', 'damage_away')


def thirds(values, ids):
    """'low' / 'middle' / 'high' of each region, a third of the regions each, by value and then by region id."""
    order = sorted(range(len(values)), key=lambda k: (values[k], ids[k])); out = [None]*len(values)
    for rank, k in enumerate(order): out[k] = ('low', 'middle', 'high')[min(2, 3*rank//max(1, len(values)))]
    return out


def pixel_measures(image, mask, inner, restored):
    """D's measures of one region (grey on [0, 1], inside the polygon)."""
    from ..baselines.restoration import inside
    from .restoration_errors import zone_counts
    a, r, g = inside(image, restored, mask, inner)
    e = pixel_edits(a, g, r)
    zone = zone_counts(np.round(255*image).astype(np.int16), mask, inner.astype(bool), np.round(255*restored).astype(np.int16))
    return {'strike': e['strike'], 'writing': e['writing'], 'strike_share': e['strike_share'],
            'residual_share': e['missed']/max(1, e['strike']), 'damage_share': e['writing_erased']/max(1, e['writing']),
            'darkened': int((((image-restored) > .2) & inner.astype(bool)).sum()),
            'residual_near': zone['stroke_near_writing'][1]/max(1, e['strike']), 'residual_away': zone['stroke_away'][1]/max(1, e['strike']),
            'damage_near': zone['writing_near_stroke'][1]/max(1, e['writing']), 'damage_away': zone['writing_away'][1]/max(1, e['writing']),
            'press_share': zone['stroke_near_writing'][0]/max(1, e['strike'])}


def table(args):
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    erased = {r['id']: r for r in json.loads((args.erased/'pack.json').read_text())['rows']}
    arms, raw = {}, {}
    for spec in args.arm:
        name, readings, files = read_arm(spec); arms[name] = readings
        raw[name] = {g['id']: g for f in files for g in map(json.loads, f.read_text(encoding='utf-8').splitlines())}
    require(set(arms) == set(ARMS), f'the arms are {ARMS}')
    out = []
    for folder in sorted(args.sets.glob('fold_*')):
        native = json.loads((folder/'heldout_native.json').read_text()); arrays = np.load(folder/'heldout_native.npz')
        for i, n in enumerate(native):
            if not all(n['id'] in raw[name] for name in ARMS): continue      # a region some arm did not read (a smoke test)
            r = rows[n['id']]; image = arrays[f'image_{i}'].astype(np.float32)/255
            restored = np.asarray(Image.open(args.erased/erased[n['id']]['images']['erased']).convert('L'), np.float32)/255
            label = prepare(r['text'] or '').replace('?', '')
            row = {'id': r['id'], 'writer': r['writer'], 'fold': r['fold'], 'readable': r['readable'], 'mode': r['mode'], 'label': r['text'],
                   'length': len(label), 'single': len(r['target']) == 1, 'fraction': '/' in r['target'],
                   **pixel_measures(image, arrays[f'mask_{i}'], arrays[f'inner_{i}'], restored)}
            for name in ARMS:
                g = raw[name][r['id']]
                row[name] = {'reading': g['text'], 'raw': g['raw'], 'finish_reason': g['finish_reason'],
                             **(counts(r['text'], g['text']) if r['readable'] else {'refused': refused(g['text'])})}
            out.append(row)
    require(out, 'no region read by every arm')
    cuts = np.percentile([r['strike'] for r in out], [100/3, 200/3]).tolist()
    press = thirds([r['press_share'] for r in out], [r['id'] for r in out])
    for r, p in zip(out, press):
        r.update(strike_share_tier=tier(r['strike_share'], STRIKE_CUTS), strike_amount_tier=tier(r['strike'], cuts), press_tier=p)
    return out, cuts


def ned(c):
    return (c['s']+c['d']+c['i'])/max(c['n'], c['m'], 1)


def rates(rows, arm):
    n, s, d, i = (sum(r[arm][k] for r in rows) for k in 'nsdi')
    return {'regions': len(rows), 'exact': float(np.mean([r[arm]['exact'] for r in rows])) if rows else None, 'ar': (n-s-d-i)/n if n else None,
            'cr': (n-s-d)/n if n else None, 'cer': (s+d+i)/n if n else None, 'substituted': s, 'deleted': d, 'inserted': i}


def resampled(rows, value, rng, draws=1000):
    """value(rows) and its writer-resampled 95% interval."""
    writers = sorted({r['writer'] for r in rows}); by = {w: [r for r in rows if r['writer'] == w] for w in writers}
    sample = [value([r for w in rng.choice(writers, len(writers)) for r in by[w]]) for _ in range(draws)]
    sample = [v for v in sample if v is not None]
    return {'value': value(rows), 'interval': [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))] if sample else [None, None]}


def verdict(a, interval_key='exact'):
    """Section four's rule: the exact interval excludes 0 and |AR difference| >= 2 points."""
    lo, hi = a[interval_key]['interval']; clear = (lo > 0 or hi < 0) and abs(a['ar']['difference']) >= .02
    return ('clearly_better' if lo > 0 else 'clearly_worse') if clear else 'cannot_tell'


def analyse(args):
    out = make_output_dir(args.output_dir); regions, cuts = table(args)
    with (out/'regions.jsonl').open('x', encoding='utf-8') as stream:
        for r in regions: stream.write(json.dumps(r, ensure_ascii=False)+'\n')
    can = [r for r in regions if r['readable']]; right = lambda r, a: bool(r[a]['exact'])
    rng = np.random.default_rng(SEED)
    # A
    cross = {}
    for r in can:
        key = '/'.join(f"{a}{'+' if right(r, a) else '-'}" for a in ('H', 'E-NAFNet', 'R0')); cross[key] = cross.get(key, 0)+1
    lost = [r for r in can if right(r, 'H') and not right(r, 'E-NAFNet')]
    outcome = {'refused': 0, '1': 0, '2': 0, '3+': 0}
    for r in lost:
        e = r['E-NAFNet']; k = e['s']+e['d']+e['i']
        outcome['refused' if refused(e['reading']) else str(k) if k < 3 else '3+'] += 1
    a_part = {'cross': dict(sorted(cross.items())), 'h_right_e_wrong': {'regions': len(lost), 'outcome': outcome,
              'characters': {k: sum(r['E-NAFNet'][c] for r in lost) for k, c in (('substituted', 's'), ('deleted', 'd'), ('inserted', 'i'))}},
              'characters_by_arm': {a: {k: v for k, v in rates(can, a).items() if k in ('substituted', 'deleted', 'inserted', 'cr')} for a in ARMS}}
    # D
    base = [r for r in can if right(r, 'H')]; d_part = {'regions': len(base)}
    for m in MEASURES:
        third = dict(zip([r['id'] for r in base], thirds([r[m] for r in base], [r['id'] for r in base])))
        exact = lambda chosen, t: float(np.mean([r['E-NAFNet']['exact'] for r in chosen if third[r['id']] == t])) if any(third[r['id']] == t for r in chosen) else None
        gap = lambda chosen: None if exact(chosen, 'low') is None or exact(chosen, 'high') is None else exact(chosen, 'low')-exact(chosen, 'high')
        low_high = resampled(base, gap, rng)
        d_part[m] = {'thirds': {t: {'regions': sum(third[r['id']] == t for r in base), 'exact': exact(base, t),
                                    'range': [min(r[m] for r in base if third[r['id']] == t), max(r[m] for r in base if third[r['id']] == t)]} for t in ('low', 'middle', 'high')},
                     'low_minus_high': low_high, 'related': None not in low_high['interval'] and (low_high['interval'][0] > 0 or low_high['interval'][1] < 0),
                     'spearman_ned': spearman([r[m] for r in base], [ned(r['E-NAFNet']) for r in base])}
    # E
    groups = {'mode': lambda r: r['mode'], 'length': lambda r: '1-2' if r['length'] <= 2 else '3-9' if r['length'] <= 9 else '10+',
              'single': lambda r: 'single' if r['single'] else None, 'fraction': lambda r: 'fraction' if r['fraction'] else None,
              'strike_share': lambda r: r['strike_share_tier'], 'strike_amount': lambda r: r['strike_amount_tier'], 'press': lambda r: r['press_tier']}
    e_part = {}
    for g, key in groups.items():
        e_part[g] = {}
        for value in sorted({key(r) for r in can} - {None}):
            chosen = [r for r in can if key(r) == value]; s = {a: rates(chosen, a) for a in ('H', 'E-NAFNet', 'R0')}
            e_part[g][value] = {'regions': len(chosen), 'not_read': len(chosen) < SMALL, **{a: {'exact': s[a]['exact'], 'ar': s[a]['ar']} for a in s},
                                'h_minus_e': {'exact': s['H']['exact']-s['E-NAFNet']['exact'], 'ar': s['H']['ar']-s['E-NAFNet']['ar']}}
    # B, C, F from eval.transcription
    scores = {}
    for spec in args.scores:
        ref, _, path = spec.partition('='); scores[ref] = json.loads(Path(path).read_text())['arms']
    vs = lambda ref, arm: scores[ref][arm]['against_reference']
    point = {a: scores['E-NAFNet'][a] for a in ARMS}
    share = lambda f, m: (point[f][m]-point['E-NAFNet'][m])/(point['H'][m]-point['E-NAFNet'][m]) if point['H'][m] != point['E-NAFNet'][m] else None
    drags = {'clearly_better': 'clearly_drags_reading', 'clearly_worse': 'putting_right_clearly_hurts', 'cannot_tell': 'cannot_tell'}
    c_part = {f: {**{k: vs('E-NAFNet', f)[k] for k in ('ar', 'exact', 'ned')}, 'verdict': drags[verdict(vs('E-NAFNet', f))],
                  'gap_closed': {m: share(f, m) for m in ('ar', 'exact')}} for f in ('F-residual', 'F-damage', 'F-both')}
    between = vs('F-residual', 'F-damage')
    both_vs_h = vs('H', 'F-both')
    result = {'regions': len(regions), 'readable': len(can), 'strike_amount_cuts': cuts, 'strike_share_cuts': STRIKE_CUTS,
              'arms': {a: {k: v for k, v in scores['E-NAFNet'][a].items() if k != 'against_reference'} for a in ARMS},
              'A': a_part,
              'B': {**{k: vs('R0', 'E-NAFNet')[k] for k in ('ar', 'exact', 'ned')}, 'verdict': {'clearly_better': 'erasing_clearly_helps', 'clearly_worse': 'erasing_clearly_hurts',
                                                                                              'cannot_tell': 'cannot_tell'}[verdict(vs('R0', 'E-NAFNet'))]},
              'C': {**c_part, 'damage_minus_residual': {**{k: between[k] for k in ('ar', 'exact')}, 'verdict': verdict(between)},
                    'both_minus_h': {k: both_vs_h[k] for k in ('ar', 'exact')}, 'rule6_left_over': abs(both_vs_h['ar']['difference']) > .02},
              'D': d_part, 'E': e_part,
              'F': {a: {k: vs('E-NAFNet', a)[k] for k in ('ar', 'exact', 'ned')} for a in ('E-later1', 'E-later2')}}
    write_json(out/'analysis.json', result)
    inputs = {str(p): digest(p) for p in [args.pack/'pack.json', args.erased/'pack.json', *(Path(s.partition('=')[2]) for s in args.scores)]}
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items() if k != 'command'}, inputs=inputs,
                   regions=len(regions), readable=len(can))
    print(json.dumps({'B': result['B']['verdict'], 'C': {f: c_part[f]['verdict'] for f in c_part}, 'gap_closed': {f: c_part[f]['gap_closed'] for f in c_part},
                      'D_related': [m for m in MEASURES if d_part[m]['related']]}), flush=True)


def kind_of(r):
    e = r['E-NAFNet']
    return 'refused' if refused(e['reading']) else 'missing' if e['d'] and not e['s'] else 'wrong'


def sheets(args):
    from PIL import ImageDraw, ImageFont
    regions = [json.loads(l) for l in args.table.read_text(encoding='utf-8').splitlines()]
    human = {r['id']: r for r in json.loads((args.human/'pack.json').read_text())['rows']}
    erased = {r['id']: r for r in json.loads((args.erased/'pack.json').read_text())['rows']}
    font = ImageFont.truetype(args.font, 18); out = make_output_dir(args.output_dir); index = {}
    order = lambda r: hashlib.sha256(f"{r['id']}:{SEED}".encode()).hexdigest()
    lost = [r for r in regions if r['readable'] and r['H']['exact'] and not r['E-NAFNet']['exact']]
    for kind in ('refused', 'missing', 'wrong'):
        chosen = sorted((r for r in lost if kind_of(r) == kind), key=order)[:SHEET_TAKE]; tiles = []
        for n, r in enumerate(chosen, 1):
            grey = lambda root, row, key: np.asarray(Image.open(root/row['images'][key]).convert('L'))
            scan, hand, model = grey(args.human, human[r['id']], 'raw'), grey(args.human, human[r['id']], 'erased'), grey(args.erased, erased[r['id']], 'erased')
            ink = scan < 200; erased_by = lambda a: ((a.astype(np.int16)-scan) > 51) & ink
            diff = np.stack([scan]*3, -1).copy(); diff[erased_by(hand) & ~erased_by(model)] = (220, 0, 0); diff[erased_by(model) & ~erased_by(hand)] = (0, 90, 255)
            k = max(1, 120//max(1, scan.shape[0]))
            pics = [Image.fromarray(a).convert('RGB').resize((scan.shape[1]*k, scan.shape[0]*k), Image.Resampling.NEAREST) for a in (scan, hand, model, diff)]
            w = sum(p.width for p in pics)+50; h = max(p.height for p in pics)+100; tile = Image.new('RGB', (max(w, 900), h), 'white'); x = 10
            for p in pics: tile.paste(p, (x, 10)); x += p.width+10
            draw = ImageDraw.Draw(tile); code = f'{kind[0].upper()}{n:02d}'
            for line, text in enumerate((f"{code}  答案：{r['label']}", f"H：{r['H']['reading']}", f"E-NAFNet：{r['E-NAFNet']['reading']}")):
                draw.text((10, h-90+28*line), text.replace('\n', ' ⏎ '), fill='black', font=font)
            tiles.append(tile); index[code] = {'id': r['id'], 'kind': kind, 'label': r['label'], 'H': r['H']['reading'], 'E-NAFNet': r['E-NAFNet']['reading']}
        if tiles:
            sheet = Image.new('RGB', (max(t.width for t in tiles), sum(t.height+6 for t in tiles)), (200, 200, 200)); y = 0
            for t in tiles: sheet.paste(t, (0, y)); y += t.height+6
            sheet.save(out/f'{kind}.png')
    write_json(out/'index.json', index)
    write_json(out/'legend.json', {'columns': ['scan', 'annotator erased', 'NAFNet erased', 'difference: red strike left, blue writing erased'],
                                   'kinds': {'refused': 'E-NAFNet refused', 'missing': 'some characters deleted, none substituted', 'wrong': 'the rest'},
                                   'order': f'sha256("<region>:{SEED}"), the first {SHEET_TAKE}', 'counts': {k: sum(kind_of(r) == k for r in lost) for k in ('refused', 'missing', 'wrong')}})
    print(json.dumps({k: sum(v['kind'] == k for v in index.values()) for k in ('refused', 'missing', 'wrong')}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('analyse')
    for key in ('pack', 'sets', 'erased', 'output-dir'): a.add_argument('--'+key, type=Path, required=True)
    a.add_argument('--arm', action='append', required=True, help='NAME=DIR[,DIR...], NAME one of '+', '.join(ARMS))
    a.add_argument('--scores', action='append', required=True, help='REF=eval.transcription output with that reference (R0, E-NAFNet, H, F-residual)')
    s = sub.add_parser('sheets')
    for key in ('table', 'human', 'erased', 'output-dir'): s.add_argument('--'+key, type=Path, required=True)
    s.add_argument('--font', default='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    args = p.parse_args(argv); {'analyse': analyse, 'sheets': sheets}[args.command](args)


if __name__ == '__main__':
    main()
