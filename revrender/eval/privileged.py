"""The rules and reports of the fourth round (plan of 2026-11-20, 第四轮_特权视图三样试验, sections 4.1-4.3): X (a reader taught on
NAFNet's held-out erased crops), V (the reader's vision tower taught to see the scan as clean) and A (a reader taught first on
synthetic regions a teacher read); and of the fifth (第五轮_X与V结合, section 5): V with a matched reader (VR-真+合, VR-真), the
adapter looking at NAFNet's crop (NV, NVR) or at the scan with NAFNet's erasure in red (MV, MVR); and of the sixth (第六轮, section
5): V-修 (the tower's positions mended), V-涂满 (with covered synthetic regions), L25 / L50 (the learning curve).

    python -m revrender.eval.privileged shares --pack VIEWS [--compare REGION_STATS.json] --output SHARES.json

``shares``: each region's strike share of its ink (``data.synth_views.window_stats`` on the pack's raw and erased windows), the
readable regions above their median the heavier half and the rest the lighter, and their quarters; with ``--compare`` region by
region against another count (the planning session's ``strike_share``): the largest difference and the regions put in another half.

    python -m revrender.eval.privileged analyse --pack VIEWS --arm NAME=DIR[,DIR...] [...] --scores REF=FILE [...]
        [--distances NAME=DIR[,DIR...] ...] [--teacher FILE ...] --output-dir OUT

``--scores``: ``eval.transcription`` runs against E-NAFNet, R0, H, V-真 and R0-长.  The rules, each as the plan writes it ("成立": the
exact interval's lower end above 0 and the AR difference at least +2; "worse": the exact interval wholly below 0):
X1 X against E-NAFNet; X2 X against R0 (exact lower end above 0); V1 V-真+合 against E-NAFNet; V2 V-真+合 against R0; V3 V-真+合
against V-真; A1 R0-长 against R0 (then A's control is R0-长, else R0); A2 A against its control.  The fifth round's four grades
(``grade``): holds, worse, exact up but AR short (the exact interval's lower end above 0, the AR difference under +2), cannot tell;
rule 1 VR-真+合 against V-真+合, rule 2 NVR against VR-真, rule 3 MVR against VR-真, rule 4 each new reading against V-真+合: of
those that hold, the highest exact is the next version, within 0.2 points a tie, broken for the one without NAFNet, then for the
shared reader (``next_version``).  Reported only: the pairs ``REPORTED``, each new reading against H, R0, E-NAFNet and X.  The
sixth round (``--shares``): rule F V-修 against V-真+合 (the four grades); rule A V-涂满 against V-修 on the heavier half (``rule_a``),
the whole and the lighter half as its floor, the four grades of the whole beside; reported the learning curve (L25, L50, V-修 on the
whole and each half, L50 - L25, V-修 - L50 and a doubling's gain carried to twice the data, marked rough) and each new reading
against V-真+合, H, R0 and X; the groups also by half and quarter of the strike share.  Reported only: exact by
annotation mode and the label's length, every arm's refusal, every arm against H; the feature distances (``train_clean_view
distance``: on the touched tokens of each region, the mean cosine to the clean crop's features of the scan, NAFNet's crop and the
student), their means and the paired differences with writer-resampled 95% intervals; the teacher's answers of A (``--teacher``:
the run manifests of ``data.synth_views answered``).  OUT/analysis.json; OUT/regions.jsonl a row per region with every arm's
reading, raw output and why it stopped.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..util import SEED, digest, make_output_dir, require, write_json, write_manifest
from .coverage import prepare
from .hint_loop import raw_rows, verdict_plus
from .transcription import counts, paired, rates

LEVELS = ('merged', 'deepstack1', 'deepstack2', 'deepstack3')
NEW = ('VR-真+合', 'VR-真', 'NV', 'NVR', 'MV', 'MVR')
WITH_NAFNET, SHARED = {'NV', 'NVR', 'MV', 'MVR'}, {'NV', 'MV'}
RULES5 = (('rule1', 'VR-真+合', 'V-真+合'), ('rule2', 'NVR', 'VR-真'), ('rule3', 'MVR', 'VR-真'))
REPORTED = (('VR-真', 'V-真'), ('NV', 'V-真'), ('MV', 'V-真'), ('MVR', 'NVR'))


def worse(a):
    return a['exact']['interval'][1] < 0


def grade(a):
    """The fifth round's four grades of a paired difference."""
    if verdict_plus(a): return 'holds'
    if worse(a): return 'worse'
    return 'exact_up_ar_short' if a['exact']['interval'][0] > 0 else 'cannot_tell'


def next_version(graded, exact):
    """Rule 4: of the new readings that hold against V-真+合 (``graded`` name -> grade), the highest exact (``exact`` name -> rate);
    within 0.2 points a tie, broken for the one without NAFNet, then for the shared reader; None when none holds."""
    held = [n for n in NEW if graded.get(n) == 'holds']
    if not held: return None
    top = max(exact[n] for n in held); tied = [n for n in held if top-exact[n] < .002]
    for prefer in (lambda n: n not in WITH_NAFNET, lambda n: n in SHARED):
        if any(prefer(n) for n in tied) and not all(prefer(n) for n in tied): tied = [n for n in tied if prefer(n)]
    return max(tied, key=lambda n: exact[n])


def rule_a(heavy, light, whole):
    """Rule A of the sixth round from the paired differences on the heavier half, the lighter half and the whole."""
    main_holds, floor = heavy['exact']['interval'][0] > 0, whole['exact']['interval'][1] >= 0 and light['exact']['interval'][1] >= 0
    if heavy['exact']['interval'][1] < 0: return 'worse'
    if main_holds: return 'useful' if floor else 'heavy_better_elsewhere_worse'
    return 'cannot_tell'


def strike_shares(pack):
    from ..data.synth_views import window_stats
    from PIL import Image
    rows = json.loads((pack/'pack.json').read_text())['rows']
    return {r['id']: window_stats(*(np.asarray(Image.open(pack/r['images'][k]).convert('L')) for k in ('raw', 'erased')))['strike_share'] for r in rows}, rows


def shares(args):
    share, rows = strike_shares(args.pack); readable = [r['id'] for r in rows if r['readable']]
    median = float(np.median([share[k] for k in readable])); quarters = [float(np.percentile([share[k] for k in readable], q)) for q in (25, 50, 75)]
    half = {k: 'heavy' if share[k] > median else 'light' for k in readable}
    tier = {k: f'q{1+sum(share[k] > q for q in quarters)}' for k in readable}
    result = {'median': median, 'quarter_cuts': quarters, 'heavy': sum(v == 'heavy' for v in half.values()), 'light': sum(v == 'light' for v in half.values()),
              'regions': {k: {'strike_share': share[k], **({'half': half[k], 'quarter': tier[k]} if k in half else {})} for k in share}}
    if args.compare:
        theirs = {k: v['strike_share'] for k, v in json.loads(args.compare.read_text()).items()}
        common = sorted(set(theirs) & set(share)); their_median = float(np.median([theirs[k] for k in readable if k in theirs]))
        moved = [k for k in readable if k in theirs and (theirs[k] > their_median) != (half[k] == 'heavy')]
        result['compared'] = {'regions': len(common), 'missing_here': sorted(set(theirs)-set(share)), 'missing_there': sorted(set(share)-set(theirs)),
                              'largest_difference': max(abs(share[k]-theirs[k]) for k in common), 'their_median': their_median, 'other_half': moved,
                              'agrees': not moved and set(common) == set(share) and max(abs(share[k]-theirs[k]) for k in common) < 1e-9}
    require(not args.output.exists(), f'{args.output} exists'); write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'regions'}), flush=True)
    require(not args.compare or result['compared']['agrees'], f'the strike shares do not agree with {args.compare}: {args.output}')


def pick(a):
    return {k: a[k] for k in ('ar', 'exact', 'ned', 'regions')}


def mean_interval(values, writers, rng, draws=1000):
    """The mean of per-region values and its writer-resampled 95% interval."""
    ids = sorted(values); names = sorted({writers[k] for k in ids}); by = {w: [k for k in ids if writers[k] == w] for w in names}
    full = float(np.mean([values[k] for k in ids]))
    sample = [np.mean([values[k] for w in rng.choice(names, len(names)) for k in by[w]]) for _ in range(draws)]
    return {'regions': len(ids), 'mean': full, 'interval': [float(np.percentile(sample, q)) for q in (2.5, 97.5)]}


def distances(spec, rng):
    """The feature distances of one arm's student (its folds' DIST folders): per view and level the three means, and the paired
    differences student - raw, student - NAFNet, NAFNet - raw."""
    rows = [json.loads(line) for folder in spec.split(',') for line in (Path(folder)/'distances.jsonl').read_text().splitlines()]
    writers = {r['id']: r['writer'] for r in rows}; out = {}
    for view in ('window', 'line'):
        out[view] = {}
        for level in LEVELS:
            got = {r['id']: r[view][level] for r in rows if r[view][level] is not None}
            if not got: out[view][level] = None; continue
            part = lambda f: {k: f(v) for k, v in got.items()}
            out[view][level] = {**{name: mean_interval(part(lambda v, n=name: v[n]), writers, rng) for name in ('raw', 'nafnet', 'student')},
                                'student_minus_raw': mean_interval(part(lambda v: v['student']-v['raw']), writers, rng),
                                'student_minus_nafnet': mean_interval(part(lambda v: v['student']-v['nafnet']), writers, rng),
                                'nafnet_minus_raw': mean_interval(part(lambda v: v['nafnet']-v['raw']), writers, rng)}
        out[view]['touched_tokens'] = {'regions': len(rows), 'with_touched': sum(r[view]['touched'] > 0 for r in rows),
                                       'touched_share_mean': float(np.mean([r[view]['touched']/max(1, r[view]['tokens']) for r in rows]))}
    return out, {r['id']: r for r in rows}


def analyse(args):
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    arms, raw = {}, {}
    for spec in args.arm:
        name, _, folders = spec.partition('='); arms[name], raw[name] = raw_rows(folders)
    scores = {}
    for spec in args.scores:
        ref, _, path = spec.partition('='); scores[ref] = json.loads(Path(path).read_text())['arms']
    vs = lambda ref, arm: scores[ref][arm]['against_reference']
    rules = {}
    fifth = 'VR-真+合' in arms; later = fifth or 'V-修' in arms        # the fourth round's arms are then references only
    if 'X' in arms and not later:
        x_e, x_r = vs('E-NAFNet', 'X'), vs('R0', 'X')
        rules['X1_vs_E-NAFNet'] = {**pick(x_e), 'verdict': 'useful' if verdict_plus(x_e) else 'worse' if worse(x_e) else 'cannot_tell'}
        rules['X2_vs_R0'] = {**pick(x_r), 'verdict': 'erase_then_read_beats_raw' if x_r['exact']['interval'][0] > 0 else 'does_not_beat_raw'}
    if 'V-真+合' in arms and not later:
        v_e, v_r = vs('E-NAFNet', 'V-真+合'), vs('R0', 'V-真+合')
        rules['V1_vs_E-NAFNet'] = {**pick(v_e), 'verdict': 'premise_holds' if verdict_plus(v_e) else 'premise_does_not_hold'}
        rules['V2_vs_R0'] = {**pick(v_r), 'verdict': 'first_version_holds' if v_r['exact']['interval'][0] > 0 else 'does_not_beat_raw'}
        if 'V-真' in scores:
            v_v = vs('V-真', 'V-真+合')
            rules['V3_vs_V-真'] = {**pick(v_v), 'verdict': 'synthetic_pairs_useful' if v_v['exact']['interval'][0] > 0 else 'cannot_tell'}
    if 'A' in arms and not later:
        long_ = vs('R0', 'R0-长'); undertrained = verdict_plus(long_)
        rules['A1_R0-长_vs_R0'] = {**pick(long_), 'verdict': 'undertrained' if undertrained else 'not_undertrained'}
        control = 'R0-长' if undertrained else 'R0'; a = vs(control, 'A')
        rules['A2_vs_control'] = {**pick(a), 'control': control, 'verdict': 'useful' if verdict_plus(a) else 'worse' if worse(a) else 'cannot_tell'}
    if fifth:
        for rule, arm, ref in RULES5:
            rules[f'{rule}_{arm}_vs_{ref}'] = {**pick(vs(ref, arm)), 'verdict': grade(vs(ref, arm))}
        graded = {n: grade(vs('V-真+合', n)) for n in NEW if n in arms}
        exact = {n: scores['V-真+合'][n]['exact'] for n in graded}
        rules['rule4_vs_V-真+合'] = {'arms': {n: {**pick(vs('V-真+合', n)), 'verdict': graded[n], 'exact_rate': exact[n]} for n in graded},
                                     'next_version': next_version(graded, exact)}
        rules['reported'] = {f'{arm}_vs_{ref}': {**pick(vs(ref, arm)), 'grade': grade(vs(ref, arm))} for arm, ref in REPORTED if arm in arms and ref in scores}
        rules['reported'].update({f'{n}_vs_{ref}': pick(vs(ref, n)) for n in graded for ref in ('H', 'R0', 'E-NAFNet', 'X') if ref in scores})
    if 'V-修' in arms:
        half = {k: v['half'] for k, v in json.loads(args.shares.read_text())['regions'].items() if 'half' in v}
        writers = {k: rows[k]['writer'] for k in rows}
        def on(part, ref, arm):
            ks = [k for k in rows if rows[k]['readable'] and half.get(k) == part and k in arms[ref] and k in arms[arm]]
            return paired({k: counts(rows[k]['text'], arms[ref][k]) for k in ks}, {k: counts(rows[k]['text'], arms[arm][k]) for k in ks}, writers, np.random.default_rng(SEED))
        rules['ruleF_V-修_vs_V-真+合'] = {**pick(vs('V-真+合', 'V-修')), 'verdict': grade(vs('V-真+合', 'V-修'))}
        if 'V-涂满' in arms:
            heavy, light, whole = on('heavy', 'V-修', 'V-涂满'), on('light', 'V-修', 'V-涂满'), vs('V-修', 'V-涂满')
            rules['ruleA_V-涂满_vs_V-修'] = {'heavy': pick(heavy), 'light': pick(light), 'whole': pick(whole), 'verdict': rule_a(heavy, light, whole), 'whole_grade': grade(whole)}
        curve = {n: {'whole': scores['V-修'][n]['exact'], 'ar': scores['V-修'][n]['ar'],
                     **{part: rates([counts(rows[k]['text'], arms[n][k]) for k in rows if half.get(k) == part and k in arms[n]])['exact'] for part in ('heavy', 'light')}}
                 for n in ('L25', 'L50', 'V-修') if n in arms}
        if len(curve) == 3:
            gain = (curve['V-修']['whole']-curve['L25']['whole'])/2
            curve['pairs'] = {'L50_vs_L25': pick(vs('L25', 'L50')), 'V-修_vs_L50': pick(vs('L50', 'V-修')),
                              'rough_200_percent': {'exact': curve['V-修']['whole']+gain, 'gain_per_doubling': gain, 'note': 'rough: the mean gain of the two doublings, carried once more'}}
        rules['learning_curve'] = curve
        rules['reported'] = {f'{n}_vs_{ref}': pick(vs(ref, n)) for n in ('V-修', 'V-涂满', 'L25', 'L50') if n in arms for ref in ('V-真+合', 'H', 'R0', 'X') if ref in scores and ref != n}
    names = list(arms)
    can = [k for k in rows if rows[k]['readable'] and all(k in arms[n] for n in names)]
    scored = lambda arm, ks: [counts(rows[k]['text'], arms[arm][k]) for k in ks]
    lengths = lambda k: (lambda n: '1-2' if n <= 2 else '3-9' if n <= 9 else '10+')(len(prepare(rows[k]['text']).replace('?', '')))
    groups = {}
    keyed = [('mode', lambda k: rows[k]['mode']), ('length', lengths), ('fold', lambda k: rows[k]['fold'])]
    if args.shares:
        cut = json.loads(args.shares.read_text())['regions']
        keyed += [('strike_half', lambda k: cut[k]['half']), ('strike_quarter', lambda k: cut[k]['quarter'])]
    for g, key in keyed:
        groups[g] = {v: {'regions': len(ks), **{n: rates(scored(n, ks))['exact'] for n in names}}
                     for v in sorted({key(k) for k in can}) for ks in [[k for k in can if key(k) == v]]}
    rng = np.random.default_rng(SEED); dist, by_region = {}, {}
    for spec in args.distances:
        name, _, folders = spec.partition('='); dist[name], by_region[name] = distances(folders, rng)
    teacher = [json.loads(Path(p).read_text()) for p in args.teacher]
    teacher = {'folds': [{k: t.get(k) for k in ('read', 'kept', 'left_out', 'refused_share', 'lengths', 'kept_by_source')} for t in teacher],
               'refused_share': (sum(t['left_out'].get('refused', 0) for t in teacher)/max(1, sum(t['read'] for t in teacher))) if teacher else None}
    result = {'arms': {n: {k: v for k, v in scores['E-NAFNet'][n].items() if k != 'against_reference'} for n in names if n in scores.get('E-NAFNet', {})},
              'rules': rules, 'vs_H': {n: pick(vs('H', n)) for n in names if n != 'H' and n in scores.get('H', {})},
              'groups': groups, 'distances': dist, 'teacher': teacher, 'regions_in_groups': len(can)}
    out = make_output_dir(args.output_dir); write_json(out/'analysis.json', result)
    with (out/'regions.jsonl').open('x', encoding='utf-8') as stream:
        for k in sorted(rows):
            r = rows[k]
            stream.write(json.dumps({'id': k, 'writer': r['writer'], 'fold': r['fold'], 'readable': r['readable'], 'mode': r['mode'], 'label': r['text'],
                                     **{n: {f: raw[n][k].get(f) for f in ('text', 'raw', 'finish_reason')} if k in raw[n] else None for n in names},
                                     **{f'distance_{n}': {v: d[k][v] for v in ('window', 'line')} if k in d else None for n, d in by_region.items()}},
                                    ensure_ascii=False)+'\n')
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items() if k != 'command'},
                   inputs={str(p): digest(p) for p in [args.pack/'pack.json', *(Path(s.partition('=')[2]) for s in args.scores)]})
    print(json.dumps({k: v.get('verdict', v.get('next_version')) for k, v in rules.items() if k != 'reported'}, ensure_ascii=False), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('analyse')
    a.add_argument('--pack', type=Path, required=True); a.add_argument('--arm', action='append', required=True)
    a.add_argument('--scores', action='append', required=True); a.add_argument('--distances', action='append', default=[])
    a.add_argument('--teacher', action='append', default=[]); a.add_argument('--output-dir', type=Path, required=True)
    a.add_argument('--shares', type=Path, help='a shares file: the sixth round\'s halves and quarters of the strike share')
    s = sub.add_parser('shares')
    s.add_argument('--pack', type=Path, required=True); s.add_argument('--compare', type=Path); s.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv); {'analyse': analyse, 'shares': shares}[args.command](args)


if __name__ == '__main__':
    main()
