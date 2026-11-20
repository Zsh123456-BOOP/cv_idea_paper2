"""The rules and reports of the fourth round (plan of 2026-11-20, 第四轮_特权视图三样试验, sections 4.1-4.3): X (a reader taught on
NAFNet's held-out erased crops), V (the reader's vision tower taught to see the scan as clean) and A (a reader taught first on
synthetic regions a teacher read); and of the fifth (第五轮_X与V结合, section 5): V with a matched reader (VR-真+合, VR-真), the
adapter looking at NAFNet's crop (NV, NVR) or at the scan with NAFNet's erasure in red (MV, MVR).

    python -m revrender.eval.privileged analyse --pack VIEWS --arm NAME=DIR[,DIR...] [...] --scores REF=FILE [...]
        [--distances NAME=DIR[,DIR...] ...] [--teacher FILE ...] --output-dir OUT

``--scores``: ``eval.transcription`` runs against E-NAFNet, R0, H, V-真 and R0-长.  The rules, each as the plan writes it ("成立": the
exact interval's lower end above 0 and the AR difference at least +2; "worse": the exact interval wholly below 0):
X1 X against E-NAFNet; X2 X against R0 (exact lower end above 0); V1 V-真+合 against E-NAFNet; V2 V-真+合 against R0; V3 V-真+合
against V-真; A1 R0-长 against R0 (then A's control is R0-长, else R0); A2 A against its control.  The fifth round's four grades
(``grade``): holds, worse, exact up but AR short (the exact interval's lower end above 0, the AR difference under +2), cannot tell;
rule 1 VR-真+合 against V-真+合, rule 2 NVR against VR-真, rule 3 MVR against VR-真, rule 4 each new reading against V-真+合: of
those that hold, the highest exact is the next version, within 0.2 points a tie, broken for the one without NAFNet, then for the
shared reader (``next_version``).  Reported only: the pairs ``REPORTED``, each new reading against H, R0, E-NAFNet and X.  Reported only: exact by
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

from ..util import SEED, digest, make_output_dir, write_json, write_manifest
from .coverage import prepare
from .hint_loop import raw_rows, verdict_plus
from .transcription import counts, rates

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
    if 'X' in arms:
        x_e, x_r = vs('E-NAFNet', 'X'), vs('R0', 'X')
        rules['X1_vs_E-NAFNet'] = {**pick(x_e), 'verdict': 'useful' if verdict_plus(x_e) else 'worse' if worse(x_e) else 'cannot_tell'}
        rules['X2_vs_R0'] = {**pick(x_r), 'verdict': 'erase_then_read_beats_raw' if x_r['exact']['interval'][0] > 0 else 'does_not_beat_raw'}
    if 'V-真+合' in arms:
        v_e, v_r = vs('E-NAFNet', 'V-真+合'), vs('R0', 'V-真+合')
        rules['V1_vs_E-NAFNet'] = {**pick(v_e), 'verdict': 'premise_holds' if verdict_plus(v_e) else 'premise_does_not_hold'}
        rules['V2_vs_R0'] = {**pick(v_r), 'verdict': 'first_version_holds' if v_r['exact']['interval'][0] > 0 else 'does_not_beat_raw'}
        if 'V-真' in scores:
            v_v = vs('V-真', 'V-真+合')
            rules['V3_vs_V-真'] = {**pick(v_v), 'verdict': 'synthetic_pairs_useful' if v_v['exact']['interval'][0] > 0 else 'cannot_tell'}
    if 'A' in arms:
        long_ = vs('R0', 'R0-长'); undertrained = verdict_plus(long_)
        rules['A1_R0-长_vs_R0'] = {**pick(long_), 'verdict': 'undertrained' if undertrained else 'not_undertrained'}
        control = 'R0-长' if undertrained else 'R0'; a = vs(control, 'A')
        rules['A2_vs_control'] = {**pick(a), 'control': control, 'verdict': 'useful' if verdict_plus(a) else 'worse' if worse(a) else 'cannot_tell'}
    if 'VR-真+合' in arms:
        for rule, arm, ref in RULES5:
            rules[f'{rule}_{arm}_vs_{ref}'] = {**pick(vs(ref, arm)), 'verdict': grade(vs(ref, arm))}
        graded = {n: grade(vs('V-真+合', n)) for n in NEW if n in arms}
        exact = {n: scores['V-真+合'][n]['exact'] for n in graded}
        rules['rule4_vs_V-真+合'] = {'arms': {n: {**pick(vs('V-真+合', n)), 'verdict': graded[n], 'exact_rate': exact[n]} for n in graded},
                                     'next_version': next_version(graded, exact)}
        rules['reported'] = {f'{arm}_vs_{ref}': {**pick(vs(ref, arm)), 'grade': grade(vs(ref, arm))} for arm, ref in REPORTED if arm in arms and ref in scores}
        rules['reported'].update({f'{n}_vs_{ref}': pick(vs(ref, n)) for n in graded for ref in ('H', 'R0', 'E-NAFNet', 'X') if ref in scores})
    names = list(arms)
    can = [k for k in rows if rows[k]['readable'] and all(k in arms[n] for n in names)]
    scored = lambda arm, ks: [counts(rows[k]['text'], arms[arm][k]) for k in ks]
    lengths = lambda k: (lambda n: '1-2' if n <= 2 else '3-9' if n <= 9 else '10+')(len(prepare(rows[k]['text']).replace('?', '')))
    groups = {}
    for g, key in (('mode', lambda k: rows[k]['mode']), ('length', lengths), ('fold', lambda k: rows[k]['fold'])):
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
    args = p.parse_args(argv); {'analyse': analyse}[args.command](args)


if __name__ == '__main__':
    main()
