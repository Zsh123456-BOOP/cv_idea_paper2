"""The rules and reports of T1 and T2 (plan of 2026-11-22, 三个不训练的检验, revision one) from their readings.

    python -m revrender.eval.untrained_checks --pack VIEWS --shares SHARES --arm NAME=DIR[,DIR...] [...] --mix-stats NAME=DIR[,DIR...]
        --output-dir OUT

Arms by name: V-修 (images 2 and 3 through the fold's V-修 adapter, read again), V-只2, V-只3, α=0 ... α=1 (the chord), M, M'.  On the
readable regions, exact and AR differences with a writer bootstrap (1,000, seed 42, 95 %; ``eval.transcription.paired``):

- rule T1: V-只2 - V-修 on exact: the lower end above 0 -> 第3张图过插件有害; the upper end below 0 -> 有用; else 说不准.
- rule T2: M - V-修: the lower end above 0 -> 弦上更好 (V 偏离直线的那部分有害); the upper end below 0 -> 偏离有用; else 一样.
- reported: V-只3 - V-修, M' - V-修, M - M'; every arm's exact, AR and refusals (the whole, the strike halves, the annotation modes);
  each pair also by half and mode; M - V-修 on the two halves of the regions cut at the median share of their window's merged tokens
  taken back (``--mix-stats`` of M), and that share's distribution.
OUT/analysis.json, OUT/regions.jsonl (every arm's reading of every region)."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..util import SEED, digest, make_output_dir, require, write_json, write_manifest
from .transcription import counts, paired, rates, read_arm, refused


def verdict_t1(d):
    lo, hi = d['exact']['interval']
    return '第3张图过插件有害' if lo > 0 else '第3张图过插件有用' if hi < 0 else '说不准'


def verdict_t2(d):
    lo, hi = d['exact']['interval']
    return '同样近的弦上特征读得更好：V偏离直线的那部分有害' if lo > 0 else 'V偏离直线的那部分是有用的' if hi < 0 else 'V和同样近的弦上点读得一样：要提高只能更近'


def taken_back(spec):
    """{region: the share of its window's (image 2) merged tokens taken back} from M's mix_stats.jsonl files."""
    out = {}
    for folder in spec.partition('=')[2].split(','):
        for line in (Path(folder)/'mix_stats.jsonl').read_text().splitlines():
            g = json.loads(line); m = g['images']['2']['merged']; out[g['id']] = m['taken_back']/m['tokens']
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--pack', type=Path, required=True); p.add_argument('--shares', type=Path, required=True)
    p.add_argument('--arm', action='append', required=True); p.add_argument('--mix-stats', required=True, help="M=DIR[,DIR...]: M's readings (mix_stats.jsonl)")
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    half = {k: v.get('half') for k, v in json.loads(args.shares.read_text())['regions'].items()}
    arms, files = {}, []
    for spec in args.arm:
        name, got, read = read_arm(spec); arms[name] = got; files += read
    require({'V-修', 'V-只2', 'M'} <= set(arms), 'the arms V-修, V-只2 and M are needed')
    require(all(set(a) == set(rows) for a in arms.values()), 'every arm reads every region')
    writers = {k: r['writer'] for k, r in rows.items()}
    readable = [k for k in sorted(rows) if rows[k]['readable']]
    parts = {'whole': lambda k: True, 'heavy': lambda k: half.get(k) == 'heavy', 'light': lambda k: half.get(k) == 'light',
             **{f'mode:{m}': (lambda k, m=m: rows[k]['mode'] == m) for m in sorted({rows[k]['mode'] for k in readable})}}

    def score(name, ks):
        r = rates([counts(rows[k]['text'], arms[name][k]) for k in ks])
        return {'regions': len(ks), 'exact': r['exact'], 'ar': r['ar'], 'readable_refused': sum(refused(arms[name][k]) for k in ks)}

    def pair(a, b, ks):
        rng = np.random.default_rng(SEED)
        return paired({k: counts(rows[k]['text'], arms[b][k]) for k in ks}, {k: counts(rows[k]['text'], arms[a][k]) for k in ks}, writers, rng)
    by = {part: [k for k in readable if inside(k)] for part, inside in parts.items()}
    result = {'arms': {n: {part: score(n, ks) for part, ks in by.items()} for n in arms},
              'refusals_all_regions': {n: sum(refused(t) for t in arms[n].values()) for n in arms}}
    t1 = pair('V-只2', 'V-修', readable); t2 = pair('M', 'V-修', readable)
    result['rule_T1'] = {**t1, 'verdict': verdict_t1(t1)}
    result['rule_T2'] = {**t2, 'verdict': verdict_t2(t2)}
    reported = [('V-只2', 'V-修'), ('V-只3', 'V-修'), ('M', 'V-修'), ("M'", 'V-修'), ('M', "M'")]
    result['pairs'] = {f'{a} - {b}': {part: pair(a, b, ks) for part, ks in by.items()} for a, b in reported if a in arms and b in arms}
    share = taken_back(args.mix_stats); values = np.array([share[k] for k in readable]); cut = float(np.median(values))
    lower, upper = [k for k in readable if share[k] <= cut], [k for k in readable if share[k] > cut]
    result['taken_back_window_merged'] = {'median': cut, 'quantiles': {q: float(np.percentile(values, q)) for q in (0, 10, 25, 50, 75, 90, 100)},
                                          'mean': float(values.mean()), 'regions_with_none': int((values == 0).sum()),
                                          'M - V-修': {'at_or_below_median': pair('M', 'V-修', lower), 'above_median': pair('M', 'V-修', upper)}}
    out = make_output_dir(args.output_dir); write_json(out/'analysis.json', result)
    with (out/'regions.jsonl').open('x', encoding='utf-8') as stream:
        for k in sorted(rows):
            r = rows[k]
            stream.write(json.dumps({'id': k, 'writer': r['writer'], 'fold': r['fold'], 'readable': r['readable'], 'mode': r['mode'], 'half': half.get(k),
                                     'label': r['text'], 'taken_back_window_merged': share.get(k), **{n: arms[n][k] for n in arms}}, ensure_ascii=False)+'\n')
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items()},
                   inputs={str(f): digest(f) for f in [args.pack/'pack.json', args.shares, *files]})
    print(json.dumps({'T1': result['rule_T1']['verdict'], 'T2': result['rule_T2']['verdict']}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
