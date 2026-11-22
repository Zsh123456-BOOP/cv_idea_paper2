"""The rules of the plug-in on another model (plan of 2026-11-21, 换模型验证插件, section four), for one model:

    python -m revrender.eval.model_switch --pack VIEWS --model NAME --arm H=DIRS --arm R0=DIRS --arm V=DIRS --scores R0=FILE
        --scores H=FILE --shares SHARES.json [--distances DIRS] --output-dir OUT

``--scores``: ``eval.transcription`` runs against R0 and H (the readable regions, writer-resampled, seed 42).  The premise: H
against R0, the exact difference's 95 % lower end above 0 -> there is a gap to close (else the model is reported apart and left out
of the transfer statement).  The transfer: V against R0 in the four grades (``eval.privileged.grade``: holds -- the exact lower end
above 0 and the AR difference at least +2 --, worse, exact up but AR short, cannot tell).  Reported only: every arm's rates, refusals
and exact by label length; V against H; each pair on the heavier and lighter halves of the strike share (``--shares``); the window
and line cosines of the adapter's features to the clean crop's (``train_clean_view distance``).  OUT/analysis.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..util import SEED, digest, make_output_dir, write_json, write_manifest
from .hint_loop import raw_rows
from .privileged import distances, grade, pick
from .transcription import counts, paired


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--pack', type=Path, required=True); p.add_argument('--model', required=True)
    p.add_argument('--arm', action='append', required=True); p.add_argument('--scores', action='append', required=True)
    p.add_argument('--shares', type=Path, required=True); p.add_argument('--distances'); p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    arms = {name: raw_rows(folders)[0] for name, _, folders in (s.partition('=') for s in args.arm)}
    scores = {ref: json.loads(Path(path).read_text())['arms'] for ref, _, path in (s.partition('=') for s in args.scores)}
    vs = lambda ref, arm: scores[ref][arm]['against_reference']
    half = {k: v['half'] for k, v in json.loads(args.shares.read_text())['regions'].items() if 'half' in v}
    writers = {k: r['writer'] for k, r in rows.items()}

    def on(part, ref, arm):
        ks = [k for k in rows if rows[k]['readable'] and half.get(k) == part and k in arms[ref] and k in arms[arm]]
        return pick(paired({k: counts(rows[k]['text'], arms[ref][k]) for k in ks}, {k: counts(rows[k]['text'], arms[arm][k]) for k in ks}, writers,
                           np.random.default_rng(SEED)))
    gap = vs('R0', 'H')['exact']['interval'][0] > 0; with_v = 'V' in arms         # LLaVA's first step (revision ten): H and R0 alone
    pairs = (('H', 'R0'), ('V', 'R0'), ('V', 'H')) if with_v else (('H', 'R0'),)
    result = {'model': args.model,
              'premise_H_vs_R0': {**pick(vs('R0', 'H')), 'verdict': 'gap_to_close' if gap else 'no_gap_to_close'},
              'transfer_V_vs_R0': ({**pick(vs('R0', 'V')), 'verdict': grade(vs('R0', 'V')) if gap else 'left_out_no_gap'} if with_v else
                                   {'verdict': 'not_taught_yet' if gap else 'left_out_no_gap'}),
              'reported': {'rates': {a: {k: scores['R0'][a][k] for k in ('exact', 'ar', 'ned', 'refusal', 'exact_by_length', 'left_wrong')} for a in arms},
                           **({'V_vs_H': pick(vs('H', 'V'))} if with_v else {}),
                           'halves': {part: {f'{a}_vs_{b}': on(part, b, a) for a, b in pairs} for part in ('heavy', 'light')}}}
    if args.distances:
        result['reported']['distances'] = distances(args.distances, np.random.default_rng(SEED))[0]
    out = make_output_dir(args.output_dir)
    write_json(out/'analysis.json', result)
    write_manifest(out, config={k: str(v) for k, v in vars(args).items()}, inputs={s.partition('=')[2]: digest(Path(s.partition('=')[2])) for s in args.scores})
    print(json.dumps({'premise': result['premise_H_vs_R0']['verdict'], 'transfer': result['transfer_V_vs_R0']['verdict']}), flush=True)


if __name__ == '__main__':
    main()
