"""The dedicated-OCR comparison's rules and reports (plan of 2026-11-21, 专用OCR对照, sections three and four).

    python -m revrender.eval.ocr_compare normalize --readings DIR[,DIR...] --output-dir OUT
    python -m revrender.eval.ocr_compare analyse --pack VIEWS --arm NAME=DIR[,DIR...] [...] --shares SHARES.json --output-dir OUT

``normalize``: readings of a reader that cannot refuse (the untaught Qwen3-VL-8B reading as it sees) with an empty reading (or
``[空]``) written as the refusal, the count in OUT/run_manifest.json (``recognize_ocr`` does the same for the OCR models).  ``analyse``:
the readable regions; differences = the former minus the latter, writer-resampled 95 % intervals (1,000 draws, seed 42); three
grades on the exact difference (``grade3``): the former more accurate (the lower end above 0), the latter (the upper end below 0),
cannot tell.  Arms named ``O-原图|<line>`` / ``O-擦线|<line>`` (an OCR model with one prompt on the scan / the annotator's erased
crop), ``Q-照读``, ``Q-照读-擦线``, ``Q-指令``, ``Q-指令+行`` (Qwen3-VL-8B untaught), ``R0``, ``当前最好``, ``H``.  P1 Q-照读 against
every O-原图 line; P2 Q-指令 against Q-照读; P3 当前最好 and R0 against the best O-原图 line (the highest exact); P4 every OCR line's
O-擦线 against its O-原图 (and against H, reported); the ladder of neighbours from the best O-原图 to H; each arm's exact on the
heavier and the lighter half.  OUT/analysis.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..util import SEED, digest, make_output_dir, write_json, write_manifest
from .transcription import counts, paired, rates, refused

LADDER = ('Q-照读', 'Q-指令', 'Q-指令+行', 'R0', '当前最好', 'H')


def grade3(a):
    lo, hi = a['exact']['interval']
    return 'former_more_accurate' if lo > 0 else 'latter_more_accurate' if hi < 0 else 'cannot_tell'


def readings(spec):
    return {g['id']: g['text'] for folder in spec.split(',') for g in map(json.loads, (Path(folder)/'predictions.jsonl').read_text().splitlines())}


def normalize(args):
    from ..recognize_lines import UNREADABLE
    out = make_output_dir(args.output_dir); changed = 0; total = 0
    with (out/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for folder in args.readings.split(','):
            for g in map(json.loads, (Path(folder)/'predictions.jsonl').read_text().splitlines()):
                total += 1
                if not g['text'].strip() or g['text'].strip() == '[空]':
                    g = {**g, 'text': UNREADABLE, 'text_before_normalize': g['text']}; changed += 1
                stream.write(json.dumps(g, ensure_ascii=False)+'\n')
    write_manifest(out, config={'readings': args.readings}, inputs={}, readings=total, empty_written_as_refusal=changed)
    print(json.dumps({'readings': total, 'empty_written_as_refusal': changed}), flush=True)


def analyse(args):
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    arms = {name: readings(dirs) for name, _, dirs in (s.partition('=') for s in args.arm)}
    half = {k: v.get('half') for k, v in json.loads(args.shares.read_text())['regions'].items()}
    writers = {k: r['writer'] for k, r in rows.items()}; readable = [k for k, r in rows.items() if r['readable']]
    scored = lambda name, ks: {k: counts(rows[k]['text'], arms[name][k]) for k in ks if k in arms[name]}
    def vs(a, b, part=None):
        ks = [k for k in readable if part is None or half.get(k) == part]
        d = paired(scored(b, ks), scored(a, ks), writers, np.random.default_rng(SEED))
        return {m: d[m] for m in ('regions', 'ar', 'exact', 'ned')} | {'grade': grade3(d)}
    exact = {n: rates(list(scored(n, readable).values()))['exact'] for n in arms}
    raw_lines = sorted(n for n in arms if n.startswith('O-原图|')); best = max(raw_lines, key=lambda n: exact[n]) if raw_lines else None
    result = {'exact': exact, 'best_O-原图': best}
    p1 = {n: vs('Q-照读', n) for n in raw_lines}
    grades = [v['grade'] for v in p1.values()]
    result['P1'] = {'pairs': p1, 'statement': 'vl_more_accurate_untaught' if grades and all(g == 'former_more_accurate' for g in grades) else
                    'ocr_more_accurate_untaught: ' + ', '.join(n for n, v in p1.items() if v['grade'] == 'latter_more_accurate') if 'latter_more_accurate' in grades else 'about_the_same'}
    result['P2'] = vs('Q-指令', 'Q-照读') if {'Q-指令', 'Q-照读'} <= set(arms) else None
    result['P3'] = {n: vs(n, best) for n in ('当前最好', 'R0') if n in arms and best}
    result['P4'] = {line.split('|', 1)[1]: {'erased_vs_raw': vs('O-擦线|'+line.split('|', 1)[1], line), 'erased_vs_H': vs('O-擦线|'+line.split('|', 1)[1], 'H')}
                    for line in raw_lines if 'O-擦线|'+line.split('|', 1)[1] in arms and 'H' in arms}
    chain = [best, *(n for n in LADDER if n in arms)] if best else [n for n in LADDER if n in arms]
    result['ladder'] = [{'from': a, 'to': b, **vs(b, a)} for a, b in zip(chain, chain[1:])]
    result['halves'] = {part: {n: rates(list(scored(n, [k for k in readable if half.get(k) == part]).values()))['exact'] for n in arms} for part in ('heavy', 'light')}
    result['refusals'] = {n: sum(refused(t) for t in arms[n].values()) for n in arms}
    out = make_output_dir(args.output_dir); write_json(out/'analysis.json', result)
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items() if k != 'command'},
                   inputs={str(args.pack/'pack.json'): digest(args.pack/'pack.json'), str(args.shares): digest(args.shares)})
    print(json.dumps({'P1': result['P1']['statement'], 'P2': result['P2'] and result['P2']['grade'], 'best_O-原图': best}, ensure_ascii=False), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); sub = p.add_subparsers(dest='command', required=True)
    n = sub.add_parser('normalize'); n.add_argument('--readings', required=True); n.add_argument('--output-dir', type=Path, required=True)
    a = sub.add_parser('analyse')
    for key in ('pack', 'shares', 'output-dir'): a.add_argument('--'+key, type=Path, required=True)
    a.add_argument('--arm', action='append', required=True)
    args = p.parse_args(argv); {'normalize': normalize, 'analyse': analyse}[args.command](args)


if __name__ == '__main__':
    main()
