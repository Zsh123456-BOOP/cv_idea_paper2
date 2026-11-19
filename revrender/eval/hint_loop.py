"""The hinted restorer's loop and the analyses of the premise test (plan of 2026-11-20, 前提检验_带提示的描字器, sections four and five).

    python -m revrender.eval.hint_loop compose --initial DIR[,DIR...] --round DIR[,DIR...] x3 --output-dir LOOP
    python -m revrender.eval.hint_loop analyse --pack VIEWS --sets SETS --loop LOOP --arm NAME=DIR[,DIR...] [...] --scores REF=FILE [...]
        --wrong WRONG.json --pixels NAME=SEPARATED [...] --output-dir OUT
    python -m revrender.eval.hint_loop sheets --analysis OUT --human VIEWS --blind SEPARATED --loop-packs SEP1,SEP2,SEP3 --wrong-pack SEP
        --output-dir SHEETS

``compose``: T-循环 from its rounds.  Round 1's hint is R0's reading (``--initial``) and its reading the first ``--round`` (T-初读);
round k+1's hint is round k's reading.  A region stops at the first round whose reading is the same as its hint (both refusals, or
equal in the ``eval.coverage.prepare`` spelling), and at round 3 at the latest; its final reading is the last one, ``converged`` when
it is the same as its hint.  LOOP/predictions.jsonl reads as a ``recognize_lines`` folder (condition ``loop``); LOOP/loop.jsonl a row
per region: each round's hint and reading, the round it stopped at, converged.

``analyse``: OUT/analysis.json -- the plan's rules (section five) from the ``eval.transcription`` runs (``--scores`` against E-NAFNet,
R0, T-初读 and H); rule 3 on the regions with a wrong hint (``--wrong``, ``data.hint_image hints --source wrong``): the share of T-错's
readings that are the wrong string (a), of T-空's (b), and the shares reading the answer under T-错, T-真 and T-空; the loop's stops,
convergence, exact share of the converged and not, and the scores when a region not converged is taken as refused; exact by annotation
mode and by the label's length; the pixel precision / recall / F1 and writing erased on the held-out folds under three hints
(``--pixels``: white, R0's reading, the answer).  OUT/regions.jsonl: a row per region with every arm's hint, reading, raw output and
why it stopped, and the loop.

``sheets``: three kinds, the first ``TAKE`` of each in the order of sha256("<region>:42"): T-循环 right and E-NAFNet wrong; T-循环
wrong and E-NAFNet right; T-错 reading its wrong hint.  Per region the scan, the hint, the traced crop, the blind one (E-NAFNet), the
annotator's, with the answer and the readings.  Student work: never in Git.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..util import SEED, digest, make_output_dir, require, write_json, write_manifest
from .coverage import prepare
from .transcription import counts, rates, read_arm, refusal, refused

ROUNDS = 3
TAKE = 12
MARGIN = .02            # rule 1 and 2: the AR difference at least +2 points
FOLLOW = .20            # rule 3: a - b above 20 points


def same(a, b):
    """Two texts the same for the loop: both refusals, or equal in the prepare spelling."""
    if refused(a) or refused(b): return refused(a) and refused(b)
    return prepare(a or '') == prepare(b or '')


def raw_rows(spec):
    name, readings, files = read_arm('X='+spec)
    return readings, {g['id']: g for f in files for g in map(json.loads, f.read_text(encoding='utf-8').splitlines())}


def compose(args):
    initial, _ = raw_rows(args.initial); rounds = [raw_rows(r) for r in args.round]
    require(len(rounds) == ROUNDS, f'{ROUNDS} rounds')
    out = make_output_dir(args.output_dir); loop, predictions = [], []
    for k in sorted(set(initial) & set.intersection(*(set(r[0]) for r in rounds))):
        hint, steps = initial[k], []
        for n, (readings, raw) in enumerate(rounds, 1):
            reading = readings[k]; steps.append({'hint': hint, 'reading': reading})
            if same(reading, hint) or n == ROUNDS: break
            hint = reading
        g = rounds[len(steps)-1][1][k]
        loop.append({'id': k, 'rounds': steps, 'stopped_at': len(steps), 'converged': same(steps[-1]['reading'], steps[-1]['hint'])})
        predictions.append({'condition': 'loop', 'id': k, 'line_id': g['line_id'], 'model': g.get('model'), 'text': g['text'], 'raw': g['raw'],
                            'generated_tokens': g.get('generated_tokens'), 'finish_reason': g['finish_reason'], 'round': len(steps)})
    for name, rows in (('loop.jsonl', loop), ('predictions.jsonl', predictions)):
        with (out/name).open('x', encoding='utf-8') as stream:
            for r in rows: stream.write(json.dumps(r, ensure_ascii=False)+'\n')
    summary = {'regions': len(loop), 'stopped_at': {n: sum(r['stopped_at'] == n for r in loop) for n in range(1, ROUNDS+1)},
               'not_converged': sum(not r['converged'] for r in loop)}
    write_json(out/'complete.json', summary)
    write_manifest(out, config={'initial': args.initial, 'rounds': args.round}, inputs={}, **summary)
    print(json.dumps(summary), flush=True)


def verdict_plus(a):
    """Rule 1's standard: the exact interval's lower end above 0 and the AR difference at least +2 points."""
    return a['exact']['interval'][0] > 0 and a['ar']['difference'] >= MARGIN


def pixel_scores(sets, pack):
    from ..baselines.restoration import inside, removal_counts, removal_metrics
    rows = {r['id']: r for r in json.loads((pack/'pack.json').read_text())['rows']}; total = np.zeros(5, np.int64)
    for folder in sorted(sets.glob('fold_*')):
        native = json.loads((folder/'heldout_native.json').read_text()); arrays = np.load(folder/'heldout_native.npz')
        for i, n in enumerate(native):
            image = arrays[f'image_{i}'].astype(np.float32)/255
            restored = np.asarray(Image.open(pack/rows[n['id']]['images']['erased']).convert('L'), np.float32)/255
            total += removal_counts(*inside(image, restored, arrays[f'mask_{i}'], arrays[f'inner_{i}']))
    return removal_metrics(total)


def analyse(args):
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    readable = {k: r['readable'] for k, r in rows.items()}
    arms, raw = {}, {}
    for spec in args.arm:
        name, _, folders = spec.partition('='); arms[name], raw[name] = raw_rows(folders)
    loop = {r['id']: r for r in map(json.loads, (args.loop/'loop.jsonl').read_text(encoding='utf-8').splitlines())}
    wrong = json.loads(args.wrong.read_text())
    scores = {}
    for spec in args.scores:
        ref, _, path = spec.partition('='); scores[ref] = json.loads(Path(path).read_text())['arms']
    vs = lambda ref, arm: scores[ref][arm]['against_reference']
    pick = lambda a: {k: a[k] for k in ('ar', 'exact', 'ned')}
    # rules 1 and 2
    rule1 = {**pick(vs('E-NAFNet', 'T-真')), 'premise': 'holds' if verdict_plus(vs('E-NAFNet', 'T-真')) else 'does_not_hold'}
    loop_e, loop_r0 = vs('E-NAFNet', 'T-循环'), vs('R0', 'T-循环')
    rule2 = {'vs_E-NAFNet': pick(loop_e), 'vs_R0': pick(loop_r0),
             'first_version': 'holds' if verdict_plus(loop_e) and loop_r0['exact']['interval'][0] > 0 else 'does_not_hold'}
    # rule 3, on the readable regions with a wrong hint
    part = [k for k, t in wrong.items() if t is not None and readable.get(k)]
    share = lambda arm, target: float(np.mean([same(arms[arm][k], target(k)) for k in part])) if part else None
    a, b = share('T-错', lambda k: wrong[k]), share('T-空', lambda k: wrong[k])
    answer = lambda k: rows[k]['text']
    rule3 = {'regions': len(part), 'left_out': sum(readable.values())-len(part), 'a_reads_the_wrong_hint': a, 'b_blank_reads_it': b, 'a_minus_b': a-b,
             'follows_the_hint': a-b > FOLLOW, 'reads_the_answer': {arm: share(arm, answer) for arm in ('T-错', 'T-真', 'T-空')}}
    # the loop
    can = [k for k in loop if readable[k]]; scored = lambda arm, ks: [counts(rows[k]['text'], arms[arm][k]) for k in ks]
    conv = [k for k in can if loop[k]['converged']]; notc = [k for k in can if not loop[k]['converged']]
    as_refused = {k: (arms['T-循环'][k] if loop[k]['converged'] else '[读不出]') for k in loop}
    loop_part = {'stopped_at': {n: sum(r['stopped_at'] == n for r in loop.values()) for n in range(1, ROUNDS+1)},
                 'not_converged': sum(not r['converged'] for r in loop.values()), 'not_converged_readable': len(notc),
                 'exact_converged': rates(scored('T-循环', conv))['exact'] if conv else None, 'exact_not_converged': rates(scored('T-循环', notc))['exact'] if notc else None,
                 'not_converged_as_refused': {**{m: v for m, v in rates([counts(rows[k]['text'], as_refused[k]) for k in can]).items() if m in ('exact', 'ar')},
                                              'refusal': refusal(as_refused, readable)}}
    # groups
    lengths = lambda k: (lambda n: '1-2' if n <= 2 else '3-9' if n <= 9 else '10+')(len(prepare(rows[k]['text']).replace('?', '')))
    names = [n for n in arms]
    groups = {}
    for g, key in (('mode', lambda k: rows[k]['mode']), ('length', lengths)):
        groups[g] = {v: {'regions': len(ks), **{n: rates(scored(n, ks))['exact'] for n in names}}
                     for v in sorted({key(k) for k in can}) for ks in [[k for k in can if key(k) == v]]}
    pixels = {}
    for spec in args.pixels:
        name, _, path = spec.partition('='); pixels[name] = pixel_scores(args.sets, Path(path))
    reports = {'T-空_vs_E-NAFNet': pick(vs('E-NAFNet', 'T-空')), 'T-初读_vs_E-NAFNet': pick(vs('E-NAFNet', 'T-初读')), 'T-初读_vs_R0': pick(vs('R0', 'T-初读')),
               'T-循环_vs_T-初读': pick(vs('T-初读', 'T-循环')), 'vs_H': {n: pick(vs('H', n)) for n in names if n != 'H'}}
    result = {'arms': {n: {k: v for k, v in scores['E-NAFNet'][n].items() if k != 'against_reference'} for n in names},
              'rule1_premise': rule1, 'rule2_first_version': rule2, 'rule3_follow': rule3, 'reports': reports, 'loop': loop_part,
              'groups': groups, 'pixels': pixels}
    out = make_output_dir(args.output_dir); write_json(out/'analysis.json', result)
    with (out/'regions.jsonl').open('x', encoding='utf-8') as stream:
        for k in sorted(loop):
            r = rows[k]
            stream.write(json.dumps({'id': k, 'writer': r['writer'], 'fold': r['fold'], 'readable': r['readable'], 'mode': r['mode'], 'label': r['text'],
                                     'wrong_hint': wrong.get(k), 'loop': loop[k],
                                     **{n: {f: raw[n][k].get(f) for f in ('text', 'raw', 'finish_reason')} for n in names}}, ensure_ascii=False)+'\n')
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items() if k != 'command'},
                   inputs={str(p): digest(p) for p in [args.pack/'pack.json', args.wrong, *(Path(s.partition('=')[2]) for s in args.scores)]})
    print(json.dumps({'premise': rule1['premise'], 'first_version': rule2['first_version'], 'follows': rule3['follows_the_hint'],
                      'a_minus_b': rule3['a_minus_b']}), flush=True)


def sheets(args):
    from PIL import ImageDraw, ImageFont
    font = ImageFont.truetype(args.font, 18); out = make_output_dir(args.output_dir)
    table = [json.loads(l) for l in (args.analysis/'regions.jsonl').read_text(encoding='utf-8').splitlines()]
    pack = lambda p: {r['id']: r for r in json.loads((Path(p)/'pack.json').read_text())['rows']}
    human, blind, wrong = pack(args.human), pack(args.blind), pack(args.wrong_pack); rounds = [pack(p) for p in args.loop_packs.split(',')]
    exact = lambda r, arm: r['readable'] and prepare(r['label'] or '').replace('?', '') == ('' if refused(r[arm]['text']) else prepare(r[arm]['text'] or ''))
    kinds = {'loop_right_blind_wrong': lambda r: exact(r, 'T-循环') and not exact(r, 'E-NAFNet'),
             'loop_wrong_blind_right': lambda r: r['readable'] and not exact(r, 'T-循环') and exact(r, 'E-NAFNet'),
             'follows_wrong_hint': lambda r: r['wrong_hint'] is not None and r['readable'] and same(r['T-错']['text'], r['wrong_hint'])}
    order = lambda r: hashlib.sha256(f"{r['id']}:{SEED}".encode()).hexdigest(); index = {}
    for kind, test in kinds.items():
        tiles = []
        for n, r in enumerate(sorted(filter(test, table), key=order)[:TAKE], 1):
            traced = wrong[r['id']] if kind == 'follows_wrong_hint' else rounds[r['loop']['stopped_at']-1][r['id']]
            root = args.wrong_pack if kind == 'follows_wrong_hint' else args.loop_packs.split(',')[r['loop']['stopped_at']-1]
            grey = lambda base, row, key: np.asarray(Image.open(Path(base)/row['images'][key]).convert('L'))
            pics = [grey(args.human, human[r['id']], 'raw'), grey(root, traced, 'hint'), grey(root, traced, 'erased'),
                    grey(args.blind, blind[r['id']], 'erased'), grey(args.human, human[r['id']], 'erased')]
            k = max(1, 120//max(1, pics[0].shape[0])); pics = [Image.fromarray(p).resize((p.shape[1]*k, p.shape[0]*k), Image.Resampling.NEAREST) for p in pics]
            lines = [f"{kind[0].upper()}{n:02d}  答案：{r['label']}", f"T-循环（第 {r['loop']['stopped_at']} 轮）：{r['T-循环']['text']}", f"E-NAFNet：{r['E-NAFNet']['text']}"]
            if kind == 'follows_wrong_hint': lines += [f"错提示：{r['wrong_hint']}", f"T-错：{r['T-错']['text']}"]
            w = sum(p.width for p in pics)+60; h = max(p.height for p in pics)+20+28*len(lines); tile = Image.new('RGB', (max(w, 900), h), 'white'); x = 10
            for p in pics: tile.paste(p, (x, 10)); x += p.width+10
            draw = ImageDraw.Draw(tile)
            for m, text in enumerate(lines): draw.text((10, h-28*len(lines)+28*m), (text or '').replace('\n', ' ⏎ '), fill='black', font=font)
            tiles.append(tile); index[f"{kind[0].upper()}{n:02d}"] = {'id': r['id'], 'kind': kind}
        if tiles:
            sheet = Image.new('RGB', (max(t.width for t in tiles), sum(t.height+6 for t in tiles)), (200, 200, 200)); y = 0
            for t in tiles: sheet.paste(t, (0, y)); y += t.height+6
            sheet.save(out/f'{kind}.png')
    write_json(out/'index.json', index)
    write_json(out/'legend.json', {'columns': ['scan', 'hint', 'traced', 'blind (E-NAFNet)', 'annotator'], 'order': f'sha256("<region>:{SEED}"), the first {TAKE}',
                                   'counts': {kind: sum(1 for r in table if test(r)) for kind, test in kinds.items()}})
    print(json.dumps({kind: sum(v['kind'] == kind for v in index.values()) for kind in kinds}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('compose')
    c.add_argument('--initial', required=True); c.add_argument('--round', action='append', required=True); c.add_argument('--output-dir', type=Path, required=True)
    a = sub.add_parser('analyse')
    for key in ('pack', 'sets', 'loop', 'wrong', 'output-dir'): a.add_argument('--'+key, type=Path, required=True)
    a.add_argument('--arm', action='append', required=True); a.add_argument('--scores', action='append', required=True)
    a.add_argument('--pixels', action='append', default=[], help='NAME=SEPARATED pack: pixel scores on the held-out folds')
    s = sub.add_parser('sheets')
    for key in ('analysis', 'human', 'blind', 'wrong-pack', 'output-dir'): s.add_argument('--'+key, type=Path, required=True)
    s.add_argument('--loop-packs', required=True, help='the three rounds\' separated packs, comma-separated')
    s.add_argument('--font', default='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    args = p.parse_args(argv); {'compose': compose, 'analyse': analyse, 'sheets': sheets}[args.command](args)


if __name__ == '__main__':
    main()
