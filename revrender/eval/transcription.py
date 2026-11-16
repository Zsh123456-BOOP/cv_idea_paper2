"""Scores of readings of struck text in the definitions of the literature, the ways of reading paired by writer.

    python -m revrender.eval.transcription --pack PACK --arm NAME=DIR[,DIR...][@CONDITION] [--arm ...] --reference NAME --output OUT.json

An arm is the readings (``recognize_lines``) of one way of reading, over one or several folders (the folds of a cross-validation, each
holding the readings of the writers its reader never trained on), of one condition when a folder holds several.  A label and a reading
are compared in the spelling of ``eval.coverage.prepare`` (the label's ``?`` dropped) by a minimal edit alignment: of the label's N
characters S are substituted and D deleted, and the reading adds I.

  CER    (S+D+I)/N          character error rate of handwriting recognition (Pathirage et al., IJDAR 2026, on crossed-out words)
  CR     (N-D-S)/N          correct rate of handwritten Chinese text recognition (SCUT-EPT, 2019): the project's earlier "read-back"
  AR     (N-D-S-I)/N        accuracy rate, its companion: what the reading adds is charged
  exact  reading = label    the expression rate of handwritten mathematical expression recognition (Uni-MuMER, NeurIPS 2025)
  NED    (S+D+I)/max(N, M)  normalised edit distance, M the reading's length (KIE-HVQA, NeurIPS 2025)

CER, CR and AR are over all the characters of the readable regions; exact and NED are means over the regions.  A region the annotator
could not read (``readable`` false in the pack) should be answered with ``recognize_lines.UNREADABLE``: refusal precision, recall and F1
(the degraded-region rule of KIE-HVQA: read what can be read, refuse what cannot); a refusal on a readable region reads nothing.  Each
arm is paired with ``--reference`` on the regions both read: the differences of AR, exact and NED with 95% intervals by resampling
writers (1,000 draws, seed 42).  ``--agreement A=B``: whether two arms giving different readings of a region tells that A's is wrong
(a reading refused is a reading of its own) -- how often they differ, A's exact share where they agree and where they differ, and, of
the regions the annotator could not read, the share on which they differ.  ``--gone NAME``: an arm read with the struck writing taken
off the images (``recognize_lines --intervention writing_off``): the share of its readings that refuse, as they all should.  ``--only
FILE:KEY``: only the regions listed under KEY in a JSON file (``data.line_set unchanged``).  Each arm's readable regions also by what
is left wrong (refused, 1, 2, 3 or more edits) and its exact share by the label's length (1-2, 3-9, 10 and more characters);
``--readings OUT.jsonl``: a row per region (writer, fold, label, each arm's reading) -- student text, it stays out of Git.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..recognize_lines import UNREADABLE
from ..util import SEED, digest, require, write_json
from .coverage import prepare


def edits(label, reading):
    """(substituted, deleted, inserted) of a minimal edit alignment of the label to the reading; ties go to the diagonal, as in
    ``coverage.alignment``, so N - substituted - deleted is the count of characters ``revision_boxes.held`` keeps."""
    n, m = len(label), len(reading)
    cost = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        cost[i][0] = i
    for j in range(m+1):
        cost[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost[i][j] = min(cost[i-1][j]+1, cost[i][j-1]+1, cost[i-1][j-1]+(label[i-1] != reading[j-1]))
    i, j, substituted, deleted, inserted = n, m, 0, 0, 0
    while i or j:
        if i and j and label[i-1] == reading[j-1] and cost[i][j] == cost[i-1][j-1]:
            i, j = i-1, j-1
        elif i and j and cost[i][j] == cost[i-1][j-1]+1:
            substituted, i, j = substituted+1, i-1, j-1
        elif i and cost[i][j] == cost[i-1][j]+1:
            deleted, i = deleted+1, i-1
        else:
            inserted, j = inserted+1, j-1
    return substituted, deleted, inserted


def refused(reading):
    return UNREADABLE.strip('[]') in (reading or '')


def counts(label, reading):
    """The edit counts of one readable region; a refusal reads nothing."""
    x, y = prepare(label).replace('?', ''), '' if refused(reading) else prepare(reading)
    s, d, i = edits(x, y)
    return {'n': len(x), 's': s, 'd': d, 'i': i, 'm': len(y), 'exact': float(x == y)}


def rates(rows):
    """The scores of the readable regions whose counts are given."""
    n, s, d, i = (sum(r[k] for r in rows) for k in 'nsdi')
    return {'cer': (s+d+i)/n, 'cr': (n-d-s)/n, 'ar': (n-d-s-i)/n, 'exact': float(np.mean([r['exact'] for r in rows])),
            'ned': float(np.mean([(r['s']+r['d']+r['i'])/max(r['n'], r['m'], 1) for r in rows]))}


def refusal(readings, readable):
    """Precision, recall and F1 of the refusals among the regions read, and the share of the readable regions refused."""
    said = {k for k, text in readings.items() if refused(text)}
    cannot, can = {k for k in readings if not readable[k]}, {k for k in readings if readable[k]}
    hit = len(said & cannot); precision, recall = hit/len(said) if said else None, hit/len(cannot) if cannot else None
    f1 = 2*precision*recall/(precision+recall) if precision and recall else (0. if cannot and said else None)
    return {'unreadable': len(cannot), 'refusals': len(said), 'precision': precision, 'recall': recall, 'f1': f1,
            'readable_refused': len(said & can)/len(can) if can else None}


LENGTHS = (('1-2', 1, 2), ('3-9', 3, 9), ('10+', 10, None))


def left_wrong(readings, scored):
    """The readable regions by what is left wrong, and the exact share by the length of the label."""
    wrong = {'right': 0, 'refused': 0, '1': 0, '2': 0, '3+': 0}
    for k, c in scored.items():
        e = c['s']+c['d']+c['i']
        wrong['right' if not e else 'refused' if refused(readings[k]) else str(e) if e < 3 else '3+'] += 1
    by_length = {}
    for name, low, high in LENGTHS:
        sel = [c['exact'] for c in scored.values() if low <= c['n'] and (high is None or c['n'] <= high)]
        by_length[name] = {'regions': len(sel), 'exact': float(np.mean(sel)) if sel else None}
    return {'left_wrong': wrong, 'exact_by_length': by_length}


def paired(reference, version, writers, rng, draws=1000):
    """The version's scores minus the reference's on the regions both have counts for, with writer-resampled 95% intervals."""
    ids = sorted(set(reference) & set(version)); names = sorted({writers[k] for k in ids}); by = {w: [k for k in ids if writers[k] == w] for w in names}
    def difference(keys):
        a, b = rates([reference[k] for k in keys]), rates([version[k] for k in keys])
        return {m: b[m]-a[m] for m in ('ar', 'exact', 'ned')}
    full = difference(ids); sample = [difference([k for w in rng.choice(names, len(names)) for k in by[w]]) for _ in range(draws)]
    return {'regions': len(ids), 'writers': len(names), **{m: {'difference': full[m], 'interval': [float(np.percentile([s[m] for s in sample], q)) for q in (2.5, 97.5)]}
                                                       for m in full}}


def agreement(first, second, rows):
    """Do two arms' differing readings mark the first one's errors?  Counts over the regions both read."""
    spelt = lambda text: prepare(UNREADABLE) if refused(text) else prepare(text)
    ids = sorted(set(first) & set(second)); differ = {k for k in ids if spelt(first[k]) != spelt(second[k])}
    can = [k for k in ids if rows[k].get('readable', True)]; cannot = [k for k in ids if not rows[k].get('readable', True)]
    right = {k for k in can if spelt(first[k]) == prepare(rows[k]['text']).replace('?', '')}
    share = lambda part, whole: len(part)/len(whole) if whole else None
    same, other = [k for k in can if k not in differ], [k for k in can if k in differ]
    return {'regions': len(ids), 'differ': share(differ, ids), 'exact_where_same': share(right & set(same), same), 'exact_where_differ': share(right & set(other), other),
            'wrong_caught': share([k for k in other if k not in right], [k for k in can if k not in right]), 'unreadable_differ': share([k for k in cannot if k in differ], cannot)}


def read_arm(spec):
    """(name, {region id: reading}, the files read) of ``NAME=DIR[,DIR...][@CONDITION]``."""
    name, _, rest = spec.partition('='); folders, _, condition = rest.partition('@'); readings, files = {}, []
    require(name and folders, f'not NAME=DIR[,DIR...][@CONDITION]: {spec}')
    for folder in folders.split(','):
        files.append(Path(folder)/'predictions.jsonl')
        for line in files[-1].read_text(encoding='utf-8').splitlines():
            g = json.loads(line)
            if condition and g['condition'] != condition:
                continue
            require(g['id'] not in readings, f'{name}: region {g["id"]} read twice (name the condition, or a fold is given twice)')
            readings[g['id']] = g['text']
    require(readings, f'{name}: no reading')
    return name, readings, files


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--pack', type=Path, required=True); p.add_argument('--arm', action='append', required=True, help='NAME=DIR[,DIR...][@CONDITION]')
    p.add_argument('--reference', required=True, help='the arm every other is paired with'); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--agreement', action='append', default=[], help='A=B: do the two arms differing mark the errors of A')
    p.add_argument('--gone', action='append', default=[], help='an arm read with the struck writing taken off: the share of its readings that refuse')
    p.add_argument('--only', help='FILE:KEY, only the regions listed under KEY in a JSON file')
    p.add_argument('--readings', type=Path, help='OUT.jsonl, a row per region with every arm\'s reading')
    args = p.parse_args(argv)
    require(not args.output.exists() and not (args.readings and args.readings.exists()), 'output exists')
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    if args.only:
        file, _, key = args.only.rpartition(':'); only = set(json.loads(Path(file).read_text())[key])
        rows = {k: r for k, r in rows.items() if k in only}
    readable = {k: r.get('readable', True) for k, r in rows.items()}; writers = {k: r.get('writer') or r['line_id'].split('_')[0] for k, r in rows.items()}
    arms, inputs = {}, {}
    for spec in args.arm:
        name, readings, files = read_arm(spec); require(name not in arms, f'arm {name} twice')
        arms[name] = {k: v for k, v in readings.items() if k in rows}; inputs.update({str(f): digest(f) for f in files})
    require(args.reference in arms, f'no arm {args.reference}')
    scored = {name: {k: counts(rows[k]['text'], text) for k, text in readings.items() if readable[k]} for name, readings in arms.items()}
    rng = np.random.default_rng(SEED); result = {}
    for name in arms:
        result[name] = {'regions': len(arms[name]), 'readable': len(scored[name]), **rates(list(scored[name].values())), 'refusal': refusal(arms[name], readable),
                        **left_wrong(arms[name], scored[name])}
        if name != args.reference:
            result[name]['against_reference'] = paired(scored[args.reference], scored[name], writers, rng)
    for name in args.gone:
        require(name in arms, f'--gone {name}: not an arm')
        result[name]['refused_with_the_writing_gone'] = float(np.mean([refused(t) for t in arms[name].values()]))
        print(f"{name}: refuses {100*result[name]['refused_with_the_writing_gone']:.1f}% of the regions whose struck writing is gone")
    agreed = {}
    for pair in args.agreement:
        a, _, b = pair.partition('='); require(a in arms and b in arms, f'--agreement {pair}: not two arms')
        agreed[pair] = agreement(arms[a], arms[b], rows)
        print('agreement', pair, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in agreed[pair].items()}))
    write_json(args.output, {'reference': args.reference, 'only': args.only, 'arms': result, 'agreement': agreed, 'inputs': inputs})
    if args.readings:
        with args.readings.open('x', encoding='utf-8') as stream:
            for k in sorted(set().union(*arms.values())):
                r = rows[k]
                stream.write(json.dumps({'region': k, 'line': r.get('line'), 'writer': writers[k], 'fold': r.get('fold'), 'readable': readable[k],
                                         'label': r.get('text'), **{name: got.get(k) for name, got in arms.items()}}, ensure_ascii=False)+'\n')
    for name, r in result.items():
        a, f = r.get('against_reference'), r['refusal']
        print(f"{name:28s} AR {100*r['ar']:6.1f} CR {100*r['cr']:5.1f} exact {100*r['exact']:5.1f} NED {100*r['ned']:5.1f} n={r['readable']}"
              + (f" refuse P/R {100*(f['precision'] or 0):.0f}/{100*(f['recall'] or 0):.0f}" if f['unreadable'] else '')
              + (f"  vs ref AR {100*a['ar']['difference']:+5.1f} ({100*a['ar']['interval'][0]:+.1f}, {100*a['ar']['interval'][1]:+.1f})"
                 f" exact {100*a['exact']['difference']:+5.1f} ({100*a['exact']['interval'][0]:+.1f}, {100*a['exact']['interval'][1]:+.1f})" if a else '  (reference)'))


if __name__ == '__main__':
    main()
