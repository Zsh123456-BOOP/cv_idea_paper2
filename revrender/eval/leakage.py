"""How much of the struck-out text a reading keeps, how much of the rewrite, and what a condition does to lines with no revision.

    python -m revrender.eval.leakage --pack PACK --predictions OUT/predictions.jsonl [...] --output scores.json

Per model and condition (``revrender.recognize_lines``), on the pack's lines:

  leak    of the lines whose struck text a person wrote down (``text_before``, readable, long enough to look for) the share whose reading
          contains it (``eval.fragments.contains``): the lower the better when the words ask for the final work only
  keep    of the lines whose rewrite a person wrote down (``text_after``) the share whose reading contains it: the higher the better
  blank   of the lines with nothing written after the strike (``no_rewrite``) the share the model read as empty: the higher the better
  change  on the controls (lines a person judged to have no revision) how far the reading moves from the same model's ``raw_neutral``
          reading: 1 - similarity, and the share of lines that move by more than ``MOVED``; a condition that harms clean lines shows here

Intervals are 95% by resampling writers (the lines of one writer are not independent), seed 42; ``difference`` is a condition's leak or
keep minus ``raw_final``'s on the lines both can be told for, with the same resampling.  A share with no line to count is None.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from ..util import SEED, digest, require, to_jsonable, write_json, provenance
from .fragments import contains, similarity

MOVED = .2                 # a reading that moved by more than this (1 - similarity) from the raw neutral one has been changed
BASELINE = 'raw_neutral'
REFERENCE = 'raw_final'
RESAMPLES = 1000


def flags(row, reading):
    """The leak / keep / blank flags of one reading of a pair line: True, False or None (cannot be told)."""
    before = None if row.get('before_unreadable') or not row.get('text_before') else contains(reading, row['text_before'])
    after = contains(reading, row['text_after']) if row.get('text_after') else None
    blank = (not reading.strip()) if row.get('no_rewrite') and row.get('text_before') else None
    return {'leak': before, 'keep': after, 'blank': blank}


def writer_of(row):
    return row['image_id'].split('_')[0]


def interval(values, writers, rng, resamples=RESAMPLES):
    """(share, 2.5th, 97.5th) of a list of 0/1 values by resampling writers; None for an empty list."""
    if not values:
        return None
    by = defaultdict(list)
    for v, w in zip(values, writers):
        by[w].append(v)
    names = sorted(by)
    shares = []
    for _ in range(resamples):
        drawn = [v for w in (rng.choice(names) for _ in names) for v in by[w]]
        shares.append(sum(drawn)/len(drawn))
    shares.sort()
    return [sum(values)/len(values), shares[int(.025*resamples)], shares[int(.975*resamples)-1]]


def difference(a, b, writers, rng, resamples=RESAMPLES):
    """(mean of a - b, 2.5th, 97.5th) over the lines both are told for (lists of 0/1/None), resampling writers."""
    pairs = [(x, y, w) for x, y, w in zip(a, b, writers) if x is not None and y is not None]
    if not pairs:
        return None
    by = defaultdict(list)
    for x, y, w in pairs:
        by[w].append(int(x)-int(y))
    names = sorted(by)
    means = []
    for _ in range(resamples):
        drawn = [d for w in (rng.choice(names) for _ in names) for d in by[w]]
        means.append(sum(drawn)/len(drawn))
    means.sort()
    return [sum(d for ds in by.values() for d in ds)/len(pairs), means[int(.025*resamples)], means[int(.975*resamples)-1]]


def score(pack_rows, predictions, seed=SEED):
    """{model: {condition: {...}}} from the pack's rows and predictions rows (condition, id, model, text)."""
    rows = {r['id']: r for r in pack_rows}
    by = defaultdict(lambda: defaultdict(dict))
    for p in predictions:
        by[p['model']][p['condition']][p['id']] = p['text']
    out = {}
    for model, conditions in by.items():
        out[model] = {}
        base = conditions.get(BASELINE, {})
        ref = {i: flags(rows[i], t) for i, t in conditions.get(REFERENCE, {}).items() if rows[i]['kind'] == 'pair'}
        for condition, texts in conditions.items():
            rng = random.Random(seed)
            pair_ids = sorted(i for i in texts if rows[i]['kind'] == 'pair')
            got = {i: flags(rows[i], texts[i]) for i in pair_ids}
            entry = {'lines': len(pair_ids)}
            for name in ('leak', 'keep', 'blank'):
                ids = [i for i in pair_ids if got[i][name] is not None]
                entry[name] = {'n': len(ids), 'share': interval([int(got[i][name]) for i in ids], [writer_of(rows[i]) for i in ids], rng)}
                if condition != REFERENCE and name != 'blank' and REFERENCE in conditions:
                    entry[name]['difference_from_raw_final'] = difference([got[i][name] for i in pair_ids], [ref[i][name] for i in pair_ids],
                                                                         [writer_of(rows[i]) for i in pair_ids], random.Random(seed))
            groups = defaultdict(list)
            for i in pair_ids:
                groups[rows[i].get('ink_kind') or 'unknown'].append(i)
            entry['by_ink_kind'] = {k: {'lines': len(v), 'leak': (lambda xs: sum(xs)/len(xs) if xs else None)([int(got[i]['leak']) for i in v if got[i]['leak'] is not None]),
                                        'keep': (lambda xs: sum(xs)/len(xs) if xs else None)([int(got[i]['keep']) for i in v if got[i]['keep'] is not None])}
                                    for k, v in sorted(groups.items())}
            control_ids = sorted(i for i in texts if rows[i]['kind'] == 'no_edit' and i in base)
            moves = [1.-similarity(texts[i], base[i]) for i in control_ids]
            entry['controls'] = {'lines': len(control_ids), 'mean_change': mean(moves) if moves else None,
                                 'share_moved': sum(m > MOVED for m in moves)/len(moves) if moves else None,
                                 'empty': sum(not texts[i].strip() for i in control_ids)/len(control_ids) if control_ids else None}
            out[model][condition] = entry
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, action='append', required=True, help='predictions.jsonl of revrender.recognize_lines (repeatable)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'create-only output exists', FileExistsError)
    pack = json.loads((args.pack/'pack.json').read_text())
    predictions = [json.loads(line) for path in args.predictions for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    scores = score(pack['rows'], predictions)
    write_json(args.output, {'scores': scores, 'definitions': {'moved': MOVED, 'baseline': BASELINE, 'reference': REFERENCE, 'resamples': RESAMPLES},
                             'provenance': provenance(to_jsonable(args), {str(args.pack/'pack.json'): digest(args.pack/'pack.json'),
                                                                          **{str(p): digest(p) for p in args.predictions}})})
    for model, conditions in scores.items():
        for condition, e in conditions.items():
            share = lambda k: ('%.2f' % e[k]['share'][0]) if e[k]['share'] else '--'
            print(f"{model} {condition:15} leak {share('leak')} (n={e['leak']['n']}) keep {share('keep')} (n={e['keep']['n']}) blank {share('blank')} "
                  f"| controls change {e['controls']['mean_change'] if e['controls']['mean_change'] is None else round(e['controls']['mean_change'], 3)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
