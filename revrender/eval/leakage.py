"""How much of the struck-out text a reading keeps, how much of the valid text it loses, and what a condition does to lines with no revision.

    python -m revrender.eval.leakage --pack PACK --predictions OUT/predictions.jsonl [...] --output scores.json

Per model and condition (``revrender.recognize_lines``), on the pack's lines labelled with the text before and after the strike
(``eval.coverage``; a line whose struck text a person could not read is left out):

  leak     the share of the struck characters the reading holds (mean over lines): the lower the better when the words ask for the
           final work only
  rewrite  the share of the rewritten characters the reading holds
  valid    the share of the characters valid before and after the strike the reading holds: the higher the better; a condition that
           whites out what it should not shows here
  blank    of the lines whose whole text was struck with nothing written after (``no_rewrite``) the share the model read as empty: the
           higher the better
  change   on the controls (lines a person judged to have no revision) how far the reading moves from the same model's ``raw_neutral``
           reading: 1 - similarity, and the share of lines that move by more than ``MOVED``

Beside each share's mean over lines (``mean``: [mean, 2.5th, 97.5th]) an entry has ``pooled`` (characters over characters, so a line
of forty struck characters weighs forty times a line of one), ``strict`` and ``lenient`` (the mean when only the characters held under
both alignments of ``eval.coverage`` count, and under either), ``masked`` and ``unmasked`` (the mean on lines RevRender gave a mask and
on those it did not: a mask-conditioned reading cannot help the second), ``by_extent`` (lines struck out whole, with nothing after, against lines
struck in part: the first is mostly a question of saying nothing, the second of reading around the strike), ``abstain`` (the share of lines read as nothing or as
placeholders only: a leak is easily lowered by saying nothing, so read it with ``valid``) and ``change_from_raw_final`` (per line, the
share that lose or gain more than ``MOVED_SHARE`` and the median change: the paired mean is carried by a few lines).

``references`` scores what no model reads: exactly the text after the strike (the ceiling), exactly the text before it (leak 1), an
empty reading and the reading of another line (the one with the nearest label length), so a share can be read against what the labels
allow and against chance.  Intervals are 95% by resampling writers (the lines of one writer are not independent), seed 42; ``difference``
is a condition's mean minus ``raw_final``'s on the lines both are told for, with the same resampling.  A share with no line to count is None.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from ..util import SEED, digest, require, to_jsonable, write_json, provenance
from .coverage import counts, prepare, similarity

MOVED_SHARE = .05          # a line whose share moves by more than this from raw_final's has lost or gained
MOVED = .2                 # a reading that moved by more than this (1 - similarity) from the raw neutral one has been changed
BASELINE = 'raw_neutral'
REFERENCE = 'raw_final'
RESAMPLES = 1000
SHARES = ('leak', 'rewrite', 'valid')


def labelled(row):
    """Whether a pack row carries the text before the strike (a line whose struck text could not be read has none to count)."""
    return row['kind'] == 'pair' and not row.get('before_unreadable') and bool(prepare(row.get('text_before')))


def line_counts(row, reading):
    """``eval.coverage.counts`` of one reading of a labelled line."""
    return counts(reading, row['text_before'], row.get('text_after'))


def flags(row, reading):
    """The shares of one reading of a labelled line (``eval.coverage``) and whether it is empty where the whole line was struck."""
    got = line_counts(row, reading)
    shares = {name: got[name]['held']/got[name]['total'] if name in got else None for name in SHARES}
    shares['blank'] = (not reading.strip()) if row.get('no_rewrite') else None
    return shares


def abstained(reading):
    """Whether a reading is nothing or placeholders only (□)."""
    return not prepare(reading).replace('□', '')


def writer_of(row):
    return row['image_id'].split('_')[0]


def draws(by, rng):
    names = sorted(by)
    return [v for w in (rng.choice(names) for _ in names) for v in by[w]]


def interval(values, writers, rng, resamples=RESAMPLES):
    """(mean, 2.5th, 97.5th) of a list of numbers by resampling writers; None for an empty list."""
    if not values:
        return None
    by = defaultdict(list)
    for v, w in zip(values, writers):
        by[w].append(v)
    means = sorted(mean(draws(by, rng)) for _ in range(resamples))
    return [mean(values), means[int(.025*resamples)], means[int(.975*resamples)-1]]


def difference(a, b, writers, rng, resamples=RESAMPLES):
    """(mean of a - b, 2.5th, 97.5th) over the lines both are told for (lists of numbers or None), resampling writers."""
    by = defaultdict(list)
    for x, y, w in zip(a, b, writers):
        if x is not None and y is not None:
            by[w].append(x-y)
    if not by:
        return None
    means = sorted(mean(draws(by, rng)) for _ in range(resamples))
    return [mean(d for ds in by.values() for d in ds), means[int(.025*resamples)], means[int(.975*resamples)-1]]


def references(pack_rows):
    """The mean shares of four readings no model gives: the text after the strike, the text before it, nothing, another line's text."""
    labelled_rows = sorted((r for r in pack_rows if labelled(r)), key=lambda r: r['id'])
    lengths = {r['id']: len(prepare(r.get('text_after'))) for r in labelled_rows}

    def other(row):
        rest = [o for o in labelled_rows if o['id'] != row['id']]
        return min(rest, key=lambda o: (abs(lengths[o['id']]-lengths[row['id']]), o['id'])).get('text_after') or ''
    out = {}
    for name, read in (('exactly_after', lambda r: r.get('text_after') or ''), ('exactly_before', lambda r: r.get('text_before') or ''), ('empty', lambda r: ''),
                       ('other_line_of_the_same_length', other)):
        got = [flags(r, read(r)) for r in labelled_rows]
        out[name] = {k: {'n': len([g for g in got if g[k] is not None]), 'mean': mean(g[k] for g in got if g[k] is not None) if any(g[k] is not None for g in got) else None}
                     for k in SHARES}
    return out


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
        ref = {i: flags(rows[i], t) for i, t in conditions.get(REFERENCE, {}).items() if labelled(rows[i])}
        for condition, texts in conditions.items():
            rng = random.Random(seed)
            pair_ids = sorted(i for i in texts if labelled(rows[i]))
            tallies = {i: line_counts(rows[i], texts[i]) for i in pair_ids}
            got = {i: flags(rows[i], texts[i]) for i in pair_ids}
            entry = {'lines': len(pair_ids), 'abstain': mean(abstained(texts[i]) for i in pair_ids) if pair_ids else None}
            for name in (*SHARES, 'blank'):
                ids = [i for i in pair_ids if got[i][name] is not None]
                entry[name] = {'n': len(ids), 'mean': interval([float(got[i][name]) for i in ids], [writer_of(rows[i]) for i in ids], rng)}
                if name in SHARES:
                    have = [i for i in pair_ids if name in tallies[i]]
                    total = sum(tallies[i][name]['total'] for i in have)
                    entry[name]['pooled'] = sum(tallies[i][name]['held'] for i in have)/total if total else None
                    for bound in ('strict', 'lenient'):
                        entry[name][bound] = mean(tallies[i][name][bound]/tallies[i][name]['total'] for i in have) if have else None
                    for label, wanted in (('masked', True), ('unmasked', False)):
                        part = [got[i][name] for i in ids if (rows[i].get('masked_pixels', 0) > 0) == wanted]
                        entry[name][label] = {'n': len(part), 'mean': mean(part) if part else None}
                if condition != REFERENCE and name != 'blank' and REFERENCE in conditions:
                    entry[name]['difference_from_raw_final'] = difference([got[i][name] for i in pair_ids], [ref[i][name] for i in pair_ids],
                                                                         [writer_of(rows[i]) for i in pair_ids], random.Random(seed))
                    moves = [got[i][name]-ref[i][name] for i in pair_ids if got[i][name] is not None and ref[i][name] is not None]
                    entry[name]['change_from_raw_final'] = {'n': len(moves), 'lose': sum(m < -MOVED_SHARE for m in moves)/len(moves) if moves else None,
                                                            'gain': sum(m > MOVED_SHARE for m in moves)/len(moves) if moves else None, 'median': median(moves) if moves else None}
            groups = defaultdict(list)
            for i in pair_ids:
                groups[rows[i].get('ink_kind') or 'unknown'].append(i)
            mean_of = lambda xs: mean(xs) if xs else None
            entry['by_ink_kind'] = {k: {'lines': len(v), **{s: mean_of([got[i][s] for i in v if got[i][s] is not None]) for s in SHARES}} for k, v in sorted(groups.items())}
            extents = {'whole_line_struck': [i for i in pair_ids if not prepare(rows[i].get('text_after'))], 'partial': [i for i in pair_ids if prepare(rows[i].get('text_after'))]}
            entry['by_extent'] = {k: {'lines': len(v), **{s: mean_of([got[i][s] for i in v if got[i][s] is not None]) for s in SHARES}, 'abstain': mean_of([abstained(texts[i]) for i in v])}
                                  for k, v in extents.items()}
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
    refs = references(pack['rows'])
    write_json(args.output, {'scores': scores, 'references': refs, 'definitions': {'moved': MOVED, 'baseline': BASELINE, 'reference': REFERENCE, 'resamples': RESAMPLES},
                             'provenance': provenance(to_jsonable(args), {str(args.pack/'pack.json'): digest(args.pack/'pack.json'),
                                                                          **{str(p): digest(p) for p in args.predictions}})})
    for name, shares in refs.items():
        print(f"reference {name:15} " + ' '.join(f"{k} {'--' if s['mean'] is None else '%.2f' % s['mean']} (n={s['n']})" for k, s in shares.items()))
    for model, conditions in scores.items():
        for condition, e in conditions.items():
            share = lambda k: ('%.2f' % e[k]['mean'][0]) if e[k]['mean'] else '--'
            pooled = lambda k: ('%.2f' % e[k]['pooled']) if e[k]['pooled'] is not None else '--'
            print(f"{model} {condition:25} leak {share('leak')} (pooled {pooled('leak')}, n={e['leak']['n']}) rewrite {share('rewrite')} (n={e['rewrite']['n']}) "
                  f"valid {share('valid')} (n={e['valid']['n']}) blank {share('blank')} abstain {e['abstain'] if e['abstain'] is None else round(e['abstain'], 2)} "
                  f"| controls change {e['controls']['mean_change'] if e['controls']['mean_change'] is None else round(e['controls']['mean_change'], 3)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
