"""Readings of the struck text of a line, by three ways of reading, scored line by line in ``eval.transcription``'s definitions.

    python -m revrender.eval.line_reading --pack LINE_PACK --regions DIR --lines DIR --output-dir OUT

The pack is one of ``data.line_set pack`` (a row per line, its label the struck text of its readable regions left to right); the
readings are ``recognize_lines``'s: DIR of ``--regions`` the pack's ``regions`` read under ``erased_revision``, DIR of ``--lines`` the
lines read under ``before_line``, ``after_line`` and ``difference_line``.  The reading of a line:

  crops       (R0) the readings of its regions' erased crops, in the order of the label, a refusal reading nothing
  difference  (L1) the characters of the reading of ``before`` that a minimal edit alignment (``coverage.alignment``) to the reading of
              ``after`` does not keep, in their order, both readings in the spelling of ``coverage.prepare``
  pair        (L2) the reading of the two images in one request, a refusal reading nothing

A line whose one region is the whole line has no ``after``: its difference and its pair are the reading of ``before``.  Each way is
scored over the lines all three read (CER, CR, AR, exact, NED), over the partial and the whole lines apart and over the lengths of the
label (1-2, 3-9, 10 and more characters); ``difference`` and ``pair`` are paired with ``crops`` and ``pair`` with ``difference`` by
resampling writers (``transcription.paired``, 1,000 draws, a generator of seed 42 for each comparison).  The crops are also scored
region by region (the regions of the lines, readable ones).  ``unmatched``: the partial lines on which the reading of ``after`` holds a
character the reading of ``before`` does not (what the difference cannot take away), and the exact share of the difference on them and
on the other partial lines.  OUT/scores.json; OUT/readings.jsonl a row per line (line, writer, fold, label, the three readings, the two
readings of the difference).  Student text: OUT stays out of Git.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..util import SEED, digest, make_output_dir, require, write_json
from .coverage import alignment, prepare
from .transcription import counts, paired, rates, refused

WAYS = ('crops', 'difference', 'pair')
LENGTHS = (('1-2', 1, 2), ('3-9', 3, 9), ('10+', 10, None))


def readings(folder, condition):
    """{id: reading} of one condition in a folder of ``recognize_lines``."""
    out = {}
    for line in (Path(folder)/'predictions.jsonl').read_text(encoding='utf-8').splitlines():
        g = json.loads(line)
        if g['condition'] == condition:
            require(g['id'] not in out, f"{g['id']} read twice under {condition}")
            out[g['id']] = g['text']
    return out


def difference(before, after):
    """The characters of ``before`` a minimal edit alignment to ``after`` does not keep, in order (both in the spelling of ``prepare``)."""
    b, a = prepare(before), prepare(after)
    kept = alignment(b, a)[0]
    return ''.join(c for i, c in enumerate(b) if i not in kept)


def unmatched(before, after):
    """Does the reading of ``after`` hold a character the alignment to the reading of ``before`` does not keep?"""
    b, a = prepare(before), prepare(after)
    return len(alignment(b, a)[1]) < len(a)


def line_readings(rows, crops, before, after, pair):
    """{way: {line: reading}} over the lines every way reads."""
    out = {way: {} for way in WAYS}
    for row in rows:
        k = row['id']; needed = [k in before, all(r in crops for r in row['readable_regions'])]
        if row['kind'] == 'partial':
            needed += [k in after, k in pair]
        if not all(needed):
            continue
        out['crops'][k] = ''.join('' if refused(crops[r]) else prepare(crops[r]) for r in row['readable_regions'])
        if row['kind'] == 'whole':
            out['difference'][k] = out['pair'][k] = before[k]
        else:
            out['difference'][k] = difference(before[k], after[k]); out['pair'][k] = pair[k]
    return out


def length_of(row):
    return len(prepare(row['text']).replace('?', ''))


def scored(rows, read):
    """{way: {line: counts}}."""
    return {way: {k: counts(rows[k]['text'], text) for k, text in got.items()} for way, got in read.items()}


def summary(rows, by_way):
    """The scores of every way over all the lines, the partial and the whole ones, and by the length of the label."""
    def part(keep):
        return {way: ({'lines': len(sel), **rates(sel)} if sel else {'lines': 0}) for way, got in by_way.items()
                for sel in [[c for k, c in got.items() if keep(rows[k])]]}
    out = {'all': part(lambda r: True), 'partial': part(lambda r: r['kind'] == 'partial'), 'whole': part(lambda r: r['kind'] == 'whole')}
    for name, low, high in LENGTHS:
        out[f'length {name}'] = part(lambda r, low=low, high=high: low <= length_of(r) and (high is None or length_of(r) <= high))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key in ('pack', 'regions', 'lines', 'output-dir'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args(argv)
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    region_rows = {r['id']: r for r in json.loads((args.pack/'regions'/'pack.json').read_text())['rows']}
    crops = readings(args.regions, 'erased_revision'); before, after, pair = (readings(args.lines, c) for c in ('before_line', 'after_line', 'difference_line'))
    read = line_readings(rows.values(), crops, before, after, pair); by_way = scored(rows, read)
    require(read['crops'], 'no line read every way')
    writers = {k: r['writer'] for k, r in rows.items()}
    comparisons = {}
    for reference, version in (('crops', 'difference'), ('crops', 'pair'), ('difference', 'pair')):
        comparisons[f'{version} - {reference}'] = paired(by_way[reference], by_way[version], writers, np.random.default_rng(SEED))
    partial = [k for k in read['difference'] if rows[k]['kind'] == 'partial']; spoilt = [k for k in partial if unmatched(before[k], after[k])]
    clean = [k for k in partial if k not in spoilt]; exact = lambda keys: float(np.mean([by_way['difference'][k]['exact'] for k in keys])) if keys else None
    regions = {k: counts(region_rows[k]['text'], text) for k, text in crops.items() if k in region_rows and region_rows[k]['readable']}
    out = make_output_dir(args.output_dir)
    result = {'lines': len(rows), 'read_every_way': len(read['crops']), 'not_read_every_way': sorted(set(rows)-set(read['crops'])),
              'scores': summary(rows, by_way), 'paired': comparisons,
              'crops_by_region': {'regions': len(regions), **rates(list(regions.values()))},
              'unmatched': {'partial_lines': len(partial), 'lines': len(spoilt), 'share': len(spoilt)/len(partial) if partial else None,
                            'difference_exact_there': exact(spoilt), 'difference_exact_elsewhere': exact(clean)},
              'inputs': {str(f): digest(f) for f in (args.pack/'pack.json', args.pack/'regions'/'pack.json', Path(args.regions)/'predictions.jsonl', Path(args.lines)/'predictions.jsonl')}}
    write_json(out/'scores.json', result)
    with (out/'readings.jsonl').open('x', encoding='utf-8') as stream:
        for k in sorted(read['crops']):
            r = rows[k]
            stream.write(json.dumps({'line': k, 'writer': r['writer'], 'fold': r['fold'], 'kind': r['kind'], 'label': r['text'], **{way: read[way][k] for way in WAYS},
                                     'before': before.get(k), 'after': after.get(k), 'crops_by_region': {x: crops.get(x) for x in r['readable_regions']}}, ensure_ascii=False)+'\n')
    for name, by in result['scores'].items():
        print(name.ljust(12), '  '.join(f"{way} {100*s['ar']:5.1f}/{100*s['exact']:5.1f} n={s['lines']}" if s['lines'] else f'{way} -' for way, s in by.items()))
    for name, c in comparisons.items():
        print(name.ljust(22), ' '.join(f"{m} {100*c[m]['difference']:+5.1f} ({100*c[m]['interval'][0]:+.1f}, {100*c[m]['interval'][1]:+.1f})" for m in ('ar', 'exact')))
    print('crops by region', json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result['crops_by_region'].items()}))
    print('unmatched', json.dumps(result['unmatched']))


if __name__ == '__main__':
    main()
