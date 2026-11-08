"""Read-back of the struck text by each (model, condition) of one or more readings of a pack (pilot of 2026-11-08).

    python -m revrender.eval.struck_reader --pack EVAL_PACK --readings DIR [DIR ...] --reference 'MODEL|CONDITION' --output OUT.json

Each reading is a ``recognize_lines`` output (``predictions.jsonl``); a (model, condition) is scored on the pack's lines by the share of
the struck characters the reading holds (``eval.revision_boxes.held``) and the share of lines read exactly, and paired with the
reference on the lines both read, with 95% intervals by resampling writers and by resampling lines (1,000, seed 42;
``restoration_diagnostics.paired``).  The pilot's rule, written before its reading: teaching the reader is worth pursuing when the taught
model reading the scan beats the reference (the untaught model reading after erasing) by more than 3 points, its writer interval above 0.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..util import SEED, digest, require, write_json
from .restoration_diagnostics import paired
from .revision_boxes import held, prepare


def scores_of(pack, readings):
    rows = {r['id']: r for r in json.loads((pack/'pack.json').read_text())['rows'] if r['kind'] == 'line'}
    shares, exact = defaultdict(dict), defaultdict(dict)
    for directory in readings:
        for line in (directory/'predictions.jsonl').read_text(encoding='utf-8').splitlines():
            g = json.loads(line); r = rows.get(g['id'])
            if r is None: continue
            key = g['model']+'|'+g['condition']; h = held(r['text'], g['text'])
            if h is not None:
                shares[key][r['line_id']] = h; exact[key][r['line_id']] = float(prepare(g['text']) == prepare(r['text']))
    return shares, exact


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack', type=Path, required=True); p.add_argument('--readings', type=Path, nargs='+', required=True)
    p.add_argument('--reference', required=True, help="MODEL|CONDITION every other is paired with"); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    require(not args.output.exists(), 'output exists')
    shares, exact = scores_of(args.pack, args.readings); require(args.reference in shares, f'no reading {args.reference}')
    rng = np.random.default_rng(SEED); result = {}
    for key in sorted(shares):
        result[key] = {'read_back': float(np.mean(list(shares[key].values()))), 'exact': float(np.mean(list(exact[key].values()))), 'lines': len(shares[key])}
        if key != args.reference: result[key]['against_reference'] = paired(shares[args.reference], shares[key], rng)
    write_json(args.output, {'reference': args.reference, 'versions': result, 'inputs': {str(d/'predictions.jsonl'): digest(d/'predictions.jsonl') for d in args.readings}})
    for key, r in result.items():
        a = r.get('against_reference')
        print(f"{key:60s} read {100*r['read_back']:5.1f} exact {100*r['exact']:5.1f} n={r['lines']}" + (
            f"  vs ref {100*a['difference']:+5.1f} writers ({100*a['writer_interval'][0]:+.1f}, {100*a['writer_interval'][1]:+.1f})"
            f" lines ({100*a['strike_interval'][0]:+.1f}, {100*a['strike_interval'][1]:+.1f})" if a else '  (reference)'))


if __name__ == '__main__':
    main()
