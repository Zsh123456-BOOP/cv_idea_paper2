"""Compare two training traces (trace.json of a smoke run) step by step: loss, every loss term, input batch hash, state hash.

    python -m revrender.tools.compare_trace OLD/trace.json NEW/trace.json

A smoke run records, for each of its first four steps, the loss, the loss terms, the SHA-256 of the input batch and of
the model state after the update.  Equal traces mean the same weights, the same batches, the same forward, loss and
optimiser step.  This is the training-equivalence check of the port against paper 1's recorded smoke trace.
"""
import json
import sys
from pathlib import Path


def compare(old, new):
    """[(step, field, equal, old value, new value)] over every compared field."""
    rows = []
    if len(old) != len(new):
        rows.append((None, 'steps', False, len(old), len(new)))
    for a, b in zip(old, new):
        step = a['step']
        for field in ('loss', 'input_sha256', 'state_sha256'):
            rows.append((step, field, a[field] == b[field], a[field], b[field]))
        names = list(a['pieces']) + [k for k in b['pieces'] if k not in a['pieces']]
        for name in names:
            rows.append((step, 'pieces.'+name, a['pieces'].get(name) == b['pieces'].get(name),
                         a['pieces'].get(name), b['pieces'].get(name)))
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    old, new = (json.loads(Path(p).read_text()) for p in argv)
    rows = compare(old, new)
    different = [r for r in rows if not r[2]]
    for step, field, equal, a, b in rows:
        if not equal or field in ('loss', 'state_sha256'):
            print(f"step {step} {field}: {'equal' if equal else 'DIFFERENT'}  old={a}  new={b}")
    print(json.dumps({'fields_compared': len(rows), 'different': len(different), 'identical': not different}))
    return 0 if not different else 1


if __name__ == '__main__':
    raise SystemExit(main())
