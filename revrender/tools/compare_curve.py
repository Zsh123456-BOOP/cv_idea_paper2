"""Compare two training curves (curve.jsonl of a formal run) at the steps both have: validation reading and loss terms.

    python -m revrender.tools.compare_curve OLD/curve.jsonl NEW/curve.jsonl

Each line is {step, validation, loss_terms}.  Equal entries at a step mean the model, the batches drawn so far, the loss, the
optimiser and the validation reading agree through that step.  A run stopped early still compares at the steps it reached;
``same_steps`` in the summary says whether both curves have exactly the same steps (a complete run against a complete run).
"""
import json
import sys
from pathlib import Path


def read(path):
    return {e['step']: e for e in (json.loads(line) for line in Path(path).read_text().splitlines() if line)}


def compare(old, new):
    """[(step, field, equal)] over the steps in both."""
    rows = []
    for step in sorted(set(old) & set(new)):
        rows.append((step, 'validation', old[step]['validation'] == new[step]['validation']))
        rows.append((step, 'loss_terms', old[step]['loss_terms'] == new[step]['loss_terms']))
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    old, new = read(argv[0]), read(argv[1])
    rows = compare(old, new)
    for step, field, equal in rows:
        print(f"step {step} {field}: {'equal' if equal else 'DIFFERENT'}")
    summary = {'steps_compared': sorted({r[0] for r in rows}), 'different': sum(not r[2] for r in rows),
               'identical': bool(rows) and all(r[2] for r in rows), 'same_steps': set(old) == set(new)}
    print(json.dumps(summary))
    return 0 if summary['identical'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
