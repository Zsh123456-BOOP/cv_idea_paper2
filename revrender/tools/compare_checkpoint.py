"""Compare two RevRender checkpoints tensor by tensor.

    python -m revrender.tools.compare_checkpoint OLD/best.pt NEW/best.pt

The same parameter names in the same order, the same chosen step, and every tensor exactly equal.  The files themselves can
differ in bytes (serialisation metadata) while every tensor is equal; the tensors are what count.
"""
import json
import sys

import torch


def compare(old, new):
    """What differs between two checkpoints' state dicts."""
    a, b = old['state_dict'], new['state_dict']
    different = [name for name in a if name in b and not torch.equal(a[name], b[name])]
    return {'same_names': set(a) == set(b), 'same_order': list(a) == list(b), 'tensors': len(a),
            'steps': [old['step'], new['step']],
            'different': {name: float((a[name].double()-b[name].double()).abs().max()) for name in different},
            'identical': set(a) == set(b) and list(a) == list(b) and not different and old['step'] == new['step']}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    old, new = (torch.load(path, map_location='cpu', weights_only=False) for path in argv)
    result = compare(old, new)
    print(json.dumps(result))
    return 0 if result['identical'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
