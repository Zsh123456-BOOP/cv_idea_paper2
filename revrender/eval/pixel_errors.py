"""Where each model's wrong pixels lie, by what the pixel is: ink or blank, inside, next to or away from a revision.

At each model's frozen point (the delivered masks after NMS and the score cut) every validation pixel falls in one class:
  revision ink       inside a gold revision, ink (grey below eval.ink.INK_BELOW)    wrong when no delivered mask covers it
  revision blank     inside a gold revision, not ink                                wrong when no delivered mask covers it
  touching ink       outside, within NEAR px of a revision, ink                     wrong when a delivered mask covers it
  near blank         outside, within NEAR px, not ink                               wrong when a delivered mask covers it
  far ink            outside, farther than NEAR px, ink                             wrong when a delivered mask covers it
  background         outside, farther, not ink                                      wrong when a delivered mask covers it
and for each class the error rate (wrong / pixels of that class) and its share of all wrong pixels are read.  Gold revisions are
drawn as regions, so the blank inside them and the strokes touching them are where a pixel's label depends on its surroundings
rather than on its own ink.  A model and its point are not changed here.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..data.bundle import gold_source_sha, load_bundle
from ..util import digest, provenance, require, to_jsonable
from . import ink as ink_view
from . import matching
from .fates import delivered
from .freeze import read_frozen

SCHEMA = 'revrender-pixel-errors-v1'
NEAR = 3
CLASSES = ('revision ink', 'revision blank', 'touching ink', 'near blank', 'far ink', 'background')
INSIDE = ('revision ink', 'revision blank')


def classes_of(ink, gold):
    """Boolean maps of the six classes for one line (``ink``: the line's ink map, ``gold``: the union of its revisions)."""
    near = (ndimage.distance_transform_edt(~gold) <= NEAR) & ~gold if gold.any() else np.zeros_like(gold)
    far = ~gold & ~near
    return {'revision ink': gold & ink, 'revision blank': gold & ~ink, 'touching ink': near & ink,
            'near blank': near & ~ink, 'far ink': far & ink, 'background': far & ~ink}


def wrong(cls, name, covered):
    """The pixels of a class that are wrong: uncovered inside a revision, covered outside it."""
    return cls & ~covered if name in INSIDE else cls & covered


def read_model(job):
    label, path, bundle_sha, source_sha, order, golds, inks = job
    provide, _, point = read_frozen(path, bundle_sha, source_sha)
    pixels = {c: 0 for c in CLASSES}
    errors = {c: 0 for c in CLASSES}
    for line, gold, ink in zip(order, golds, inks):
        masks, scores = provide(line)
        chosen = delivered(masks, np.asarray(scores, np.float64), point)
        covered = np.logical_or.reduce([masks[i] for i in chosen]) if chosen else np.zeros(gold.shape, bool)
        for name, cls in classes_of(ink, gold).items():
            pixels[name] += int(cls.sum())
            errors[name] += int(wrong(cls, name, covered).sum())
    total = sum(errors.values())
    return label, {name: {'pixels': pixels[name], 'wrong': errors[name],
                          'error_rate': errors[name]/pixels[name] if pixels[name] else None,
                          'share_of_wrong': errors[name]/total if total else None} for name in CLASSES}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--freeze', action='append', required=True, help='label=freeze.json')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'create-only output exists', FileExistsError)
    labels = [e.split('=', 1)[0] for e in args.freeze]
    require(len(set(labels)) == len(labels), 'duplicate freeze label')
    source = gold_source_sha(args.bundle, args.bundle_sha)
    bundle = load_bundle(args.bundle, args.bundle_sha)
    dev, order, _ = matching.validation_order(bundle['dev'])
    golds = [np.asarray(r['mask'], bool) for r in dev]
    inks = ink_view.ink_maps(dev)
    jobs = [(label, path, args.bundle_sha, source, order, golds, inks) for label, path in (e.split('=', 1) for e in args.freeze)]
    models = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for label, reading in pool.map(read_model, jobs):
            models[label] = reading
            print(json.dumps({label: {c: r['error_rate'] and round(r['error_rate'], 3) for c, r in reading.items()}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps({'schema': SCHEMA, 'test_touched': False, 'bundle_sha256': args.bundle_sha,
                            'ink_below': ink_view.INK_BELOW, 'near_px': NEAR, 'classes': list(CLASSES), 'models': models,
                            'provenance': provenance(to_jsonable(args), {str(args.bundle): args.bundle_sha, **{
                                e.split('=', 1)[1]: digest(e.split('=', 1)[1]) for e in args.freeze}})},
                           indent=1, sort_keys=True)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
