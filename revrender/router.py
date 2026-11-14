"""Route each struck region to the reading whose evidence is trusted most: the first form of a conflict-calibrated reader.

    python -m revrender.router --pack PACK --source NAME=DIR[,DIR...][@CONDITION] [--source ...] --output-dir OUT

A source is the readings of every region by one way of reading (the scan alone, the erased crop alone, both ...), over the folders of
a cross-validation, each region read by a reader that never trained on its writer (``recognize_lines --backend hf``: the readings
carry their confidence).  The sources disagree on about half the regions, and one of them is often right where the best one is wrong;
which to believe is learned apart from how to read: for each source a critic -- a logistic regression -- gives the chance its reading
of a region is right (it is the label; for a region nobody could read: it refuses) from ``FEATURES``: the reading's own confidence,
whether it refuses, its length, how far it stands from the other sources' readings (normalised edit distance) and whether it equals
any, and what the separator took of the region's ink (the pack's ``map``).  A fold's critics learn on the other folds' regions only.
The router gives each region the reading of the source its critic trusts most, with that trust as its confidence.

OUT/predictions.jsonl holds the routed readings (condition ``routed``; ``eval.transcription`` scores them as one more arm) and
OUT/router.json how often each source is right, chosen and trusted rightly, and the selective scores (``eval.selective``) of the routed
readings and of each source by its own confidence: the area under the risk-coverage curve, the risk at 90% and 80% coverage (a
region's loss is 1 unless its reading is right) and how well low trust tells the regions nobody can read.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .eval.coverage import prepare
from .eval.selective import auroc, aurc, risk_at
from .eval.transcription import edits, refused
from .recognize_lines import UNREADABLE
from .util import digest, make_output_dir, require, write_json, write_manifest

FEATURES = ('confidence', 'refuses', 'length', 'distance to the others', 'equals another', 'ink taken away', 'ink')


def spelt(text):
    return prepare(UNREADABLE) if refused(text) else prepare(text)


def right(row, text):
    """Is a reading of a region right: the label, or, of a region nobody could read, a refusal."""
    return spelt(text) == prepare(row['text']).replace('?', '') if row.get('readable', True) else refused(text)


def distance(a, b):
    return sum(edits(a, b))/max(len(a), len(b), 1)


def taken(root, row):
    """(the share of the region's ink the separator took away, the share of the crop that is ink)."""
    scan = np.asarray(Image.open(root/row['images']['raw']).convert('L')); share = np.asarray(Image.open(root/row['images']['map']).convert('L'), np.float32)/255
    ink = scan < 200
    return (float(share[ink].mean()) if ink.any() else 0., float(ink.mean()))


def features(name, readings, row, ink):
    """One source's reading of a region as the critic sees it (``FEATURES``)."""
    mine = spelt(readings[name]['text']); others = [spelt(r['text']) for other, r in readings.items() if other != name]
    return [readings[name].get('confidence', 0.), float(refused(readings[name]['text'])), np.log1p(len(mine)),
            float(np.mean([distance(mine, o) for o in others])) if others else 0., float(any(mine == o for o in others)), *ink]


def fit(x, y, ridge=1., steps=50):
    """A logistic regression (standardised features, a ridge on the weights, Newton's steps): (mean, scale, weights with the bias last)."""
    mean, scale = x.mean(0), x.std(0)+1e-6; z = np.c_[(x-mean)/scale, np.ones(len(x))]; w = np.zeros(z.shape[1]); penalty = ridge*np.eye(z.shape[1]); penalty[-1, -1] = 0
    for _ in range(steps):
        p = 1/(1+np.exp(-z@w)); step = np.linalg.solve(z.T@(z*(p*(1-p))[:, None])+penalty, z.T@(y-p)-penalty@w); w += step
        if np.abs(step).max() < 1e-8:
            break
    return mean, scale, w


def trust(model, x):
    mean, scale, w = model
    return 1/(1+np.exp(-np.c_[(x-mean)/scale, np.ones(len(x))]@w))


def read_source(spec):
    """(name, {region id: its reading row}, the files) of ``NAME=DIR[,DIR...][@CONDITION]``."""
    name, _, rest = spec.partition('='); folders, _, condition = rest.partition('@'); readings, files = {}, []
    for folder in folders.split(','):
        files.append(Path(folder)/'predictions.jsonl')
        for line in files[-1].read_text(encoding='utf-8').splitlines():
            g = json.loads(line)
            if not condition or g['condition'] == condition:
                require(g['id'] not in readings, f'{name}: region {g["id"]} read twice'); readings[g['id']] = g
    return name, readings, files


def route(rows, sources, inks):
    """({region id: (source, trust)}, {source: {region id: trust}}): every fold's critics taught on the other folds, then asked."""
    names = list(sources); ids = sorted(rows); folds = sorted({rows[k]['fold'] for k in ids})
    x = {n: np.array([features(n, {m: sources[m][k] for m in names}, rows[k], inks[k]) for k in ids]) for n in names}
    y = {n: np.array([float(right(rows[k], sources[n][k]['text'])) for k in ids]) for n in names}
    fold = np.array([rows[k]['fold'] for k in ids]); trusted = {n: np.zeros(len(ids)) for n in names}
    for f in folds:
        for n in names:
            trusted[n][fold == f] = trust(fit(x[n][fold != f], y[n][fold != f]), x[n][fold == f])
    best = np.argmax(np.stack([trusted[n] for n in names]), 0)
    return {k: (names[best[i]], float(trusted[names[best[i]]][i])) for i, k in enumerate(ids)}, {n: dict(zip(ids, trusted[n].tolist())) for n in names}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--pack', type=Path, required=True); p.add_argument('--source', action='append', required=True, help='NAME=DIR[,DIR...][@CONDITION]')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}; sources, inputs = {}, {str(args.pack/'pack.json'): digest(args.pack/'pack.json')}
    for spec in args.source:
        name, readings, files = read_source(spec); require(set(readings) == set(rows), f'{name}: not a reading of every region of the pack')
        sources[name] = readings; inputs.update({str(f): digest(f) for f in files})
    require(len(sources) > 1, 'nothing to route between')
    inks = {k: taken(args.pack, r) for k, r in rows.items()}
    chosen, trusted = route(rows, sources, inks); ids = sorted(rows); out = make_output_dir(args.output_dir)
    with (out/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for k in ids:
            name, value = chosen[k]
            stream.write(json.dumps({'condition': 'routed', 'id': k, 'line_id': k, 'model': 'router', 'text': sources[name][k]['text'], 'source': name, 'confidence': value}, ensure_ascii=False)+'\n')
    loss = lambda name_of: np.array([1.-right(rows[k], sources[name_of(k)][k]['text']) for k in ids])
    cannot = np.array([not rows[k].get('readable', True) for k in ids])
    def selective(confidence, losses):
        return {'error': float(losses.mean()), 'aurc': aurc(confidence, losses), 'risk_at_90': risk_at(confidence, losses, .9), 'risk_at_80': risk_at(confidence, losses, .8),
                'unreadable_auroc': auroc(-np.asarray(confidence), cannot)}
    report = {'regions': len(ids), 'unreadable': int(cannot.sum()),
              'routed': {**selective([chosen[k][1] for k in ids], loss(lambda k: chosen[k][0])), 'chosen': {n: sum(chosen[k][0] == n for k in ids) for n in sources}},
              'any_source_right': float(np.mean([any(right(rows[k], sources[n][k]['text']) for n in sources) for k in ids])),
              'sources': {n: {'own_confidence': selective([sources[n][k].get('confidence', 0.) for k in ids], loss(lambda k, n=n: n)),
                              'critic': selective([trusted[n][k] for k in ids], loss(lambda k, n=n: n))} for n in sources}}
    write_json(out/'router.json', report)
    write_manifest(out, config={'pack': str(args.pack), 'sources': args.source, 'features': FEATURES}, inputs=inputs, regions=len(ids))
    print(json.dumps({'routed': report['routed'], 'any_source_right': report['any_source_right'],
                      'sources': {n: {k: round(v['error'], 4) for k, v in s.items()} for n, s in report['sources'].items()}}), flush=True)


if __name__ == '__main__':
    main()
