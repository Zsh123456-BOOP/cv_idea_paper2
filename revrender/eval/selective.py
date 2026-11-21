"""Selective reading: how the error falls when only the readings one is surest of are kept.

``risk_coverage`` sorts the regions by confidence and gives, for every coverage (the share of regions answered), the risk (the mean
loss of the answered ones); ``aurc`` is the area under that curve, lower the better, and ``risk_at`` the risk at one coverage -- the
selective-prediction scores of Geifman and El-Yaniv (SelectiveNet, ICML 2019), as Reliable VQA (ECCV 2022) reports risk and coverage
for answers a model may withhold.  ``auroc`` says how well a score ranks one kind of region above the rest (here: the regions nobody
can read, by how little any reading of them is trusted).

    python -m revrender.eval.selective --pack VIEWS --arm NAME=DIR[,DIR...] [--arm ...] --other R0=DIRS --other X=DIRS
        [--head HEAD_DIR] --output-dir OUT

A region the reader refused is not answered; the others are answered from the surest down, a score below a threshold not answered.
An answer is wrong when the region cannot be read or the reading is not exactly right (``transcription.counts``).  Over all the
pack's regions: coverage the share answered, selective risk the share of the answered that are wrong, AURC the area under risk
against coverage (Geifman et al.; here the sum of the risk at each answered count over all regions), AUROC how well the score tells
right from wrong among the answered (Hendrycks & Gimpel).  The scores (``FEATURES``): (a) the reader's own confidence (the geometric
mean of its tokens' probabilities) and the reading's length; (b) and whether it agrees with R0's reading; (c) and with X's;
(a+head) (a) and the score of ``train_clean_view head``.  Each fold's scores come from a logistic regression fitted on the other folds'
answered regions (standardised features).  Reported: AUROC, AURC, the risk at coverage 60 / 50 / 40 / 30 %; for a target risk of 30 %
and 20 %, a threshold set on the other four folds and used on the fifth, the empirical one and one whose risk's one-sided 95 %
Clopper-Pearson upper bound meets the target (Geifman & El-Yaniv 2017, without their correction over thresholds), the risk and the
coverage it gives.  B1 (reported only): every arm, scores (a), (b), (c).  B2 (the first arm, with ``--head``): AURC(a+head) - AURC(a)
with writer-resampled 95 % interval (1,000 draws, seed 42): upper end below 0 useful, lower end above 0 worse, else cannot tell.
OUT/selective.json; OUT/scores.jsonl every region's scores.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..util import SEED, digest, make_output_dir, write_json, write_manifest
from .coverage import prepare
from .transcription import counts, refused

def risk_coverage(confidence, loss):
    """(coverage, risk): the regions in the order of falling confidence (ties in the given order), the mean loss of the first k."""
    order = np.argsort(-np.asarray(confidence, float), kind='stable'); losses = np.asarray(loss, float)[order]
    k = np.arange(1, len(losses)+1)
    return k/len(losses), np.cumsum(losses)/k


def aurc(confidence, loss):
    return float(np.mean(risk_coverage(confidence, loss)[1]))


def risk_at(confidence, loss, coverage):
    """The risk when the surest ``coverage`` of the regions are answered."""
    covered, risk = risk_coverage(confidence, loss)
    return float(risk[min(len(risk)-1, max(0, int(np.ceil(coverage*len(risk)))-1))])


def auroc(score, positive):
    """The chance a positive region has a higher score than a negative one (ties count a half)."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    if positive.all() or not positive.any():
        return None
    ranks = np.empty(len(score)); order = np.argsort(score, kind='stable'); ranks[order] = np.arange(1, len(score)+1)
    for value in np.unique(score):                                   # the mean rank of ties
        same = score == value
        ranks[same] = ranks[same].mean()
    n, m = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum()-n*(n+1)/2)/(n*m))


FEATURES = {'a': ('confidence', 'length'), 'b': ('confidence', 'length', 'agrees_R0'), 'c': ('confidence', 'length', 'agrees_R0', 'agrees_X'),
            'a+head': ('confidence', 'length', 'head')}
COVERAGES, TARGETS = (.6, .5, .4, .3), (.3, .2)


def readings(spec):
    """{id: (text, confidence)} of an arm's folders."""
    out = {}
    for folder in spec.split(','):
        for line in (Path(folder)/'predictions.jsonl').read_text().splitlines():
            g = json.loads(line); out[g['id']] = (g['text'], g.get('confidence'))
    return out


def same(a, b):
    return (refused(a) and refused(b)) or (not refused(a) and not refused(b) and prepare(a) == prepare(b))


def fitted_scores(table, folds, names):
    """Each answered region's score: a logistic regression on ``names`` fitted on the other folds' answered regions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    out = {}
    for k in sorted(set(folds.values())):
        train = [i for i in table if folds[i] != k]; test = [i for i in table if folds[i] == k]
        x = np.array([[table[i][n] for n in names] for i in train]); y = np.array([table[i]['right'] for i in train])
        scaler = StandardScaler().fit(x); model = LogisticRegression(max_iter=1000).fit(scaler.transform(x), y)
        for i, p in zip(test, model.predict_proba(scaler.transform(np.array([[table[i][n] for n in names] for i in test])))[:, 1]):
            out[i] = float(p)
    return out


def risk_curve(scores, right):
    """Answered regions from the surest down (ties by id): the number wrong after each."""
    order = sorted(scores, key=lambda i: (-scores[i], i)); wrong = np.cumsum([0 if right[i] else 1 for i in order])
    return order, wrong


def answered_aurc(scores, right, total):
    """``aurc`` of the answered regions, as the area over the coverage of all ``total`` regions (a refusal is never answered)."""
    ids = sorted(scores)
    return aurc([scores[i] for i in ids], [0. if right[i] else 1. for i in ids])*len(ids)/total if ids else 0.


def upper_bound(wrong, n):
    from scipy.stats import beta
    return 1. if wrong >= n else float(beta.ppf(.95, wrong+1, n-wrong))


def threshold(scores, right, target, bounded):
    """The lowest score whose answers above it keep the risk (or its upper bound) within the target; None when none does."""
    order, wrong = risk_curve(scores, right); best = None
    for n in range(1, len(order)+1):
        if n < len(order) and scores[order[n]] == scores[order[n-1]]:
            continue                                                 # a threshold falls between scores
        risk = upper_bound(int(wrong[n-1]), n) if bounded else wrong[n-1]/n
        if risk <= target: best = scores[order[n-1]]
    return best


def evaluate(scores, right, folds, total):
    ids = sorted(scores)
    out = {'auroc': auroc([scores[i] for i in ids], [right[i] for i in ids]), 'aurc': answered_aurc(scores, right, total), 'answered_by_reader': len(scores)}
    order, wrong = risk_curve(scores, right)
    out['risk_at_coverage'] = {str(c): (float(wrong[int(round(c*total))-1]/int(round(c*total))) if int(round(c*total)) <= len(order) else None) for c in COVERAGES}
    for target in TARGETS:
        for bounded in (False, True):
            chosen, wrong_n = 0, 0
            for k in sorted(set(folds[i] for i in scores)):
                cal = {i: scores[i] for i in scores if folds[i] != k}; t = threshold(cal, right, target, bounded)
                if t is None: continue
                picked = [i for i in scores if folds[i] == k and scores[i] >= t]; chosen += len(picked); wrong_n += sum(not right[i] for i in picked)
            out[f'target_{target}_{"bound" if bounded else "empirical"}'] = {'coverage': chosen/total, 'risk': (wrong_n/chosen) if chosen else None, 'answered': chosen}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--pack', type=Path, required=True); p.add_argument('--arm', action='append', required=True)
    p.add_argument('--other', action='append', default=[], help='NAME=DIRS of R0 and X, the readings agreed with')
    p.add_argument('--head', type=Path); p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}; total = len(rows)
    folds = {k: r['fold'] for k, r in rows.items()}; writers = {k: r['writer'] for k, r in rows.items()}
    others = {name: readings(dirs) for name, _, dirs in (s.partition('=') for s in args.other)}
    head = {g['id']: g['score'] for g in map(json.loads, (args.head/'scores.jsonl').read_text().splitlines())} if args.head else None
    result, per_region = {}, {}
    for n, spec in enumerate(args.arm):
        name, _, dirs = spec.partition('='); read = readings(dirs)
        assert set(read) == set(rows), f'{name} did not read every region'
        table = {}
        for k, (text, confidence) in read.items():
            if refused(text): continue
            table[k] = {'confidence': confidence, 'length': math.log1p(len(prepare(text))), 'right': bool(rows[k]['readable'] and counts(rows[k]['text'], text)['exact']),
                        **{f'agrees_{o}': float(same(text, others[o][k][0])) for o in others}, **({'head': head[k]} if head else {})}
        right = {k: v['right'] for k, v in table.items()}
        kinds = [f for f in FEATURES if all(x in next(iter(table.values())) for x in FEATURES[f]) and (f != 'a+head' or n == 0)]
        scores = {f: fitted_scores(table, folds, FEATURES[f]) for f in kinds}
        result[name] = {f: evaluate(scores[f], right, folds, total) for f in kinds}
        result[name]['answered_right_share'] = sum(right.values())/max(1, len(right))
        for k in rows: per_region.setdefault(k, {})[name] = {f: scores[f].get(k) for f in kinds} | {'right': right.get(k)}
        if n == 0 and head is not None:
            rng = np.random.default_rng(SEED); names = sorted(set(writers.values())); by = {w: [k for k in rows if writers[k] == w] for w in names}
            def difference(ids):
                pick = lambda f: {f'{i}#{j}': scores[f][i] for j, i in enumerate(ids) if i in scores[f]}
                r = {f'{i}#{j}': right[i] for j, i in enumerate(ids) if i in right}
                return answered_aurc(pick('a+head'), r, len(ids))-answered_aurc(pick('a'), r, len(ids))
            full = difference(list(rows)); draws = [difference([k for w in rng.choice(names, len(names)) for k in by[w]]) for _ in range(1000)]
            interval = [float(np.percentile(draws, q)) for q in (2.5, 97.5)]
            result['B2'] = {'arm': name, 'aurc_difference': full, 'interval': interval,
                            'verdict': 'useful' if interval[1] < 0 else 'worse' if interval[0] > 0 else 'cannot_tell'}
    out = make_output_dir(args.output_dir); write_json(out/'selective.json', result)
    with (out/'scores.jsonl').open('x', encoding='utf-8') as stream:
        for k in sorted(per_region): stream.write(json.dumps({'id': k, **per_region[k]})+'\n')
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items()},
                   inputs={str(args.pack/'pack.json'): digest(args.pack/'pack.json'), **({str(args.head/'scores.jsonl'): digest(args.head/'scores.jsonl')} if args.head else {})})
    print(json.dumps({k: (v if k == 'B2' else {f: {m: v[f][m] for m in ('auroc', 'aurc')} for f in v if isinstance(v[f], dict)}) for k, v in result.items()}), flush=True)


if __name__ == '__main__':
    main()
