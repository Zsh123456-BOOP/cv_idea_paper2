"""Selective reading: how the error falls when only the readings one is surest of are kept.

``risk_coverage`` sorts the regions by confidence and gives, for every coverage (the share of regions answered), the risk (the mean
loss of the answered ones); ``aurc`` is the area under that curve, lower the better, and ``risk_at`` the risk at one coverage -- the
selective-prediction scores of Geifman and El-Yaniv (SelectiveNet, ICML 2019), as Reliable VQA (ECCV 2022) reports risk and coverage
for answers a model may withhold.  ``auroc`` says how well a score ranks one kind of region above the rest (here: the regions nobody
can read, by how little any reading of them is trusted).
"""
import numpy as np


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
