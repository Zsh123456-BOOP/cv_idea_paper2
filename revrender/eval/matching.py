"""Component matching and F1: greedy one-to-one at IoU >= 0.5, mask NMS, score cuts, the precision/recall frontier.

The metric every model is read by.  A model delivers, per line, candidate masks with scores.  Mask NMS drops a
candidate that overlaps a higher-scored kept one by more than ``level`` (1.0 = off); a score cut keeps the candidates
at or above it; the kept masks are matched to the gold components by IoU-ordered greedy one-to-one matching at 0.5.
Everything is computed once per (line, NMS level) as a prefix table over the score-ordered candidates, so every score
threshold is a lookup, and the whole two-knob frontier is cheap.

  provider(line_id) -> (list of bool masks, float64 scores)     see packed_provider / native_provider
  frontier(provide, order, truths, gold)                         the non-dominated (NMS, cut) readings
  frontier max-F1 point (ties: fewer false positives)            what a model's operating point is chosen as
  true_positives_at(points, budget)                              the best reading within a false-positive budget
"""
from pathlib import Path

import numpy as np

from ..util import require

MATCH_IOU = .50
NMS_LEVELS = (1., .9, .7, .5, .3)             # 1.0 = suppression off: no pair can exceed an IoU of one
FALSE_POSITIVE_BUDGETS = (41, 55, 83, 111)    # what the baselines spend (MaskDINO 41, PointRend 111, ours 55 / 83)


# ---------------------------------------------------------------- providers (what a model archived)

def native_provider(directory):
    """Per line id, the archived instance masks and scores: one ``*.npz`` per line (baselines, component methods).

    The archive holds sample_id, shape (n, H, W), masks_packed (np.packbits per instance) and scores.
    """
    directory = Path(directory)
    index = {}
    for path in sorted(directory.glob('*.npz')):
        line = str(np.load(path, allow_pickle=True)['sample_id']).split(':', 1)[1]
        require(line not in index, f'two archives for {line}')
        index[line] = path

    def provide(line):
        require(line in index, f'no archive for {line}')
        payload = np.load(index[line], allow_pickle=True)
        shape = tuple(int(v) for v in payload['shape'])
        masks = np.unpackbits(payload['masks_packed'], axis=1)
        masks = masks[:, :shape[1]*shape[2]].reshape(*shape).astype(bool)
        return [masks[i] for i in range(shape[0])], np.asarray(payload['scores'], np.float64)

    return provide, set(index)


def packed_provider(path, mask_threshold):
    """The same, from RevRender's packed dev predictions (one npz, 7 mask thresholds, 32 queries per line)."""
    with np.load(path, allow_pickle=False) as archive:
        thresholds = archive['mask_thresholds'].tolist()
        matches = [i for i, t in enumerate(thresholds) if abs(t - mask_threshold) < 1e-9]
        require(len(matches) == 1, f'{path}: mask {mask_threshold} not stored once')
        bits = np.array(archive['packed_masks'][matches[0]])
        scores = np.asarray(archive['scores'], np.float64)
        shape = tuple(int(v) for v in archive['mask_shape'])
        rows = {str(s).split(':', 1)[1]: i for i, s in enumerate(archive['sample_ids'].tolist())}
    height, width = shape[-2], shape[-1]

    def provide(line):
        require(line in rows, f'no packed row for {line}')
        row = rows[line]
        masks = np.unpackbits(bits[row], axis=-1, count=height*width).reshape(-1, height, width).astype(bool)
        return [masks[i] for i in range(len(masks))], scores[row]

    return provide, set(rows)


def provider_for(archive, model, mask_threshold):
    """RevRender's archive (a file) or a native-format archive (a directory), as a freeze record names them."""
    if model == 'point_context':
        return packed_provider(Path(archive), float(mask_threshold))
    return native_provider(Path(archive))


# ---------------------------------------------------------------- prefix tables

def prefix_outcomes(masks, scores, truth):
    """Matched and kept counts after admitting each score-ordered prefix, and the scores in that order.

    The IoU-ordered matching is redone from scratch for every prefix: admitting a lower-scored candidate can move an
    earlier match, so extending the previous prefix would not equal matching the filtered list.
    """
    order = sorted(range(len(masks)), key=lambda i: (-float(scores[i]), i))
    areas = [int(masks[i].sum()) for i in order]
    pairs_by_rank = []
    for rank, index in enumerate(order):
        row = []
        if areas[rank]:
            for t_index, t_mask in enumerate(truth):
                intersection = int(np.logical_and(masks[index], t_mask).sum())
                if not intersection:
                    continue
                union = areas[rank] + int(t_mask.sum()) - intersection
                iou = intersection/union
                if iou >= MATCH_IOU:
                    row.append((iou, rank, t_index))
        pairs_by_rank.append(row)
    matched = np.zeros(len(masks)+1, np.int64)
    kept = np.zeros(len(masks)+1, np.int64)
    pairs = []
    running_kept = 0
    for rank in range(len(masks)):
        pairs.extend(pairs_by_rank[rank])
        running_kept += 1 if areas[rank] else 0
        ordered = sorted(pairs, key=lambda item: (-item[0], item[1], item[2]))
        used_p, used_t = set(), set()
        for _iou, p_index, t_index in ordered:
            if p_index in used_p or t_index in used_t:
                continue
            used_p.add(p_index)
            used_t.add(t_index)
        matched[rank+1] = len(used_t)
        kept[rank+1] = running_kept
    return matched, kept, np.array([float(scores[i]) for i in order], np.float64)


def suppressed(masks, scores, level):
    """Indices surviving mask NMS at this level, in descending score order (empty masks stay and suppress nothing)."""
    order = sorted(range(len(masks)), key=lambda i: (-float(scores[i]), i))
    if level >= 1.:
        return order
    kept = []
    areas = {i: int(masks[i].sum()) for i in order}
    for index in order:
        if areas[index]:
            duplicate = False
            for other in kept:
                if not areas[other]:
                    continue
                intersection = int(np.logical_and(masks[index], masks[other]).sum())
                if not intersection:
                    continue
                union = areas[index] + areas[other] - intersection
                if intersection/union > level:
                    duplicate = True
                    break
            if duplicate:
                continue
        kept.append(index)
    return kept


def tabulate_at(provide, order, truths, level):
    """One prefix table per line after NMS at this level, and every distinct score the model produced."""
    tables, cuts = [], set()
    for line, truth in zip(order, truths):
        masks, scores = provide(line)
        if truth and masks and masks[0].shape != truth[0].shape:
            raise ValueError(f'{line}: candidate {masks[0].shape} against gold {truth[0].shape}')
        keep = suppressed(masks, scores, level)
        table = prefix_outcomes([masks[i] for i in keep], np.array([float(scores[i]) for i in keep]), truth)
        tables.append(table)
        cuts.update(float(v) for v in table[2])
    return tables, sorted(cuts)


# ---------------------------------------------------------------- readings

def sweep_every_cut(tables, gold_total):
    """Every distinct score cut in one pass: lowering the cut past a candidate extends exactly one line's prefix."""
    events = []
    for row, (matched, kept, ordered) in enumerate(tables):
        for rank, score in enumerate(ordered):
            events.append((-float(score), row, rank))
    events.sort()
    readings, true_positive, kept_total, index = [], 0, 0, 0
    while index < len(events):
        cut = -events[index][0]
        while index < len(events) and events[index][0] == -cut:
            _, row, rank = events[index]
            matched, kept, _ = tables[row]
            true_positive += int(matched[rank+1]) - int(matched[rank])
            kept_total += int(kept[rank+1]) - int(kept[rank])
            index += 1
        false_positive = kept_total - true_positive
        precision = true_positive/(true_positive+false_positive) if true_positive+false_positive else 0.
        recall = true_positive/gold_total if gold_total else 0.
        readings.append({'score_threshold': cut, 'precision': precision, 'recall': recall,
                         'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.,
                         'true_positive': true_positive, 'false_positive': false_positive,
                         'false_negative': gold_total - true_positive})
    return readings


def totals_at(tables, cut):
    true_positive = kept_total = 0
    for matched, kept, ordered in tables:
        admitted = int(np.searchsorted(-ordered, -cut, side='right'))
        true_positive += int(matched[admitted])
        kept_total += int(kept[admitted])
    return true_positive, kept_total - true_positive


def reading(tables, cut, gold_total):
    true_positive, false_positive = totals_at(tables, cut)
    precision = true_positive/(true_positive+false_positive) if true_positive+false_positive else 0.
    recall = true_positive/gold_total if gold_total else 0.
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
    return {'score_threshold': cut, 'f1': f1, 'precision': precision, 'recall': recall,
            'true_positive': true_positive, 'false_positive': false_positive, 'false_negative': gold_total - true_positive}


def best_over(tables, cuts, gold_total):
    best = None
    for cut in sorted(set(float(c) for c in cuts)):
        candidate = reading(tables, cut, gold_total)
        if best is None or (candidate['f1'], -candidate['score_threshold']) > (best['f1'], -best['score_threshold']):
            best = candidate
    require(best is not None, 'no score threshold to read')
    return best


def frontier(provide, order, truths, gold_total):
    """Every (NMS, cut) reading, reduced to the precision/recall frontier (a dominated point is dropped)."""
    points = []
    for level in NMS_LEVELS:
        tables, _ = tabulate_at(provide, order, truths, level)
        for reading_ in sweep_every_cut(tables, gold_total):
            reading_['nms'] = level
            points.append(reading_)
    points.sort(key=lambda p: (-p['precision'], -p['recall']))
    kept, best_recall = [], -1.
    for point in points:
        if point['recall'] > best_recall:
            kept.append(point)
            best_recall = point['recall']
    return kept


def best_point(points):
    """The operating point: best F1, ties to fewer false positives."""
    return dict(max(points, key=lambda p: (p['f1'], -p['false_positive'])))


def recall_at_precision(points, floor):
    reachable = [p for p in points if p['precision'] >= floor]
    if not reachable:
        return None
    return max(reachable, key=lambda p: (p['recall'], p['precision']))


def true_positives_at(points, budget):
    """The reading with the most true positives that spends at most ``budget`` false positives."""
    affordable = [p for p in points if p['false_positive'] <= budget]
    if not affordable:
        return None
    return max(affordable, key=lambda p: (p['true_positive'], -p['false_positive']))


def pixel_iou(provide, order, truths, point):
    """Pixels of the kept union against pixels of the gold union, at one stated point."""
    intersection = predicted = gold_pixels = 0
    for line, truth in zip(order, truths):
        masks, scores = provide(line)
        keep = suppressed(masks, scores, point['nms'])
        union = None
        for index in keep:
            if float(scores[index]) < point['score_threshold']:
                continue
            union = masks[index] if union is None else np.logical_or(union, masks[index])
        if union is None and not truth:
            continue
        shape = truth[0].shape if truth else union.shape
        union = np.zeros(shape, bool) if union is None else union
        gold = np.zeros(shape, bool)
        for component in truth:
            gold = np.logical_or(gold, component)
        intersection += int(np.logical_and(union, gold).sum())
        predicted += int(union.sum())
        gold_pixels += int(gold.sum())
    denominator = predicted + gold_pixels - intersection
    return {'pixel_iou': intersection/denominator if denominator else 0.,
            'pixel_precision': intersection/predicted if predicted else 0.,
            'pixel_recall': intersection/gold_pixels if gold_pixels else 0.}


def sweep(provide, order, truths, gold_total):
    """Best reading per NMS level and the best over the whole grid."""
    per_level, best = {}, None
    for level in NMS_LEVELS:
        tables, cuts = tabulate_at(provide, order, truths, level)
        reading_ = best_over(tables, cuts, gold_total)
        reading_['nms'] = level
        per_level[f'{level:.2f}'] = reading_
        if best is None or (reading_['f1'], reading_['nms']) > (best['f1'], best['nms']):
            best = dict(reading_)
    return {'best': best, 'per_nms_level': per_level, 'suppression_bought': best['f1'] - per_level['1.00']['f1']}


def validation_order(dev):
    """Validation rows as every reading takes them: revised lines first (stable), ids without the bundle prefix."""
    dev = sorted(dev, key=lambda r: not r['is_positive'])
    return dev, [r['sample_id'].split(':', 1)[1] for r in dev], [list(r['instances']) for r in dev]
