"""The in-loop validation reading that picks RevRender's checkpoint and its early stop.

Validation predictions are kept as exact threshold bit-masks (7 mask thresholds x N lines x 32 queries x packed
pixels) plus the 32 scores per line.  The reading takes the best of the 7 x 5 (mask threshold, score threshold) grid by
component F1 at IoU 0.5 (ties: recall, precision, fewer false positives, then the higher thresholds) and reports the
pixel metrics at that same point.  The checkpoint is chosen by (F1@0.5, pixel IoU).  This is the tuning-time reading;
the paper's numbers come from eval.matching on the frozen point.
"""
import numpy as np

MASK_THRESHOLDS = (.2, .3, .4, .5, .6, .7, .8)
SCORE_THRESHOLDS = (.45, .55, .65, .75, .85)
MATCH_IOUS = (.25, .5, .75)


def match_pairs(ious, keep, threshold=.5):
    """IoU-ordered one-to-one matching of the kept predictions to the gold: (used predictions, used gold)."""
    edges = sorted([(float(ious[p, g]), int(p), int(g)) for p in keep
                    for g in range(ious.shape[1]) if ious[p, g] >= threshold],
                   key=lambda value: (-value[0], value[1], value[2]))
    used_p, used_g = set(), set()
    for _, p, g in edges:
        if p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
    return used_p, used_g


def metric_from_ious(iou_rows, area_rows, scores, threshold, match_iou, positive_count):
    tp = fp = fn = clean_fp = 0
    for row, (ious, areas, row_scores) in enumerate(zip(iou_rows, area_rows, scores)):
        keep = np.flatnonzero((row_scores >= threshold) & (areas > 0))
        pairs = [(float(ious[p, g]), int(p), int(g)) for p in keep for g in range(ious.shape[1]) if ious[p, g] >= match_iou]
        pairs.sort(key=lambda value: (-value[0], value[1], value[2]))
        used_p, used_g = set(), set()
        for _, p, g in pairs:
            if p not in used_p and g not in used_g:
                used_p.add(p)
                used_g.add(g)
        tp += len(used_p)
        fp += len(keep) - len(used_p)
        fn += ious.shape[1] - len(used_g)
        if row >= positive_count:
            clean_fp += len(keep)
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return {'precision': precision, 'recall': recall, 'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'clean_false_positive': clean_fp,
            'positive_false_positive': fp - clean_fp, 'fp_components_per_line': fp/len(scores)}


def evaluate(masks, scores, truths, positive_count, packed=False):
    """Best of the grid by F1@0.5, the pixel metrics at that point, and the whole grid.

    ``packed``: ``masks`` is 7 x N x Q x ceil(H*W/8) uint8 exact threshold bitmasks.
    """
    best, grid = None, []
    pixels = int(np.prod(truths[0].shape[-2:]))
    for m_index, mask_threshold in enumerate(MASK_THRESHOLDS):
        iou_rows, area_rows = [], []
        for row, truth in enumerate(truths):
            binary = (np.unpackbits(masks[m_index, row], axis=-1, count=pixels).astype(bool) if packed
                      else (masks[row] >= mask_threshold).reshape(len(scores[row]), pixels))
            gold = truth.reshape(len(truth), pixels)
            areas = binary.sum(1)
            intersections = (np.stack([binary[:, target].sum(1) for target in gold], 1) if len(gold)
                             else np.zeros((len(binary), 0)))
            denominator = areas[:, None] + gold.sum(1)[None] - intersections
            iou_rows.append(intersections/np.maximum(denominator, 1))
            area_rows.append(areas)
        for score_threshold in SCORE_THRESHOLDS:
            metrics = metric_from_ious(iou_rows, area_rows, scores, score_threshold, .5, positive_count)
            grid.append({'mask_threshold': mask_threshold, 'score_threshold': score_threshold, **metrics})
            key = (metrics['f1'], metrics['recall'], metrics['precision'], -metrics['false_positive'],
                   mask_threshold, score_threshold)
            if best is None or key > best[0]:
                readings = {f'{level:.2f}': metric_from_ious(iou_rows, area_rows, scores, score_threshold, level,
                                                              positive_count) for level in MATCH_IOUS}
                best = key, m_index, mask_threshold, score_threshold, readings
    _, m_index, mask_threshold, score_threshold, readings = best
    intersection = predicted_area = gold_area = 0
    for row, truth in enumerate(truths):
        binary = (np.unpackbits(masks[m_index, row], axis=-1, count=pixels).astype(bool) if packed
                  else (masks[row] >= mask_threshold).reshape(len(scores[row]), pixels))
        pred = binary[scores[row] >= score_threshold].any(0)
        gold = truth.reshape(len(truth), pixels).any(0)
        intersection += int((pred & gold).sum())
        predicted_area += int(pred.sum())
        gold_area += int(gold.sum())
    return {'instance': readings, 'mask_threshold': mask_threshold, 'score_threshold': score_threshold,
            'pixel_iou': intersection/max(1, predicted_area + gold_area - intersection),
            'pixel_precision': intersection/max(1, predicted_area),
            'pixel_recall': intersection/max(1, gold_area),
            'pixel_metric_scope': 'same instance operating point, union of retained independent masks',
            'grid': grid, 'gold_instances': sum(len(row) for row in truths)}
