"""RevRender's training loss:  L = L_det + L_pix + L_pt + L_q.

  L_det  every decoder stage (the encoder selection and the four layers) on its own Hungarian assignment:
         focal class x4, mask BCE x5 + Dice x5, box L1 x5 + GIoU x2, each normalised by the batch's gold count;
         plus the denoising groups' stage losses (each divided by the number of groups)
  L_pix  per-pixel Dice + BCE of the detail map against the union of the gold revisions (zero in 'nodense')
  L_pt   BCE of the renderer's ROI points against the gold at their coordinates, on the final stage's assignment
  L_q    BCE of the quality head against the IoU each rendered mask reaches (detached from the trunk), weight 1

The Hungarian cost (focal x4, BCE x5, Dice x5, L1 x5, -GIoU x2 on 4,096 random points per row) is computed on the
device for every row and moved to the host once per stage; the assignment is the SciPy solver's, so the order of draws
(stage, then row) is the recipe.  Terms are returned as detached tensors; ``materialize`` turns them into floats.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from . import ops
from .quality import overlap_targets

QUERIES = 32
CLASS_WEIGHT = 4.
MASK_IOU_WEIGHT = 1.


def materialize(pieces):
    """Detached scalar terms copied to the host together, as Python floats."""
    names = [name for name, value in pieces.items() if torch.is_tensor(value)]
    values = torch.stack([pieces[name].detach().reshape(()) for name in names]).cpu().tolist() if names else []
    return {name: (values[names.index(name)] if name in names else value) for name, value in pieces.items()}


def dice_bce(selected, gold):
    """Binary cross-entropy plus soft Dice, each averaged over the matched masks."""
    probability = selected.sigmoid()
    dimensions = tuple(range(1, selected.ndim))
    numerator = 2.0*(probability*gold).sum(dimensions) + 1.0
    denominator = probability.sum(dimensions) + gold.sum(dimensions) + 1.0
    region = (1.0 - numerator/denominator).mean()
    return F.binary_cross_entropy_with_logits(selected, gold) + region


def matching_cost(stage, row, truth, cache):
    """The hybrid matching cost of one row's queries against its gold components (None for a clean row)."""
    masks, scores, boxes = stage
    if not truth:
        return None
    with torch.no_grad():
        key = id(truth)
        if key not in cache:
            gold = torch.stack(truth)
            cache[key] = gold, ops.mask_boxes(gold)
        gold, gt_boxes = cache[key]
        coordinates = torch.rand(1, 4096, 2, device=masks.device)
        # Native forward on freshly generated, finite coordinates; no adjoint is needed inside no_grad.
        target = F.grid_sample(gold[None], coordinates[:, None]*2-1, mode='bilinear', padding_mode='border',
                               align_corners=False)[:, :, 0][0]
        logits = F.grid_sample(masks[row:row+1], coordinates[:, None]*2-1, mode='bilinear', padding_mode='border',
                               align_corners=False)[:, :, 0][0]
        probability = logits.sigmoid()
        bce = F.softplus(logits).mean(1)[:, None] - logits@target.T/logits.shape[1]
        dice = 1 - (2*probability@target.T+1)/(probability.sum(1)[:, None]+target.sum(1)[None]+1)
        p = scores[row].sigmoid()
        focal = .25*(1-p).square()*(-(p+1e-8).log()) - .75*p.square()*(-(1-p+1e-8).log())
        return (CLASS_WEIGHT*focal[:, None] + 5*bce + 5*dice + 5*torch.cdist(boxes[row], gt_boxes, p=1)
                - 2*ops.giou(ops.xyxy(boxes[row]), ops.xyxy(gt_boxes)))


def assign(stage, targets, cache):
    """Hungarian assignment of every row of one stage: [(query indices, gold indices)] as host arrays."""
    costs = [matching_cost(stage, row, truth, cache) for row, truth in enumerate(targets)]
    nonempty = [cost for cost in costs if cost is not None]
    if not nonempty:
        return [([], []) for _ in targets]
    flat = torch.cat([cost.flatten() for cost in nonempty]).cpu().numpy()
    pairs, offset = [], 0
    for cost in costs:
        if cost is None:
            pairs.append(([], []))
        else:
            end = offset + cost.numel()
            pairs.append(linear_sum_assignment(flat[offset:end].reshape(cost.shape)))
            offset = end
    return pairs


def stage_term(stage, targets, pairs):
    """One stage's loss under a given assignment."""
    masks, objects, boxes = stage
    pairs = [(torch.as_tensor(q, device=masks.device, dtype=torch.long), g) for q, g in pairs]
    labels = torch.zeros_like(objects)
    count = max(1, sum(len(t) for t in targets))
    mask_term, box_term = masks.sum()*0, boxes.sum()*0
    for row, truth in enumerate(targets):
        q, g = pairs[row]
        if not len(g):
            continue
        labels[row, q] = 1
        gold = torch.stack([truth[int(i)] for i in g])
        logits = F.interpolate(masks[row, q, None], size=gold.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
        probability = logits.sigmoid().flatten(1)
        target = gold.flatten(1)
        bce = F.binary_cross_entropy_with_logits(logits, gold, reduction='none').flatten(1).mean(1)
        dice = 1 - (2*(probability*target).sum(1)+1)/(probability.sum(1)+target.sum(1)+1)
        mask_term = mask_term + (5*bce+5*dice).sum()/count
        gold_boxes = ops.mask_boxes(gold)
        box_term = box_term + (5*(boxes[row, q]-gold_boxes).abs().sum(1)
                               + 2*(1-ops.giou(ops.xyxy(boxes[row, q]), ops.xyxy(gold_boxes)).diag())).sum()/count
    p = objects.sigmoid()
    pt = p*labels + (1-p)*(1-labels)
    alpha = .25*labels + .75*(1-labels)
    classification = 4*(alpha*(1-pt).square()*F.binary_cross_entropy_with_logits(objects, labels, reduction='none')).sum()/count
    return mask_term + box_term + classification


def compute_loss(model, rendered, targets):
    """(total, {term: detached tensor}) of one step.  ``rendered`` is the model's first output, ``targets`` the gold
    components of every line as lists of float masks on the device."""
    if any(len(t) > QUERIES for t in targets):
        raise ValueError('targets exceed query budget')
    cache, stored = {}, model.cache
    masks = rendered
    total = masks.sum()*0
    pieces = {}
    pairs = None
    for index, stage in enumerate(stored['stages']):
        pairs = assign(stage, targets, cache)
        term = stage_term(stage, targets, pairs)
        total = total + term
        pieces['stage_'+str(index)] = term.detach()
    dn = stored['dn']
    if dn is not None:
        dn_loss = masks.sum()*0
        for stage in stored['dn_stages']:
            dn_loss = dn_loss + stage_term(stage, targets, dn[3])/dn[4]
        total = total + dn_loss
        pieces['denoising'] = dn_loss.detach()
    union = torch.stack([torch.stack(t).amax(0) if t else torch.zeros_like(masks[0, 0]) for t in targets])[:, None]
    dense = dice_bce(stored['dense'], union) if model.dense_supervision else stored['dense'].sum()*0
    total = total + dense
    pieces['dense'] = dense.detach()
    coords, logits = stored['roi_points']
    terms = []
    for row, (q, g) in enumerate(pairs):
        if len(g):
            truth = torch.stack([targets[row][int(i)] for i in g])[:, None]
            gold = ops.sample_points(truth, coords[row, q])[:, 0]
            terms.append(F.binary_cross_entropy_with_logits(logits[row, q], gold, reduction='none').mean(1))
    point = torch.cat(terms).mean() if terms else logits.sum()*0
    total = total + point
    pieces['roi_point'] = point.detach()
    truth = overlap_targets(stored['mask_iou_delivered'], targets)
    overlap = F.binary_cross_entropy_with_logits(stored['mask_iou'], truth)
    pieces['mask_iou'] = overlap.detach()
    pieces['mask_iou_target_mean'] = truth.mean().detach()
    return total + MASK_IOU_WEIGHT*overlap, pieces


class Clipper:
    """Gradient clipping per declared group (parameter-name prefix), everything else together, each at ``limit``.

    One global norm would couple an added head loss to the trunk through the clip coefficient even when the head is
    detached; clipping the quality head on its own keeps the trunk's update a function of the trunk's gradients."""

    def __init__(self, model, prefixes, limit=5.):
        named = list(model.named_parameters())
        claimed, self.groups = set(), []
        self.limit = limit
        for prefix in prefixes:
            selected = [p for name, p in named if name.startswith(prefix) and name not in claimed]
            if not selected:
                raise ValueError('no parameter matches clip group ' + prefix)
            claimed.update(name for name, _ in named if name.startswith(prefix))
            self.groups.append(selected)
        self.groups.append([p for name, p in named if name not in claimed])

    def __call__(self):
        for group in self.groups:
            if group:
                torch.nn.utils.clip_grad_norm_(group, self.limit, foreach=False, error_if_nonfinite=True)
