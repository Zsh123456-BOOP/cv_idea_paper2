"""The mask-quality head: predict the IoU a query's own delivered mask will reach, and deliver with it.

After Mask Scoring R-CNN (Huang et al., CVPR 2019).  The head reads the query, a summary of its coarse mask (pooled
detail features plus the mask's area and peak) and is trained jointly (BCE against the IoU the rendered mask reaches
with its best gold piece, recomputed every step) but detached from the trunk, so the trunk sees exactly the gradients it
saw without it.  At inference the delivered score is the geometric mean of the objectness and the predicted quality.
"""
import torch
from torch import nn

DIMENSION, CHANNELS = 160, 32
MASK_LEVEL = .5
RULES = ('product', 'product_rescaled', 'quality_only')
RULE = 'product_rescaled'


class MaskIoUHead(nn.Module):
    """Reads the query and a summary of the mask that query just drew."""

    def __init__(self, dimension=DIMENSION, channels=CHANNELS):
        super().__init__()
        self.project = nn.Linear(channels, dimension, bias=False)
        # The pooled feature is a weighted average, so it is unchanged by how much of the map the mask covers; these two
        # statistics put the extent back in.
        self.extent = nn.Linear(2, dimension, bias=False)
        self.score = nn.Sequential(nn.LayerNorm(dimension*3), nn.Linear(dimension*3, dimension), nn.ReLU(),
                                   nn.Linear(dimension, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, query, coarse, pixels, delivered=None, detach=True):
        """``delivered`` is the mask that will be reported: the pooled feature comes from the cheap quarter-resolution
        mask, but the extent has to describe the mask whose overlap is being predicted."""
        if detach:
            query, coarse, pixels = query.detach(), coarse.detach(), pixels.detach()
            delivered = None if delivered is None else delivered.detach()
        weight = coarse.sigmoid()
        total = weight.flatten(2).sum(-1).clamp(min=1e-4)
        pooled = torch.einsum('bqhw,bchw->bqc', weight, pixels)/total[..., None]
        source = weight if delivered is None else delivered.sigmoid()
        area = source.flatten(2).sum(-1)
        extent = torch.stack([area/(source.shape[-1]*source.shape[-2]), source.flatten(2).amax(-1)], -1)
        return self.score(torch.cat([query, self.project(pooled), self.extent(extent)], -1)).squeeze(-1)


def overlap_targets(masks, targets, level=MASK_LEVEL):
    """The IoU each query's delivered mask reaches against its best gold piece (zero for an empty mask or a clean row)."""
    batch, queries = masks.shape[:2]
    coarse = masks
    values = torch.zeros(batch, queries, device=coarse.device)
    binary = (coarse.detach().sigmoid() >= level)
    areas = binary.flatten(2).sum(-1).to(coarse.dtype)
    for index, truth in enumerate(targets):
        if not len(truth):
            continue
        gold = torch.stack([t.to(coarse.dtype) for t in truth]).flatten(1)
        if gold.shape[-1] != binary.shape[-2]*binary.shape[-1]:
            raise ValueError('gold and coarse mask resolutions differ')
        flat = binary[index].flatten(1).to(coarse.dtype)
        intersection = flat @ gold.t()
        union = areas[index][:, None] + gold.sum(-1)[None] - intersection
        values[index] = (intersection/union.clamp(min=1.)).max(-1).values
    return values


def delivered_score(objectness, quality, rule=RULE):
    """The score logit delivered: objectness and predicted quality combined (``product_rescaled`` = their geometric mean).

    Returns a logit, not a probability: the pipeline turns it into the probability the score thresholds are written
    against, and a probability would be squashed a second time."""
    if rule not in RULES:
        raise ValueError('undeclared scoring rule: ' + str(rule))
    predicted = quality.sigmoid()
    if rule == 'quality_only':
        combined = predicted
    else:
        combined = objectness.sigmoid()*predicted
        if rule == 'product_rescaled':
            combined = combined.clamp(min=0.).sqrt()
    return torch.logit(combined.clamp(1e-6, 1. - 1e-6))
