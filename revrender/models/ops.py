"""Spatial operators shared by the detector, the renderer and the losses.

Coordinates are normalised (x, y) in [0, 1], pixel-centre convention.  Boxes are (cx, cy, w, h) unless a function says
xyxy.  ``sample_points`` is looked up through this module at call time: training replaces it with an operator whose
backward is deterministic (revrender.determinism); ``sample_learned`` differentiates coordinates as well and is
deterministic on its own (four explicit gathers, no atomics).
"""
import torch
from torch.nn import functional as F

_CONSTANTS = {}


def constant(like, values):
    """A small immutable device constant, built once per (device, dtype, values)."""
    key = like.device, like.dtype, tuple(values)
    if key not in _CONSTANTS:
        _CONSTANTS[key] = like.new_tensor(values)
    return _CONSTANTS[key]


def sample_points(features, coordinates, mode='bilinear'):
    """B,C,H,W and B,P,2 normalised x,y pixel-centre coordinates -> B,C,P (fixed coordinates, border padding)."""
    return F.grid_sample(features, coordinates[:, None]*2 - 1, mode=mode, padding_mode='border', align_corners=False)[:, :, 0]


def index_coordinates(indices, height, width):
    return torch.stack(((indices % width + .5)/width, (indices//width + .5)/height), -1)


def grid(h, w, like):
    return index_coordinates(torch.arange(h*w, device=like.device)[None], h, w).to(like.dtype)


def sample_learned(features, coordinates):
    """Bilinear, zero padded, align_corners=False; deterministic gather adjoint.

    Unlike ``sample_points`` this differentiates the coordinates as well as the values; four explicit gathers avoid the
    native CUDA grid_sample backward atomics.
    """
    b, c, h, w = features.shape
    if coordinates.ndim != 3 or coordinates.shape[0] != b or coordinates.shape[-1] != 2:
        raise ValueError('expected matching NCHW and NP2')
    if not torch.isfinite(coordinates).all():
        raise ValueError('nonfinite coordinates')
    xy = ((coordinates*2 - 1 + 1)*constant(coordinates, [w, h]) - 1)/2
    x, y = xy.unbind(-1)
    x0, y0 = x.floor(), y.floor()
    dx, dy = x - x0, y - y0
    flat = features.flatten(2)
    output = 0
    for xx, yy, weight in ((x0, y0, (1-dx)*(1-dy)), (x0+1, y0, dx*(1-dy)),
                           (x0, y0+1, (1-dx)*dy), (x0+1, y0+1, dx*dy)):
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        index = yy.long().clamp(0, h-1)*w + xx.long().clamp(0, w-1)
        output = output + flat.gather(2, index[:, None].expand(-1, c, -1))*(weight*valid)[:, None]
    return output


def xyxy(boxes):
    return torch.cat((boxes[..., :2]-boxes[..., 2:]/2, boxes[..., :2]+boxes[..., 2:]/2), -1)


def giou(a, b):
    lower = torch.maximum(a[:, None, :2], b[None, :, :2])
    upper = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (upper-lower).clamp_min(0).prod(-1)
    union = (a[:, 2:]-a[:, :2]).clamp_min(0).prod(-1)[:, None] + (b[:, 2:]-b[:, :2]).clamp_min(0).prod(-1)[None] - inter
    enclosing = (torch.maximum(a[:, None, 2:], b[None, :, 2:])-torch.minimum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    return inter/union.clamp_min(1e-7) - (enclosing-union)/enclosing.clamp_min(1e-7)


def mask_boxes(masks):
    """cxcywh from nonempty binary masks; an empty mask gets the full image."""
    h, w = masks.shape[-2:]
    xs, ys = masks.bool().any(-2), masks.bool().any(-1)
    x, y = torch.arange(w, device=masks.device), torch.arange(h, device=masks.device)
    left = x.expand_as(xs).masked_fill(~xs, w).amin(-1)/w
    right = (x.expand_as(xs).masked_fill(~xs, -1).amax(-1)+1)/w
    top = y.expand_as(ys).masked_fill(~ys, h).amin(-1)/h
    bottom = (y.expand_as(ys).masked_fill(~ys, -1).amax(-1)+1)/h
    boxes = torch.stack(((left+right)/2, (top+bottom)/2, right-left, bottom-top), -1)
    return torch.where(xs.any(-1)[..., None], boxes, constant(boxes, [.5, .5, 1., 1.]))


def expanded_roi(boxes, size):
    """The ROI of each box: grown 1.5x (at least 4 px), clipped to the image; returns (x0, y0, x1, y1)."""
    boxes = boxes.detach()
    minimum = constant(boxes, [4/size[1], 4/size[0]])
    extent = torch.maximum(boxes[..., 2:]*1.5, minimum)
    low = (boxes[..., :2]-extent/2).clamp(0, 1)
    high = (boxes[..., :2]+extent/2).clamp(0, 1)
    return torch.cat((low, high), -1)


def paste_roi_probabilities(logits, local, size):
    """As in native mask pasting: interpolate probabilities, zero outside the ROI."""
    probability = sample_learned(logits.sigmoid(), local).reshape(len(local), *size)
    inside = ((local >= 0) & (local <= 1)).all(-1).reshape(len(local), *size)
    return torch.logit(probability.clamp(1e-6, 1-1e-6)).masked_fill(~inside, -20.)
