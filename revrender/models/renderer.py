"""The draw stage: an in-box point renderer with a context residual.

Each query's box is grown 1.5x into an ROI.  The renderer starts from the quarter-resolution coarse mask resampled onto
a 14x56 canvas and subdivides it 14x56 -> 28x112 -> 56x224; at each step it re-judges the 1,024 most uncertain points
with a small MLP (detail features + coarse value), then pastes the ROI back to full resolution (zero outside).

The context residual adds to every point logit a 32-token term: the point's detail features at three scales (1, 1/2,
1/4), its coarse value and its position inside the ROI are encoded, and (``contextual``) the point cross-attends to a
fixed 4x8 lattice of points of the same ROI.  The last layer starts at zero, so the residual starts as the identity.
``contextual=False`` is the 'nocontext' ablation: the same residual and multi-scale features without the attention.
"""
import torch
from torch import nn
from torch.nn import functional as F

from . import ops

CHANNELS = 32
RESIDUAL_SEED = 81601


class PointResidual(nn.Module):
    """32-dimensional tokens; queries attend only to 32 fixed points in their ROI."""

    def __init__(self, contextual):
        super().__init__()
        self.contextual = contextual
        self.encode = nn.Sequential(nn.Linear(99, 32), nn.ReLU())
        # Shared layers are built first, so the contextual and non-contextual variants start from identical weights.
        self.output = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
        if contextual:
            self.query = nn.Linear(32, 32, bias=False)
            self.key = nn.Linear(32, 32, bias=False)
            self.value = nn.Linear(32, 32, bias=False)
            self.norm = nn.LayerNorm(32)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, features, memory=None):
        token = self.encode(features)
        if self.contextual:
            if memory is None:
                raise ValueError('contextual points require ROI memory')
            mem = self.encode(memory)
            attention = (self.query(token) @ self.key(mem).transpose(-1, -2))/(32**.5)
            token = self.norm(token+attention.softmax(-1) @ self.value(mem))
        return self.output(token).squeeze(-1)


class RoiPointDraw(nn.Module):
    """Per-instance ROI renderer; ``alignment_residual`` is attached after the rest of the model is built."""

    def __init__(self):
        super().__init__()
        self.point_head = nn.Sequential(nn.Linear(CHANNELS+1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.contextual = True
        self._pyramid = self._roi = self._memory = None

    def attach_residual(self, contextual):
        """Add the context residual.  Its weights are drawn from their own seed and the generators are restored, so
        building it changes no other weight and no later random draw."""
        import torch as _torch
        with _torch.random.fork_rng(devices=list(range(_torch.cuda.device_count()))):
            _torch.manual_seed(RESIDUAL_SEED)
            self.alignment_residual = PointResidual(contextual)
        self.contextual = contextual

    # ------------------------------------------------------------------ points
    def point_features(self, coarse, coordinates):
        b, q, p = coordinates.shape[:3]
        features = [ops.sample_points(f, coordinates.reshape(b, q*p, 2)).transpose(1, 2).reshape(b, q, p, 32)
                    for f in self._pyramid]
        mask = ops.sample_learned(coarse.reshape(b*q, 1, *coarse.shape[-2:]),
                                  coordinates.reshape(b*q, p, 2)).reshape(b, q, p, 1)
        roi = self._roi
        relative = (coordinates-roi[:, :, None, :2])/(roi[:, :, None, 2:]-roi[:, :, None, :2]).clamp_min(1e-5)
        return torch.cat([*features, mask, relative], -1)

    def at_points(self, coarse, fine, coords):
        b, q, p = coords.shape[:3]
        feature = ops.sample_points(fine, coords.reshape(b, q*p, 2)).transpose(1, 2).reshape(b, q, p, CHANNELS)
        values = ops.sample_learned(coarse.reshape(b*q, 1, *coarse.shape[-2:]), coords.reshape(b*q, p, 2)).reshape(b, q, p)
        base = self.point_head(torch.cat((feature, values[..., None]), -1)).squeeze(-1)
        features = self.point_features(coarse, coords)
        memory = None
        if self.contextual:
            if self._memory is None:
                roi = self._roi
                lattice = ops.grid(4, 8, coarse)[None]
                lattice_coords = roi[:, :, None, :2]+lattice*(roi[:, :, None, 2:]-roi[:, :, None, :2])
                self._memory = self.point_features(coarse, lattice_coords)
            memory = self._memory
        return base+self.alignment_residual(features, memory)

    # ------------------------------------------------------------------ forward
    def forward(self, coarse, fine, boxes, size):
        self._pyramid = (fine, F.avg_pool2d(fine, 2), F.avg_pool2d(fine, 4))
        self._roi = ops.expanded_roi(boxes, size)
        self._memory = None
        try:
            return self._render(coarse, fine, boxes, size)
        finally:
            # Tensors returned by forward retain their graphs; the module must not.
            self._pyramid = self._roi = self._memory = None

    def _render(self, coarse, fine, boxes, size):
        b, q = coarse.shape[:2]
        roi = ops.expanded_roi(boxes, size)
        low, extent = roi[..., :2], (roi[..., 2:]-roi[..., :2]).clamp_min(1e-5)
        if self.training:
            with torch.no_grad():
                local = torch.rand(b, q, 768, 2, device=coarse.device)
                global_coords = low[:, :, None]+local*extent[:, :, None]
                sampled = ops.sample_learned(coarse.reshape(b*q, 1, *coarse.shape[-2:]),
                                             global_coords.reshape(b*q, 768, 2)).reshape(b, q, 768)
                top = torch.argsort(-sampled.abs(), dim=-1, descending=True, stable=True)[..., :192]
                uncertain = local.gather(2, top[..., None].expand(-1, -1, -1, 2))
                random = torch.rand(b, q, 64, 2, device=coarse.device)
                coords = low[:, :, None]+torch.cat((uncertain, random), 2)*extent[:, :, None]
            logits = self.at_points(coarse, fine, coords)
            return F.interpolate(coarse, size=size, mode='bilinear', align_corners=False), {'roi_points': (coords, logits)}
        local = ops.grid(14, 56, coarse).expand(b*q, -1, -1).reshape(b, q, -1, 2)
        coords = low[:, :, None]+local*extent[:, :, None]
        rendered = ops.sample_learned(coarse.reshape(b*q, 1, *coarse.shape[-2:]), coords.reshape(b*q, -1, 2)).reshape(b*q, 1, 14, 56)
        for h, w in ((28, 112), (56, 224)):
            rendered = F.interpolate(rendered, size=(h, w), mode='bilinear', align_corners=False)
            indices = torch.argsort(-rendered.flatten(1).abs(), dim=-1, descending=True, stable=True)[:, :1024]
            local = ops.index_coordinates(indices, h, w).reshape(b, q, -1, 2)
            coords = low[:, :, None]+local*extent[:, :, None]
            logits = self.at_points(coarse, fine, coords).reshape(b*q, 1, -1)
            rendered = rendered.flatten(2).scatter(2, indices[:, None], logits).reshape(b*q, 1, h, w)
        # Paste in small groups to limit temporary allocation; outside the ROI is background.
        full_grid = ops.grid(*size, coarse)
        masks = []
        for row in range(b):
            local = (full_grid-low[row, :, None])/extent[row, :, None]
            masks.append(ops.paste_roi_probabilities(rendered[row*q:(row+1)*q], local, size))
        return torch.stack(masks), {}
