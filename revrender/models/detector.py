"""The find stage: a two-stage query detector with deformable decoding and denoising groups.

The 4 feature levels (1/4 ... 1/32) are scored per position; the 32 best positions become the queries, with their
anchor boxes.  Four decoder layers (self-attention, a 4-head multi-scale deformable read, FFN) refine the queries and
boxes.  Every query predicts an objectness logit, a box and a quarter-resolution coarse mask (a 32-channel embedding
dotted with the detail map pooled to a quarter).  In training, up to four groups of noisy gold boxes enter as extra
denoising queries that cannot be seen by the delivered ones.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from . import ops

DIMENSION, CHANNELS, QUERIES, LAYERS = 160, 32, 32, 4


class DeformableRead(nn.Module):
    def __init__(self, dim=DIMENSION, heads=4, levels=4, points=4):
        super().__init__()
        self.heads, self.levels, self.points = heads, levels, points
        self.offsets = nn.Linear(dim, heads*levels*points*2)
        self.weights = nn.Linear(dim, heads*levels*points)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        nn.init.zeros_(self.offsets.weight)
        angles = torch.arange(heads)*2*math.pi/heads
        direction = torch.stack((angles.cos(), angles.sin()), -1)
        direction /= direction.abs().amax(-1, keepdim=True)
        bias = direction[:, None, None]*torch.arange(1, points+1)[None, None, :, None]
        with torch.no_grad():
            self.offsets.bias.copy_(bias.expand(heads, levels, points, 2).reshape(-1))
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)

    def forward(self, query, features, reference):
        b, q, d = query.shape
        offsets = self.offsets(query).reshape(b, q, self.heads, self.levels, self.points, 2)
        locations = reference[:, :, None, None, None, :2] + offsets/self.points*reference[:, :, None, None, None, 2:]*.5
        weights = self.weights(query).reshape(b, q, self.heads, self.levels*self.points).softmax(-1)
        weights = weights.reshape(b, q, self.heads, self.levels, self.points)
        result = 0
        for level, feature in enumerate(features):
            h, w = feature.shape[-2:]
            values = self.value(feature.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b*self.heads, d//self.heads, h, w)
            coordinates = locations[:, :, :, level].permute(0, 2, 1, 3, 4).reshape(b*self.heads, q*self.points, 2)
            sampled = ops.sample_learned(values, coordinates).reshape(b, self.heads, d//self.heads, q, self.points).permute(0, 3, 1, 4, 2)
            result = result + (sampled*weights[:, :, :, level, :, None]).sum(3)
        return self.output(result.reshape(b, q, d))


class RegionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = nn.Sequential(nn.Linear(4, DIMENSION), nn.ReLU(), nn.Linear(DIMENSION, DIMENSION))
        self.self_attention = nn.MultiheadAttention(DIMENSION, 4, dropout=0., batch_first=True)
        self.cross_attention = DeformableRead()
        self.norms = nn.ModuleList([nn.LayerNorm(DIMENSION) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(DIMENSION, 640), nn.ReLU(), nn.Linear(640, DIMENSION))

    def forward(self, query, features, reference, attention_mask):
        position = self.position(torch.logit(reference.clamp(.001, .999)))
        value = self.self_attention(query+position, query+position, query, attn_mask=attention_mask, need_weights=False)[0]
        query = self.norms[0](query+value)
        query = self.norms[1](query+self.cross_attention(query+position, features, reference))
        return self.norms[2](query+self.ffn(query))


class RegionFinder(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_count = QUERIES
        self.enc_output = nn.Sequential(nn.Linear(DIMENSION, DIMENSION), nn.LayerNorm(DIMENSION))
        self.level_embed = nn.Parameter(torch.randn(4, DIMENSION)*.02)
        self.output_norm = nn.LayerNorm(DIMENSION)
        self.class_embed = nn.Linear(DIMENSION, 1)
        self.mask_embed = nn.Sequential(nn.Linear(DIMENSION, DIMENSION), nn.ReLU(), nn.Linear(DIMENSION, DIMENSION), nn.ReLU(),
                                        nn.Linear(DIMENSION, CHANNELS))
        self.box_embed = nn.Sequential(nn.Linear(DIMENSION, DIMENSION), nn.ReLU(), nn.Linear(DIMENSION, DIMENSION), nn.ReLU(),
                                       nn.Linear(DIMENSION, 4))
        nn.init.zeros_(self.box_embed[-1].weight)
        nn.init.zeros_(self.box_embed[-1].bias)
        nn.init.constant_(self.class_embed.bias, math.log(.01/.99))
        self.label_embedding = nn.Embedding(1, DIMENSION)
        self.layers = nn.ModuleList([RegionLayer() for _ in range(LAYERS)])

    def prediction(self, query, pixels, boxes):
        value = self.output_norm(query)
        return (torch.einsum('bqc,bchw->bqhw', self.mask_embed(value), pixels), self.class_embed(value).squeeze(-1), boxes)

    def denoising(self, targets, batch, like):
        maximum = max((len(t) for t in targets), default=0)
        groups = min(4, QUERIES//maximum) if maximum else 0
        count = maximum*groups
        if not count:
            return None
        boxes = like.new_full((batch, count, 4), .5)
        pairs = []
        for row, truth in enumerate(targets):
            n = len(truth)
            if n:
                boxes[row, torch.cat([torch.arange(n, device=like.device)+group*maximum for group in range(groups)])] = \
                    ops.mask_boxes(torch.stack(truth)).repeat(groups, 1)
            pairs.append((torch.cat([torch.arange(n, device=like.device)+g*maximum for g in range(groups)]),
                          torch.arange(n, device=like.device).repeat(groups)))
        extent = torch.cat((boxes[..., 2:]/2, boxes[..., 2:]), -1)
        noisy = (boxes+(torch.rand_like(boxes)*2-1)*extent*.4).clamp(.001, .999)
        content = self.label_embedding.weight[0][None, None].expand(batch, count, -1)
        attention = torch.zeros(count+QUERIES, count+QUERIES, dtype=torch.bool, device=like.device)
        attention[count:, :count] = True
        for group in range(groups):
            attention[group*maximum:(group+1)*maximum, :count] = True
            attention[group*maximum:(group+1)*maximum, group*maximum:(group+1)*maximum] = False
        return content, noisy, attention, pairs, groups

    def forward(self, levels, fine, targets=None):
        b = fine.shape[0]
        pixels = F.adaptive_avg_pool2d(fine, levels[0].shape[-2:])
        memories, anchors = [], []
        for level, feature in enumerate(levels):
            h, w = feature.shape[-2:]
            memories.append(feature.flatten(2).transpose(1, 2)+self.level_embed[level])
            centre = ops.grid(h, w, feature).expand(b, -1, -1)
            anchors.append(torch.cat((centre, torch.full_like(centre, .05*2**level)), -1))
        memory = self.enc_output(torch.cat(memories, 1))
        anchors = torch.cat(anchors, 1)
        boxes = (torch.logit(anchors.clamp(.001, .999))+self.box_embed(memory)).sigmoid()
        scores = self.class_embed(memory).squeeze(-1)
        indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :QUERIES]
        content = memory.gather(1, indices[..., None].expand(-1, -1, DIMENSION))
        reference = boxes.gather(1, indices[..., None].expand(-1, -1, 4))
        # The selected encoder predictions receive their own hybrid supervision.
        stages = [self.prediction(content, pixels, reference)]
        query, reference = content.detach(), reference.detach()
        dn = self.denoising(targets, b, fine) if self.training and targets is not None else None
        count, attention = 0, None
        if dn is not None:
            dc, db, attention, _, _ = dn
            count = dc.shape[1]
            query, reference = torch.cat((dc, query), 1), torch.cat((db, reference), 1)
        dn_stages = []
        for layer in self.layers:
            query = layer(query, levels, reference, attention)
            updated = (torch.logit(reference.clamp(.001, .999))+self.box_embed(query)).sigmoid()
            logits, objects, _ = self.prediction(query, pixels, updated)
            stages.append((logits[:, count:], objects[:, count:], updated[:, count:]))
            if count:
                dn_stages.append((logits[:, :count], objects[:, :count], updated[:, :count]))
            reference = updated.detach()
        return query[:, count:], stages, dn_stages, dn
