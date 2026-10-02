"""RevRender: see (ResNet-50 + detail pathway) -> find (query detector) -> draw (in-box point renderer) + quality head.

``build_model`` makes the paper-1 model and its three ablations:

  control    the full model
  nofine     the detail pathway's image skips are fed zeros (the pixel decoder sees a blank image)
  nocontext  the renderer's residual has no cross-attention to the 32 fixed points of its box
  nodense    the per-pixel Dice supervision of the detail map is an exact zero (a loss-side change: see losses.py)

Construction order and the global random stream are part of the model: see models/legacy_init.py.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .backbone import MultiScaleSee, build_segmenter
from .detector import RegionFinder
from .legacy_init import build_see_consuming_legacy_draws
from .quality import MaskIoUHead, RULE, delivered_score
from .renderer import RoiPointDraw

REGION_SEED, DRAW_SEED = 67401, 67402
QUERIES = 32
VARIANTS = ('control', 'nofine', 'nocontext', 'nodense')


class RevRender(nn.Module):
    def __init__(self, see, variant='control', rule=RULE):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('undeclared RevRender variant: ' + str(variant))
        self.variant, self.rule, self.query_count = variant, rule, QUERIES
        # Registration order (see, find, draw) fixes parameter order, hence state digests and gradient-clip sums.
        self.see, self.find, self.draw = see, nn.Identity(), nn.Identity()
        # The leftover CUDA reseed of these forks (devices=[]) is deliberate: see models/legacy_init.py.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(REGION_SEED)
            self.see, self.find = MultiScaleSee(see), RegionFinder()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(DRAW_SEED)
            self.draw = RoiPointDraw()
        self.find.mask_iou = MaskIoUHead()
        self.draw.attach_residual(contextual=variant != 'nocontext')
        self.see.pixel_decoder.blank_image = variant == 'nofine'
        self.dense_supervision = variant != 'nodense'
        self.cache = {}

    def forward(self, image, targets=None):
        quarter, fine, dense, levels = self.see(image)
        query, stages, dn_stages, dn = self.find(levels, fine, targets)
        coarse, objects, boxes = stages[-1]
        rendered, points = self.draw(coarse, fine, boxes, image.shape[-2:])
        self.cache = {'coarse': coarse, 'dense': dense, 'boxes': boxes, 'stages': stages, 'dn_stages': dn_stages,
                      'dn': dn, 'auxiliary': [], 'seed_coordinates': boxes[..., :2], **points}
        pixels = F.adaptive_avg_pool2d(fine, coarse.shape[-2:])
        quality = self.find.mask_iou(query, coarse, pixels, rendered, detach=True)
        self.cache['mask_iou'] = quality
        self.cache['mask_iou_delivered'] = rendered
        if self.training:
            return rendered, objects, boxes
        return rendered, delivered_score(objects, quality, self.rule), boxes


def build_model(variant='control', backbone_weights=None):
    """The model with its initial weights.  ``backbone_weights`` is the ImageNet R-50 .pkl (training); at inference the
    checkpoint supplies every weight, so it can be left out."""
    segmenter = build_segmenter(backbone_weights)
    return RevRender(build_see_consuming_legacy_draws(segmenter), variant)


def configuration(variant='control'):
    """What the model is, for the run manifest."""
    return {'model': 'RevRender', 'variant': variant, 'encoder': 'ResNet-50 (ImageNet), SMP Segformer decoder',
            'queries': QUERIES, 'decoder_layers': 4, 'dimension': 160, 'mask_channels': 32,
            'roi_expansion': 1.5, 'roi_canvas': [14, 56], 'roi_subdivision': [[28, 112], [56, 224]],
            'roi_points_per_step': 1024, 'roi_train_points_per_query': 256,
            'context_residual': {'tokens': 32, 'roi_memory_points': 32 if variant != 'nocontext' else 0,
                                 'point_scales': [1, 2, 4], 'seed': 81601},
            'detail_pathway_image_skips': variant != 'nofine',
            'dense_supervision': variant != 'nodense',
            'score': {'rule': RULE, 'mask_level': .5, 'detached_from_the_trunk': True},
            'denoising': 'up to 4 isolated positive noisy-box groups, training only',
            'region_supervision': 'focal class 4, mask BCE 5 / Dice 5, box L1 5 / GIoU 2; each layer and the encoder selection',
            'seeds': {'region': REGION_SEED, 'draw': DRAW_SEED}}
