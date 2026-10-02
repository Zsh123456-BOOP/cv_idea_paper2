"""The per-pixel supervision RevRender's detail map gets, given to a native baseline (the "fairness" check).

A small head (3x3 conv to 64, ReLU, 1x1 conv) on the baseline's 1/4-resolution features (MaskDINO's pixel-decoder mask
features, PointRend's FPN P2); its logits are upsampled bilinearly to the line and trained with the same soft Dice
against the union of the gold revisions, weight 1, added to the baseline's own losses.  The head is dropped from every
saved state, so freezing and re-thresholding read the model unchanged.  The question: does the extra supervision alone
close the gap to RevRender?
"""
from collections import OrderedDict

from ...util import require

HEAD_SEED = 91201
DENSE_WEIGHT = 1.


def soft_dice(logits, target):
    """1 - (2 sum(p t) + 1) / (sum p + sum t + 1) per image, averaged."""
    p = logits.sigmoid().flatten(1)
    t = target.flatten(1).to(p.dtype)
    return (1-(2*(p*t).sum(1)+1)/(p.sum(1)+t.sum(1)+1)).mean()


def attach_dense(model, method='maskdino'):
    """The head on the baseline's 1/4-resolution features; its loss is added in training and it is left out of state_dict."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    inner = getattr(model, 'native_model', model)          # the factory wraps the detectron2 model
    captured = {}
    if method == 'maskdino':
        decoder = inner.sem_seg_head.pixel_decoder
        require(isinstance(decoder.mask_features, nn.Conv2d), 'unexpected MaskDINO pixel decoder')
        channels = decoder.mask_features.out_channels
        decoder.mask_features.register_forward_hook(lambda module, inputs, output: captured.__setitem__('features', output))
    else:
        require(method == 'pointrend', 'the dense head is declared for MaskDINO and PointRend only')
        shapes = inner.backbone.output_shape()
        require('p2' in shapes and shapes['p2'].stride == 4, 'unexpected PointRend backbone: no stride-4 P2')
        channels = shapes['p2'].channels
        inner.backbone.register_forward_hook(lambda module, inputs, output: captured.__setitem__('features', output['p2']))
    device = next(model.parameters()).device
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        torch.manual_seed(HEAD_SEED)
        model.dense_head = nn.Sequential(nn.Conv2d(channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1)).to(device)
    forward = model.forward

    def with_dense(batched_inputs):
        captured.clear()
        out = forward(batched_inputs)
        if not model.training:
            return out
        features = captured['features']
        h, w = batched_inputs[0]['image'].shape[-2:]
        logits = F.interpolate(model.dense_head(features), size=(4*features.shape[-2], 4*features.shape[-1]),
                               mode='bilinear', align_corners=False)[..., :h, :w]
        targets = []
        for item in batched_inputs:
            masks = item['instances'].gt_masks
            masks = masks.tensor if hasattr(masks, 'tensor') else masks
            targets.append(masks.any(0) if len(masks) else torch.zeros((h, w), dtype=torch.bool))
        target = torch.stack(targets).to(logits.device)[:, None]
        require(logits.shape == target.shape, f'dense logits {tuple(logits.shape)} against target {tuple(target.shape)}')
        return dict(out, loss_dense=DENSE_WEIGHT*soft_dice(logits, target))
    model.forward = with_dense
    state_dict = model.state_dict

    def without_head(*args, **kwargs):
        state = state_dict(*args, **kwargs)
        return OrderedDict((k, v) for k, v in state.items() if not k.startswith('dense_head.'))
    model.state_dict = without_head
    load_state = model.load_state_dict

    def load_without_head(state, strict=True):
        result = load_state(state, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith('dense_head.')]
        require(not (strict and (missing or result.unexpected_keys)), f'state mismatch: {missing} {result.unexpected_keys}')
        return result
    model.load_state_dict = load_without_head
    return model
