"""The see stage: a ResNet-50 encoder, a Segformer decoder to quarter resolution and a full-resolution detail pathway.

  encoder         ImageNet ResNet-50 (the detectron2 R-50-torchvision.pkl, converted), one grey input channel
  dense_decoder   SMP's Segformer decoder: 256 channels at 1/4 resolution
  shared_projection  3x3 conv to 160 channels: the quarter-resolution features the detector reads
  pixel_decoder   the detail pathway: the image and its 1/2-resolution skip are added back at 1/2 and full resolution,
                  giving a 32-channel full-resolution detail map ``fine``
  dense_head      1x1 conv on ``fine``: a per-pixel revision logit, supervised with Dice every step, unused at inference
  laterals        1x1 conv + GroupNorm lifting the three coarser encoder maps to 160 channels (MultiScaleSee)
"""
import pickle
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DIMENSION, CHANNELS = 160, 32
RESNET50_SHA = '16fd03f0b0123e85b959cb9b64aaf9eeb4a7dc68bd9b0b134a9220d4b24c15c6'
RESNET50_KEYS = 265
_STAGE = re.compile(r'^res(\d)\.(\d+)\.(conv(\d)|shortcut)(\.norm)?\.(.+)$')


class ConvNormAct(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU())


class PixelDecoder(nn.Module):
    """Learned full- and half-resolution image skips with 32 mask channels.

    ``blank_image`` (the 'nofine' ablation) feeds the skips zeros, so the detail map carries only what the
    quarter-resolution features hold; every module and parameter stays.
    """

    def __init__(self, dimension, channels):
        super().__init__()
        self.skip_full = ConvNormAct(1, 32)
        self.skip_half = ConvNormAct(1, 64)
        self.reduce_half = nn.Conv2d(dimension, 64, 1, bias=False)
        self.refine_half = ConvNormAct(64, 64)
        self.reduce_full = nn.Conv2d(64, 32, 1, bias=False)
        self.refine_full = ConvNormAct(32, 32)
        self.output = nn.Conv2d(32, channels, 1)
        self.blank_image = False

    def forward(self, quarter, image):
        if self.blank_image:
            image = torch.zeros_like(image)
        half = F.interpolate(quarter, size=(image.shape[-2]//2, image.shape[-1]//2), mode='bilinear', align_corners=False)
        half = self.refine_half(self.reduce_half(half) + self.skip_half(F.avg_pool2d(image, 2)))
        full = F.interpolate(half, size=image.shape[-2:], mode='bilinear', align_corners=False)
        return self.output(self.refine_full(self.reduce_full(full) + self.skip_full(image)))


class SeeDetails(nn.Module):
    """Image encoder, scale fusion and full-resolution detail features."""

    def __init__(self, encoder, decoder, decoder_channels, dimension=DIMENSION, channels=CHANNELS):
        super().__init__()
        self.encoder = encoder
        self.dense_decoder = decoder
        self.shared_projection = ConvNormAct(decoder_channels, dimension)
        self.pixel_decoder = PixelDecoder(dimension, channels)
        self.dense_head = nn.Conv2d(channels, 1, 1)


class MultiScaleSee(nn.Module):
    """SeeDetails' parts plus three laterals; returns (quarter, fine, dense logit, the four feature levels)."""

    def __init__(self, base):
        super().__init__()
        for name, part in base.named_children():
            setattr(self, name, part)
        self.laterals = nn.ModuleList([nn.Sequential(nn.Conv2d(c, DIMENSION, 1), nn.GroupNorm(8, DIMENSION))
                                       for c in self.encoder.out_channels[-3:]])

    def forward(self, image):
        encoded = self.encoder(image)
        quarter = self.shared_projection(self.dense_decoder(encoded))
        levels = [quarter] + [layer(value) for layer, value in zip(self.laterals, encoded[-3:])]
        fine = self.pixel_decoder(quarter, image)
        return quarter, fine, self.dense_head(fine), levels


def torchvision_names(state):
    """detectron2's converted ResNet names -> torchvision's; the classifier is dropped."""
    out = {}
    for key, value in state.items():
        if key.startswith('stem.fc.'):
            continue
        if key == 'stem.conv1.weight':
            name = 'conv1.weight'
        elif key.startswith('stem.conv1.norm.'):
            name = 'bn1.' + key[len('stem.conv1.norm.'):]
        else:
            match = _STAGE.match(key)
            if match is None:
                raise ValueError('unrecognised ResNet key: ' + key)
            stage, block, part, conv, norm, field = match.groups()
            layer = f'layer{int(stage) - 1}.{block}'
            if part == 'shortcut':
                name = f'{layer}.downsample.{1 if norm else 0}.{field}'
            else:
                name = f'{layer}.{"bn" if norm else "conv"}{conv}.{field}'
        out[name] = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    return out


def load_resnet50(encoder, path):
    """Load the ImageNet R-50 into an SMP ResNet encoder that reads one channel (conv1 averaged over RGB)."""
    with open(path, 'rb') as handle:
        payload = pickle.load(handle, encoding='latin1')
    state = torchvision_names(payload['model'] if 'model' in payload else payload)
    if len(state) != RESNET50_KEYS:
        raise ValueError(f'{len(state)} ResNet tensors, expected {RESNET50_KEYS}')
    if encoder.conv1.weight.shape[1] == 1 and state['conv1.weight'].shape[1] == 3:
        state['conv1.weight'] = state['conv1.weight'].mean(1, keepdim=True)
    missing, unexpected = torch.nn.Module.load_state_dict(encoder, state, strict=False)
    if unexpected or any(not key.endswith('num_batches_tracked') for key in missing):
        raise ValueError(f'ResNet state mismatch: missing={missing}, unexpected={unexpected}')
    return len(state)


def build_segmenter(weights=None):
    """SMP's Segformer on a one-channel ResNet-50, with the ImageNet weights if a path is given."""
    import segmentation_models_pytorch as smp

    segmenter = smp.Segformer(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1)
    if weights is not None:
        load_resnet50(segmenter.encoder, weights)
    return segmenter
