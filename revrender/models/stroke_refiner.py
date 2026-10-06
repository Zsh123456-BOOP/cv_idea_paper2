"""Small conditional pixel refiner: scan, previous erase mask, add/keep clicks.

This is an annotation aid, trained separately from the paper-1 detector. Its target
is the union of removable strike pixels, not an object instance or an OCR answer.
"""
import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU())


class StrokeRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.ModuleList([block(4, 16), block(16, 32), block(32, 64), block(64, 96)])
        self.up = nn.ModuleList([block(96+64, 64), block(64+32, 32), block(32+16, 16)])
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, image):
        x, skips = image, []
        for k, layer in enumerate(self.down):
            if k:
                x = F.avg_pool2d(x, 2)
            x = layer(x)
            skips.append(x)
        for layer, skip in zip(self.up, reversed(skips[:-1])):
            x = layer(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='nearest'), skip], 1))
        prior = image[:, 1:2].clamp(.08, .92)
        return self.out(x)+torch.logit(prior)
