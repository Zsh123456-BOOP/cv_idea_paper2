"""The strike GAN: a generator that places real struck lines over a clean line of writing, and a discriminator that tells such lines
from lines a student struck out.

The shapes are not the generator's: they are the struck lines cut out of paper 1's gold (``data.strike_bank``), each kept as an
opacity map.  For each of ``SLOTS`` places the generator reads the line (a small CNN, its features kept along the width so it knows
where the words are), a noise vector and the shape it was dealt (its aspect and its length in heights of its own line), and gives
whether to use the place, the shape's centre on the line, how much longer and taller than at its own scale to draw it, a small turn
and how dark.  ``place`` warps every shape onto the line with an affine sampling grid, so the picture is differentiable in every
placement, and the pixels each placed shape covers are known exactly: the struck pixels of what it drew.  The first place is always
used.

Canvas: the line letterboxed onto ``CANVAS`` (paper 1's 96 x 768 strip), grey in [0, 1] with 1 paper.  The discriminator sees both
kinds of lines through the same random warp (``augment``: a shift, sometimes a light blur, brightness and contrast) so that it judges
where and how the strokes lie rather than the scan's grain.  Hinge losses; an R1 penalty on real lines keeps the discriminator smooth;
a diversity term (two noises on one line should place its strikes differently) keeps the generator from striking every line the
same way.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

CANVAS = (96, 768)         # height, width: paper 1's strip
SLOTS = 3                  # places the generator may strike on a line
NOISE = 32
SHAPE = (64, 256)          # every real shape is resampled to this (height, width) opacity map; its own aspect is kept apart
STRETCH = (.5, 2.)         # a shape is drawn this much shorter .. longer (and lower .. taller) than at its own scale
TURN = .35                 # radians a shape may be turned either way
DARKNESS = (.6, 1.)        # how dark it is drawn (1: the line's own darkest ink)
OUTPUTS = 7                # per place: use, centre x, centre y, stretch along, stretch across, turn, darkness


def shape_size(aspect, length, line_height):
    """(width, height) on the canvas of a real shape at its own scale: ``length`` in heights of its own line, ``aspect`` width/height,
    on a line ``line_height`` canvas pixels high."""
    width = length*line_height
    return width, width/aspect


def place(shapes, centre, size, turn, canvas=CANVAS):
    """The opacity [B, S, H, W] of shapes [B, S, h, w] placed on a canvas: ``centre`` [B, S, 2] and ``size`` [B, S, 2] (width, height) in
    canvas pixels, ``turn`` [B, S] in radians."""
    batch, slots = shapes.shape[:2]
    height, width = canvas
    cx, cy = centre[..., 0]/width*2-1, centre[..., 1]/height*2-1          # normalised centre
    sw, sh = (size[..., 0]/width).clamp(min=1e-3), (size[..., 1]/height).clamp(min=1e-3)
    cos, sin = torch.cos(turn), torch.sin(turn)
    # input = A (output - c): turn back by the turn (in pixel units), then scale to the shape's own extent
    aspect = width/height
    a11 = cos/sw
    a12 = sin/(sw*aspect)
    a21 = -sin*aspect/sh
    a22 = cos/sh
    theta = torch.stack([torch.stack([a11, a12, -(a11*cx+a12*cy)], -1), torch.stack([a21, a22, -(a21*cx+a22*cy)], -1)], -2)
    grid = F.affine_grid(theta.reshape(-1, 2, 3), (batch*slots, 1, height, width), align_corners=False)
    out = F.grid_sample(shapes.reshape(-1, 1, *shapes.shape[2:]), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return out.reshape(batch, slots, height, width)


def composite(line, opacity, ink):
    """The line [B, H, W] (1 paper) with ``opacity`` [B, H, W] drawn in ``ink`` [B] (0 black)."""
    return line*(1-opacity)+ink[:, None, None]*opacity


def ink_of(line):
    """The darkness of a line's ink: the 1st percentile of its grey, at most 0.45."""
    return torch.quantile(line.reshape(line.shape[0], -1), .01, dim=1).clamp(max=.45)


def block(c_in, c_out, stride=2, norm=True):
    layers = [nn.Conv2d(c_in, c_out, 3, stride, 1)]
    if norm:
        layers.append(nn.GroupNorm(8, c_out))
    layers.append(nn.LeakyReLU(.2))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """A clean line, noise and the shapes it was dealt -> where and how to draw them."""

    def __init__(self, canvas=CANVAS):
        super().__init__()
        self.canvas = canvas
        self.encoder = nn.Sequential(block(1, 32, norm=False), block(32, 64), block(64, 64), block(64, 32))   # 1/16: 6 x 48
        columns = canvas[1]//16
        self.head = nn.Sequential(nn.Linear(32*columns+NOISE+2*SLOTS, 512), nn.LeakyReLU(.2), nn.Linear(512, 256), nn.LeakyReLU(.2),
                                  nn.Linear(256, SLOTS*OUTPUTS))

    def placements(self, line, noise, aspect, length, line_height):
        """(use [B, S] in {0, 1} with its sigmoid's gradient, centre [B, S, 2], size [B, S, 2], turn [B, S], darkness [B, S]) for lines
        [B, H, W], noise [B, NOISE], the dealt shapes' aspect and length [B, S] and the lines' heights on the canvas [B]."""
        height, width = self.canvas
        features = self.encoder(line[:, None]*2-1).mean(dim=2).flatten(1)
        descriptors = torch.cat([torch.log(aspect), torch.log(length)], 1)
        raw = self.head(torch.cat([features, noise, descriptors], 1)).reshape(-1, SLOTS, OUTPUTS)
        soft = torch.sigmoid(raw[..., 0])
        use = (soft > .5).float()+soft-soft.detach()
        use = torch.cat([torch.ones_like(use[:, :1]), use[:, 1:]], 1)
        centre = torch.stack([torch.sigmoid(raw[..., 1])*width, torch.sigmoid(raw[..., 2])*height], -1)
        own_w, own_h = shape_size(aspect, length, line_height[:, None])
        log_stretch = lambda x: math.log(STRETCH[0])+(math.log(STRETCH[1])-math.log(STRETCH[0]))*torch.sigmoid(x)
        size = torch.stack([own_w*torch.exp(log_stretch(raw[..., 3])), own_h*torch.exp(log_stretch(raw[..., 4]))], -1)
        turn = TURN*torch.tanh(raw[..., 5])
        darkness = DARKNESS[0]+(DARKNESS[1]-DARKNESS[0])*torch.sigmoid(raw[..., 6])
        return use, centre, size, turn, darkness

    def forward(self, line, noise, shapes, aspect, length, line_height):
        """(struck lines [B, H, W], each place's opacity [B, S, H, W] as drawn)."""
        use, centre, size, turn, darkness = self.placements(line, noise, aspect, length, line_height)
        each = place(shapes, centre, size, turn, self.canvas)*(use*darkness)[..., None, None]
        opacity = 1-torch.prod(1-each, dim=1)
        return composite(line, opacity, ink_of(line)), each


class Discriminator(nn.Module):
    """A line [B, H, W] -> a realness score [B] (spectral norm throughout)."""

    def __init__(self):
        super().__init__()
        layers, c = [], 1
        for c_out in (32, 64, 128, 256, 256):
            layers += [spectral_norm(nn.Conv2d(c, c_out, 4, 2, 1)), nn.LeakyReLU(.2)]
            c = c_out
        self.features = nn.Sequential(*layers, spectral_norm(nn.Conv2d(c, 256, 3, 1, 1)), nn.LeakyReLU(.2))
        self.score = spectral_norm(nn.Linear(256, 1))

    def forward(self, line):
        return self.score(self.features(line[:, None]*2-1).mean(dim=(2, 3)))[:, 0]


def augment(lines, generator):
    """The same kind of random warp for real and drawn lines: a shift (up to 1/16 of the width and of the height, paper coming in),
    sometimes a light blur, brightness and contrast within 10%."""
    batch, height, width = lines.shape
    device = lines.device
    rand = lambda *shape: torch.rand(*shape, generator=generator, device=device)
    shift_x = ((rand(batch)*2-1)*width/16).round().long()
    shift_y = ((rand(batch)*2-1)*height/16).round().long()
    out = torch.ones_like(lines)
    for b in range(batch):
        sx, sy = int(shift_x[b]), int(shift_y[b])
        src = lines[b, max(0, -sy):height-max(0, sy), max(0, -sx):width-max(0, sx)]
        out[b, max(0, sy):max(0, sy)+src.shape[0], max(0, sx):max(0, sx)+src.shape[1]] = src
    blur = rand(batch) < .3
    if blur.any():
        kernel = torch.tensor([.25, .5, .25], device=device)
        kernel = (kernel[:, None]*kernel[None, :])[None, None]
        out[blur] = F.conv2d(out[blur][:, None], kernel, padding=1)[:, 0]
    contrast = 1+(rand(batch, 1, 1)*2-1)*.1
    brightness = (rand(batch, 1, 1)*2-1)*.1
    return ((out-1)*contrast+1+brightness).clamp(0, 1)


def d_loss(real_scores, fake_scores):
    return F.relu(1-real_scores).mean()+F.relu(1+fake_scores).mean()


def g_loss(fake_scores):
    return -fake_scores.mean()


def r1_penalty(discriminator, real):
    real = real.detach().requires_grad_(True)
    grad, = torch.autograd.grad(discriminator(real).sum(), real, create_graph=True)
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


def diversity(each_a, each_b, noise_a, noise_b, cap=.05):
    """How much two noises on the same lines move the strikes, per unit of noise, up to ``cap`` (to be made large: its negative is
    the loss, so the generator does not strike every line the same way)."""
    change = (each_a-each_b).abs().mean(dim=tuple(range(1, each_a.dim())))/((noise_a-noise_b).abs().mean(1)+1e-5)
    return -change.clamp(max=cap).mean()


def seeded(seed, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
