"""RevSep: a restorer told where the revision is, which decides what to whiten and what to redraw (plan of 2026-10-25).

The body is the official NAFNet, built exactly as the baseline (``baselines.restoration_torch.build``), fed three channels in place
of the grey crop three times: the grey crop, the hint (``data.separation``: where the detector put the revision) and zeros.  The first
channel of its output is the restored image (H1).  A 1x1 convolution on the decoder's last features (the input of NAFNet's
``ending``) gives two logits per pixel: erase (H2: a strike pixel on paper, made white) and cross (H3: a strike pixel over writing,
redrawn from H1).  The output is composed (``compose``): a crossing takes H1, an erased pixel is white, every other pixel keeps the
scan -- with the heads' probabilities while training, at one half when inferring -- so away from the strike the scan passes through.

Loss (``separation_loss``): NAFNet's PSNR loss on the composed output and on H1 (H1 stays a whole restorer: the composition can be
taken off as an ablation), BCE plus Dice on erase and BCE on cross, both over the pixels their labels count, and optionally a reading
term (``reading``: the frozen reader's CTC loss on the output, ``reader_ctc``).  Without heads (``heads=False``) the loss is the PSNR
loss of H1 and the output H1: NAFNet with the hint channel.

The stroke version (``stroke=True``, plan of 2026-10-25 "the strike as an object") adds a head A on the same features: a centreline
logit and a half-width (0 to ``RADIUS`` pixels), which ``render`` draws -- differentiably, each pixel covered by every centreline
pixel within its half-width -- into the strike's mask S, crossings included.  Where the second pass takes its range from (``run``):
``detector`` -- one pass, head A only supervised; ``stroke`` -- a second pass with S (detached) in place of the detector's range;
``both`` -- the detector's range and S side by side.  Head A's loss: BCE plus Dice on the centreline, L1 on the half-width along it,
Dice of S against the strike's mask (``data.separation.stroke_labels``).

The writing gate (``gate=True``, plan 保字门 of 2026-10-25): a further logit, whether a pixel's ink is writing, independent of erase
(both hold where a strike crosses writing).  While training the composition erases with sigmoid(erase)*(1-sigmoid(writing)); its loss
is BCE on the writing labels (``data.separation.writing_labels``) whose writing pixels near the strike weigh ``near_weight``.  Its
inference (``gated``) is hard and never redraws: a pixel is made white when erase is above its threshold and writing below its own,
every other pixel keeps the scan.
"""
import numpy as np
import torch
import torch.nn.functional as F

IGNORE = 255        # data.separation.IGNORE
RADIUS = 6          # pixels: the widest half-width head A draws
SOFTNESS = .3       # pixels: the edge of a drawn disc (two pixels beyond the half-width it covers under 0.2%)


class RevSep(torch.nn.Module):
    def __init__(self, body, width=32, stroke=False, gate=False):
        super().__init__()
        self.body = body; self.heads = torch.nn.Conv2d(width, 2+2*stroke+gate, 1); self.features = None; self.stroke, self.gate = stroke, gate
        body.ending.register_forward_pre_hook(self.keep)

    def keep(self, module, inputs):
        self.features = inputs[0]

    def forward(self, grey, hint, extra=None):
        """(H1, erase logits, cross logits, head A, writing logits) of grey crops, their hints and a third channel (zeros when None), each
        (N, 1, H, W) on [0, 1]; head A is (centreline logits, half-widths) of the stroke version, the writing logits the gate's; else None."""
        h, w = grey.shape[-2:]
        restored = self.body(torch.cat([grey, hint, torch.zeros_like(grey) if extra is None else extra], 1))[:, :1]
        logits = self.heads(self.features)[..., :h, :w]; self.features = None
        stroke = (logits[:, 2:3], RADIUS*torch.sigmoid(logits[:, 3:4])) if self.stroke else None
        return restored, logits[:, :1], logits[:, 1:2], stroke, logits[:, -1:] if self.gate else None


def render(centre, half):
    """The strike mask drawn from centreline probabilities and half-widths (N, 1, H, W): a pixel is covered by each centreline pixel
    within its half-width (a disc with a ``SOFTNESS`` edge), the covers joined as independent events."""
    h, w = centre.shape[-2:]; c = F.pad(centre, [RADIUS]*4); r = F.pad(half, [RADIUS]*4); missed = torch.zeros_like(centre)
    for dy in range(-RADIUS, RADIUS+1):
        for dx in range(-RADIUS, RADIUS+1):
            d = (dy*dy+dx*dx)**.5
            if d > RADIUS: continue
            cq = c[..., RADIUS+dy:RADIUS+dy+h, RADIUS+dx:RADIUS+dx+w]; rq = r[..., RADIUS+dy:RADIUS+dy+h, RADIUS+dx:RADIUS+dx+w]
            missed = missed + torch.log1p(-(cq*torch.sigmoid((rq-d)/SOFTNESS)).clamp(max=1-1e-4))
    return 1-torch.exp(missed)


def run(network, grey, hint, hint_from='detector'):
    """(H1, erase, cross, head A, S, writing) of a batch: one pass, or for the stroke version with ``hint_from`` stroke / both a second
    pass whose range is S (detached), alone or beside the detector's."""
    restored, erase, cross, stroke, writing = network(grey, hint)
    if stroke is None: return restored, erase, cross, None, None, writing
    drawn = render(torch.sigmoid(stroke[0]), stroke[1])
    if hint_from == 'detector': return restored, erase, cross, stroke, drawn, writing
    second = network(grey, drawn.detach()) if hint_from == 'stroke' else network(grey, hint, drawn.detach())
    return (*second[:3], stroke, drawn, second[4])


def compose(grey, restored, erase, cross, hard=False, writing=None, gate_hard=False):
    """The output: H1 where a strike crosses writing, white where it is erased, the scan elsewhere (logits; ``hard``: at one half).
    With the gate's ``writing`` logits the erase probability is erase*(1-writing) (``gate_hard``: erase above a half and writing not)."""
    e, c = ((erase > 0).float(), (cross > 0).float()) if hard else (torch.sigmoid(erase), torch.sigmoid(cross))
    if writing is not None: e = ((erase > 0) & (writing <= 0)).float() if gate_hard else e*(1-torch.sigmoid(writing))
    return c*restored + (1-c)*(e + (1-e)*grey)


def gated(grey, erase, writing, erase_threshold=.5, writing_threshold=.5):
    """The gate's inference: white where erase is above its threshold and writing below its own, the scan everywhere else."""
    white = (torch.sigmoid(erase) > erase_threshold) & (torch.sigmoid(writing) < writing_threshold)
    return torch.where(white, torch.ones_like(grey), grey)


def psnr_loss(value, target, weight=None):
    """NAFNet's PSNR loss (the baseline's, ``restoration_torch.train``); with a per-pixel ``weight``, its weighted mean error."""
    error = ((value-target)**2).mean((1, 2, 3)) if weight is None else ((value-target)**2*weight).mean((1, 2, 3))/weight.mean((1, 2, 3))
    return 10/np.log(10)*torch.log(error+1e-8).mean()


def masked_bce(logits, labels):
    counted = labels != IGNORE
    if not counted.any(): return logits.sum()*0
    return F.binary_cross_entropy_with_logits(logits[counted], labels[counted].float())


def masked_dice(logits, labels, probabilities=False):
    counted = (labels != IGNORE).float(); p = (logits if probabilities else torch.sigmoid(logits))*counted; y = (labels == 1).float()
    return 1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)


def separation_loss(network, grey, target, hint, erase, cross, heads=True, erase_weight=1., cross_weight=1., reading=None,
                    strokes=None, hint_from='detector', stroke_weights=(1., 1., 1.), writing=None, gate_weights=(1., 1.), gate_hard=False):
    """The loss of a batch (tensors (N, 1, H, W): grey, target and hint on [0, 1], erase and cross labels uint8) and its terms;
    ``reading``: a function of the output (composed, or H1 without heads) giving the weighted reading term, or None; ``strokes``: the
    stroke version's (mask, centreline, half-width) labels, weighted by ``stroke_weights``; ``writing``: the gate's (labels, near the
    strike) with ``gate_weights`` (the term's, the near writing's)."""
    restored, erase_logits, cross_logits, stroke, drawn, writing_logits = run(network, grey, hint, hint_from)
    terms = {'restored': psnr_loss(restored, target)}; output = restored
    if stroke is not None:
        mask, centre, half = strokes; on = centre == 1; wc, ww, ws = stroke_weights
        terms['centre'] = wc*(masked_bce(stroke[0], centre)+masked_dice(stroke[0], centre))
        terms['width'] = ww*((stroke[1][on]-half[on]).abs().mean() if on.any() else stroke[1].sum()*0)
        terms['stroke'] = ws*masked_dice(drawn, mask, probabilities=True)
    if writing_logits is not None:
        labels, near = writing; counted = labels != IGNORE; weight = torch.where(near & (labels == 1), gate_weights[1], 1.)*counted
        terms['writing'] = gate_weights[0]*(F.binary_cross_entropy_with_logits(writing_logits, (labels == 1).float(), weight=weight, reduction='sum')
                                            / weight.sum().clamp(min=1))
    if heads:
        output = compose(grey, restored, erase_logits, cross_logits, writing=writing_logits, gate_hard=gate_hard); terms['composed'] = psnr_loss(output, target)
        terms['erase'] = erase_weight*(masked_bce(erase_logits, erase)+masked_dice(erase_logits, erase))
        terms['cross'] = cross_weight*masked_bce(cross_logits, cross)
    if reading is not None:
        value = reading(output)
        if value is not None: terms['reading'] = value
    return sum(terms.values()), {k: float(v.detach()) for k, v in terms.items()}


@torch.no_grad()
def centreline(network, grey, hint):
    """Head A's centreline probabilities (N, H, W numpy) of grey crops and their hints (the stroke version's first pass)."""
    network.eval()
    g = torch.from_numpy(np.ascontiguousarray(grey[:, None], np.float32)).cuda()
    h = torch.from_numpy(np.ascontiguousarray(hint[:, None], np.float32)).cuda()
    return torch.sigmoid(network(g, h)[3][0])[:, 0].cpu().numpy()


@torch.no_grad()
def separate(network, grey, hint, heads=True, composed=True, hint_from='detector', soften=None, erase_threshold=.5, writing_threshold=.5):
    """The restored crops (N, H, W numpy on [0, 1]) of grey crops and their hints: composed at one half (the erase head at
    ``erase_threshold``), or H1 alone; ``soften`` (a temperature T): the scan lightened by the erase probability at T,
    scan + sigmoid(erase/T)*(1-scan), no H1 and no crossing.  The gated version answers by ``gated`` at the two thresholds."""
    network.eval()
    g = torch.from_numpy(np.ascontiguousarray(grey[:, None], np.float32)).cuda()
    h = torch.from_numpy(np.ascontiguousarray(hint[:, None], np.float32)).cuda()
    restored, erase, cross, _, _, writing = run(network, g, h, hint_from); restored = restored.clamp(0, 1)
    if writing is not None: return gated(g, erase, writing, erase_threshold, writing_threshold)[:, 0].cpu().numpy()
    if soften: return (g+torch.sigmoid(erase/soften)*(1-g))[:, 0].clamp(0, 1).cpu().numpy()
    if erase_threshold != .5: erase = erase-float(np.log(erase_threshold/(1-erase_threshold)))   # its logit at the threshold is 0
    out = compose(g, restored, erase, cross, hard=True) if heads and composed else restored
    return out[:, 0].clamp(0, 1).cpu().numpy()
