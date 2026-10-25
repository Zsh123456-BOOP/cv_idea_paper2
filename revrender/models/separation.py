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
"""
import numpy as np
import torch
import torch.nn.functional as F

IGNORE = 255        # data.separation.IGNORE


class RevSep(torch.nn.Module):
    def __init__(self, body, width=32):
        super().__init__()
        self.body = body; self.heads = torch.nn.Conv2d(width, 2, 1); self.features = None
        body.ending.register_forward_pre_hook(self.keep)

    def keep(self, module, inputs):
        self.features = inputs[0]

    def forward(self, grey, hint):
        """(H1, erase logits, cross logits), each (N, 1, H, W), of grey crops and their hints (N, 1, H, W) on [0, 1]."""
        h, w = grey.shape[-2:]
        restored = self.body(torch.cat([grey, hint, torch.zeros_like(grey)], 1))[:, :1]
        logits = self.heads(self.features)[..., :h, :w]; self.features = None
        return restored, logits[:, :1], logits[:, 1:]


def compose(grey, restored, erase, cross, hard=False):
    """The output: H1 where a strike crosses writing, white where it is erased, the scan elsewhere (logits; ``hard``: at one half)."""
    e, c = ((erase > 0).float(), (cross > 0).float()) if hard else (torch.sigmoid(erase), torch.sigmoid(cross))
    return c*restored + (1-c)*(e + (1-e)*grey)


def psnr_loss(value, target):
    """NAFNet's PSNR loss (the baseline's, ``restoration_torch.train``)."""
    return 10/np.log(10)*torch.log(((value-target)**2).mean((1, 2, 3))+1e-8).mean()


def masked_bce(logits, labels):
    counted = labels != IGNORE
    if not counted.any(): return logits.sum()*0
    return F.binary_cross_entropy_with_logits(logits[counted], labels[counted].float())


def masked_dice(logits, labels):
    counted = (labels != IGNORE).float(); p = torch.sigmoid(logits)*counted; y = (labels == 1).float()
    return 1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)


def separation_loss(network, grey, target, hint, erase, cross, heads=True, erase_weight=1., cross_weight=1., reading=None):
    """The loss of a batch (tensors (N, 1, H, W): grey, target and hint on [0, 1], erase and cross labels uint8) and its terms;
    ``reading``: a function of the output (composed, or H1 without heads) giving the weighted reading term, or None."""
    restored, erase_logits, cross_logits = network(grey, hint)
    terms = {'restored': psnr_loss(restored, target)}; output = restored
    if heads:
        output = compose(grey, restored, erase_logits, cross_logits); terms['composed'] = psnr_loss(output, target)
        terms['erase'] = erase_weight*(masked_bce(erase_logits, erase)+masked_dice(erase_logits, erase))
        terms['cross'] = cross_weight*masked_bce(cross_logits, cross)
    if reading is not None:
        value = reading(output)
        if value is not None: terms['reading'] = value
    return sum(terms.values()), {k: float(v.detach()) for k, v in terms.items()}


@torch.no_grad()
def separate(network, grey, hint, heads=True, composed=True):
    """The restored crops (N, H, W numpy on [0, 1]) of grey crops and their hints: composed at one half, or H1 alone."""
    network.eval()
    g = torch.from_numpy(np.ascontiguousarray(grey[:, None], np.float32)).cuda()
    h = torch.from_numpy(np.ascontiguousarray(hint[:, None], np.float32)).cuda()
    restored, erase, cross = network(g, h); restored = restored.clamp(0, 1)
    out = compose(g, restored, erase, cross, hard=True) if heads and composed else restored
    return out[:, 0].clamp(0, 1).cpu().numpy()
