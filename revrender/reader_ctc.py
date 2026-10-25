"""R: a small line reader (CNN + BiLSTM + CTC) whose only use is RevSep's recognition loss (plan of 2026-10-25); it is never evaluated.

    python -m revrender.reader_ctc --corpus CORPUS/corpus.jsonl --pages DIR [--pages DIR] --output-dir OUT

Taught on the clean lines of the reader corpus (``data.reader_corpus``: lines of paper 1's training writers with no edit, read alike by
two strong models; its own validation writers apart), each cut from its page (``train_reader.Crops``), cleaned as the restorers' crops
are (``data.clean_pages.review_clean``) and scaled to ``HEIGHT``.  Its alphabet is the characters of the training texts in the spelling
read-back counts (``eval.coverage.prepare``).  Kept: the step with the lowest character error on the validation lines (OUT/reader.pt,
with the alphabet).  OUT holds student text only as weights.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data.clean_pages import review_clean
from .eval.coverage import alignment, prepare
from .util import SEED, digest, make_output_dir, require, to_jsonable, write_manifest

HEIGHT = 64
MAX_WIDTH = 1024


class Reader(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        def block(a, b, pool): return [torch.nn.Conv2d(a, b, 3, padding=1), torch.nn.BatchNorm2d(b), torch.nn.ReLU(), torch.nn.MaxPool2d(pool)]
        self.cnn = torch.nn.Sequential(*block(1, 64, 2), *block(64, 128, 2), *block(128, 256, (2, 1)), *block(256, 256, (2, 1)))
        self.rnn = torch.nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.out = torch.nn.Linear(256, classes+1)       # class 0 is CTC's blank

    def forward(self, images):
        """Log-probabilities (T, N, classes+1) of white-padded images (N, 1, HEIGHT, W) on [0, 1]; T = W/4."""
        x = self.cnn(1-images).mean(2).transpose(1, 2)
        return self.out(self.rnn(x)[0]).log_softmax(-1).transpose(0, 1)


def scaled(image):
    """A grey crop (H, W, [0, 1] tensor) at ``HEIGHT``, its width in proportion (at least 4, at most ``MAX_WIDTH``)."""
    h, w = image.shape
    width = int(min(MAX_WIDTH, max(4, round(w*HEIGHT/h))))
    return F.interpolate(image[None, None], size=(HEIGHT, width), mode='bilinear', align_corners=False)[0, 0]


def batch_of(images):
    """Scaled crops padded with white to one width: (N, 1, HEIGHT, W) and each one's sequence length."""
    scaled_images = [scaled(im) for im in images]; width = max(im.shape[1] for im in scaled_images); width += -width % 4
    batch = torch.ones(len(images), 1, HEIGHT, width, device=images[0].device)
    for k, im in enumerate(scaled_images): batch[k, 0, :, :im.shape[1]] = im
    return batch, torch.tensor([im.shape[1]//4 for im in scaled_images], dtype=torch.long)


def encoded(text, alphabet):
    """The classes of a text in read-back's spelling; characters outside the alphabet and ``?`` dropped."""
    return [alphabet.index(c)+1 for c in prepare(text) if c in alphabet]


def ctc(reader, images, texts, alphabet):
    """The mean CTC loss of crops (grey [0, 1] tensors, any size, the graph kept) read against their texts.  Without cuDNN: the frozen
    reader stays in eval mode, where cuDNN's RNN has no backward."""
    batch, lengths = batch_of(images); targets = [encoded(t, alphabet) for t in texts]
    with torch.backends.cudnn.flags(enabled=False):
        log_probs = reader(batch)
    flat = torch.tensor([c for t in targets for c in t], dtype=torch.long)
    return F.ctc_loss(log_probs, flat, lengths, torch.tensor([len(t) for t in targets]), zero_infinity=True)


def greedy(log_probs, lengths, alphabet):
    best = log_probs.argmax(-1).transpose(0, 1).cpu().numpy(); texts = []
    for row, n in zip(best, lengths.tolist()):
        keep = [c for k, c in enumerate(row[:n]) if c and (k == 0 or c != row[k-1])]
        texts.append(''.join(alphabet[c-1] for c in keep))
    return texts


def load(path, device='cuda'):
    state = torch.load(path, map_location=device, weights_only=False)
    reader = Reader(len(state['alphabet'])).to(device); reader.load_state_dict(state['model']); reader.eval().requires_grad_(False)
    return reader, state['alphabet']


def main(argv=None):
    from .data.reader_corpus import val_writer
    from .train_reader import Crops, read_rows
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True); p.add_argument('--pages', type=Path, action='append', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=12000); p.add_argument('--batch-size', type=int, default=16); p.add_argument('--eval-every', type=int, default=500)
    args = p.parse_args(argv)
    require(torch.cuda.is_available(), 'CUDA required'); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rows = read_rows(args.corpus, kinds=('clean',)); crops = Crops(args.pages, rows)
    require(all((r['split'] == 'val') == val_writer(r['writer']) for r in rows), 'the corpus split moved')
    def grey(r): return torch.from_numpy(review_clean(np.asarray(Image.fromarray(crops.get(r['line_id'], r['box_1000'])).convert('L'))).astype(np.float32)/255)
    data = {s: [(grey(r), r['host_text']) for r in rows if r['split'] == s] for s in ('train', 'val')}
    alphabet = sorted({c for _, t in data['train'] for c in prepare(t) if c != '?'})
    out = make_output_dir(args.output_dir); reader = Reader(len(alphabet)).cuda()
    opt = torch.optim.Adam(reader.parameters(), 1e-3); schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    rng = random.Random(SEED); best = (2., 0); log = (out/'events.jsonl').open('w'); start = time.monotonic()
    def error(split):
        reader.eval(); wrong = total = 0
        with torch.no_grad():
            for k in range(0, len(data[split]), 32):
                part = data[split][k:k+32]; batch, lengths = batch_of([im.cuda() for im, _ in part])
                for (_, text), reading in zip(part, greedy(reader(batch), lengths, alphabet)):
                    x = prepare(text); kept, _ = alignment(x, reading); wrong += len(x)-len(kept)+len(reading)-len(kept); total += len(x)
        reader.train(); return wrong/max(1, total)
    for step in range(1, args.steps+1):
        part = rng.sample(data['train'], args.batch_size)
        images = []
        for im, _ in part:      # a little scale and contrast jitter
            im = im.cuda(); h = max(8, round(im.shape[0]*rng.uniform(.85, 1.15))); w = max(8, round(im.shape[1]*rng.uniform(.85, 1.15)))
            images.append(F.interpolate(im[None, None], size=(h, w), mode='bilinear', align_corners=False)[0, 0].clamp(0, 1)**rng.uniform(.8, 1.25))
        loss = ctc(reader, images, [t for _, t in part], alphabet)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(reader.parameters(), 5.); opt.step(); schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            cer = error('val'); record = {'step': step, 'loss': float(loss), 'val_character_error': cer, 'seconds': round(time.monotonic()-start)}
            if cer < best[0]:
                best = (cer, step); torch.save({'model': reader.state_dict(), 'alphabet': alphabet, 'step': step}, out/'reader.pt')
            log.write(json.dumps(record)+'\n'); log.flush(); print(json.dumps(record), flush=True)
    write_manifest(out, config=to_jsonable(args), inputs={str(args.corpus): digest(args.corpus)}, alphabet=len(alphabet),
                   lines={s: len(v) for s, v in data.items()}, selected_step=best[1], val_character_error=best[0],
                   files={'reader.pt': digest(out/'reader.pt')})


if __name__ == '__main__':
    main()
