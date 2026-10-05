"""The box remover's training set drawn by the strike GAN: real struck lines placed by its generator on clean lines, a crop round each.

    python -m revrender.data.gan_strikes --generator generator_last.pt --sets GAN_SETS --bank STRIKE_BANK --pages DIR [--pages DIR ...]
        [--copies 2] [--clean-share .1] [--batch 64] --output-dir OUT

Every host line of ``--sets`` (``data.gan_sets``: clean lines of the writers training may use) is struck ``--copies`` times: the
generator places the shapes it is dealt on the line's strip as it learned to (``models.strike_gan``), and those placements are drawn
again at the line's own resolution (the strip's coordinates taken back through the letterbox), in the line's own ink.  Each place it
used is a revision; removing it takes its pixels at least ``REMOVE_ALPHA`` opaque that are not writing (``paste_strikes``: the writing a
struck line crosses keeps its pixels).  A line whose strikes would remove nothing is skipped.  Each revision gives a crop round the
writing it voids, framed as ``synthetic_set --crops`` frames one; with probability ``--clean-share`` a host also gives a crop of one to
three of its words, unstruck.  A small share of the writers (``reader_corpus.val_writer``) is the set's validation split.

The output is a synthetic set (``synthetic_set``'s layout: images.npy, labels.npy, records.json, summary.json, sheet.png), so
``train --synthetic`` trains on it.  Student handwriting: OUT stays out of Git.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .. import pages as pages_module
from ..eval.ink import INK_BELOW
from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .paste_strikes import REMOVE_ALPHA, WRITING_BELOW, word_spans
from .reader_corpus import val_writer
from .synthetic_set import cropped, region_of, rows_round, sheet


def native(centre, size, line_size):
    """Placements on the strip -> on the line of ``line_size`` (width, height) that was letterboxed onto it."""
    from ..models.strike_gan import CANVAS
    width, height = line_size
    scale = min(CANVAS[1]/width, CANVAS[0]/height)
    left, top = (CANVAS[1]-max(1, round(width*scale)))//2, (CANVAS[0]-max(1, round(height*scale)))//2
    shift = centre.new_tensor([left, top])
    return (centre-shift)/scale, size/scale


def struck(grey, each, use_dark):
    """(the line struck in its own ink, the revisions' numbers) from a grey line [h, w] uint8 and its places' opacity [S, h, w] (a
    torch tensor) times each place's use and darkness [S]."""
    import torch
    opacity = each*use_dark[:, None, None]
    total = (1-torch.prod(1-opacity, dim=0)).cpu().numpy()
    ink = min(.45, float(np.percentile(grey, 1))/255)
    picture = ((grey/255.)*(1-total)+ink*total)*255
    writing = grey < WRITING_BELOW
    numbers = np.zeros(grey.shape, np.uint8)
    used = [k for k in range(opacity.shape[0]) if float(use_dark[k]) > 0]
    xs = {k: float((opacity[k] > REMOVE_ALPHA).nonzero()[:, 1].float().mean()) if (opacity[k] > REMOVE_ALPHA).any() else 0. for k in used}
    for n, k in enumerate(sorted(used, key=lambda k: xs[k]), 1):
        mine = (opacity[k] >= REMOVE_ALPHA).cpu().numpy() & ~writing & (numbers == 0)
        numbers[mine] = n
    renumbered = np.zeros_like(numbers)
    for n, k in enumerate(sorted(set(np.unique(numbers)) - {0}), 1):
        renumbered[numbers == k] = n
    return Image.fromarray(np.clip(picture, 0, 255).astype(np.uint8)), renumbered


def main(argv=None):
    import torch
    from ..models import strike_gan as G_
    from ..train_gan import shapes_of
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--generator', type=Path, required=True)
    parser.add_argument('--sets', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--pages', type=Path, action='append', required=True)
    parser.add_argument('--copies', type=int, default=2)
    parser.add_argument('--clean-share', type=float, default=.1)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    device = 'cuda'
    torch.manual_seed(SEED)
    generator = G_.Generator().to(device).eval()
    generator.load_state_dict(torch.load(args.generator, map_location=device, weights_only=False)['generator'])
    shapes, aspect, length = shapes_of(args.bank, device)
    canvases = np.load(args.sets/'hosts.npy', mmap_mode='r')
    hosts = json.loads((args.sets/'hosts.json').read_text())
    require(len(hosts) == len(canvases), 'hosts.json does not match hosts.npy')
    files = pages_module.page_files(args.pages, sorted({h['image_id'] for h in hosts}))
    order = sorted(range(len(hosts)), key=lambda i: hosts[i]['image_id'])
    rng, torch_rng = random.Random(SEED), G_.seeded(SEED, device)
    drawn, opened, skipped = [], {}, 0
    for copy in range(args.copies):
        for start in range(0, len(order), args.batch):
            chunk = order[start:start+args.batch]
            canvas = torch.from_numpy(np.stack([canvases[i] for i in chunk])).to(device).float()/255
            si = torch.randint(len(shapes), (len(chunk), G_.SLOTS), generator=torch_rng, device=device)
            noise = torch.randn(len(chunk), G_.NOISE, generator=torch_rng, device=device)
            heights = torch.tensor([hosts[i]['strip_height'] for i in chunk], dtype=torch.float32, device=device)
            with torch.no_grad():
                use, centre, size, turn, darkness = generator.placements(canvas, noise, aspect[si], length[si], heights)
            for j, i in enumerate(chunk):
                h = hosts[i]
                if h['image_id'] not in opened:
                    opened.clear()
                    with Image.open(files[h['image_id']]) as page:
                        opened[h['image_id']] = page.convert('L')
                grey = np.asarray(opened[h['image_id']].crop(tuple(h['box'])))
                c, s = native(centre[j], size[j], (grey.shape[1], grey.shape[0]))
                with torch.no_grad():
                    each = G_.place(shapes[si[j]][None], c[None], s[None], turn[j][None], canvas=grey.shape)[0]
                image, numbers = struck(grey, each, use[j].detach().round()*darkness[j])
                split = 'val' if val_writer(h['writer']) else 'train'
                base = {'line_id': h['line_id'], 'group_uuid': h['writer'], 'split': split}
                if not numbers.any():
                    skipped += 1
                else:
                    ink = np.asarray(image) < INK_BELOW
                    for k in sorted(set(np.unique(numbers)) - {0}):
                        strip, strip_numbers, crop = cropped(image, numbers, region_of(numbers, k, ink), rng)
                        drawn.append((strip, strip_numbers, {**base, 'id': f'{h["line_id"]}~{copy}#{k}', 'kinds': ['gan'], 'styles': ['gan'],
                                                             'revisions': int(len(np.unique(strip_numbers[strip_numbers > 0]))), 'crop': crop}))
                if rng.random() < args.clean_share:
                    ink = grey < INK_BELOW
                    spans = word_spans(ink, ink.shape[0])
                    if spans:
                        first = rng.randrange(len(spans))
                        last = min(len(spans), first+rng.randint(1, 3))-1
                        columns = (spans[first][0], spans[last][1])
                        top, bottom = rows_round(ink, columns, int(np.median(np.flatnonzero(ink[:, columns[0]:columns[1]].any(1)))))
                        strip, strip_numbers, crop = cropped(Image.fromarray(grey), np.zeros(grey.shape, np.uint8), [columns[0], top, columns[1], bottom], rng)
                        drawn.append((strip, strip_numbers, {**base, 'id': f'{h["line_id"]}~{copy}#c', 'kinds': ['clean'], 'styles': [None],
                                                             'revisions': 0, 'crop': crop}))
    require(drawn, 'nothing drawn')
    out = make_output_dir(args.output_dir)
    np.save(out/'images.npy', np.stack([g for g, _, _ in drawn]))
    np.save(out/'labels.npy', np.stack([n for _, n, _ in drawn]))
    records = [r for _, _, r in drawn]
    write_json(out/'records.json', records)
    sheet(np.load(out/'images.npy', mmap_mode='r'), np.load(out/'labels.npy', mmap_mode='r'), records, out/'sheet.png')
    summary = {'samples': len(records), 'by_split': dict(Counter(r['split'] for r in records)), 'positive': sum(r['revisions'] > 0 for r in records),
               'two_revisions': sum(r['revisions'] >= 2 for r in records), 'clean': sum(r['kinds'] == ['clean'] for r in records),
               'skipped_lines': skipped, 'writers': len({r['group_uuid'] for r in records})}
    write_json(out/'summary.json', summary)
    files_ = {name: digest(out/name) for name in ('images.npy', 'labels.npy', 'records.json')}
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in (args.generator, args.sets/'run_manifest.json', args.bank/'run_manifest.json')},
                   files=files_, **summary)
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
