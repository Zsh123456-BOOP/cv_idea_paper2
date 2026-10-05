"""Train the strike GAN (``models.strike_gan``) on lines a student struck out and clean lines (``data.gan_sets``), the shapes it places
being the real struck lines of ``data.strike_bank``.

    python -m revrender.train_gan --sets GAN_SETS --bank STRIKE_BANK [--steps 200000] [--max-hours 9] [--batch 32] --output-dir OUT

Each step: a batch of real lines and of host lines, each host dealt ``SLOTS`` random shapes; the discriminator learns real from
struck hosts (an R1 penalty every ``R1_EVERY`` steps), then the generator learns to fool it, with the diversity term on a second
noise.  Adam (betas 0, 0.99); an exponential moving average of the generator (``--ema``) is what is saved and drawn.  Every
``--sample-every`` steps: OUT/sheet_<step>.png (struck hosts with what was drawn in red, and real lines) and OUT/generator_<step>.pt;
OUT/log.jsonl every 100 steps (both losses, the discriminator's mean score on real and struck lines, places used, darkness);
OUT/generator_last.pt and OUT/summary.json at the end (``--steps`` reached or ``--max-hours`` spent).  Seed 42.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

R1_EVERY = 4


def shapes_of(bank_dir, device):
    """(opacity maps [N, h, w] at the GAN's shape size, aspect [N], length in heights of their own line [N]) of a strike bank."""
    import torch
    from torch.nn import functional as F
    from .data.paste_strikes import Bank
    from .models.strike_gan import SHAPE
    bank = Bank(bank_dir)
    maps, aspect, length = [], [], []
    for p in bank.pieces:
        grey = bank.grey[p['id']].astype(np.float32)
        alpha = np.clip((255.-grey)/(255.-min(p['ink_level'], 200)), 0, 1)
        maps.append(F.interpolate(torch.from_numpy(alpha)[None, None], size=SHAPE, mode='bilinear', align_corners=False)[0, 0])
        aspect.append(p['size'][0]/p['size'][1])
        length.append(p['size'][0]/p['line_height'])
    return (torch.stack(maps).to(device), torch.tensor(aspect, dtype=torch.float32, device=device),
            torch.tensor(length, dtype=torch.float32, device=device))


def sheet(fake, each, real, path, rows=8):
    """Struck hosts (what was drawn tinted red) above real lines."""
    tiles = []
    for k in range(min(rows, fake.shape[0])):
        grey = (fake[k].cpu().numpy()*255).astype(np.uint8)
        rgb = np.repeat(grey[..., None], 3, 2).astype(np.float32)
        drawn = each[k].max(0).values.cpu().numpy()
        rgb[..., 1:] *= (1-.8*drawn)[..., None]
        tiles.append(rgb.astype(np.uint8))
    for k in range(min(rows, real.shape[0])):
        tiles.append(np.repeat((real[k].cpu().numpy()*255).astype(np.uint8)[..., None], 3, 2))
    gap = np.full((4, tiles[0].shape[1], 3), 120, np.uint8)
    Image.fromarray(np.concatenate([x for t in tiles for x in (t, gap)])).save(path)


def run(args):
    import torch
    from .models import strike_gan as G_
    torch.manual_seed(SEED)
    device = 'cuda'
    require(torch.cuda.is_available(), 'CUDA required')
    real = torch.from_numpy(np.load(args.sets/'real.npy')).to(device)
    hosts = torch.from_numpy(np.load(args.sets/'hosts.npy')).to(device)
    host_height = torch.tensor([r['strip_height'] for r in json.loads((args.sets/'hosts.json').read_text())], dtype=torch.float32, device=device)
    shapes, aspect, length = shapes_of(args.bank, device)
    generator, discriminator = G_.Generator().to(device), G_.Discriminator().to(device)
    average = copy.deepcopy(generator).eval()
    for p in average.parameters():
        p.requires_grad_(False)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0., .99))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0., .99))
    rng = G_.seeded(SEED, device)
    out = make_output_dir(args.output_dir)
    write_manifest(out, config=to_jsonable(args), inputs={str(args.sets/'run_manifest.json'): digest(args.sets/'run_manifest.json'),
                   str(args.bank/'run_manifest.json'): digest(args.bank/'run_manifest.json')},
                   real=int(len(real)), hosts=int(len(hosts)), shapes=int(len(shapes)))
    batch, started, step = args.batch, time.monotonic(), 0
    fixed = {k: torch.randint(n, shape, generator=G_.seeded(SEED+1, device), device=device)
             for k, n, shape in (('h', len(hosts), (8,)), ('s', len(shapes), (8, G_.SLOTS)), ('r', len(real), (8,)))}
    fixed_noise = torch.randn(8, G_.NOISE, generator=G_.seeded(SEED+2, device), device=device)
    log = (out/'log.jsonl').open('x')

    def draw(model, h, s, noise):
        with torch.no_grad():
            return model(hosts[h].float()/255, noise, shapes[s], aspect[s], length[s], host_height[h])

    while step < args.steps and time.monotonic()-started < args.max_hours*3600:
        step += 1
        ri = torch.randint(len(real), (batch,), generator=rng, device=device)
        hi = torch.randint(len(hosts), (batch,), generator=rng, device=device)
        si = torch.randint(len(shapes), (batch, G_.SLOTS), generator=rng, device=device)
        line, host = real[ri].float()/255, hosts[hi].float()/255
        noise_a = torch.randn(batch, G_.NOISE, generator=rng, device=device)
        noise_b = torch.randn(batch, G_.NOISE, generator=rng, device=device)
        dealt = (shapes[si], aspect[si], length[si], host_height[hi])
        fake, each = generator(host, noise_a, *dealt)
        real_scores = discriminator(G_.augment(line, rng))
        fake_scores = discriminator(G_.augment(fake.detach(), rng))
        loss_d = G_.d_loss(real_scores, fake_scores)
        if step % R1_EVERY == 0:
            loss_d = loss_d+args.r1*R1_EVERY/2*G_.r1_penalty(discriminator, G_.augment(line, rng))
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()
        _, each_b = generator(host, noise_b, *dealt)
        spread = G_.diversity(each, each_b, noise_a, noise_b)
        loss_g = G_.g_loss(discriminator(G_.augment(fake, rng)))+args.diversity*spread
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        with torch.no_grad():
            for a, g in zip(average.parameters(), generator.parameters()):
                a.lerp_(g, 1-args.ema)
        require(torch.isfinite(loss_d) and torch.isfinite(loss_g), f'a loss is not finite at step {step}', RuntimeError)
        if step % 100 == 0:
            used = generator.placements(host, noise_a, aspect[si], length[si], host_height[hi])[0].sum(1).mean()
            entry = {'step': step, 'seconds': round(time.monotonic()-started, 1), 'loss_d': float(loss_d), 'loss_g': float(loss_g),
                     'real_score': float(real_scores.mean()), 'fake_score': float(fake_scores.mean()), 'spread': float(-spread),
                     'places_used': float(used)}
            log.write(json.dumps(entry)+'\n')
            log.flush()
        if step % args.sample_every == 0:
            fake_s, each_s = draw(average, fixed['h'], fixed['s'], fixed_noise)
            sheet(fake_s, each_s, real[fixed['r']].float()/255, out/f'sheet_{step:06d}.png')
            torch.save({'generator': average.state_dict(), 'step': step, 'config': to_jsonable(args)}, out/f'generator_{step:06d}.pt')
    log.close()
    fake_s, each_s = draw(average, fixed['h'], fixed['s'], fixed_noise)
    sheet(fake_s, each_s, real[fixed['r']].float()/255, out/'sheet_last.png')
    torch.save({'generator': average.state_dict(), 'step': step, 'config': to_jsonable(args)}, out/'generator_last.pt')
    summary = {'steps': step, 'seconds': round(time.monotonic()-started, 1), 'stopped_by': 'steps' if step >= args.steps else 'time'}
    write_json(out/'summary.json', summary)
    print(json.dumps(summary))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--sets', type=Path, required=True, help='data.gan_sets output')
    parser.add_argument('--bank', type=Path, required=True, help='data.strike_bank output')
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--max-hours', type=float, default=9.)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr-g', type=float, default=1e-4)
    parser.add_argument('--lr-d', type=float, default=2e-4)
    parser.add_argument('--r1', type=float, default=1.)
    parser.add_argument('--diversity', type=float, default=1.)
    parser.add_argument('--ema', type=float, default=.999)
    parser.add_argument('--sample-every', type=int, default=5000)
    parser.add_argument('--output-dir', type=Path, required=True)
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
