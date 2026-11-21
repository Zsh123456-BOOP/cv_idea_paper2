"""Teach a reader's vision tower to see a struck crop as its clean twin: a student (the tower with a low-rank adapter) looks at the scan,
the teacher (the same tower, frozen) at the clean image, and the student is pulled to the teacher token by token (plan of
2026-11-20, 第四轮_特权视图三样试验, section 4.2).

    python -m revrender.train_clean_view train --model MODEL --pack VIEWS --scan-pack SCAN_VIEWS --fold K [--synthetic SYNTH_DIR]
        [--steps N] [--batch-size B] [--eval-every E] --output-dir OUT
    python -m revrender.train_clean_view distance --model MODEL --student OUT/student --pack VIEWS --raw-pack RAW_VIEWS
        --erased-pack ERASED_VIEWS [--scan-pack SCAN_VIEWS] --fold K --output-dir DIST
    python -m revrender.train_clean_view agree --model MODEL --pack PACK --students PATTERN --fold K [--regions N] --output OUT.json
    python -m revrender.train_clean_view subsets --pack VIEWS --scan-pack SCAN_VIEWS --shares 0.25 0.5 --output OUT.json
    python -m revrender.train_clean_view same-tower --model MODEL --pack VIEWS --output OUT.json
    python -m revrender.train_clean_view head --model MODEL --pack VIEWS --students PATTERN --output-dir HEAD
    python -m revrender.train_clean_view boards-distance --model MODEL --student OUT/student --boards BOARDS --kinds answer wrong blank --fold K --output-dir DIST
    python -m revrender.train_clean_view spread --model MODEL --pack VIEWS --student NAME=PATTERN [...] --folds 0 1 2 3 4 --output OUT.json

The features: everything the tower hands the language model -- the merged tokens and the three deepstack levels of Qwen3-VL
(``deepstack_visual_indexes``), each a token of the merged grid.  A scan and its clean twin have one size, so their tokens pair one
to one (asserted).  The loss of a level: the mean over the tokens of 1 - cosine, plus the squared error over the teacher's mean
square; the loss is the mean of the four levels.  The adapter: rank 16, alpha 32, dropout .05, on every linear layer of the tower's
blocks (the mergers into the language model stay as they are), AdamW at 1e-4, 30 warm-up steps then a cosine to a tenth.

``train``: the pairs of a region are its window (what the student sees, ``SCAN_VIEWS`` erased -> the annotator's erased window) and
its line (``SCAN_VIEWS`` context -> the line with the erased window, ``VIEWS`` context), of the regions of the other folds.  With the
raw views pack (its ``erased`` the scan, its ``context`` the line with the scan in the box) the student sees the scan (V, fourth
round); with NAFNet's held-out erased views it sees NAFNet's crop (N), with ``data.struck_pack tinted`` the scan with NAFNet's erasure
in red (M; plan of 2026-11-20, 第五轮_X与V结合, sections 4.2-4.3).  Writers
ordered by sha256("<writer>:42"), the first holding a tenth of those regions validate (their loss only).  ``--synthetic``: a
``data.synth_views build`` folder whose fold_K_raw / fold_K_clean packs give the synthetic pairs (scan -> clean host, window and
line); each sample of a batch is then real or synthetic with even chances.  ``--covered``: a ``data.synth_views covered`` folder
(the same packs), and a sample is real, synthetic or covered with chances 1/2, 1/4, 1/4 (sixth round, section 4.2).
``--real-share S``: only the real pairs of the training writers ``real_subset`` takes for S (the learning curve, section 4.3; the
validation writers and the synthetic pairs as ever).  Every ``--eval-every`` steps the validation loss; the adapter of the lowest is
kept (OUT/student), the curve in OUT/train_log.jsonl.

``subsets``: every fold's learning-curve writers for each share (OUT.json: writers, regions, the share reached, whether each share's
writers hold the smaller ones'), as ``train --real-share`` takes them.  ``same-tower`` (the sixth round's check 1): one batch of
the pack's images (eight regions, three images each) through the frozen tower alone (``vision_tower``) and through the whole model's
tower: whether the four levels are the same bit for bit, and their mean and lowest token cosines.

``head`` (the sixth round's B2, section 4.4): a small net that tells from what is there at test time how clean an adapter made a
window.  Every region's window (the pack's ``raw``) through its own fold's adapter (PATTERN, ``{fold}``: the one that never trained
on it) and through the frozen tower, the merged tokens of both side by side are the input; the target of a token is the cosine of
the adapter's token to the frozen tower's token of the annotator's erased window (there only in training).  Per token: the two
features -> 512 -> 1, squashed to 0-1, trained against the target clipped to 0-1 (mean squared error); a region's score is the mean
over its tokens.  For each fold K a head learns on the other folds' regions (AdamW 1e-3, 16 regions a batch, 30 epochs, seed 42)
and scores fold K's.  HEAD/scores.jsonl per region its score and the true mean cosine; HEAD/summary.json their Pearson and Spearman
correlations per fold and over all.

``train --discriminator`` (V-判别, the sixth round's update, section two): MAA's token discriminator (``TokenDiscriminator``) learns
the teacher's merged tokens of the clean images as real and the adapter's as fake (0.5 x the two BCEs, AdamW at ``DISCRIMINATOR_LR``,
float32; ``--discriminator-lr`` another rate), then the adapter learns the alignment loss plus ``ADVERSARIAL_WEIGHT`` x the BCE of its tokens judged real, each step in
that order, windows and lines, real and synthetic pairs alike; the discriminator is built with the random streams put back as they
were and never saved.  ``--window-only``: the window's pair alone (a board pack of ``data.hint_image boards``).

``boards-distance`` (the update, section three): fold K's regions with each kind of board (BOARDS/<kind>/teacher): the frozen tower
on the scan with the board and on the erased window with the same board, the student on the scan with the board; the touched tokens
are those of the window's columns (``window_columns``) where the frozen tower's merged tokens of the two have cosine below .99; per
level the mean cosine to the clean features over them of the scan (raw) and of the student (DIST/distances.jsonl).  ``spread``: on
the touched tokens of each fold's windows (the pack's raw and erased windows, the frozen tower), the merged tokens of the scan, of
each student (each region through the fold's adapter, PATTERN with ``{fold}``) against the clean ones: MMD squared (Gaussian kernel,
its width the median distance of the pooled clean and scan tokens) and the energy distance, on up to ``SPREAD`` tokens of each (seed 42).

``distance``: on the regions of fold K, the teacher's features of the scan, of the annotator's erased crop and of NAFNet's
(``ERASED_VIEWS``), and the student's of the scan, for the window and for the line.  The tokens the strike touches: where the
teacher's merged features of the scan and of the clean crop have cosine below .99.  Per region and level, the mean cosine to the
clean crop's features over those tokens of the scan (raw), NAFNet's crop and the student (DIST/distances.jsonl).  The student sees
the scan, or with ``--scan-pack`` that pack's ``erased`` window and ``context`` line (what it was taught on).

``agree`` (the fifth round's check 2, as revised: plan, section six, second revision; OUT.json): 2a the features the training
path (``train_reader --view-students``) gives a batch of the regions of the other folds, a few of each, each through its own fold's
adapter, bit for bit the same as a pass written apart over the very batch with each adapter switched on by hand (the first image:
all adapters off); 2b with the reader in training mode, every module of the tower and the adapters out of training and the same
batch twice the same; 2c one fold's regions through its adapter by the training model and by the reading ``Reader``: per level the
mean token cosine at least that of one image alone against in a batch (0.9988, 0.9999, 0.9995, 0.9994), the lowest and the count
below .999 reported.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .models.internvl import processing
from .util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

TOUCHED = .99
DISCRIMINATOR_LR, ADVERSARIAL_WEIGHT = 1e-6, .1      # MAA's values (its train_maa.py defaults)
TARGETS = r'.*(blocks|encoder\.layers)\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2|fc1|fc2))'     # Qwen's blocks, InternViT's layers


def vision_tower(model_dir):
    from .models.internvl import is_internvl, tower
    if is_internvl(model_dir):                                           # InternVL3.5: its vision model and mlp1 (models.internvl.tower)
        return tower(model_dir)
    return qwen_tower(model_dir)


def qwen_tower(model_dir):
    """The vision tower of a Qwen3-VL or Qwen3.5 checkpoint alone (bf16 weights, sdpa), its patch embedding run as a matrix product (the
    two towers are built alike; Qwen3.5's has no deepstack levels, so its features are the merged tokens alone).  Its buffers
    stay as built, as the whole model keeps them: the rotary ``inv_freq`` in float32 (a cast of the tower to bf16 coarsened the
    positions of the fourth and fifth rounds' adapters; sixth round, section 4.1)."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from .models.vlm_tokens import patch_embed_as_linear
    whole = AutoConfig.from_pretrained(str(model_dir)); config = whole.vision_config; config._attn_implementation = 'sdpa'
    if whole.model_type == 'qwen3_5':
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel as Tower
    else:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel as Tower
    tower = Tower._from_config(config, dtype=torch.bfloat16)
    state = {}
    for path in sorted(Path(model_dir).glob('*.safetensors')):
        with safe_open(str(path), 'pt') as f:
            for key in f.keys():
                if key.startswith('model.visual.'): state[key[len('model.visual.'):]] = f.get_tensor(key)
    missing, unexpected = tower.load_state_dict(state, strict=False)
    require(not missing and not unexpected, f'vision tower weights: missing {missing[:3]}, unexpected {unexpected[:3]}')
    patch_embed_as_linear(tower.patch_embed)
    return tower


def with_student(tower, rank=16, alpha=32, dropout=.05, adapter=None):
    """The tower with a low-rank adapter on its blocks' linear layers (new, or ``adapter`` loaded): a peft model whose
    ``disable_adapter()`` is the teacher."""
    from peft import LoraConfig, PeftModel, get_peft_model
    if adapter:
        return PeftModel.from_pretrained(tower, str(adapter))
    return get_peft_model(tower, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias='none', target_modules=TARGETS))


def levels(tower, pixel_values, grid):
    """[merged, deepstack 1..3] (Qwen3.5: [merged]): each [tokens, width] float32."""
    from .models.vlm_tokens import tower_levels
    merged, deep = tower_levels(tower(pixel_values, grid_thw=grid))
    return [merged.float(), *(d.float() for d in deep)]


def encode(processor, images, device):
    import torch
    batch = processor.image_processor(images=images, return_tensors='pt')
    return batch['pixel_values'].to(device, torch.bfloat16), batch['image_grid_thw'].to(device)


class Totals:
    """Per level: the sum of 1 - cosine and the token count, the squared error and the teacher's square."""

    def __init__(self, n):
        self.cos, self.tokens, self.sq, self.t2 = [0.]*n, [0]*n, [0.]*n, [0.]*n

    def add(self, student, teacher):
        import torch
        for k, (s, t) in enumerate(zip(student, teacher)):
            self.cos[k] += float((1-torch.nn.functional.cosine_similarity(s, t, dim=-1)).sum()); self.tokens[k] += s.shape[0]
            self.sq[k] += float(((s-t)**2).sum()); self.t2[k] += float((t**2).sum())

    def loss(self):
        return float(np.mean([c/max(1, n)+q/max(1e-12, t) for c, n, q, t in zip(self.cos, self.tokens, self.sq, self.t2)]))


def pair_loss(student, teacher):
    import torch
    parts = [(1-torch.nn.functional.cosine_similarity(s, t, dim=-1)).mean()+((s-t)**2).mean()/(t**2).mean().clamp_min(1e-12) for s, t in zip(student, teacher)]
    return torch.stack(parts).mean()


def regions_of(pack, scan_pack, keep, window_only=False):
    """[(id, writer, [(what the student sees, the clean twin) image paths of the window and the line])] of the pack's rows kept: the
    scan pack's erased window and context line against the pack's; ``window_only``: the window's pair alone."""
    scans = {r['id']: r for r in json.loads((scan_pack/'pack.json').read_text())['rows']}
    out = []
    for r in json.loads((pack/'pack.json').read_text())['rows']:
        if not keep(r): continue
        q = scans[r['id']]
        pairs = [(scan_pack/q['images']['erased'], pack/r['images']['erased'])]
        out.append((r['id'], r['writer'], pairs if window_only else pairs+[(scan_pack/q['images']['context'], pack/r['images']['context'])]))
    return out


class TokenDiscriminator:
    """MAA's token discriminator (Li et al., ICML 2026, ``TokenDiscriminator`` of its public code): a token -> 2048 (LeakyReLU 0.2) ->
    five residual blocks (Linear, LayerNorm, LeakyReLU 0.2, Linear) -> one logit; float32."""

    def __new__(cls, width, hidden=2048, blocks=5):
        import torch
        from torch import nn

        class Block(nn.Module):
            def __init__(self):
                super().__init__(); self.fc1, self.norm, self.act, self.fc2 = nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(.2), nn.Linear(hidden, hidden)

            def forward(self, x):
                return x+self.fc2(self.act(self.norm(self.fc1(x))))
        return nn.Sequential(nn.Linear(width, hidden), nn.LeakyReLU(.2), *(Block() for _ in range(blocks)), nn.Linear(hidden, 1)).float()


def opened(path):
    return Image.open(path).convert('RGB')


def writer_key(writer):
    return hashlib.sha256(f'{writer}:{SEED}'.encode()).hexdigest()


def real_subset(regions, share):
    """The learning curve's writers (sixth round, section 4.3): the training writers in sha256("<writer>:42") order, taken while that
    brings their regions nearer ``share`` of all; a smaller share's writers are a prefix of a larger one's."""
    writers = sorted({w for _, w, _ in regions}, key=writer_key); target = share*len(regions); chosen, count = [], 0
    for w in writers:
        n = sum(x == w for _, x, _ in regions)
        if chosen and abs(count+n-target) >= abs(count-target):
            break
        chosen.append(w); count += n
    return chosen


def validation_writers(regions, share=.1):
    writers = sorted({w for _, w, _ in regions}, key=writer_key)
    chosen, count, need = set(), 0, share*len(regions)
    for w in writers:
        if count >= need: break
        chosen.add(w); count += sum(x == w for _, x, _ in regions)
    return chosen


def train(args):
    import torch
    from .models.vlm_tokens import processor_for
    real = regions_of(args.pack, args.scan_pack, lambda r: r['fold'] != args.fold, args.window_only)
    require(not (args.window_only and (args.synthetic or args.covered)), '--window-only goes with real pairs only')
    held = validation_writers(real); val = [r for r in real if r[1] in held]; real = [r for r in real if r[1] not in held]
    subset = real_subset(real, args.real_share) if args.real_share < 1 else sorted({w for _, w, _ in real}, key=writer_key)
    everyone = len(real); real = [r for r in real if r[1] in set(subset)]
    covered = regions_of(args.covered/f'fold_{args.fold}_clean', args.covered/f'fold_{args.fold}_raw', lambda r: True) if args.covered else []
    require(not covered or args.synthetic, '--covered goes with --synthetic')
    synthetic = []
    if args.synthetic:
        # the clean pack: erased the clean window, context the line with the clean one; the raw pack: erased the synthetic window,
        # context the line with the synthetic window
        synthetic = regions_of(args.synthetic/f'fold_{args.fold}_clean', args.synthetic/f'fold_{args.fold}_raw', lambda r: True)
    out = make_output_dir(args.output_dir)
    torch.manual_seed(SEED); rng = random.Random(SEED)
    processor = processor_for(args.model)
    model = with_student(vision_tower(args.model).to('cuda'), args.rank, args.alpha, args.dropout)
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0., betas=(.9, .999))
    judge = None
    if args.discriminator:                       # built without touching the adapter's random streams (sixth round's update, section two)
        state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        judge = TokenDiscriminator(model.base_model.model.config.out_hidden_size).to("cuda")
        judge_optimizer = torch.optim.AdamW(judge.parameters(), lr=args.discriminator_lr, weight_decay=0.)
        torch.set_rng_state(state[0]); torch.cuda.set_rng_state_all(state[1])
        bce = lambda logits, label: torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.full_like(logits, label))
    schedule = lambda step: min(1., (step+1)/args.warmup)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in (args.pack/'pack.json', args.scan_pack/'pack.json',
                   *((args.synthetic/f'fold_{args.fold}_raw'/'pack.json',) if args.synthetic else ()))},
                   real_training=len(real), synthetic_training=len(synthetic), validation=len(val), validation_writers=sorted(held),
                   trainable_parameters=sum(q.numel() for q in params), targets=TARGETS, model_processing=processing(args.model),
                   lora_modules=sorted({n.split('.lora_')[0].rsplit('.', 1)[-1] for n, q in model.named_parameters() if q.requires_grad and '.lora_' in n}), features='merged + deepstack (all four levels; Qwen3.5: the merged tokens alone)',
                   synthetic_share=(.25 if covered else .5) if synthetic else 0., covered_training=len(covered), covered_share=.25 if covered else 0.,
                   real_share=args.real_share, real_regions_all=everyone, real_writers=subset, real_regions_kept=len(real), window_only=args.window_only,
                   **({'discriminator': {'structure': 'token -> 2048 LeakyReLU(0.2) -> 5 residual blocks -> 1 (MAA TokenDiscriminator), float32',
                                         'level': 'merged tokens', 'lr': args.discriminator_lr, 'weight': ADVERSARIAL_WEIGHT, 'loss': '0.5 (BCE(real, 1) + BCE(fake, 0)); adapter + weight x BCE(fake, 1)',
                                         'parameters': sum(q.numel() for q in judge.parameters()), 'saved': 'never: it exists only while training'}} if judge is not None else {}))

    def pairs_of(chosen):
        scans, cleans = [], []
        for _, _, ps in chosen:
            for scan, clean in ps: scans.append(opened(scan)); cleans.append(opened(clean))
        return scans, cleans

    def forward(scans, cleans):
        xs, gs = encode(processor, scans, 'cuda'); xc, gc = encode(processor, cleans, 'cuda')
        require(torch.equal(gs, gc), 'a scan and its clean twin give different grids')
        with torch.no_grad(), model.disable_adapter():
            teacher = levels(model, xc, gc)
        student = levels(model, xs, gs)
        require(all(s.shape == t.shape for s, t in zip(student, teacher)), 'token counts differ')
        return student, teacher

    def validate():
        model.eval(); totals = Totals(1+len(getattr(model.base_model.model.config, 'deepstack_visual_indexes', None) or ()))
        with torch.no_grad():
            for start in range(0, len(val), args.batch_size):
                student, teacher = forward(*pairs_of(val[start:start+args.batch_size])); totals.add(student, teacher)
        model.train(); return totals.loss()
    stream = (out/'train_log.jsonl').open('x', encoding='utf-8'); started = time.monotonic(); best = (math.inf, None)
    first = validate(); stream.write(json.dumps({'step': 0, 'val_loss': first})+'\n'); stream.flush(); print(json.dumps({'step': 0, 'val_loss': first}), flush=True)
    model.train()
    for step in range(args.stop_after or args.steps):
        for group in optimizer.param_groups: group['lr'] = args.lr*schedule(step)
        if covered:                                # real 1/2, the earlier synthetic 1/4, the covered 1/4 (sixth round, section 4.2)
            drawn = [(lambda u: 'covered' if u >= .75 else 'synthetic' if u >= .5 else 'real')(rng.random()) for _ in range(args.batch_size)]
            chosen = [rng.choice({'real': real, 'synthetic': synthetic, 'covered': covered}[d]) for d in drawn]
        else:                                      # as in the fourth round (the random stream unchanged)
            drawn = [('synthetic' if bool(synthetic) and rng.random() < .5 else 'real') for _ in range(args.batch_size)]
            chosen = [rng.choice(synthetic) if d == 'synthetic' else rng.choice(real) for d in drawn]
        student, teacher = forward(*pairs_of(chosen))
        extra = {}
        if judge is not None:                    # MAA's order: the discriminator first, then the adapter fooling it
            judged = .5*(bce(judge(teacher[0].detach()), 1.)+bce(judge(student[0].detach()), 0.)); judged.backward()
            torch.nn.utils.clip_grad_norm_(judge.parameters(), 1.); judge_optimizer.step(); judge_optimizer.zero_grad(set_to_none=True)
            for q in judge.parameters(): q.requires_grad_(False)
            align = pair_loss(student, teacher); fooled = bce(judge(student[0]), 1.); loss = align+ADVERSARIAL_WEIGHT*fooled
            for q in judge.parameters(): q.requires_grad_(True)
            extra = {'align': float(align), 'adversarial': float(fooled), 'discriminator': float(judged)}
        else:
            loss = pair_loss(student, teacher)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        row = {'step': step+1, 'loss': float(loss), 'lr': optimizer.param_groups[0]['lr'], 'synthetic': drawn.count('synthetic'), 'covered': drawn.count('covered'),
               **extra, 'seconds': round(time.monotonic()-started, 1)}
        if (step+1) % args.eval_every == 0 or step+1 == args.steps:
            row['val_loss'] = validate()
            if row['val_loss'] < best[0]:
                best = (row['val_loss'], step+1)
                if (out/'student').exists():
                    import shutil; shutil.rmtree(out/'student')
                model.save_pretrained(out/'student')
        stream.write(json.dumps(row)+'\n'); stream.flush(); print(json.dumps(row), flush=True)
    stream.close()
    write_json(out/'complete.json', {'steps': args.steps, 'seconds': round(time.monotonic()-started, 1), 'initial_val_loss': first,
                                     'kept_step': best[1], 'kept_val_loss': best[0] if best[1] else None})


def distance(args):
    import torch
    from .models.vlm_tokens import processor_for
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows'] if r['fold'] == args.fold}
    raw = {r['id']: r for r in json.loads((args.raw_pack/'pack.json').read_text())['rows']}
    erased = {r['id']: r for r in json.loads((args.erased_pack/'pack.json').read_text())['rows']}
    scans = {r['id']: r for r in json.loads((args.scan_pack/'pack.json').read_text())['rows']} if args.scan_pack else None
    out = make_output_dir(args.output_dir)
    processor = processor_for(args.model)
    model = with_student(vision_tower(args.model).to('cuda'), adapter=args.student).eval()
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in (args.pack/'pack.json', args.raw_pack/'pack.json', args.erased_pack/'pack.json',
                   *((args.scan_pack/'pack.json',) if args.scan_pack else ()), *sorted(Path(args.student).glob('*.safetensors')))}, touched_below=TOUCHED)
    names = ('merged', 'deepstack1', 'deepstack2', 'deepstack3')
    with (out/'distances.jsonl').open('x', encoding='utf-8') as stream, torch.no_grad():
        for k, r in sorted(rows.items()):
            row = {'id': k, 'writer': r['writer'], 'fold': r['fold'], 'readable': r['readable'], 'mode': r['mode']}
            views = {'window': (args.pack/r['images']['raw'], args.pack/r['images']['erased'], args.erased_pack/erased[k]['images']['erased']),
                     'line': (args.raw_pack/raw[k]['images']['context'], args.pack/r['images']['context'], args.erased_pack/erased[k]['images']['context'])}
            seen = {'window': args.scan_pack/scans[k]['images']['erased'], 'line': args.scan_pack/scans[k]['images']['context']} if scans else {}
            for view, (scan, clean, naf) in views.items():
                x, g = encode(processor, [opened(scan), opened(clean), opened(naf)], 'cuda')
                require(len({tuple(t) for t in g.tolist()}) == 1, f'{k}: the three crops give different grids')
                with model.disable_adapter():
                    t = levels(model, x, g)
                if view in seen:
                    xs, gs = encode(processor, [opened(seen[view])], 'cuda')
                    require(torch.equal(gs, g[:1]), f'{k}: what the student sees has another grid')
                    s = levels(model, xs, gs)
                else:
                    s = levels(model, x[:x.shape[0]//3], g[:1])
                n = t[0].shape[0]//3
                split = lambda f: (f[:n], f[n:2*n], f[2*n:])
                cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
                t_raw, t_clean, _ = split(t[0]); touched = cos(t_raw, t_clean) < TOUCHED
                entry = {'tokens': int(n), 'touched': int(touched.sum())}
                for name, tl, sl in zip(names, t, s):
                    a, c, f = split(tl)
                    entry[name] = ({'raw': float(cos(a, c)[touched].mean()), 'nafnet': float(cos(f, c)[touched].mean()), 'student': float(cos(sl, c)[touched].mean())}
                                   if touched.any() else None)
                row[view] = entry
            stream.write(json.dumps(row)+'\n')
    print(json.dumps({'regions': len(rows)}), flush=True)


def agree(args):
    """The fifth round's check 2 as revised (plan, section six, second revision; see the module's words)."""
    import torch
    from .models.vlm_tokens import processor_for
    from .models.vlm_tokens import Reader, install_students
    from .recognize_lines import image_keys, prompt
    from .train_reader import Collate, PackSamples, lora_model
    names = ('merged', 'deepstack1', 'deepstack2', 'deepstack3'); places = (2, 3); keys = image_keys('context_revision')
    rows = [r for r in json.loads((args.pack/'pack.json').read_text())['rows'] if r['fold'] != args.fold]
    folds = sorted({r['fold'] for r in rows}); per = max(1, args.regions//len(folds))
    mixed = [r for j in folds for r in sorted((r for r in rows if r['fold'] == j), key=lambda r: r['id'])[:per]]
    one = sorted((r for r in rows if r['fold'] == folds[0]), key=lambda r: r['id'])[:args.regions]
    processor = processor_for(args.model); processor.tokenizer.padding_side = 'right'
    adapters = {f'fold{j}': Path(args.students.format(fold=j)) for j in folds}
    collate = Collate(processor, prompt('context_revision', {'boxes': [], 'problem': ''}), places)
    batch_of = lambda chosen: collate([PackSamples(args.pack, chosen, keys, condition='context_revision', latex=True)[i] for i in range(len(chosen))])
    model = lora_model(args.model, 16, 32, .05).to('cuda')
    inner = model.get_base_model().model; students = install_students(inner, adapters)
    model.train()                                                       # where train_reader puts the reader before its steps
    plain = type(inner).get_image_features                              # the model's own, without the adapters' routing
    from .models.vlm_tokens import image_levels as split
    same = lambda a, b: all(torch.equal(x, y) for x, y in zip(a, b))
    result = {}
    # 2b: the tower and the adapters in eval while the reader trains; the same batch twice gives the same features
    modules = list(inner.visual.modules())
    result['2b'] = {'modules': len(modules), 'training': sum(m.training for m in modules), 'reader_modules_training': sum(m.training for m in model.modules())}
    batch = batch_of(mixed); flags = batch.pop('students')
    x, g = batch['pixel_values'].to('cuda', torch.bfloat16), batch['image_grid_thw'].to('cuda')
    students.flags = flags; first = split(inner.get_image_features(x, g)); second = split(inner.get_image_features(x, g))
    result['2b']['student_images_same_twice'] = all(same(a, b) for a, b, f in zip(first, second, flags) if f)
    result['2b']['passed'] = result['2b']['training'] == 0 and result['2b']['student_images_same_twice']
    # 2a: against a pass written apart -- each adapter switched on by hand over the very batch the training path feeds the tower
    reference = {}
    with torch.no_grad():
        with students.view.disable_adapter():
            reference[None] = split(plain(inner, x, g))
        for name in sorted({f for f in flags if f}):
            students.view.set_adapter(name, inference_mode=True)
            reference[name] = split(plain(inner, x, g))
    per_image = [{'id': mixed[i//len(keys)]['id'], 'image': i % len(keys)+1, 'adapter': f, 'bitwise': same(first[i], reference[f][i])} for i, f in enumerate(flags)]
    result['2a'] = {'regions': len(mixed), 'folds': {f'fold{j}': sum(r['fold'] == j for r in mixed) for j in folds}, 'images': per_image,
                    'passed': all(e['bitwise'] for e in per_image) and len({f for f in flags if f}) == len(folds)}
    # 2c: one fold's regions through its adapter, by the training model and by the reading Reader
    batch = batch_of(one); flags = batch.pop('students'); x, g = batch['pixel_values'].to('cuda', torch.bfloat16), batch['image_grid_thw'].to('cuda')
    students.flags = flags; taught = [[t.cpu() for t in image] for image in split(inner.get_image_features(x, g))]
    import gc
    del model, inner, students, modules, first, second, reference, batch; gc.collect(); torch.cuda.empty_cache()
    reader = Reader(args.model, adapter=args.reader_adapter, student=adapters[f'fold{folds[0]}'])
    reader.students.flags = ['default' if f else None for f in flags]
    read = split(reader.model.model.get_image_features(x, g))
    bars = dict(zip(names, (.9988, .9999, .9995, .9994))); levels_ = {}
    for k, name in enumerate(names):
        c = torch.cat([torch.nn.functional.cosine_similarity(a[k].float(), b[k].cpu().float(), dim=-1) for a, b in zip(taught, read)])
        levels_[name] = {'mean': float(c.mean()), 'min': float(c.min()), 'below_0.999': int((c < .999).sum()), 'tokens': int(c.numel()), 'bar': bars[name],
                         'bitwise': all(torch.equal(a[k], b[k].cpu()) for a, b in zip(taught, read))}
    result['2c'] = {'regions': len(one), 'fold': folds[0], 'levels': levels_, 'passed': all(v['mean'] >= v['bar'] for v in levels_.values())}
    result['passed'] = all(result[part]['passed'] for part in ('2a', '2b', '2c'))
    require(not args.output.exists(), f'{args.output} exists'); write_json(args.output, result)
    print(json.dumps({part: result[part]['passed'] for part in ('2a', '2b', '2c')}), flush=True)
    require(result['passed'], f'check 2 failed: {args.output}')


def subsets(args):
    rows = json.loads((args.pack/'pack.json').read_text())['rows']; out = {}
    for k in sorted({r['fold'] for r in rows}):
        real = regions_of(args.pack, args.scan_pack, lambda r: r['fold'] != k)
        held = validation_writers(real); train_ = [r for r in real if r[1] not in held]; fold = {'validation_writers': sorted(held), 'training_regions': len(train_)}
        for share in args.shares:
            writers = real_subset(train_, share); n = sum(r[1] in set(writers) for r in train_)
            fold[str(share)] = {'writers': writers, 'regions': n, 'target': share*len(train_), 'off_target': n/(share*len(train_))-1}
        shares = sorted(args.shares); fold['nested'] = all(set(fold[str(a)]['writers']) <= set(fold[str(b)]['writers']) for a, b in zip(shares, shares[1:]))
        out[k] = fold
    require(not args.output.exists(), f'{args.output} exists'); write_json(args.output, out)
    print(json.dumps({k: {s: v[str(s)]['regions'] for s in args.shares} | {'nested': v['nested']} for k, v in out.items()}), flush=True)


def same_tower(args):
    import torch
    from transformers import AutoModelForImageTextToText
    from .models.vlm_tokens import processor_for
    from .models.vlm_tokens import patch_embed_as_linear
    rows = json.loads((args.pack/'pack.json').read_text())['rows'][:8]
    processor = processor_for(args.model)
    x, g = encode(processor, [opened(args.pack/r['images'][k]) for r in rows for k in ('raw', 'erased', 'context')], 'cuda')
    with torch.no_grad():
        alone = levels(vision_tower(args.model).to('cuda').eval(), x, g)
        model = AutoModelForImageTextToText.from_pretrained(str(args.model), dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
        patch_embed_as_linear(model.model.visual.patch_embed)
        whole = levels(model.model.visual, x, g)
    names = ('merged', 'deepstack1', 'deepstack2', 'deepstack3'); result = {'images': int(g.shape[0]), 'levels': {}}
    for name, a, b in zip(names, alone, whole):
        c = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        result['levels'][name] = {'bitwise': bool(torch.equal(a, b)), 'mean_cosine': float(c.mean()), 'lowest_cosine': float(c.min())}
    result['passed'] = all(v['bitwise'] for v in result['levels'].values())
    require(not args.output.exists(), f'{args.output} exists'); write_json(args.output, result)
    print(json.dumps(result), flush=True)
    require(result['passed'], f'the tower alone and the whole model\'s differ: {args.output}')


def head(args):
    import torch
    from scipy.stats import pearsonr, spearmanr
    from .models.vlm_tokens import processor_for
    rows = json.loads((args.pack/'pack.json').read_text())['rows']; folds = sorted({r['fold'] for r in rows})
    adapters = {f'fold{j}': Path(args.students.format(fold=j)) for j in folds}
    out = make_output_dir(args.output_dir); torch.manual_seed(SEED)
    processor = processor_for(args.model)
    from peft import PeftModel
    tower = PeftModel.from_pretrained(vision_tower(args.model).to('cuda'), str(adapters['fold0']), adapter_name='fold0')
    for name in list(adapters)[1:]:
        tower.load_adapter(str(adapters[name]), adapter_name=name)
    tower.eval()
    write_manifest(out, config=to_jsonable(args), inputs={str(w): digest(w) for v in adapters.values() for w in sorted(v.glob('*.safetensors'))} | {str(args.pack/'pack.json'): digest(args.pack/'pack.json')},
                   input_features="the window (the pack's raw) through its own fold's adapter and through the frozen tower: merged tokens, side by side",
                   target="per token: cosine of the adapter's token to the frozen tower's token of the annotator's erased window (training only)",
                   own_fold_adapter='each region through the adapter of its own fold (never trained on it)', trained_on='for fold K, the regions of the other folds only')
    features, targets = {}, {}
    with torch.no_grad():
        for r in rows:
            x, g = encode(processor, [opened(args.pack/r['images']['raw']), opened(args.pack/r['images']['erased'])], 'cuda')
            n = int(g[0].prod())
            with tower.disable_adapter():
                frozen = levels(tower, x, g)[0]
            tower.set_adapter(f'fold{r["fold"]}')
            student = levels(tower, x[:n], g[:1])[0]
            m = student.shape[0]
            features[r['id']] = torch.cat([student, frozen[:m]], 1).half().cpu()
            targets[r['id']] = torch.nn.functional.cosine_similarity(student, frozen[m:], dim=-1).clamp(0, 1).cpu()
    width = next(iter(features.values())).shape[1]; scores = {}
    for k in folds:
        torch.manual_seed(SEED); order = random.Random(SEED)
        net = torch.nn.Sequential(torch.nn.Linear(width, 512), torch.nn.GELU(), torch.nn.Linear(512, 1)).to('cuda')
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        train_ = [r['id'] for r in rows if r['fold'] != k]
        for _ in range(30):
            order.shuffle(train_)
            for start in range(0, len(train_), 16):
                ids = train_[start:start+16]
                x = torch.cat([features[i] for i in ids]).float().cuda(); y = torch.cat([targets[i] for i in ids]).cuda()
                loss = ((torch.sigmoid(net(x)[:, 0])-y)**2).mean(); loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for r in (r for r in rows if r['fold'] == k):
                scores[r['id']] = (float(torch.sigmoid(net(features[r['id']].float().cuda())[:, 0]).mean()), float(targets[r['id']].mean()))
    with (out/'scores.jsonl').open('x', encoding='utf-8') as stream:
        for r in rows:
            stream.write(json.dumps({'id': r['id'], 'fold': r['fold'], 'writer': r['writer'], 'score': scores[r['id']][0], 'true_mean_cosine': scores[r['id']][1]})+'\n')
    corr = lambda ids: {'regions': len(ids), 'pearson': float(pearsonr(*zip(*(scores[i] for i in ids)))[0]), 'spearman': float(spearmanr(*zip(*(scores[i] for i in ids)))[0])}
    summary = {'all': corr([r['id'] for r in rows]), 'folds': {k: corr([r['id'] for r in rows if r['fold'] == k]) for k in folds}}
    write_json(out/'summary.json', summary); print(json.dumps(summary), flush=True)


def boards_distance(args):
    import torch
    from .models.vlm_tokens import processor_for
    packs = {k: {r['id']: r for r in json.loads((args.boards/k/'teacher'/'pack.json').read_text())['rows']} for k in args.kinds}
    ids = sorted(i for i, r in packs[args.kinds[0]].items() if r['fold'] == args.fold)
    out = make_output_dir(args.output_dir); processor = processor_for(args.model)
    model = with_student(vision_tower(args.model).to('cuda'), adapter=args.student).eval()
    write_manifest(out, config=to_jsonable(args), inputs={str(args.boards/k/'teacher'/'pack.json'): digest(args.boards/k/'teacher'/'pack.json') for k in args.kinds}
                   | {str(w): digest(w) for w in sorted(Path(args.student).glob('*.safetensors'))}, touched_below=TOUCHED,
                   window_tokens="the merged-token columns wholly over the window (the board pack's window_columns); the board's tokens are not counted",
                   clean="the frozen tower on the annotator's erased window with the same board the student sees")
    names = ('merged', 'deepstack1', 'deepstack2', 'deepstack3'); cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
    with (out/'distances.jsonl').open('x', encoding='utf-8') as stream, torch.no_grad():
        for i in ids:
            row = {'id': i, 'fold': args.fold}
            for k in args.kinds:
                r = packs[k][i]; row.update({'writer': r['writer'], 'readable': r['readable'], 'mode': r['mode']})
                x, g = encode(processor, [opened(r['images']['raw']), opened(r['images']['erased'])], 'cuda')
                require(len({tuple(t) for t in g.tolist()}) == 1, f'{i}: the two boards give different grids')
                with model.disable_adapter():
                    t = levels(model, x, g)
                st = levels(model, x[:x.shape[0]//2], g[:1]); n = t[0].shape[0]//2
                cols = int(g[0][2])//2; window = (torch.arange(n, device='cuda') % cols) < r['window_columns']
                touched = (cos(t[0][:n], t[0][n:]) < TOUCHED) & window
                entry = {'board_blank': r['board_blank'], 'window_tokens': int(window.sum()), 'touched': int(touched.sum())}
                for name, tl, sl in zip(names, t, st):
                    entry[name] = {'raw': float(cos(tl[:n], tl[n:])[touched].mean()), 'student': float(cos(sl, tl[n:])[touched].mean())} if touched.any() else None
                row[k] = entry
            stream.write(json.dumps(row)+'\n')
    print(json.dumps({'regions': len(ids)}), flush=True)


SPREAD = 3000


def spread(args):
    import torch
    from peft import PeftModel
    from .models.vlm_tokens import processor_for
    rows = json.loads((args.pack/'pack.json').read_text())['rows']; processor = processor_for(args.model)
    named = [s.partition('=')[::2] for s in args.student]; tower = vision_tower(args.model).to('cuda'); model = None
    for name, pattern in named:
        for k in args.folds:
            path = str(Path(pattern.format(fold=k)))
            if model is None: model = PeftModel.from_pretrained(tower, path, adapter_name=f'{name}|{k}')
            else: model.load_adapter(path, adapter_name=f'{name}|{k}')
    model.eval(); cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1); result = {'folds': {}}
    def distances(a, b, width):
        kernel = lambda x, y: torch.exp(-torch.cdist(x, y).pow(2)/(2*width**2))
        mmd = kernel(a, a).mean()+kernel(b, b).mean()-2*kernel(a, b).mean()
        energy = 2*torch.cdist(a, b).mean()-torch.cdist(a, a).mean()-torch.cdist(b, b).mean()
        return {'mmd2': float(mmd), 'energy': float(energy)}
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for k in args.folds:
            pools = {'clean': [], 'raw': [], **{name: [] for name, _ in named}}
            for r in (r for r in rows if r['fold'] == k):
                x, g = encode(processor, [opened(args.pack/r['images']['raw']), opened(args.pack/r['images']['erased'])], 'cuda'); n = int(g[0].prod())
                with model.disable_adapter():
                    merged = levels(model, x, g)[0]
                m = merged.shape[0]//2; touched = cos(merged[:m], merged[m:]) < TOUCHED
                if not touched.any(): continue
                pools['raw'].append(merged[:m][touched].cpu()); pools['clean'].append(merged[m:][touched].cpu())
                for name, _ in named:
                    model.set_adapter(f'{name}|{k}'); pools[name].append(levels(model, x[:n], g[:1])[0][touched].cpu())
            pools = {key: torch.cat(v) for key, v in pools.items()}
            pick = torch.randperm(len(pools['clean']), generator=gen)[:SPREAD]
            sample = {key: v[pick].cuda() for key, v in pools.items()}         # the same tokens in every pool (they pair one to one)
            width = float(torch.cdist(torch.cat([sample['clean'], sample['raw']]), torch.cat([sample['clean'], sample['raw']])).median())
            result['folds'][k] = {'tokens': len(pools['clean']), 'sampled': len(pick), 'width': width,
                                  **{key: distances(sample[key], sample['clean'], width) for key in pools if key != 'clean'}}
    result['mean'] = {key: {m: float(np.mean([result['folds'][k][key][m] for k in args.folds])) for m in ('mmd2', 'energy')} for key in ['raw', *(n for n, _ in named)]}
    require(not args.output.exists(), f'{args.output} exists'); write_json(args.output, result)
    print(json.dumps(result['mean']), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    for key in ('model', 'pack', 'scan-pack', 'output-dir'): t.add_argument('--'+key, type=Path, required=True)
    t.add_argument('--fold', type=int, required=True); t.add_argument('--synthetic', type=Path)
    t.add_argument('--covered', type=Path, help='a data.synth_views covered folder: real 1/2, synthetic 1/4, covered 1/4')
    t.add_argument('--real-share', type=float, default=1., help='the share of the real training regions kept (the learning curve)')
    t.add_argument('--window-only', action='store_true', help='the window pairs alone (the boards of data.hint_image)')
    t.add_argument('--discriminator', action='store_true', help="MAA's token discriminator on the merged tokens (V-判别)")
    t.add_argument('--discriminator-lr', type=float, default=DISCRIMINATOR_LR, help="the discriminator's learning rate (MAA's 1e-6 by default)")
    t.add_argument('--stop-after', type=int, help='stop after this many steps of the --steps schedule (a check)')
    t.add_argument('--steps', type=int, default=600); t.add_argument('--batch-size', type=int, default=8, help='regions a step (each a window and a line)')
    t.add_argument('--eval-every', type=int, default=50); t.add_argument('--rank', type=int, default=16); t.add_argument('--alpha', type=int, default=32)
    t.add_argument('--dropout', type=float, default=.05); t.add_argument('--lr', type=float, default=1e-4); t.add_argument('--warmup', type=int, default=30)
    d = sub.add_parser('distance')
    for key in ('model', 'student', 'pack', 'raw-pack', 'erased-pack', 'output-dir'): d.add_argument('--'+key, type=Path, required=True)
    d.add_argument('--fold', type=int, required=True); d.add_argument('--scan-pack', type=Path, help='what the student sees (default: the scan)')
    a = sub.add_parser('agree')
    for key in ('model', 'pack', 'output'): a.add_argument('--'+key, type=Path, required=True)
    a.add_argument('--students', required=True, help="a fold's adapter, {fold} for the fold"); a.add_argument('--fold', type=int, required=True)
    a.add_argument('--regions', type=int, default=8); a.add_argument('--reader-adapter', type=Path, help="the reading Reader's adapter (2c)")
    u = sub.add_parser('subsets')
    for key in ('pack', 'scan-pack', 'output'): u.add_argument('--'+key, type=Path, required=True)
    u.add_argument('--shares', type=float, nargs='+', required=True)
    w = sub.add_parser('same-tower')
    for key in ('model', 'pack', 'output'): w.add_argument('--'+key, type=Path, required=True)
    h = sub.add_parser('head')
    for key in ('model', 'pack', 'output-dir'): h.add_argument('--'+key, type=Path, required=True)
    h.add_argument('--students', required=True, help="a fold's adapter, {fold} for the fold")
    bd = sub.add_parser('boards-distance')
    for key in ('model', 'student', 'boards', 'output-dir'): bd.add_argument('--'+key, type=Path, required=True)
    bd.add_argument('--kinds', nargs='+', required=True); bd.add_argument('--fold', type=int, required=True)
    sp = sub.add_parser('spread')
    for key in ('model', 'pack', 'output'): sp.add_argument('--'+key, type=Path, required=True)
    sp.add_argument('--student', action='append', required=True, help='NAME=PATTERN, {fold} for the fold'); sp.add_argument('--folds', type=int, nargs='+', required=True)
    args = p.parse_args(argv)
    {'train': train, 'distance': distance, 'agree': agree, 'subsets': subsets, 'same-tower': same_tower, 'head': head, 'boards-distance': boards_distance, 'spread': spread}[args.command](args)


if __name__ == '__main__':
    main()
