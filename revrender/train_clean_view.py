"""Teach a reader's vision tower to see a struck crop as its clean twin: a student (the tower with a low-rank adapter) looks at the scan,
the teacher (the same tower, frozen) at the clean image, and the student is pulled to the teacher token by token (plan of
2026-11-20, 第四轮_特权视图三样试验, section 4.2).

    python -m revrender.train_clean_view train --model MODEL --pack VIEWS --scan-pack SCAN_VIEWS --fold K [--synthetic SYNTH_DIR]
        [--steps N] [--batch-size B] [--eval-every E] --output-dir OUT
    python -m revrender.train_clean_view distance --model MODEL --student OUT/student --pack VIEWS --raw-pack RAW_VIEWS
        --erased-pack ERASED_VIEWS [--scan-pack SCAN_VIEWS] --fold K --output-dir DIST
    python -m revrender.train_clean_view agree --model MODEL --pack PACK --students PATTERN --fold K [--regions N] --output OUT.json

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
line); each sample of a batch is then real or synthetic with even chances.  Every ``--eval-every`` steps the validation loss; the
adapter of the lowest is kept (OUT/student), the curve in OUT/train_log.jsonl.

``distance``: on the regions of fold K, the teacher's features of the scan, of the annotator's erased crop and of NAFNet's
(``ERASED_VIEWS``), and the student's of the scan, for the window and for the line.  The tokens the strike touches: where the
teacher's merged features of the scan and of the clean crop have cosine below .99.  Per region and level, the mean cosine to the
clean crop's features over those tokens of the scan (raw), NAFNet's crop and the student (DIST/distances.jsonl).  The student sees
the scan, or with ``--scan-pack`` that pack's ``erased`` window and ``context`` line (what it was taught on).

``agree`` (the fifth round's check 2): the features ``train_reader --view-students`` gives a training batch (the regions of the
other folds, a few of each, mixed in one batch, each through its own fold's adapter) against the reading path's (each region alone,
its images 2 and 3 flagged as ``recognize_lines --view-student`` flags them) and against the adapter as it was taught (the vision
tower alone, ``levels``): per image and level the lowest token cosine (OUT.json); below .999 fails.
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

from .util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

TOUCHED = .99
TARGETS = r'.*blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))'


def vision_tower(model_dir):
    """The vision tower of a Qwen3-VL checkpoint alone (bf16, sdpa), its patch embedding run as a matrix product."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from .models.vlm_tokens import patch_embed_as_linear
    config = AutoConfig.from_pretrained(str(model_dir)).vision_config; config._attn_implementation = 'sdpa'
    tower = Qwen3VLVisionModel._from_config(config, torch_dtype=torch.bfloat16)
    state = {}
    for path in sorted(Path(model_dir).glob('*.safetensors')):
        with safe_open(str(path), 'pt') as f:
            for key in f.keys():
                if key.startswith('model.visual.'): state[key[len('model.visual.'):]] = f.get_tensor(key)
    missing, unexpected = tower.load_state_dict(state, strict=False)
    require(not missing and not unexpected, f'vision tower weights: missing {missing[:3]}, unexpected {unexpected[:3]}')
    patch_embed_as_linear(tower.patch_embed)
    return tower.to(torch.bfloat16)


def with_student(tower, rank=16, alpha=32, dropout=.05, adapter=None):
    """The tower with a low-rank adapter on its blocks' linear layers (new, or ``adapter`` loaded): a peft model whose
    ``disable_adapter()`` is the teacher."""
    from peft import LoraConfig, PeftModel, get_peft_model
    if adapter:
        return PeftModel.from_pretrained(tower, str(adapter))
    return get_peft_model(tower, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias='none', target_modules=TARGETS))


def levels(tower, pixel_values, grid):
    """[merged, deepstack 1..3]: each [tokens, width] float32."""
    merged, deep = tower(pixel_values, grid_thw=grid)
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


def regions_of(pack, scan_pack, keep):
    """[(id, writer, [(what the student sees, the clean twin) image paths of the window and the line])] of the pack's rows kept: the
    scan pack's erased window and context line against the pack's."""
    scans = {r['id']: r for r in json.loads((scan_pack/'pack.json').read_text())['rows']}
    out = []
    for r in json.loads((pack/'pack.json').read_text())['rows']:
        if not keep(r): continue
        q = scans[r['id']]
        out.append((r['id'], r['writer'], [(scan_pack/q['images']['erased'], pack/r['images']['erased']), (scan_pack/q['images']['context'], pack/r['images']['context'])]))
    return out


def opened(path):
    return Image.open(path).convert('RGB')


def validation_writers(regions, share=.1):
    writers = sorted({w for _, w, _ in regions}, key=lambda w: hashlib.sha256(f'{w}:{SEED}'.encode()).hexdigest())
    chosen, count, need = set(), 0, share*len(regions)
    for w in writers:
        if count >= need: break
        chosen.add(w); count += sum(x == w for _, x, _ in regions)
    return chosen


def train(args):
    import torch
    from transformers import AutoProcessor
    real = regions_of(args.pack, args.scan_pack, lambda r: r['fold'] != args.fold)
    held = validation_writers(real); val = [r for r in real if r[1] in held]; real = [r for r in real if r[1] not in held]
    synthetic = []
    if args.synthetic:
        # the clean pack: erased the clean window, context the line with the clean one; the raw pack: erased the synthetic window,
        # context the line with the synthetic window
        synthetic = regions_of(args.synthetic/f'fold_{args.fold}_clean', args.synthetic/f'fold_{args.fold}_raw', lambda r: True)
    out = make_output_dir(args.output_dir)
    torch.manual_seed(SEED); rng = random.Random(SEED)
    processor = AutoProcessor.from_pretrained(str(args.model))
    model = with_student(vision_tower(args.model).to('cuda'), args.rank, args.alpha, args.dropout)
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0., betas=(.9, .999))
    schedule = lambda step: min(1., (step+1)/args.warmup)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in (args.pack/'pack.json', args.scan_pack/'pack.json',
                   *((args.synthetic/f'fold_{args.fold}_raw'/'pack.json',) if args.synthetic else ()))},
                   real_training=len(real), synthetic_training=len(synthetic), validation=len(val), validation_writers=sorted(held),
                   trainable_parameters=sum(q.numel() for q in params), targets=TARGETS, features='merged + deepstack (all four levels)',
                   synthetic_share=.5 if synthetic else 0.)

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
        model.eval(); totals = Totals(4)
        with torch.no_grad():
            for start in range(0, len(val), args.batch_size):
                student, teacher = forward(*pairs_of(val[start:start+args.batch_size])); totals.add(student, teacher)
        model.train(); return totals.loss()
    stream = (out/'train_log.jsonl').open('x', encoding='utf-8'); started = time.monotonic(); best = (math.inf, None)
    first = validate(); stream.write(json.dumps({'step': 0, 'val_loss': first})+'\n'); stream.flush(); print(json.dumps({'step': 0, 'val_loss': first}), flush=True)
    model.train()
    for step in range(args.steps):
        for group in optimizer.param_groups: group['lr'] = args.lr*schedule(step)
        drawn = [bool(synthetic) and rng.random() < .5 for _ in range(args.batch_size)]
        chosen = [rng.choice(synthetic) if d else rng.choice(real) for d in drawn]
        student, teacher = forward(*pairs_of(chosen))
        loss = pair_loss(student, teacher); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        row = {'step': step+1, 'loss': float(loss), 'lr': optimizer.param_groups[0]['lr'], 'synthetic': sum(drawn),
               'seconds': round(time.monotonic()-started, 1)}
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
                                     'kept_step': best[1], 'kept_val_loss': best[0]})


def distance(args):
    import torch
    from transformers import AutoProcessor
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows'] if r['fold'] == args.fold}
    raw = {r['id']: r for r in json.loads((args.raw_pack/'pack.json').read_text())['rows']}
    erased = {r['id']: r for r in json.loads((args.erased_pack/'pack.json').read_text())['rows']}
    scans = {r['id']: r for r in json.loads((args.scan_pack/'pack.json').read_text())['rows']} if args.scan_pack else None
    out = make_output_dir(args.output_dir)
    processor = AutoProcessor.from_pretrained(str(args.model))
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
    """The fifth round's check 2 (see the module's words)."""
    import torch
    from transformers import AutoProcessor
    from .models.vlm_tokens import install_students
    from .recognize_lines import image_keys, prompt
    from .train_reader import Collate, PackSamples, lora_model
    rows = [r for r in json.loads((args.pack/'pack.json').read_text())['rows'] if r['fold'] != args.fold]
    folds = sorted({r['fold'] for r in rows}); per = max(1, args.regions//len(folds))
    chosen = [r for j in folds for r in sorted((r for r in rows if r['fold'] == j), key=lambda r: r['id'])[:per]]
    processor = AutoProcessor.from_pretrained(str(args.model)); processor.tokenizer.padding_side = 'right'
    adapters = {f'fold{j}': Path(args.students.format(fold=j)) for j in folds}
    model = lora_model(args.model, 16, 32, .05).to('cuda'); model.train()
    inner = model.get_base_model().model; students = install_students(inner, adapters)
    keys = image_keys('context_revision'); places = (2, 3)
    samples = PackSamples(args.pack, chosen, keys, condition='context_revision', latex=True)
    batch = Collate(processor, prompt('context_revision', {'boxes': [], 'problem': ''}), places)([samples[i] for i in range(len(chosen))])
    students.flags = batch['students']
    embeds, deep = inner.get_image_features(batch['pixel_values'].to('cuda', torch.bfloat16), batch['image_grid_thw'].to('cuda'))
    sizes = [e.shape[0] for e in embeds]; training = [[e, *(d.split(sizes)[i] for d in deep)] for i, e in enumerate(embeds)]
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).min())
    names = ('merged', 'deepstack1', 'deepstack2', 'deepstack3'); result = []; tower = {}
    for n, r in enumerate(chosen):
        images = [opened(args.pack/r['images'][key]) for key in keys]
        x, g = encode(processor, images, 'cuda')
        students.flags = [f'fold{r["fold"]}' if i+1 in places else None for i in range(len(images))]
        e2, d2 = inner.get_image_features(x, g); s2 = [e.shape[0] for e in e2]
        if r['fold'] not in tower:
            tower.clear(); tower[r['fold']] = with_student(vision_tower(args.model).to('cuda'), adapter=adapters[f'fold{r["fold"]}']).eval()
        starts = np.cumsum([0, *(int(q.prod()) for q in g)])
        for i in places:
            t = training[n*len(keys)+i-1]; reading = [e2[i-1], *(d.split(s2)[i-1] for d in d2)]
            with torch.no_grad():
                taught = levels(tower[r['fold']], x[starts[i-1]:starts[i]], g[i-1:i])
            result.append({'id': r['id'], 'fold': r['fold'], 'image': i,
                           'reading': {name: cos(a, b) for name, a, b in zip(names, t, reading)},
                           'taught': {name: cos(a, b) for name, a, b in zip(names, t, taught)}})
    lowest = min(v for e in result for part in ('reading', 'taught') for v in e[part].values())
    summary = {'regions': len(chosen), 'folds': {f'fold{j}': sum(r['fold'] == j for r in chosen) for j in folds}, 'lowest_cosine': lowest,
               'passed': lowest >= .999, 'encoded': students.encoded, 'images': result}
    require(not args.output.exists(), f'{args.output} exists'); write_json(args.output, summary)
    print(json.dumps({k: summary[k] for k in ('regions', 'lowest_cosine', 'passed')}), flush=True)
    require(summary['passed'], f'the training and reading features differ (lowest cosine {lowest:.5f}): {args.output}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    for key in ('model', 'pack', 'scan-pack', 'output-dir'): t.add_argument('--'+key, type=Path, required=True)
    t.add_argument('--fold', type=int, required=True); t.add_argument('--synthetic', type=Path)
    t.add_argument('--steps', type=int, default=600); t.add_argument('--batch-size', type=int, default=8, help='regions a step (each a window and a line)')
    t.add_argument('--eval-every', type=int, default=50); t.add_argument('--rank', type=int, default=16); t.add_argument('--alpha', type=int, default=32)
    t.add_argument('--dropout', type=float, default=.05); t.add_argument('--lr', type=float, default=1e-4); t.add_argument('--warmup', type=int, default=30)
    d = sub.add_parser('distance')
    for key in ('model', 'student', 'pack', 'raw-pack', 'erased-pack', 'output-dir'): d.add_argument('--'+key, type=Path, required=True)
    d.add_argument('--fold', type=int, required=True); d.add_argument('--scan-pack', type=Path, help='what the student sees (default: the scan)')
    a = sub.add_parser('agree')
    for key in ('model', 'pack', 'output'): a.add_argument('--'+key, type=Path, required=True)
    a.add_argument('--students', required=True, help="a fold's adapter, {fold} for the fold"); a.add_argument('--fold', type=int, required=True)
    a.add_argument('--regions', type=int, default=8)
    args = p.parse_args(argv); {'train': train, 'distance': distance, 'agree': agree}[args.command](args)


if __name__ == '__main__':
    main()
