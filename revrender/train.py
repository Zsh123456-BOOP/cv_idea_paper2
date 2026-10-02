"""Train RevRender, seed 42, under strict determinism: smoke -> replay -> train-only learning gate -> formal.

    rvr-train --stage smoke    --bundle B --bundle-sha S [--corrected-bundle C --corrected-sha T]
              --backbone-weights R-50-torchvision.pkl --output-dir OUT/smoke_a
    rvr-train --stage smoke    ... --output-dir OUT/smoke_b --replay-of OUT/smoke_a     (must equal smoke_a step by step)
    rvr-train --stage learning ... --output-dir OUT/learning                             (512 steps on 8 training lines)
    rvr-train --stage formal   ... --output-dir OUT/formal --gate-root OUT               (needs the three gates above)

The recipe is configs/revrender.toml.  The run draws batches of 4 revised + 4 clean lines from the bundle's training
split (plus the corrected rows if given), validates every ``validation_interval`` steps (every 4 x until
``select_from_step``), keeps the best checkpoint by (F1@0.5, pixel IoU) from ``select_from_step`` on, and stops early
after ``early_stop_patience`` validations in a row without a better one.  Output directories are create-only; the run
manifest records the recipe, seed, git commit and the hashes of the input files.  A formal run refuses to start unless
the smoke run, its exact replay and the learning gate passed under the same profile (configuration + code commit).
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

from .config import load_recipe
from .data.bundle import excluded_writers, load_bundle, train_pools
from .data.extra import extra_rows
from .data.sampling import BatchStream
from .eval.inloop import MASK_THRESHOLDS, evaluate, match_pairs
from .infer import BATCH_SIZE, predict_packed, save_predictions
from .util import (ROOT, SEED, RefusedError, digest, git_state, input_hashes, make_output_dir, require, safe_path,
                   to_jsonable, write_json, write_manifest)

STAGES = ('smoke', 'learning', 'formal')
CANVAS = (96, 768)


def learning_rows(rows):
    """The fixed rows of the learning gate: 2 single-revision, 2 multi-revision and 4 clean training lines."""
    single = [r for r in rows if len(r['instances']) == 1][:2]
    multi = [r for r in rows if len(r['instances']) >= 2][:2]
    clean = [r for r in rows if len(r['instances']) == 0][:4]
    require(list(map(len, (single, multi, clean))) == [2, 2, 4], 'missing TRAIN learning strata')
    return single+multi+clean


def learning_gate(initial, final, tp, fp, gold, multi_tp, clean_fired):
    return (final < .7*initial and tp >= int(np.ceil(.7*gold)) and tp/max(1, tp+fp) >= .7 and multi_tp >= 2
            and clean_fired <= 1)


def validation_is_due(step, steps, interval, start_select, multiplier):
    if step == steps:
        return True
    return step % ((multiplier if step < start_select else 1)*interval) == 0


def learning_rate_at(step, train):
    warmup, steps, rate = train['warmup_steps'], train['optimizer_steps'], train['learning_rate']
    if step <= warmup:
        return rate*step/max(1, warmup)
    progress = (step-warmup)/max(1, steps-warmup)
    return rate*(.05+.95*.5*(1+math.cos(math.pi*min(1., progress))))


def state_digest(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def training_parts(model, initial):
    return {name: {'parameters': sum(p.numel() for p in part.parameters()),
                   'all_require_grad': all(p.requires_grad for p in part.parameters()),
                   'changed': state_digest(part) != initial[name], 'sha256': state_digest(part)}
            for name, part in model.named_children()}


def device_batch(batch):
    """A prepared (images, gold blocks) batch on the GPU: images N x 1 x H x W, targets one list of masks per line."""
    import torch
    values, blocks = batch
    images = torch.from_numpy(values).to('cuda')
    targets = [list(torch.from_numpy(block).to('cuda').unbind(0)) for block in blocks]
    return images, targets


def validate_evidence(path, profile):
    """A finished gate stage: complete, not failed, same profile, files unchanged since they were hashed."""
    path = Path(path)
    c = json.loads((path/'complete.json').read_text())
    m = json.loads((path/'run_manifest.json').read_text())
    if (path/'failure.json').exists() or m['profile'] != profile or m['frozen_test_touched']:
        raise RefusedError('gate profile mismatch or failure')
    for name, sha in c['files'].items():
        if Path(name).name != name or digest(path/name) != sha:
            raise RefusedError('gate evidence changed: '+name)
    return c, json.loads((path/'summary.json').read_text())


def make_profile(torch, recipe, variant, args, configuration, determinism_description):
    """Everything that must be the same across a run's gate stages and its formal run."""
    from .models.backbone import RESNET50_SHA
    git = git_state()
    profile = {'seed': SEED, 'variant': variant, 'dataset_sha256': args.bundle_sha,
               'extra_sha256': args.corrected_sha, 'pretrained_sha256': RESNET50_SHA,
               'configuration': configuration, 'recipe': recipe, 'git_commit': git['commit'],
               'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': platform.python_version(),
               'gpu': torch.cuda.get_device_name(0), 'cudnn': torch.backends.cudnn.version(),
               'strict_determinism': True, 'autograd_multithreading': False, 'determinism': determinism_description,
               'optimizer': f"AdamW lr={recipe['train']['learning_rate']:g} wd={recipe['train']['weight_decay']:g} "
                            f"foreach=False fused=False; clip {recipe['train']['grad_clip']:g}; "
                            f"cosine over {recipe['train']['optimizer_steps']} steps",
               'environment': {n: os.environ.get(n) for n in ('CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED',
                                                                'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                                'OPENBLAS_NUM_THREADS')},
               'early_stopping': {'patience_validations': recipe['train']['early_stop_patience'],
                                  'from_step': recipe['train']['select_from_step'],
                                  'key': 'validation F1@0.50, then pixel IoU; stop only in formal'}}
    return json.loads(json.dumps(profile))        # compared after a JSON round trip: lists, not tuples


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument('--stage', choices=STAGES, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--bundle-sha', required=True)
    p.add_argument('--corrected-bundle', type=Path, help='the user-corrected rows added to the training pools')
    p.add_argument('--corrected-sha')
    p.add_argument('--backbone-weights', type=Path, required=True, help='ImageNet R-50 .pkl (R-50-torchvision.pkl)')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gate-root', type=Path, help='formal: the directory holding smoke_a, smoke_b and learning')
    p.add_argument('--replay-of', type=Path, help='smoke: an earlier smoke run this one must equal step by step')
    p.add_argument('--recipe', type=Path, help='default configs/revrender.toml')
    p.add_argument('--variant', choices=('control', 'nofine', 'nocontext', 'nodense'),
                   help="default: the recipe's [model] variant")
    p.add_argument('--allow-dirty', action='store_true', help='run from an uncommitted tree (local development only)')
    a = p.parse_args(argv)
    if (a.corrected_bundle is None) != (a.corrected_sha is None):
        p.error('--corrected-bundle and --corrected-sha go together')
    if a.stage == 'formal' and a.gate_root is None:
        p.error('formal requires completed learning and replay gates')
    if a.replay_of and a.stage != 'smoke':
        p.error('replay-of is smoke only')
    return a


def run(args):
    import torch

    from . import determinism
    from .models.backbone import RESNET50_SHA
    from .models.losses import Clipper, compute_loss, materialize
    from .models.revrender import build_model, configuration

    recipe = load_recipe(args.recipe)
    variant = args.variant or recipe['model']['variant']
    train, stages = recipe['train'], recipe['stages']
    run_steps = {'smoke': stages['smoke'], 'learning': stages['learning'], 'formal': train['optimizer_steps']}[args.stage]
    for path in (args.bundle, args.backbone_weights, args.output_dir, args.gate_root, args.replay_of, args.corrected_bundle):
        if path is not None:
            safe_path(path)
    require(not Path(args.output_dir).exists() and not Path(args.output_dir).is_symlink(),
            f'create-only output exists: {args.output_dir}', FileExistsError)
    require(not Path(args.output_dir).resolve().is_relative_to(ROOT), 'runtime outputs must be outside the source checkout')
    require(digest(args.backbone_weights) == RESNET50_SHA, 'the ImageNet ResNet-50 weights are not the declared ones')
    state = git_state()
    require(not state['dirty'] or args.allow_dirty, 'source tree is dirty: commit first (or --allow-dirty to develop)')
    determinism.configure()
    torch.autograd.set_multithreading_enabled(False)
    determinism.install()
    require(torch.cuda.is_available(), 'CUDA required')
    profile = make_profile(torch, recipe, variant, args, configuration(variant), determinism.description())
    expected = None
    if args.replay_of:
        c, _ = validate_evidence(args.replay_of, profile)
        require(c['stage'] == 'smoke' and c['steps'] == stages['smoke'], 'four-step smoke replay required')
        expected = json.loads((Path(args.replay_of)/'trace.json').read_text())
    if args.stage == 'formal':
        a, _ = validate_evidence(Path(args.gate_root)/'smoke_a', profile)
        b, replay = validate_evidence(Path(args.gate_root)/'smoke_b', profile)
        c, learn = validate_evidence(Path(args.gate_root)/'learning', profile)
        require((a['stage'], b['stage'], c['stage']) == ('smoke', 'smoke', 'learning') and replay['exact_replay']
                and learn['passed'], 'replay and training-only learning gates required')
        expected = json.loads((Path(args.gate_root)/'smoke_a'/'trace.json').read_text())

    bundle = load_bundle(args.bundle, args.bundle_sha)
    excluded = excluded_writers(args.bundle)
    extra = extra_info = None
    if args.corrected_bundle is not None:
        extra, extra_info = extra_rows(args.corrected_bundle, args.corrected_sha, excluded)
    data = train_pools(bundle, excluded, extra)
    # Seeds, then the model: the draws of the build are part of the recipe (models/legacy_init.py).
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = build_model(variant, args.backbone_weights).cuda()
    require(list(dict(model.named_children())) == ['see', 'find', 'draw'] and all(p.requires_grad for p in model.parameters()),
            'three fully trainable parts required')
    initial = {name: state_digest(part) for name, part in model.named_children()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=train['learning_rate'], weight_decay=train['weight_decay'],
                                  foreach=False, fused=False)
    clipper = Clipper(model, train['clip_groups'], train['grad_clip'])
    out = make_output_dir(args.output_dir)
    inputs = input_hashes([args.bundle, args.backbone_weights] + ([args.corrected_bundle] if extra else []))
    write_manifest(out, config=to_jsonable(args), inputs=inputs, profile=profile, stage=args.stage, arm=variant,
                   recipe=recipe, git_revision=state['commit'], command=[sys.executable, *sys.argv],
                   parameters=sum(p.numel() for p in model.parameters()), initial_part_sha256=initial,
                   full_joint_training=True, frozen_test_touched=False, extra_rows=extra_info,
                   gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                   numa_node=os.environ.get('HME_NUMA_NODE'), encoder_class=type(model.see.encoder).__name__)
    if args.stage == 'learning':
        selected = learning_rows(bundle['train'])
        dev = [{**r, 'image': r['image'][0]} for r in selected]
        fixed_images = torch.from_numpy(np.stack([r['image'] for r in dev])[:, None].astype(np.float32)/127.5-1).cuda()
        fixed_targets = [[torch.from_numpy(t.astype(np.float32)).cuda() for t in r['instances']] for r in selected]
        stream, positive_count = None, 4
    else:
        stream = BatchStream(data, random.Random(SEED), train['optimizer_steps'], train['batch_size'], train['negative_ratio'])
        dev = data['validation_positive']+data['validation_clean']
        positive_count = len(data['validation_positive'])
        if args.stage == 'smoke':
            dev = data['validation_positive'][:4]+data['validation_clean'][:4]
            positive_count = 4
    truths = [r['instances'] for r in dev]
    curve, trace, losses = [], [], []
    best_key = best_step = best_reading = final_reading = stopped_at = None
    stale = 0
    started = time.monotonic()
    if stream:
        stream.start()
    try:
        for step in range(1, run_steps+1):
            model.train()
            rate = train['learning_rate'] if args.stage == 'learning' else learning_rate_at(step, train)
            for group in optimizer.param_groups:
                group['lr'] = rate
            images, targets = device_batch(stream.next()) if stream else (fixed_images, fixed_targets)
            prediction, scores, _ = model(images, targets)
            loss, pieces = compute_loss(model, prediction, targets)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clipper()
            if step == 1:
                checks = {name: {'all_require_grad': all(p.requires_grad for p in part.parameters()),
                                 'nonzero_gradient': any(p.grad is not None and bool((p.grad != 0).any()) for p in part.parameters())}
                          for name, part in model.named_children()}
                checks['backbone'] = {'nonzero_gradient': any(p.grad is not None and bool((p.grad != 0).any())
                                                              for p in model.see.encoder.parameters())}
                if not all(v['nonzero_gradient'] for v in checks.values()):
                    raise RuntimeError('missing joint learning gradient')
                write_json(out/'gradient_check.json', checks)
            optimizer.step()
            if args.stage == 'learning':
                losses.append(loss.detach())
            due = (validation_is_due(step, train['optimizer_steps'], train['validation_interval'], train['select_from_step'],
                                     train['early_validation_multiplier']) if args.stage == 'formal' else step == run_steps)
            if due or (step <= 4 and args.stage != 'learning'):
                pieces = materialize(pieces)
            if step <= 4 and args.stage != 'learning':
                entry = {'step': step, 'loss': float(loss.detach()), 'pieces': pieces,
                         'input_sha256': hashlib.sha256(images.detach().cpu().numpy().tobytes()).hexdigest(),
                         'state_sha256': state_digest(model)}
                trace.append(entry)
                if expected is not None and entry != expected[step-1]:
                    raise RuntimeError('independent run differs at step '+str(step))
            if step == 1 or step % 64 == 0 or step == run_steps:
                progress = {'variant': variant, 'stage': args.stage, 'step': step, 'steps': run_steps,
                            'loss': float(loss.detach()), 'elapsed_seconds': time.monotonic()-started,
                            'gpu_gib': torch.cuda.max_memory_allocated()/2**30}
                (out/'progress.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress), flush=True)
            if not due:
                continue
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            bits, scores_np, seconds = predict_packed(model, dev, BATCH_SIZE)
            reading = evaluate(bits, scores_np, truths, positive_count, packed=True)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            curve.append({'step': step, 'validation': reading, 'loss_terms': pieces})
            with (out/'curve.jsonl').open('a') as f:
                f.write(json.dumps(curve[-1], allow_nan=False)+'\n')
            key = reading['instance']['0.50']['f1'], reading['pixel_iou']
            if (args.stage != 'formal' or step >= train['select_from_step']) and (best_key is None or key > best_key):
                best_key, best_step, best_reading = key, step, reading
                torch.save({'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            'step': step, 'profile': profile}, out/'best.pt')
            if args.stage == 'formal' and step >= train['select_from_step']:
                stale = 0 if best_step == step else stale+1
            if step == run_steps:
                final_reading = reading
                save_predictions(out/'final_predictions.npz', bits, scores_np, dev)
            print(json.dumps({'event': 'validation', 'step': step, 'best_step': best_step, **reading['instance']['0.50']}),
                  flush=True)
            if train['early_stop_patience'] and args.stage == 'formal' and stale >= train['early_stop_patience']:
                stopped_at = step
                print(json.dumps({'event': 'early_stop', 'step': step, 'best_step': best_step}), flush=True)
                break
        changed = training_parts(model, initial)
        require(all(v['changed'] for v in changed.values()), 'one of the three parts did not update', RuntimeError)
        checkpoint = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        bits, scores_np, seconds = predict_packed(model, dev, BATCH_SIZE)
        reading = evaluate(bits, scores_np, truths, positive_count, packed=True)
        require(reading == best_reading, 'checkpoint reload changed metrics', RuntimeError)
        save_predictions(out/'dev_predictions.npz', bits, scores_np, dev)
        summary = {'variant': variant, 'seed': SEED, 'stage': args.stage, 'steps': run_steps, 'best_step': best_step,
                   'stopped_early_at': stopped_at, 'validation': reading, 'final_validation': final_reading,
                   'late_mean_f1': float(np.mean([r['validation']['instance']['0.50']['f1'] for r in curve
                                                  if args.stage != 'formal' or r['step'] >= train['select_from_step']])),
                   'parameters': sum(p.numel() for p in model.parameters()), 'parts_after_training': changed,
                   'gpu_gib': torch.cuda.max_memory_allocated()/2**30, 'elapsed_seconds': time.monotonic()-started,
                   'checkpoint_reload_verified': True, 'exact_replay': bool(args.replay_of), 'frozen_test_touched': False}
        if args.stage == 'learning':
            losses = torch.stack(losses).cpu().tolist()
            queries = model.query_count
            binary = np.unpackbits(bits[MASK_THRESHOLDS.index(.5)], axis=-1, count=CANVAS[0]*CANVAS[1]).reshape(
                len(dev), queries, *CANVAS).astype(bool)
            tp = fp = multi_tp = clean_fired = 0
            for row, gold in enumerate(truths):
                pred = binary[row].reshape(queries, -1)
                target = gold.reshape(len(gold), CANVAS[0]*CANVAS[1])
                inter = np.stack([pred[:, t].sum(1) for t in target], 1) if len(gold) else np.zeros((queries, 0))
                iou = inter/np.maximum(1, pred.sum(1)[:, None]+target.sum(1)[None]-inter)
                keep = np.flatnonzero((scores_np[row] >= .45) & pred.any(1))
                _, found = match_pairs(iou, keep)
                tp += len(found)
                fp += len(keep)-len(found)
                multi_tp += len(found) if len(gold) >= 2 else 0
                clean_fired += int(len(keep) > 0) if not len(gold) else 0
            initial_loss, final_loss = float(np.mean(losses[:16])), float(np.mean(losses[-16:]))
            gold_total = sum(map(len, truths))
            passed = learning_gate(initial_loss, final_loss, tp, fp, gold_total, multi_tp, clean_fired)
            summary.update(initial_loss=initial_loss, final_loss=final_loss, tp=tp, fp=fp, fn=gold_total-tp,
                           multi_tp=multi_tp, clean_fired=clean_fired, passed=passed, training_only=True)
        write_json(out/'trace.json', trace)
        write_json(out/'summary.json', summary)
        if args.stage == 'learning' and not summary['passed']:
            raise RuntimeError('train-only learning gate failed; no formal training allowed')
        end = git_state()
        require(end['commit'] == state['commit'] and (end['dirty'] == state['dirty']), 'source changed during run', RuntimeError)
        write_json(out/'complete.json', {'stage': args.stage, 'steps': run_steps,
                                         'files': {p.name: digest(p) for p in out.iterdir() if p.is_file()}})
        print(json.dumps({'event': 'complete', 'variant': variant, 'stage': args.stage}), flush=True)
    except BaseException as exc:
        write_json(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        if stream:
            stream.close()


def main(argv=None):
    run(parse(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
