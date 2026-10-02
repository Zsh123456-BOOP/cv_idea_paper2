"""Train one native instance-segmentation baseline under the common protocol, seed 42.

    rvr-native-train --method maskdino --bundle B --bundle-sha S [--corrected-bundle C --corrected-sha T]
                     --upstream-root U --weights-dir W --output-dir OUT [--smoke-steps 4]

The seven native models (Mask R-CNN, PointRend, CondInst, SOLOv2, Mask2Former, MaskDINO, DI-MaskDINO) and Mask
Transfiner (``--method maskrcnn --transfiner-pin P``) run their upstream architecture and losses under one declared
task budget (configs/baselines.toml): AdamW 3e-4 / weight decay 1e-4 (transformer backbones at 0.1x, norms and biases
without decay), cosine with warm-up, gradient clip 5, effective batch 8 (4 revised + 4 clean lines, micro-batches of 2),
the same augmentation as RevRender, checkpoint chosen by (component F1, pixel F1) from step 14,150 with early stopping
after 6 validations without a better one.  Journal-run options: ``--fraction`` (nested writer subsets), ``--dense`` (the
per-pixel head of dense.py), ``--data synthetic`` (public synthetic revisions as the only positives).

This is a task adaptation using upstream architectures and losses, not a reproduction of each paper's COCO recipe.  Run it
in the native environment (detectron2 and friends); the test split is never read.
"""
import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np

from ...config import load_baselines
from ...data.augment import paired_affine
from ...data.bundle import excluded_writers, instances_from_mask, load_bundle
from ...data.extra import extra_rows
from ...data.journal import FRACTIONS, kept_writers, synthetic_positives
from ...util import (SEED, digest, git_state, input_hashes, make_output_dir, require, safe_path,
                     to_jsonable, write_json, write_manifest)
from . import metrics
from .build import METHODS, build, weights_path
from .dense import DENSE_WEIGHT, HEAD_SEED


class EarlyStop:
    """Stop once ``patience`` eligible validations in a row have not beaten the best."""

    def __init__(self, patience):
        self.patience, self.best, self.best_step, self.stale = patience, None, None, 0

    def update(self, step, key):
        if self.best is None or key > self.best:
            self.best, self.best_step, self.stale = key, step, 0
            return True
        self.stale += 1
        return False

    @property
    def stop(self):
        return self.stale >= self.patience


def learning_rate(step, total, base=3e-4, warmup=500):
    if step <= warmup:
        return base*step/warmup
    progress = min(1, (step-warmup)/max(1, total-warmup))
    return base*(.05+.95*.5*(1+math.cos(math.pi*progress)))


class Cycle:
    """Rows in a fresh random order each pass, taken one at a time."""

    def __init__(self, rows, rng):
        self.rows, self.rng, self.order = rows, rng, []

    def take(self):
        if not self.order:
            self.order = list(range(len(self.rows)))
            self.rng.shuffle(self.order)
        return self.rows[self.order.pop()]


def native_rows(rows):
    """Extra pool rows ({'fit_positive': [...], 'fit_clean': [...]}) in the native form: a three-channel strip, a boolean
    mask and the polarity."""
    out = []
    for pool in ('fit_positive', 'fit_clean'):
        for row in sorted(rows[pool], key=lambda r: r['line_id']):
            mask = np.asarray(row['mask']) > 0
            require(bool(mask.any()) == (pool == 'fit_positive'), 'an extra row disagrees with its pool')
            out.append({'sample_id': row['sample_id'], 'line_id': row['line_id'], 'group_uuid': row['group_uuid'],
                        'split': 'train', 'is_positive': pool == 'fit_positive', 'legacy_split': pool,
                        'image': np.repeat(np.asarray(row['image'], np.uint8)[None], 3, axis=0), 'mask': mask,
                        'instances': instances_from_mask(mask)})
    return out


def make_input(record, augment=False, rng=None):
    import torch
    from detectron2.structures import BitMasks, Instances

    image = np.asarray(record['image']).copy()
    mask = np.asarray(record['mask']).copy()
    if augment:
        gray, label = paired_affine(image[0], mask.astype(np.uint8)*255, rng)
        image, mask = np.repeat(gray[None], 3, axis=0), label > 0
    masks = instances_from_mask(mask)
    instance = Instances(mask.shape)
    instance.gt_masks = BitMasks(torch.from_numpy(masks.copy()))
    instance.gt_boxes = instance.gt_masks.get_bounding_boxes()
    instance.gt_classes = torch.zeros(len(masks), dtype=torch.int64)
    return {'image': torch.from_numpy(image.astype(np.float32)), 'height': mask.shape[0], 'width': mask.shape[1],
            'instances': instance}


def predict_record(model, record, device):
    import torch
    item = {'image': torch.from_numpy(np.asarray(record['image'], dtype=np.float32).copy()).to(device),
            'height': record['mask'].shape[0], 'width': record['mask'].shape[1]}
    result = model([item])[0]['instances'].to('cpu')
    return {'sample_id': record['sample_id'], 'masks': result.pred_masks.numpy().astype(bool),
            'scores': result.scores.float().numpy()}


def archive_prediction(directory, index, row, prediction):
    """One line's instances in the archive format eval.matching.native_provider reads."""
    masks = prediction['masks']
    np.savez_compressed(directory/f'{index:04d}.npz', sample_id=np.asarray(row['sample_id']),
                        scores=prediction['scores'], shape=np.asarray(masks.shape, dtype=np.int64),
                        masks_packed=np.packbits(masks.reshape(len(masks), -1), axis=1) if len(masks)
                        else np.zeros((0, math.ceil(row['mask'].size/8)), np.uint8))


def evaluate(model, rows, device, archive=None):
    import torch
    model.eval()
    prepared = []
    if archive is not None:
        archive.mkdir()
    with torch.no_grad():
        for index, row in enumerate(rows):
            prediction = predict_record(model, row, device)
            prepared.append(metrics.prepare_record(row, prediction))
            if archive is not None:
                archive_prediction(archive, index, row, prediction)
    reading = metrics.select_operating_point(prepared)
    model.train()
    return reading


def probe_native_head(model, method):
    """Smoke-only: a synthetic rectangle checks the actual head, even for very thin data gold.

    No optimizer update or reported validation metric uses these fixtures; this catches a registered-but-disconnected
    point / dynamic / mask head."""
    import torch
    positive = np.zeros((96, 768), bool)
    positive[24:64, 256:336] = True
    records = []
    for mask in (positive, np.zeros_like(positive)):
        gray = np.where(mask, 0, 255).astype(np.uint8)
        records.append({'image': np.repeat(gray[None], 3, axis=0), 'mask': mask})
    model.train()
    model.zero_grad(set_to_none=True)
    losses = model([make_input(record) for record in records])
    required = {'maskrcnn': {'loss_cls', 'loss_box_reg', 'loss_mask', 'loss_rpn_cls', 'loss_rpn_loc'},
                'solov2': {'loss_ins', 'loss_cate'},
                'dimaskdino': {'loss_ce', 'loss_mask', 'loss_dice', 'loss_bbox', 'loss_giou'},
                'mask2former': {'loss_ce', 'loss_mask', 'loss_dice'},
                'maskdino': {'loss_ce', 'loss_mask', 'loss_dice', 'loss_bbox', 'loss_giou'},
                'pointrend': {'loss_mask', 'loss_mask_point'},
                'condinst': {'loss_mask', 'loss_fcos_cls', 'loss_fcos_loc', 'loss_fcos_ctr'}}[method]
    if not required <= losses.keys() or not torch.isfinite(sum(losses.values())):
        raise RuntimeError(f'native {method} losses missing or nonfinite: {list(losses)}')
    sum(losses.values()).backward()
    tokens = {'mask2former': ('mask_embed',), 'maskdino': ('mask_embed',), 'pointrend': ('point_head',),
              'condinst': ('controller',), 'maskrcnn': ('backbone', 'proposal_generator', 'box_predictor', 'mask_head'),
              'solov2': ('backbone', 'ins_head.cate_pred', 'ins_head.kernel_pred', 'mask_head'),
              'dimaskdino': ('backbone', 'pixel_decoder', 'mask_embed', 'class_embed', 'bbox_embed', 'oldm',
                             'query_guide_feature_update')}[method]
    gradients = {name: float(value.grad.norm()) for name, value in model.named_parameters()
                 if any(token in name for token in tokens) and value.grad is not None}
    for token in tokens:
        values = [value for name, value in gradients.items() if token in name]
        if not values or not all(math.isfinite(value) for value in values) or not any(value > 0 for value in values):
            raise RuntimeError(f'native {method} component {token} has no finite nonzero gradient')
    result = {'smoke_only': True, 'fixture': 'one synthetic rectangle and one clean image', 'optimizer_updates': 0,
              'losses': {k: float(v.detach()) for k, v in losses.items()}, 'native_head_gradients': gradients}
    model.zero_grad(set_to_none=True)
    return result


def training_event_storage():
    from detectron2.utils.events import EventStorage
    return EventStorage(0)


def write_json_replace(path, value):
    """Progress files that grow during a run (the curve): written next to the target, then moved over it."""
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--bundle-sha', required=True)
    p.add_argument('--corrected-bundle', type=Path)
    p.add_argument('--corrected-sha')
    p.add_argument('--upstream-root', type=Path, required=True)
    p.add_argument('--weights-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--microbatch', type=int, default=2)
    p.add_argument('--smoke-steps', type=int, default=0, help='0 for a formal run, 4 for the smoke run')
    p.add_argument('--transfiner-pin', type=Path, help='Mask Transfiner: SHA-256 record of the authors\' files')
    p.add_argument('--fraction', type=float, default=1., choices=FRACTIONS)
    p.add_argument('--dense', action='store_true')
    p.add_argument('--data', choices=('corrected', 'synthetic'), default='corrected')
    p.add_argument('--synthetic-root', type=Path)
    p.add_argument('--allow-dirty', action='store_true')
    return p


def training_rows(args, bundle, extra, held):
    """The bundle's training lines, the extra rows, and the journal options applied in that order."""
    rows, record = list(bundle['train']), {'fraction': args.fraction, 'dense': args.dense, 'data': args.data}
    if extra is not None:
        lines = {r['line_id'] for r in bundle['train']+bundle['dev']}
        require(not any(r['line_id'] in lines for r in extra), 'an extra row is already in the bundle')
        rows = rows+extra
    if args.data == 'synthetic':
        require(args.synthetic_root is not None, '--data synthetic needs --synthetic-root')
        mapping = json.loads((args.bundle/'index_manifest.json').read_text())
        heldout = [r for r in mapping['rows'] if r['split'] in ('dev', 'test')]
        synthetic, meta = synthetic_positives(args.synthetic_root, {r['group_uuid'] for r in heldout},
                                              {r.get('line_id') for r in heldout if r.get('line_id')})
        synthetic = native_rows({'fit_positive': synthetic, 'fit_clean': []})
        clean = [r for r in rows if not r['is_positive']]
        record.update(replaced_positives=len(rows)-len(clean), synthetic=meta,
                      training_rows={'clean': len(clean), 'synthetic_positive': len(synthetic)})
        rows = clean+synthetic
    elif args.fraction < 1.:
        keep = kept_writers([r['group_uuid'] for r in rows], args.fraction)
        subset = [r for r in rows if r['group_uuid'] in keep]
        record['subset'] = {'writers_kept': len(keep), 'writers_all': len({r['group_uuid'] for r in rows}),
                            'rows_before': len(rows), 'rows_after': len(subset),
                            'positive_after': sum(r['is_positive'] for r in subset)}
        rows = subset
    require(not ({r['group_uuid'] for r in rows} & held), 'a validation or test writer is in the training rows')
    return rows, record


def run(args):
    import torch

    from . import factory
    recipe = load_baselines()
    require(args.microbatch in (1, 2, 4, 8) and args.smoke_steps in (0, recipe['smoke_steps']),
            'the protocol requires a divisible microbatch and smoke=0/4')
    require(not args.smoke_steps or args.microbatch <= 4, 'smoke microbatch<=4 is required to exercise all-clean and all-positive batches')
    require(not args.dense or args.method in ('maskdino', 'pointrend'), '--dense is declared for MaskDINO and PointRend only')
    require(args.transfiner_pin is None or args.method == 'maskrcnn', 'Transfiner builds on the Mask R-CNN')
    require(args.data == 'corrected' or not (args.dense or args.fraction < 1.), 'synthetic data is declared alone')
    require((args.corrected_bundle is None) == (args.corrected_sha is None), '--corrected-bundle and --corrected-sha go together')
    require(args.output_dir.expanduser().is_symlink() is False, 'output symlink refused')
    for path in (args.bundle, args.upstream_root, args.weights_dir, args.output_dir, args.corrected_bundle):
        if path is not None:
            safe_path(path)
    require(not args.output_dir.exists(), f'create-only run output already exists: {args.output_dir}', FileExistsError)
    state = git_state()
    require(not state['dirty'] or args.allow_dirty, 'source tree is dirty: commit first')
    bundle = load_bundle(args.bundle, args.bundle_sha)
    held = excluded_writers(args.bundle)
    extra = extra_info = None
    if args.corrected_bundle is not None:
        pool_rows, extra_info = extra_rows(args.corrected_bundle, args.corrected_sha, held)
        extra = native_rows(pool_rows)
    rows, journal = training_rows(args, bundle, extra, held)
    weight, kind = weights_path(args.method, args.weights_dir)
    require(weight.is_file(), f'missing pretrained weights: {weight}', FileNotFoundError)
    require(torch.cuda.is_available(), 'native baseline requires CUDA')
    output = make_output_dir(args.output_dir)
    torch.set_num_threads(2)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = 'cuda'
    started = time.time()
    model = build(args.method, args.upstream_root, args.weights_dir, seed=SEED, smoke=bool(args.smoke_steps),
                  transfiner_pin=args.transfiner_pin, dense=args.dense, device=device)
    config_yaml = getattr(model, 'native_config_yaml', None)
    if config_yaml:
        (output/'native_config.yaml').write_text(config_yaml)
    groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        multiplier = recipe['backbone_multiplier'] if 'backbone' in name and args.method in ('mask2former', 'maskdino', 'dimaskdino') else 1.
        no_decay = parameter.ndim == 1 or 'relative_position_bias_table' in name or 'absolute_pos_embed' in name
        groups.append({'params': [parameter], 'lr': recipe['learning_rate']*multiplier, 'lr_multiplier': multiplier,
                       'weight_decay': 0. if no_decay else recipe['weight_decay']})
    optimiser = torch.optim.AdamW(groups, lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    smoke = bool(args.smoke_steps)
    total = args.smoke_steps or recipe['total_steps']
    warmup = 1 if smoke else recipe['warmup_steps']
    select_start = 2 if smoke else recipe['select_from_step']
    patience = 1 if smoke else recipe['early_stop_patience']
    expected_components = sum(len(r['instances']) for r in bundle['dev'])
    inputs = input_hashes([args.bundle, weight] + ([args.corrected_bundle] if extra is not None else []))
    write_manifest(output, config=to_jsonable(args), inputs=inputs, method=args.method, smoke_only=smoke,
                   formal_comparison_eligible=not smoke, git_revision=state['commit'],
                   dataset_manifest_sha256=args.bundle_sha, pretraining={'kind': kind, 'path': str(weight), 'sha256': digest(weight)},
                   native_description=factory.describe_model(args.method),
                   native_metadata=getattr(model, 'baseline_metadata', {}),
                   upstream_compatibility=json.loads((args.upstream_root/'compatibility_manifest.json').read_text()),
                   corrected_rows=extra_info, journal=journal,
                   protocol={'seed': SEED, 'max_steps': total, 'recipe': recipe,
                             'early_stopping': {'from_step': select_start, 'patience_validations': patience, 'interval_steps': 500},
                             'validation_components': expected_components, 'effective_batch_size': 8, 'microbatch': args.microbatch,
                             'negative_ratio': 1, 'optimizer': 'AdamW', 'warmup_steps': warmup, 'amp': False, 'canvas': [96, 768],
                             'loss': 'native upstream instance losses including configured auxiliaries',
                             'augmentation': 'paired affine + photometry; components re-labelled after the warp',
                             'evaluation': f"all {len(bundle['dev'])} validation rows; mask=.5; shared score grid; clean FP<=.15",
                             'validation_interval_early': recipe['validation_interval_early'],
                             'validation_interval_late': recipe['validation_interval_late'], 'test_accessed': False},
                   environment_native={'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                                       'gpu': torch.cuda.get_device_name(0), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                       'parameters': sum(p.numel() for p in model.parameters())},
                   dense={'head_seed': HEAD_SEED, 'weight': DENSE_WEIGHT} if args.dense else None)
    with training_event_storage() as storage:
        if smoke:
            write_json(output/'native_head_smoke.json', probe_native_head(model, args.method))
        positives = [r for r in rows if r['is_positive']]
        clean = [r for r in rows if not r['is_positive']]
        rng = random.Random(SEED)
        pos_cycle, neg_cycle = Cycle(positives, rng), Cycle(clean, rng)
        dev = bundle['dev']
        if smoke:
            dev = [r for r in dev if r['is_positive']][:2]+[r for r in dev if not r['is_positive']][:2]
        curve, stopper, stopped_at = [], EarlyStop(patience), None
        print(json.dumps({'event': 'model_ready', 'method': args.method, 'manifest': str(output/'run_manifest.json')}), flush=True)
        for step in range(1, total+1):
            model.train()
            lr = learning_rate(step, total, base=recipe['learning_rate'], warmup=warmup)
            for group in optimiser.param_groups:
                group['lr'] = lr*group['lr_multiplier']
            batch = [pos_cycle.take() for _ in range(4)]+[neg_cycle.take() for _ in range(4)]
            if not smoke:
                rng.shuffle(batch)
            optimiser.zero_grad(set_to_none=True)
            loss_report = {}
            for start in range(0, 8, args.microbatch):
                inputs_ = [make_input(row, True, rng) for row in batch[start:start+args.microbatch]]
                losses = model(inputs_)
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite native loss at step {step}: {losses}')
                (loss*args.microbatch/8).backward()
                for key, value in losses.items():
                    loss_report[key] = loss_report.get(key, 0.)+float(value.detach())*args.microbatch/8
            if step == 1:
                gradients = {name: float(parameter.grad.detach().norm()) for name, parameter in model.named_parameters()
                             if parameter.grad is not None and parameter.requires_grad}
                if not gradients or not any(value > 0 for value in gradients.values()):
                    raise RuntimeError('no native model gradients')
                write_json(output/'first_step_gradients.json', gradients)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['grad_clip'])
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'nonfinite gradients at step {step}')
            optimiser.step()
            storage.step()
            if step == 1 or step % 100 == 0 or smoke:
                print(json.dumps({'event': 'train', 'method': args.method, 'step': step, 'loss': sum(loss_report.values()),
                                  'elapsed_seconds': round(time.time()-started, 2),
                                  'max_allocated_mib': round(torch.cuda.max_memory_allocated()/2**20)}), flush=True)
            interval = 2 if smoke else (recipe['validation_interval_early'] if step < select_start else recipe['validation_interval_late'])
            if step % interval and step != total:
                continue
            reading = evaluate(model, dev, device)
            if not smoke and (reading['rows'] != len(bundle['dev']) or reading['gold_components'] != expected_components):
                raise RuntimeError('validation coverage changed')
            eligible = step >= select_start
            curve.append({'step': step, 'checkpoint_eligible': eligible, 'loss': loss_report, 'learning_rate': lr,
                          'elapsed_seconds': time.time()-started, **reading})
            write_json_replace(output/'validation_curve.json', curve)
            if eligible and stopper.update(step, (reading['component_f1'], reading['pixel_f1'])):
                torch.save({'model': model.state_dict(), 'step': step, 'threshold': reading['threshold']},
                           output/'best_model.pth.partial')
                (output/'best_model.pth.partial').replace(output/'best_model.pth')
            print(json.dumps({'event': 'validation', 'method': args.method, 'step': step, 'component_f1': reading['component_f1'],
                              'pixel_f1': reading['pixel_f1'], 'clean_fp': reading['clean_fp'], 'threshold': reading['threshold'],
                              'best_step': stopper.best_step, 'stale': stopper.stale}), flush=True)
            if eligible and stopper.stop:
                stopped_at = step
                break
        best = torch.load(output/'best_model.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(best['model'])
        final = evaluate(model, dev, device, archive=output/'best_dev_instances')
        eligible_curve = [item for item in curve if item['checkpoint_eligible']]
        summary = {'status': 'complete', 'smoke_only': smoke, 'method': args.method, 'seed': SEED, 'max_steps': total,
                   'stopped_early_at': stopped_at, 'best_step': stopper.best_step, 'best_dev': final,
                   'late_mean_component_f1': float(np.mean([x['component_f1'] for x in eligible_curve])),
                   'max_allocated_mib': torch.cuda.max_memory_allocated()/2**20, 'elapsed_seconds': time.time()-started,
                   'formal_comparison_eligible': not smoke, 'run_manifest_sha256': digest(output/'run_manifest.json')}
        write_json(output/'summary.json', summary)
        write_json(output/('smoke_complete.json' if smoke else 'train_complete.json'), summary)
        print(json.dumps({'event': 'complete', 'method': args.method, 'best_step': stopper.best_step,
                          'stopped_early_at': stopped_at}), flush=True)


def main(argv=None):
    run(parser().parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
