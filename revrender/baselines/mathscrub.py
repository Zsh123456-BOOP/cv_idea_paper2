"""MathScrub's component classifier (Shen et al., IEEE TLT 2026) re-implemented on ExamRev under the paper's protocol.

No author code was found (searched 2026-09-05), so this follows the paper's description:
  components  BT.601 grey, Otsu's threshold, 8-connected components; a training component is 'delete' when at least half of
              its ink lies on the gold (LABEL_FRACTION)
  classifier  the paper's ViT -- 128 px input, 16 px patches, width 160, 16 blocks, 8 heads -- on the component isolated on
              white in its tight box and resized to 128x128, with two geometry channels (log area, log longest side)
              standardised on training components; from scratch, binary focal loss (gamma 2), AdamW 1e-4 / weight decay 0.01,
              batch 32, 10 epochs of 5 warm-up then a cosine, classes sampled 1:1
  masks       each component is one scored instance.  MathScrub deletes a component whole, so its ink is the first
              conversion; the component classifier's four (holes filled, dilated 2 px, hull, box) are offered as well, and the
              epoch (6-10), the conversion and the NMS / score cut are chosen together on validation by the frontier -- best
              F1, ties fewer false positives -- as every model's operating point
  ceiling     on validation, a perfect classifier (scores = labels) under each conversion, and how many gold revisions a single
              component can match at IoU 0.5

Training lines are the component classifier's (the bundle's training split and the user-corrected rows).  Writes an archive of
the validation lines and a freeze record.  This does NOT add MathScrub's other stages (OCR, token grouping, inpainting,
synthetic strikes), use MathStrike data, read the test split, choose anything on data other than validation, train more than one
seed, overwrite an output, or contact the network.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Sequence

from ..eval import matching
from ..util import SEED, digest, input_hashes, make_output_dir, to_jsonable, write_json, write_manifest
from .common import freeze_record, read_archive, training_and_validation, write_line_archive
from .component_cls import convert

SCHEMA = 'hme-revision-mathscrub-v1'
CONVERSIONS = ('ink', 'fill', 'dil2', 'hull', 'box')
EPOCHS, WARMUP_EPOCHS, SELECT_FROM, BATCH, LR, WD, FOCAL_GAMMA = 10, 5, 6, 32, 1e-4, .01, 2.
H, W = 96, 768
WORKERS = 16
IMAGE_SIZE = 128
LABEL_FRACTION = 0.5
MODEL_CONFIG = {"image_size": 128, "patch_size": 16, "embed_dim": 160,
                "depth": 16, "num_heads": 8, "dropout": 0.1,
                "mlp_ratio": 4.0}
PAPER_URL = "https://ieeexplore.ieee.org/ielx8/4620076/11353273/11409990.pdf"
PAPER_SHA256 = "94d612de1c9a7fa626a950fe571bc29a644199af84c1f1a90f37381e5908a1ad"
ADAPTATIONS = {
    "status": "paper_guided_task_adaptation_not_exact_reproduction",
    "official_code": "No author code repository found in narrow title/author search or author dataset README on 2026-09-05; not proof of absence.",
    "binarization": "Per-image Otsu on rounded BT.601 grayscale; dark foreground <= threshold; constant image foreground iff intensity <128.",
    "connectivity": 8,
    "label_mapping": "delete iff corrected training GT occupies >=0.5 of CC foreground pixels; retain otherwise; mixed CCs remain in training.",
    "crop": "Isolate original CC foreground on white, crop tight bbox, bilinear-resize to 128x128; no grouping context.",
    "geometry": "log1p(original CC foreground area), log1p(max original bbox side); population mean/std on training components only, std floor 1e-6.",
    "fusion": "Three independent 1-channel patch embeddings, additive learned channel/spatial embeddings, concatenate 3x64 tokens plus one CLS.",
    "transformer_unspecified_choices": "Pre-LN, GELU, MLP ratio 4, learned position embeddings, standard torch random initialization, LayerNorm+linear binary head.",
    "pretraining": "None; no classifier pretrained checkpoint is identified in paper.",
    "class_balance": "Inverse-class-frequency replacement sampler with epoch length equal to number of training CCs; expected 1:1 sampling.",
    "augmentation": "No added synthetic scratches or external MathStrike data; same project training images as other task baselines.",
    "selection": "Common validation component F1 (baselines.native.metrics) under clean-FP<=0.15, epochs 6-10 eligible; paper instead early-stops on CC validation accuracy.",
    "optimizer_unspecified_choices": "AdamW weight_decay=0.01, betas=(0.9,0.999), eps=1e-8; no focal alpha, no gradient clipping.",
    "prediction": "One scored binary mask per image CC, exactly its original foreground; no intra-CC pixel ownership, splitting, recovery, or token merging.",
    "representation_limit": "A mixed CC cannot preserve some of its foreground while deleting the rest. Ink excluded by binarization can never be predicted.",
}


class MathScrubError(RuntimeError):
    """A run does not satisfy the declared development-only contract."""


def require_(condition: bool, message: str) -> None:
    if not condition:
        raise MathScrubError(message)


def grayscale(image: Any):
    import numpy as np

    image = np.asarray(image)
    require_(image.dtype == np.uint8 and image.ndim == 3 and image.shape[0] == 3,
            "image must be uint8 RGB CHW")
    return np.rint(0.299 * image[0].astype(np.float32)
                   + 0.587 * image[1] + 0.114 * image[2]).clip(0, 255).astype(np.uint8)


def otsu_foreground(gray: Any):
    import numpy as np

    gray = np.asarray(gray)
    require_(gray.ndim == 2 and gray.dtype == np.uint8, "Otsu expects 2D uint8 grayscale")
    if int(gray.min()) == int(gray.max()):
        return gray < 128, int(gray.flat[0])
    counts = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    cumulative = np.cumsum(counts)
    weighted = np.cumsum(counts * np.arange(256))
    denominator = cumulative * (gray.size - cumulative)
    score = np.zeros(256, dtype=np.float64)
    valid = denominator > 0
    score[valid] = (weighted[-1] * cumulative[valid] - weighted[valid] * gray.size) ** 2 / denominator[valid]
    threshold = int(np.argmax(score))
    return gray <= threshold, threshold


def extract_components(record: dict, with_labels: bool = False) -> dict:
    """No GT is accessed on the prediction path. CC IDs are image-derived."""
    import numpy as np
    from scipy import ndimage

    gray = grayscale(record["image"])
    foreground, threshold = otsu_foreground(gray)
    labels, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    objects = ndimage.find_objects(labels, max_label=count)
    target = None
    if with_labels:
        target = np.asarray(record["mask"])
        require_(target.dtype == np.bool_ and target.shape == gray.shape, "corrected GT must be bool HxW")
        overlap = np.bincount(labels[target], minlength=count + 1)
    components = []
    for component_id, slices in enumerate(objects, 1):
        if slices is None:
            continue
        y, x = slices
        area = int(areas[component_id])
        item = {"component_id": component_id, "bbox": (x.start, y.start, x.stop, y.stop),
                "area": area, "max_side": max(x.stop - x.start, y.stop - y.start)}
        if with_labels:
            intersection = int(overlap[component_id])
            item.update({"overlap": intersection, "gt_fraction": intersection / area,
                         "label": int(intersection / area >= LABEL_FRACTION),
                         "mixed": 0 < intersection < area})
        components.append(item)
    return {"sample_id": record["sample_id"], "gray": gray, "labels": labels,
            "components": components, "threshold": threshold}


def fit_geometry_statistics(pages: Sequence[dict]) -> dict:
    """Caller supplies training components only; no dev records are accepted."""
    import numpy as np

    values = np.asarray([[math.log1p(component["area"]), math.log1p(component["max_side"])]
                         for page in pages for component in page["components"]], dtype=np.float64)
    require_(values.size > 0, "no training foreground components")
    return {"mean": values.mean(axis=0).tolist(), "std": np.maximum(values.std(axis=0), 1e-6).tolist(),
            "count": len(values), "fitted_on": "train_components_only", "ddof": 0}


def resize_bilinear(image: Any, size: int = IMAGE_SIZE):
    """Explicit endpoint-aligned bilinear resize; no hidden image dependency."""
    import numpy as np

    height, width = image.shape
    yy, xx = np.linspace(0, height - 1, size), np.linspace(0, width - 1, size)
    y0, x0 = yy.astype(np.int64), xx.astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, height - 1), np.minimum(x0 + 1, width - 1)
    wy, wx = (yy - y0)[:, None], (xx - x0)[None, :]
    top = image[y0[:, None], x0[None, :]] * (1 - wx) + image[y0[:, None], x1[None, :]] * wx
    bottom = image[y1[:, None], x0[None, :]] * (1 - wx) + image[y1[:, None], x1[None, :]] * wx
    return ((1 - wy) * top + wy * bottom).astype(np.float32)


def build_model(config: dict | None = None):
    """Construct the declared architecture lazily; no pretrained downloads."""
    import torch
    from torch import nn

    cfg = dict(MODEL_CONFIG if config is None else config)

    class ComponentViT(nn.Module):
        def __init__(self):
            super().__init__()
            dimension, patch = cfg["embed_dim"], cfg["patch_size"]
            patches = (cfg["image_size"] // patch) ** 2
            self.patch_embeddings = nn.ModuleList([nn.Conv2d(1, dimension, patch, stride=patch) for _ in range(3)])
            self.channel_embedding = nn.Parameter(torch.empty(1, 3, 1, dimension))
            self.position_embedding = nn.Parameter(torch.empty(1, 1, patches, dimension))
            self.cls_token = nn.Parameter(torch.empty(1, 1, dimension))
            self.input_dropout = nn.Dropout(cfg["dropout"])
            self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
                d_model=dimension, nhead=cfg["num_heads"],
                dim_feedforward=int(dimension * cfg["mlp_ratio"]), dropout=cfg["dropout"],
                activation="gelu", batch_first=True, norm_first=True)
                for _ in range(cfg["depth"])])
            self.norm = nn.LayerNorm(dimension)
            self.head = nn.Linear(dimension, 1)
            for parameter in (self.channel_embedding, self.position_embedding, self.cls_token):
                nn.init.trunc_normal_(parameter, std=0.02)
            self.config = cfg

        def forward(self, inputs):
            streams = [embedding(inputs[:, index:index + 1]).flatten(2).transpose(1, 2)
                       for index, embedding in enumerate(self.patch_embeddings)]
            patches = torch.stack(streams, dim=1) + self.channel_embedding + self.position_embedding
            tokens = patches.flatten(1, 2)
            tokens = torch.cat((self.cls_token.expand(len(inputs), -1, -1), tokens), dim=1)
            tokens = self.input_dropout(tokens)
            for block in self.blocks:
                tokens = block(tokens)
            return self.head(self.norm(tokens[:, 0])).squeeze(-1)

    return ComponentViT()


def record_of(row):
    """A training or validation row as the classifier reads it: three-channel uint8 image and a boolean gold mask."""
    import numpy as np
    image = np.asarray(row['image'], np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[None], 3, axis=0)
    require_(image.shape == (3, H, W), f'unexpected line shape {image.shape}')
    return {'sample_id': row['sample_id'], 'image': image, 'mask': np.asarray(row['mask']) > 0}


def box_slices(bbox):
    x0, y0, x1, y1 = bbox
    return slice(y0, y1), slice(x0, x1)


def delivered(component, bbox, how):
    return component if how == 'ink' else convert(component, box_slices(bbox), how)


RECORDS = []


def prepare(index):
    """One line: per component the 128x128 view (channel 0 of its input), raw geometry, label; validation masks packed."""
    import numpy as np
    record, validation = RECORDS[index]
    page = extract_components(record, with_labels=True)
    views, geometry, labels, packed = [], [], [], {how: [] for how in CONVERSIONS}
    for item in page['components']:
        x0, y0, x1, y1 = item['bbox']
        owned = page['labels'][y0:y1, x0:x1] == item['component_id']
        isolated = np.where(owned, page['gray'][y0:y1, x0:x1], 255).astype(np.uint8)
        views.append((resize_bilinear(isolated)/255.).astype(np.float16))
        geometry.append((math.log1p(item['area']), math.log1p(item['max_side'])))
        labels.append(item['label'])
        if validation:
            component = page['labels'] == item['component_id']
            for how in CONVERSIONS:
                packed[how].append(np.packbits(delivered(component, item['bbox'], how).ravel()))
    sizes = [{'area': item['area'], 'max_side': item['max_side']} for item in page['components']]
    return index, views, geometry, labels, sizes, (packed if validation else None)


def gathered(records, validation):
    import multiprocessing
    import numpy as np
    global RECORDS
    RECORDS = [(r, validation) for r in records]
    with multiprocessing.get_context('fork').Pool(WORKERS) as pool:
        results = sorted(pool.imap_unordered(prepare, range(len(records)), chunksize=16), key=lambda x: x[0])
    RECORDS = []
    return {'owner': np.concatenate([np.full(len(r[1]), r[0], np.int64) for r in results]),
            'view': np.stack([v for r in results for v in r[1]]),
            'geometry': np.asarray([g for r in results for g in r[2]], np.float64),
            'target': np.asarray([t for r in results for t in r[3]], bool),
            'sizes': [s for r in results for s in r[4]],
            'packed': [r[5] for r in results] if validation else None}


def batch_tensor(torch, part, index, statistics):
    """The classifier's input stacked on the GPU: the view, then the two standardised geometry planes."""
    view = torch.from_numpy(part['view'][index].astype('float32')).cuda()
    geometry = (part['geometry'][index]-statistics['mean'])/statistics['std']
    geometry = torch.from_numpy(geometry.astype('float32')).cuda()[:, :, None, None].expand(-1, -1, *view.shape[1:])
    return torch.cat([view[:, None], geometry], 1)


def scored(torch, model, part, statistics):
    import numpy as np
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(part['target']), 512):
            index = np.arange(start, min(start+512, len(part['target'])))
            out.append(model(batch_tensor(torch, part, index, statistics)).sigmoid().float().cpu().numpy())
    model.train()
    return np.concatenate(out).astype(np.float32)


def provider(part, scores, sample_ids, how):
    import numpy as np
    by_line = {sid.split(':', 1)[1]: i for i, sid in enumerate(sample_ids)}

    def provide(line):
        i = by_line[line]
        masks = [np.unpackbits(p)[:H*W].reshape(H, W).astype(bool) for p in part['packed'][i][how]]
        return masks, np.asarray(scores[part['owner'] == i], np.float64)
    return provide


def reading(provide, order, truths, gold):
    points = matching.frontier(provide, order, truths, gold)
    return points, matching.best_point(points)


def ceiling(part, sample_ids, order, truths, gold):
    result = {}
    perfect = part['target'].astype('float32')
    for how in CONVERSIONS:
        provide = provider(part, perfect, sample_ids, how)
        reachable = 0
        for line, truth in zip(order, truths):
            masks, _ = provide(line)
            for g in truth:
                best = max([float((m & g).sum())/max(float((m | g).sum()), 1.) for m in masks] or [0.])
                reachable += best >= .5
        _, best = reading(provide, order, truths, gold)
        result[how] = {'gold_reachable_at_iou_0.5': int(reachable), 'perfect_classifier': best}
    return result


def learning_rate_factor(step, warmup, total):
    """The schedule: linear warm-up over WARMUP_EPOCHS, then a cosine to zero."""
    return (step+1)/max(1, warmup) if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1, total-warmup)))


def main(argv=None):
    global WORKERS, EPOCHS, WARMUP_EPOCHS, SELECT_FROM
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--corrected', type=Path, required=True)
    parser.add_argument('--corrected-sha', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--smoke', action='store_true', help='2 epochs on 64 training and 20 validation lines: checks the pipeline end to end')
    args = parser.parse_args(argv)
    WORKERS = args.workers
    try:
        require_(not args.output_dir.exists(), 'create-only output exists')
        bundle, train, validation, order, truths, extra_info = training_and_validation(
            args.bundle, args.bundle_sha, args.corrected, args.corrected_sha)
        if args.smoke:
            EPOCHS, WARMUP_EPOCHS, SELECT_FROM = 2, 1, 1
            train = [r for r in train if r['mask'].any()][:32]+[r for r in train if not r['mask'].any()][:32]
            validation, order, truths = validation[:20], order[:20], truths[:20]
        train_records = [record_of(dict(r, image=r['image'])) for r in train]
        dev_records = [record_of(dict(r, image=r['image'])) for r in validation]
        gold = sum(map(len, truths))
        sample_ids = [r['sample_id'] for r in dev_records]
        import numpy as np
        train = gathered(train_records, False)
        val = gathered(dev_records, True)
        statistics = fit_geometry_statistics([{'components': train['sizes']}])
        statistics = {'mean': np.asarray(statistics['mean']), 'std': np.asarray(statistics['std']), 'count': statistics['count']}
        counts = {'training_lines': len(train_records), 'training_components': int(len(train['target'])),
                  'training_positive': int(train['target'].sum()), 'validation_lines': len(dev_records),
                  'validation_components': int(len(val['target'])), 'validation_positive': int(val['target'].sum()),
                  'gold_revisions': gold, 'corrected': extra_info}
        print(json.dumps({'counts': {k: v for k, v in counts.items() if k != 'corrected'}}), flush=True)
        out = make_output_dir(args.output_dir)
        bound = ceiling(val, sample_ids, order, truths, gold)
        write_json(out/'ceiling.json', bound)
        print(json.dumps({'ceiling': {h: [b['gold_reachable_at_iou_0.5'], round(b['perfect_classifier']['f1'], 4)]
                                      for h, b in bound.items()}}), flush=True)

        import copy
        import torch
        from torch.nn import functional
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = build_model().cuda()
        optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        labels = torch.from_numpy(train['target'].astype(np.int64))
        class_counts = torch.bincount(labels, minlength=2).double()
        require_(bool((class_counts > 0).all()), 'a class is empty')
        generator = torch.Generator().manual_seed(SEED)
        n = len(labels)
        steps_per_epoch = math.ceil(n/BATCH)
        total, warmup = EPOCHS*steps_per_epoch, WARMUP_EPOCHS*steps_per_epoch
        targets = torch.from_numpy(train['target'].astype(np.float32))
        best, best_state, log, step = None, None, [], 0
        model.train()
        for epoch in range(1, EPOCHS+1):
            order_epoch = torch.multinomial(1./class_counts[labels], n, replacement=True, generator=generator).numpy()
            losses = []
            for start in range(0, n, BATCH):
                index = order_epoch[start:start+BATCH]
                for group in optimiser.param_groups:
                    group['lr'] = LR*learning_rate_factor(step, warmup, total)
                logits = model(batch_tensor(torch, train, index, statistics))
                bce = functional.binary_cross_entropy_with_logits(logits.float(), targets[index].cuda(), reduction='none')
                loss = ((1-torch.exp(-bce))**FOCAL_GAMMA*bce).mean()
                require_(bool(torch.isfinite(loss)), 'non-finite focal loss')
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                losses.append(float(loss.detach()))
                step += 1
                if step % 500 == 0:
                    print(json.dumps({'epoch': epoch, 'step': step, 'loss': round(losses[-1], 5),
                                      'lr': optimiser.param_groups[0]['lr']}), flush=True)
            entry_loss = float(np.mean(losses))
            if epoch < SELECT_FROM:
                log.append({'epoch': epoch, 'step': step, 'loss': entry_loss, 'eligible': False})
                continue
            scores = scored(torch, model, val, statistics)
            for how in CONVERSIONS:
                _, point = reading(provider(val, scores, sample_ids, how), order, truths, gold)
                entry = {'epoch': epoch, 'step': step, 'loss': entry_loss, 'eligible': True, 'conversion': how, **point}
                log.append(entry)
                print(json.dumps({'epoch': epoch, 'conversion': how, 'f1': round(point['f1'], 4),
                                  'tp_fp': [point['true_positive'], point['false_positive']]}), flush=True)
                if best is None or (point['f1'], -point['false_positive']) > (best['f1'], -best['false_positive']):
                    best = entry
                    best_state = (epoch, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))
        require_(best_state[0] == best['epoch'], 'the kept state is not the best reading')
        model.load_state_dict(best_state[1])
        checkpoint = out/'best.pth'
        torch.save({'state_dict': best_state[1], 'epoch': best['epoch'], 'conversion': best['conversion'],
                    'geometry_statistics': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in statistics.items()}},
                   checkpoint)
        scores = scored(torch, model, val, statistics)
        archive = out/'dev_instances'
        archive.mkdir()
        how = best['conversion']
        for i, record in enumerate(dev_records):
            masks = [np.unpackbits(p)[:H*W].reshape(H, W).astype(bool) for p in val['packed'][i][how]]
            write_line_archive(archive, i, record['sample_id'], masks, scores[val['owner'] == i])
        points, point = read_archive(archive, order, truths)
        require_((point['true_positive'], point['false_positive']) == (best['true_positive'], best['false_positive']),
                 'the archive does not reproduce the chosen reading')
        write_json(out/'freeze.json', freeze_record('mathscrub', args.bundle_sha, archive, point, truths, points,
                                                     corrected_sha256=args.corrected_sha, smoke_only=args.smoke, chosen_step=best['step'],
                                                     chosen_epoch=best['epoch'], checkpoint=str(checkpoint.resolve()),
                                                     checkpoint_sha256=digest(checkpoint), conversion=how))
        (out/'readings.jsonl').write_text(''.join(json.dumps(e, sort_keys=True)+'\n' for e in log))
        write_manifest(out, config=to_jsonable(args), inputs=input_hashes([args.bundle, args.corrected]), **{
            'schema': SCHEMA, 'test_touched': False, 'create_only': True, 'counts': counts, 'smoke_only': args.smoke,
            'recipe': {'label_fraction': LABEL_FRACTION, 'model': MODEL_CONFIG, 'epochs': EPOCHS, 'warmup_epochs': WARMUP_EPOCHS,
                       'select_from_epoch': SELECT_FROM, 'batch': BATCH, 'lr': LR, 'weight_decay': WD, 'focal_gamma': FOCAL_GAMMA,
                       'conversions': list(CONVERSIONS),
                       'sampling': 'inverse class frequency, with replacement, one epoch = every training component',
                       'paper_adaptations': ADAPTATIONS},
            'geometry_statistics': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in statistics.items()},
            'declared_inputs': {'bundle': args.bundle_sha, 'corrected': args.corrected_sha},
            'outputs': {name: digest(out/name) for name in ('ceiling.json', 'freeze.json', 'readings.jsonl', 'best.pth')},
            'chosen': {'epoch': best['epoch'], 'conversion': how}, 'python': platform.python_version(),
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})
    except MathScrubError as exc:
        raise SystemExit('mathscrub refused: '+str(exc))
    print(json.dumps({'epoch': best['epoch'], 'conversion': how, 'f1': round(point['f1'], 4),
                      'tp_fp': [point['true_positive'], point['false_positive']], 'pixel_iou': round(point['pixel_iou'], 4)},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
