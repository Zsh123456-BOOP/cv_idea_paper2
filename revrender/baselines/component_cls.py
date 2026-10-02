"""Component classification, adapted: ink connected components judged revised or not by a CNN trained on our training split.

The family of Shivakumara et al. (ESWA 2022) and MathStrike (IEEE TLT 2026) binarises the page and classifies each
connected component as struck out or not.  Adapted to ExamRev's lines and to the paper's instance metric:

  components  every line binarised at grey < INK (the background sits at 248-255), 8-connected components of at least
              MIN_PIXELS pixels
  labels      a training component is positive when at least ON_GOLD of its ink lies on the line's gold; training lines
              are the bundle's training split and the user-corrected rows, as RevRender's
  classifier  ImageNet ResNet-34 on a crop of the full line height around the component (MARGIN px either side; a wider
              crop is squeezed to CROP_W, a narrower one centred on white), three channels: grey, the component, grey;
              batches half positive, AdamW 3e-4 / weight decay 1e-4, warm-up and a cosine over STEPS, random gamma, seed 42
  masks       each component is one candidate instance scored by the classifier; how a component becomes a mask is one of
              CONVERSIONS (its ink with holes filled, dilated by 2 px, its convex hull, its box).  The checkpoint (every
              EVAL_EVERY steps), the conversion and the NMS / score cut are chosen together on validation by the frontier
              -- best F1, ties fewer false positives -- as every model's operating point
  ceiling     before training, on validation: how many gold revisions any single component matches at IoU 0.5 under each
              conversion, and the reading of a perfect classifier (scores = labels)

A probe on validation found at most about 100 of the 365 gold revisions reachable by a single component (10 of 194
strike-only ones: a stroke touches the characters it crosses, so they form one component), which is why the ceiling is
recorded next to the reading.  Writes an archive of the validation lines and a freeze record, so the evaluation reads it
as it reads every other model.

This does NOT read the test split, choose anything on data other than validation, train with more than one seed,
overwrite an output, or contact the network.
"""
import argparse
import json
import platform
from pathlib import Path

from ..eval import matching
from ..util import SEED, digest, input_hashes, make_output_dir, require, to_jsonable, write_json, write_manifest
from .common import freeze_record, read_archive, training_and_validation, write_line_archive

SCHEMA = 'hme-revision-component-classification-v1'
RESNET34_SHA = 'b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f'
INK, MIN_PIXELS, ON_GOLD = 200, 12, .2
H, MARGIN, CROP_W = 96, 48, 256
CONVERSIONS = ('fill', 'dil2', 'hull', 'box')
STEPS, WARMUP, EVAL_EVERY, BATCH = 8000, 300, 2000, 128
LR, WD = 3e-4, 1e-4
GAMMA = (.7, 1.4)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
WORKERS = 16


def components(grey):
    """The line's label map and its kept components as (label, slices), in label order."""
    import numpy as np
    from scipy import ndimage
    labels, count = ndimage.label(grey < INK, np.ones((3, 3), bool))
    sizes = np.bincount(labels.ravel(), minlength=count+1)
    return labels, [(i, box) for i, box in enumerate(ndimage.find_objects(labels), 1)
                    if box is not None and sizes[i] >= MIN_PIXELS]


def crop(grey, component, box):
    """The classifier's view: full line height, MARGIN px either side, squeezed or centred to CROP_W."""
    import numpy as np
    from PIL import Image
    require(grey.shape[0] == H, f'a line is {grey.shape[0]} px high, not {H}')
    a, b = max(box[1].start-MARGIN, 0), min(box[1].stop+MARGIN, grey.shape[1])
    g, m = grey[:, a:b], component[:, a:b]
    if b-a > CROP_W:
        g = np.asarray(Image.fromarray(np.ascontiguousarray(g)).resize((CROP_W, H), Image.BILINEAR))
        m = np.asarray(Image.fromarray(np.ascontiguousarray(m).astype(np.uint8)*255).resize((CROP_W, H), Image.BILINEAR)) > 0
        return g, m
    out_g, out_m = np.full((H, CROP_W), 255, np.uint8), np.zeros((H, CROP_W), bool)
    start = (CROP_W-(b-a))//2
    out_g[:, start:start+b-a], out_m[:, start:start+b-a] = g, m
    return out_g, out_m


def convert(component, box, how):
    """One component as a delivered mask; worked on its box grown by 3 px, then placed on the line."""
    import numpy as np
    from scipy import ndimage
    require(how in CONVERSIONS, 'unknown conversion '+how)
    y0, y1 = max(box[0].start-3, 0), min(box[0].stop+3, component.shape[0])
    x0, x1 = max(box[1].start-3, 0), min(box[1].stop+3, component.shape[1])
    local = component[y0:y1, x0:x1]
    if how == 'fill':
        local = ndimage.binary_fill_holes(local)
    elif how == 'dil2':
        local = ndimage.binary_dilation(local, np.ones((5, 5), bool))
    elif how == 'hull':
        from skimage.morphology import convex_hull_image
        local = convex_hull_image(local)
    else:
        ys, xs = np.nonzero(local)
        local = np.zeros_like(local)
        local[ys.min():ys.max()+1, xs.min():xs.max()+1] = True
    mask = np.zeros(component.shape, bool)
    mask[y0:y1, x0:x1] = local
    return mask


def positive(component, gold):
    return float((component & gold).sum()) >= ON_GOLD*float(component.sum())


ROWS = []


def prepare(index):
    """Crops and labels of one line (and, for validation, every conversion packed)."""
    import numpy as np
    row = ROWS[index]
    grey = np.asarray(row['grey'])
    gold = np.asarray(row['gold'], bool)
    labels, kept = components(grey)
    crops, masks, targets, packed = [], [], [], {how: [] for how in CONVERSIONS}
    for label, box in kept:
        component = labels == label
        g, m = crop(grey, component, box)
        crops.append(g)
        masks.append(m)
        targets.append(positive(component, gold))
        if row['validation']:
            for how in CONVERSIONS:
                packed[how].append(np.packbits(convert(component, box, how).ravel()))
    return index, crops, masks, targets, (packed if row['validation'] else None)


def gathered(rows, validation):
    """Every component of these lines, in line then label order, prepared by WORKERS processes."""
    import multiprocessing
    import numpy as np
    global ROWS
    ROWS = [{'grey': r['grey'], 'gold': r['gold'], 'validation': validation} for r in rows]
    with multiprocessing.get_context('fork').Pool(WORKERS) as pool:
        results = sorted(pool.imap_unordered(prepare, range(len(rows)), chunksize=16), key=lambda x: x[0])
    ROWS = []
    owner = np.concatenate([np.full(len(r[1]), r[0], np.int64) for r in results])
    grey = np.stack([c for r in results for c in r[1]])
    mask = np.stack([c for r in results for c in r[2]])
    target = np.asarray([t for r in results for t in r[3]], bool)
    packed = [r[4] for r in results] if validation else None
    return {'owner': owner, 'grey': grey, 'mask': mask, 'target': target, 'packed': packed}


def batch_tensor(torch, part, index, gammas=None):
    grey = torch.from_numpy(part['grey'][index]).cuda().float()/255.
    if gammas is not None:
        grey = grey.clamp_min(1e-6)**torch.from_numpy(gammas).cuda().float()[:, None, None]
    mask = torch.from_numpy(part['mask'][index]).cuda().float()
    x = torch.stack([(grey-MEAN[0])/STD[0], (mask-MEAN[1])/STD[1], (grey-MEAN[2])/STD[2]], 1)
    return x.contiguous(memory_format=torch.channels_last)


def scored(torch, model, part):
    import numpy as np
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(part['target']), 512):
            index = np.arange(start, min(start+512, len(part['target'])))
            out.append(torch.sigmoid(model(batch_tensor(torch, part, index)).squeeze(1)).float().cpu().numpy())
    model.train()
    return np.concatenate(out).astype(np.float32)


def provider(part, scores, sample_ids, how):
    """The provider (eval.matching) over the validation components: one conversion, these scores."""
    import numpy as np
    by_line = {}
    for i, sid in enumerate(sample_ids):
        by_line[sid.split(':', 1)[1]] = i

    def provide(line):
        i = by_line[line]
        packed = part['packed'][i][how]
        masks = [np.unpackbits(p)[:H*768].reshape(H, 768).astype(bool) for p in packed]
        return masks, np.asarray(scores[part['owner'] == i], np.float64)
    return provide


def reading(provide, order, truths, gold):
    points = matching.frontier(provide, order, truths, gold)
    return points, matching.best_point(points)


def ceiling(part, rows, order, truths, gold):
    """Per conversion: gold revisions a single component matches at IoU 0.5, and a perfect classifier's reading."""
    import numpy as np
    result = {}
    sample_ids = [r['sample_id'] for r in rows]
    perfect = part['target'].astype(np.float32)
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


def main(argv=None):
    global WORKERS, STEPS, WARMUP, EVAL_EVERY, BATCH
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--corrected', type=Path, required=True)
    parser.add_argument('--corrected-sha', required=True)
    parser.add_argument('--resnet34', type=Path, required=True, help='torchvision ResNet-34 ImageNet weights (.pth)')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--smoke', action='store_true', help='8 steps on 64 training and 20 validation lines: checks the pipeline end to end')
    args = parser.parse_args(argv)
    WORKERS = args.workers
    require(not args.output_dir.exists(), 'create-only output exists', FileExistsError)
    require(digest(args.resnet34) == RESNET34_SHA, 'the ResNet-34 weights changed')
    import copy
    import math
    import numpy as np
    import torch
    import torchvision
    bundle, train_rows, dev_rows, order, truths, extra_info = training_and_validation(
        args.bundle, args.bundle_sha, args.corrected, args.corrected_sha)
    to_part = lambda rows: [{'sample_id': r['sample_id'], 'grey': r['image'], 'gold': r['mask']} for r in rows]
    train_rows, dev_rows = to_part(train_rows), to_part(dev_rows)
    if args.smoke:
        STEPS, WARMUP, EVAL_EVERY, BATCH = 8, 2, 4, 16
        train_rows = [r for r in train_rows if r['gold'].any()][:32]+[r for r in train_rows if not r['gold'].any()][:32]
        dev_rows, order, truths = dev_rows[:20], order[:20], truths[:20]
    gold = sum(map(len, truths))
    train = gathered(train_rows, False)
    val = gathered(dev_rows, True)
    counts = {'training_lines': len(train_rows), 'training_components': int(len(train['target'])),
              'training_positive': int(train['target'].sum()), 'validation_lines': len(dev_rows),
              'validation_components': int(len(val['target'])), 'validation_positive': int(val['target'].sum()),
              'gold_revisions': gold, 'corrected': extra_info}
    print(json.dumps({'counts': {k: v for k, v in counts.items() if k != 'corrected'}}), flush=True)
    out = make_output_dir(args.output_dir)
    bound = ceiling(val, dev_rows, order, truths, gold)
    write_json(out/'ceiling.json', bound)
    print(json.dumps({'ceiling': {h: [b['gold_reachable_at_iou_0.5'], round(b['perfect_classifier']['f1'], 4)]
                                  for h, b in bound.items()}}), flush=True)

    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = torchvision.models.resnet34()
    model.load_state_dict(torch.load(args.resnet34, map_location='cpu', weights_only=True))
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model = model.cuda().to(memory_format=torch.channels_last)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: (s+1)/WARMUP if s < WARMUP
                                                 else .5*(1+math.cos(math.pi*(s-WARMUP)/(STEPS-WARMUP))))
    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(train['target']), np.flatnonzero(~train['target'])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    sample_ids = [r['sample_id'] for r in dev_rows]
    best, best_state, log = None, None, []
    model.train()
    for step in range(1, STEPS+1):
        index = np.concatenate([rng.choice(pos, BATCH//2), rng.choice(neg, BATCH//2)])
        target = torch.from_numpy(train['target'][index].astype(np.float32)).cuda()
        logits = model(batch_tensor(torch, train, index, rng.uniform(*GAMMA, len(index)))).squeeze(1)
        loss = loss_fn(logits, target)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()
        if step % 200 == 0:
            print(json.dumps({'step': step, 'loss': round(float(loss), 4), 'lr': schedule.get_last_lr()[0]}), flush=True)
        if step % EVAL_EVERY == 0:
            scores = scored(torch, model, val)
            for how in CONVERSIONS:
                _, point = reading(provider(val, scores, sample_ids, how), order, truths, gold)
                entry = {'step': step, 'conversion': how, **point}
                log.append(entry)
                print(json.dumps({'step': step, 'conversion': how, 'f1': round(point['f1'], 4),
                                  'tp_fp': [point['true_positive'], point['false_positive']]}), flush=True)
                if best is None or (point['f1'], -point['false_positive']) > (best['f1'], -best['false_positive']):
                    best = entry
                    best_state = (step, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))
    require(best_state[0] == best['step'], 'the kept state is not the best reading')
    model.load_state_dict(best_state[1])
    checkpoint = out/'best.pth'
    torch.save({'state_dict': best_state[1], 'step': best['step'], 'conversion': best['conversion']}, checkpoint)
    scores = scored(torch, model, val)
    archive = out/'dev_instances'
    archive.mkdir()
    how = best['conversion']
    for i, row in enumerate(dev_rows):
        masks = [np.unpackbits(p)[:H*768].reshape(H, 768).astype(bool) for p in val['packed'][i][how]]
        write_line_archive(archive, i, row['sample_id'], masks, scores[val['owner'] == i])
    points, point = read_archive(archive, order, truths)
    require((point['true_positive'], point['false_positive']) == (best['true_positive'], best['false_positive']),
            'the archive does not reproduce the chosen reading')
    write_json(out/'freeze.json', freeze_record('component_cls', args.bundle_sha, archive, point, truths, points,
                                                 corrected_sha256=args.corrected_sha, smoke_only=args.smoke, chosen_step=best['step'],
                                                 checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest(checkpoint),
                                                 conversion=how))
    (out/'readings.jsonl').write_text(''.join(json.dumps(e, sort_keys=True)+'\n' for e in log))
    write_manifest(out, config=to_jsonable(args), inputs=input_hashes([args.bundle, args.corrected, args.resnet34]), **{
        'schema': SCHEMA, 'test_touched': False, 'create_only': True, 'counts': counts, 'smoke_only': args.smoke,
        'recipe': {'ink': INK, 'min_pixels': MIN_PIXELS, 'on_gold': ON_GOLD, 'crop': [H, CROP_W], 'margin': MARGIN,
                   'conversions': list(CONVERSIONS), 'steps': STEPS, 'warmup': WARMUP, 'eval_every': EVAL_EVERY,
                   'batch': BATCH, 'lr': LR, 'weight_decay': WD, 'gamma': list(GAMMA)},
        'declared_inputs': {'bundle': args.bundle_sha, 'corrected': args.corrected_sha, 'resnet34': RESNET34_SHA},
        'outputs': {name: digest(out/name) for name in ('ceiling.json', 'freeze.json', 'readings.jsonl', 'best.pth')},
        'chosen': {'step': best['step'], 'conversion': how}, 'python': platform.python_version(),
        'torch': torch.__version__, 'torchvision': torchvision.__version__, 'gpu': torch.cuda.get_device_name(0)})
    print(json.dumps({'step': best['step'], 'conversion': how, 'f1': round(point['f1'], 4),
                      'tp_fp': [point['true_positive'], point['false_positive']], 'pixel_iou': round(point['pixel_iou'], 4)},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
