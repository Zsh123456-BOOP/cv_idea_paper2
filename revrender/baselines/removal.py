"""Strikethrough removal, adapted: Heil et al.'s released CycleGAN generators remove strokes, and what they remove is the revision.

Heil, Vats and Hast (ICDAR 2021) train CycleGANs that map a struck-through word image to its clean version; the authors
release the code and 20 struck-to-clean generators (four variants x five runs, MIT).  Adapted to ExamRev's lines, with
nothing trained:

  input      each line cut into windows of 96 x 384 at stride 192 -- the generators' 1:4 canvas, so the authors' PadToSize
             scales a window to 256 x 1024 without distortion -- and each window put through the authors' own test
             transform as their checkpoint configs set it: RGB -> PadToSize(256, 1024, 255) -> Resize(128, 512) -> grey ->
             invert -> ToTensor
  removal    removed ink = max(0, input - output) in the inverted space, resized back to the window and averaged where
             windows overlap
  instances  the removal map thresholded at one of THRESHOLDS, closed by a square of one of CLOSINGS px, then 8-connected
             components of at least MIN_PIXELS pixels; each is one instance, scored by its mean removal
  choice     generator, threshold and closing are chosen on validation together with the score cut by the frontier (best
             F1, ties fewer false positives), as every model's operating point

Writes an archive and a freeze record, so the evaluation reads it as it reads every other model.  This does NOT train or
fine-tune a generator, read the training or the test split, use a checkpoint other than the authors' release (its SHA-256 is
checked), overwrite an output, or contact the network.  The authors' code (released with the checkpoints) is imported from
``--heil-code``.
"""
import argparse
import configparser
import io
import json
import platform
import sys
import zipfile
from pathlib import Path

from ..data.bundle import load_bundle
from ..eval import matching
from ..util import digest, input_hashes, make_output_dir, to_jsonable, write_json, write_manifest
from .common import freeze_record, read_archive, write_line_archive

SCHEMA = 'hme-revision-strikethrough-removal-v1'
RELEASE_SHA = '2d737a640c3717d9d492b69affede66c6e3935c626c3c28899f2b4fd0330c71f'
GENERATOR = 'genStrikeToClean_best_fmeasure.pth'
TRANSFORM = {'padscale': 'True', 'invert_images': 'True', 'imageheight': '128', 'imagewidth': '512',
             'padheight': '256', 'padwidth': '1024', 'model': 'dense', 'blockcount': '1'}
H, W, WINDOW, STRIDE = 96, 768, 384, 192
THRESHOLDS = (.1, .2, .3, .4, .5)
CLOSINGS = (1, 3, 7)
MIN_PIXELS = 12
WORKERS = 20


class RemovalError(RuntimeError):
    pass


def require_(condition, message):
    if not condition:
        raise RemovalError(message)


def windows():
    starts = list(range(0, W-WINDOW+1, STRIDE))
    require_(starts[-1]+WINDOW == W, 'the windows do not cover the line')
    return starts


def generators(release):
    """The release's struck-to-clean generators, by name, each with its config's transform fields checked."""
    found = {}
    with zipfile.ZipFile(release) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('/'+GENERATOR):
                continue
            parser = configparser.ConfigParser()
            parser.read_string(archive.read(name.rsplit('/', 1)[0]+'/config.cfg').decode())
            section = parser[parser.sections()[0]]
            require_(all(section.get(k) == v for k, v in TRANSFORM.items()), name+': a transform differs from the others')
            found[name.split('/')[-2]] = name
    require_(len(found) == 20, f'{len(found)} generators in the release, not 20')
    return found


def instances(removal, threshold, closing):
    """One line's delivered masks and scores from its removal map."""
    import numpy as np
    from scipy import ndimage
    binary = removal >= threshold
    if closing > 1:
        binary = ndimage.binary_closing(binary, np.ones((closing, closing), bool), border_value=0)
    labels, count = ndimage.label(binary, np.ones((3, 3), bool))
    masks, scores = [], []
    for i, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None:
            continue
        mask = labels == i
        if mask.sum() < MIN_PIXELS:
            continue
        masks.append(mask)
        scores.append(float(removal[mask].mean()))
    return masks, np.asarray(scores, np.float64)


def removal_stage(args, dev, limit=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from torchvision.transforms import Compose, Grayscale, Resize, ToTensor
    sys.path.insert(0, str(args.heil_code))
    from src.network.generators import DenseGenerator
    from src.utils import PadToSize
    transform = Compose([PadToSize(256, 1024, 255), Resize((128, 512)), Grayscale(num_output_channels=1), ImageOps.invert, ToTensor()])
    starts = windows()
    inputs = torch.stack([transform(Image.fromarray(np.ascontiguousarray(np.asarray(r['image'][0])[:, s:s+WINDOW])).convert('RGB'))
                          for r in dev for s in starts])
    require_(inputs.shape[1:] == (1, 128, 512), 'the transform did not give 1 x 128 x 512')
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    names = generators(args.release)
    if limit is not None:
        names = dict(list(names.items())[:limit])
    (args.output_dir/'removal').mkdir()
    with zipfile.ZipFile(args.release) as archive:
        for key, name in names.items():
            state = torch.load(io.BytesIO(archive.read(name)), map_location='cpu', weights_only=True)
            model = DenseGenerator(1, 1, norm_layer=torch.nn.BatchNorm2d, n_blocks=1)
            model.load_state_dict(state.get('model_state_dict', state), strict=True)
            model = model.cuda().eval()
            removed = []
            with torch.no_grad():
                for start in range(0, len(inputs), 96):
                    x = inputs[start:start+96].cuda()
                    removed.append(F.interpolate((x-model(x)).clamp_min(0), size=(H, WINDOW), mode='bilinear',
                                                 align_corners=False).squeeze(1).cpu())
            removed = torch.cat(removed).numpy().reshape(len(dev), len(starts), H, WINDOW)
            total, count = np.zeros((len(dev), H, W), np.float32), np.zeros(W, np.float32)
            for k, s in enumerate(starts):
                total[:, :, s:s+WINDOW] += removed[:, k]
                count[s:s+WINDOW] += 1
            np.save(args.output_dir/'removal'/f'{key}.npy', (total/count).astype(np.float16))
            print(json.dumps({'generator': key, 'mean_removal': float((total/count).mean())}), flush=True)
    return {'generators': sorted(names), 'windows': starts, 'torch': torch.__version__}


STATE = {}


def read_one(config):
    """The best reading of one generator x threshold x closing (the components are disjoint: NMS changes nothing)."""
    import numpy as np
    key, threshold, closing = config
    maps = np.load(STATE['dir']/f'{key}.npy', mmap_mode='r')

    def provide(line):
        return instances(np.asarray(maps[STATE['index'][line]], np.float32), threshold, closing)
    tables, _ = matching.tabulate_at(provide, STATE['order'], STATE['truths'], 1.)
    best = max(matching.sweep_every_cut(tables, STATE['gold']), key=lambda p: (p['f1'], -p['false_positive']))
    return dict(best, generator=key, threshold=threshold, closing=closing)


def select_stage(args, order, truths, dev, removal_info):
    import multiprocessing
    import numpy as np
    STATE.update(dir=args.output_dir/'removal', order=order, truths=truths, index={line: i for i, line in enumerate(order)})
    STATE['gold'] = sum(map(len, truths))
    configs = [(key, t, c) for key in removal_info['generators'] for t in THRESHOLDS for c in CLOSINGS]
    with multiprocessing.get_context('fork').Pool(WORKERS) as pool:
        readings = pool.map(read_one, configs, chunksize=1)
    best = min(readings, key=lambda r: (-r['f1'], r['false_positive'], r['generator'], r['threshold'], r['closing']))
    print(json.dumps({'chosen': {k: best[k] for k in ('generator', 'threshold', 'closing', 'f1', 'true_positive', 'false_positive')}}), flush=True)
    maps = np.load(STATE['dir']/f'{best["generator"]}.npy', mmap_mode='r')
    archive = args.output_dir/'dev_instances'
    archive.mkdir()
    for i, row in enumerate(dev):
        masks, scores = instances(np.asarray(maps[i], np.float32), best['threshold'], best['closing'])
        write_line_archive(archive, i, row['sample_id'], masks, scores)
    points, point = read_archive(archive, order, truths)
    point['mask_threshold'] = best['threshold']
    require_((point['true_positive'], point['false_positive']) == (best['true_positive'], best['false_positive']),
             'the archive does not reproduce the chosen reading')
    return readings, best, point, points, archive


def main(argv=None):
    global WORKERS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--release', type=Path, required=True, help="the authors' checkpoints.zip")
    parser.add_argument('--heil-code', type=Path, required=True, help="the authors' strikethrough_removal package directory")
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--smoke', action='store_true', help='2 generators on 20 validation lines: checks the pipeline end to end')
    args = parser.parse_args(argv)
    WORKERS = args.workers
    try:
        require_(not args.output_dir.exists(), 'create-only output exists')
        require_(digest(args.release) == RELEASE_SHA, "the checkpoint archive is not the authors' v1.0 release")
        bundle = load_bundle(args.bundle, args.bundle_sha)
        dev, order, truths = matching.validation_order(bundle['dev'])
        require_(not any(r.get('split') == 'test' for r in dev), 'a test row reached the validation split')
        if args.smoke:
            dev, order, truths = dev[:20], order[:20], truths[:20]
        out = make_output_dir(args.output_dir)
        args.output_dir = out
        removal_info = removal_stage(args, dev, limit=2 if args.smoke else None)
        readings, best, point, points, archive = select_stage(args, order, truths, dev, removal_info)
        (out/'readings.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in readings))
        write_json(out/'freeze.json', freeze_record(
            'heil_removal', args.bundle_sha, archive, point, truths, points, release_sha256=RELEASE_SHA, smoke_only=args.smoke,
            generator=best['generator'], removal_threshold=best['threshold'], closing=best['closing']) | {
            'mask_threshold': best['threshold'], 'mask_threshold_rule': 'the removal-map threshold, chosen on validation among '+str(list(THRESHOLDS))})
        per_generator = {}
        for r in readings:
            if r['generator'] not in per_generator or r['f1'] > per_generator[r['generator']]['f1']:
                per_generator[r['generator']] = {k: r[k] for k in ('f1', 'threshold', 'closing', 'true_positive', 'false_positive')}
        write_manifest(out, config=to_jsonable(args), inputs=input_hashes([args.bundle, args.release]), **{
            'schema': SCHEMA, 'test_touched': False, 'create_only': True, 'smoke_only': args.smoke,
            'recipe': {'window': [H, WINDOW], 'stride': STRIDE, 'windows': removal_info['windows'], 'transform': TRANSFORM,
                       'thresholds': list(THRESHOLDS), 'closings': list(CLOSINGS), 'min_pixels': MIN_PIXELS},
            'declared_inputs': {'bundle': args.bundle_sha, 'release': RELEASE_SHA}, 'best_per_generator': per_generator,
            'outputs': {name: digest(out/name) for name in ('freeze.json', 'readings.jsonl')},
            'python': platform.python_version(), 'torch': removal_info['torch']})
    except RemovalError as exc:
        raise SystemExit('removal refused: '+str(exc))
    print(json.dumps({'generator': best['generator'], 'threshold': best['threshold'], 'closing': best['closing'],
                      'f1': round(point['f1'], 4), 'tp_fp': [point['true_positive'], point['false_positive']],
                      'pixel_iou': round(point['pixel_iou'], 4)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
