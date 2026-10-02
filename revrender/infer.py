"""Inference: run a trained RevRender on lines and deliver instance masks.

Two layers:

  predict_packed(model, rows, batch_size)    exact threshold bit-masks (7 mask thresholds) and the 32 scores per line,
                                             the archive format the evaluation reads (dev_predictions.npz)
  deliver(probabilities, scores, point)      post-processing of one line at a frozen point: mask threshold, mask NMS,
                                             score cut, empty masks dropped

and a CLI that writes the validation archive of a checkpoint:

    rvr-infer --checkpoint best.pt --bundle <bundle> --bundle-sha <sha> --output-dir <new dir>

Numbers in the paper were produced under strict determinism (determinism.configure, batch size 8), which this CLI uses;
the timing of eval.speed turns it off.  Only the validation split of a bundle is ever read here.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data.bundle import load_bundle, make_rows
from .eval.inloop import MASK_THRESHOLDS
from .eval.matching import suppressed
from .util import SEED, digest, make_output_dir, require, safe_path, to_jsonable, write_json, write_manifest

BATCH_SIZE = 8
CANVAS = (96, 768)


def predict_packed(model, rows, batch_size=BATCH_SIZE):
    """Exact threshold bits (7 x N x Q x ceil(H*W/8) uint8) and scores (N x Q float32); the model is left in eval mode."""
    import torch
    pixels = CANVAS[0]*CANVAS[1]
    bits = np.empty((len(MASK_THRESHOLDS), len(rows), model.query_count, (pixels+7)//8), np.uint8)
    scores = np.empty((len(rows), model.query_count), np.float32)
    model.eval()
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            values = np.stack([r['image'] for r in rows[start:start+batch_size]])
            images = torch.from_numpy(values[:, None].astype(np.float32)/127.5-1).cuda()
            masks, objects, _ = model(images)
            probability = masks.sigmoid().cpu().numpy()
            scores[start:start+len(images)] = objects.sigmoid().cpu().numpy()
            for index, threshold in enumerate(MASK_THRESHOLDS):
                bits[index, start:start+len(images)] = np.packbits(
                    (probability >= threshold).reshape(len(images), model.query_count, -1), axis=-1)
    return bits, scores, (time.monotonic()-started)/len(rows)


def save_predictions(path, bits, scores, rows):
    np.savez_compressed(path, packed_masks=bits, scores=scores,
                        mask_shape=np.array([len(rows), 32, *CANVAS]), mask_thresholds=np.array(MASK_THRESHOLDS),
                        sample_ids=np.array([r['sample_id'] for r in rows]))


def deliver(probabilities, scores, point):
    """One line's delivered instances at a frozen point.

    ``probabilities`` are the rendered masks' sigmoid (Q x H x W); ``point`` has mask_threshold, nms and
    score_threshold (a freeze record's fields).  Returns [(bool mask, score)] in descending score order."""
    scores = np.asarray(scores, np.float64)
    masks = [p >= point['mask_threshold'] for p in probabilities]
    keep = suppressed(masks, scores, point['nms'])
    return [(masks[i], float(scores[i])) for i in keep if scores[i] >= point['score_threshold'] and masks[i].any()]


def load_checkpoint(path, variant=None, expected_sha=None):
    """A RevRender in eval mode on the GPU with the checkpoint's weights (strict).  The variant defaults to the one the
    checkpoint's profile records."""
    import torch
    from .models.revrender import build_model
    path = Path(path)
    if expected_sha is not None:
        require(digest(path) == expected_sha, 'the checkpoint is not the declared one')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    variant = variant or payload.get('profile', {}).get('variant', 'control')
    model = build_model(variant)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.cuda().eval(), payload


def validation_rows(bundle):
    """The validation rows in the order every archive uses (revised pool first, then clean)."""
    data = make_rows(bundle)
    return data['validation_positive']+data['validation_clean']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha')
    parser.add_argument('--variant', choices=('control', 'nofine', 'nocontext', 'nodense'))
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    import torch
    from . import determinism
    safe_path(args.bundle)
    safe_path(args.checkpoint)
    require(torch.cuda.is_available(), 'CUDA required')
    determinism.configure()
    determinism.install()
    torch.manual_seed(SEED)
    bundle = load_bundle(args.bundle, args.bundle_sha)
    rows = validation_rows(bundle)
    out = make_output_dir(args.output_dir)
    model, payload = load_checkpoint(args.checkpoint, args.variant, args.checkpoint_sha)
    bits, scores, seconds = predict_packed(model, rows, args.batch_size)
    save_predictions(out/'dev_predictions.npz', bits, scores, rows)
    write_manifest(out, config=to_jsonable(args),
                   inputs={str(args.checkpoint): digest(args.checkpoint), str(args.bundle): args.bundle_sha},
                   lines=len(rows), seconds_per_line=seconds, checkpoint_step=payload.get('step'),
                   frozen_test_touched=False, determinism=determinism.description())
    write_json(out/'complete.json', {'files': {p.name: digest(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'lines': len(rows), 'output': str(out/'dev_predictions.npz'), 'seconds_per_line': round(seconds, 4)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
