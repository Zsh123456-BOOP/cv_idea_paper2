"""A separator inside the reader's training: its erased crop is made on the fly, so the reader's loss can teach it what to take away.

The separator is a restorer of ``baselines.restoration`` (NAFNet).  ``hint`` gives, for a grey crop in [0, 1], the crop it restores,
never darker than the scan (``max(scan, restored)``: a hint may only take ink away), on the white 256 canvas ``restore_native`` centres
a short region on; a crop larger than the canvas is not made here (the caller keeps the erased crop it has).  ``patches_of`` cuts a
hint into the reader's patches as the reader's image processor would -- resized (bicubic, antialiased) to the grid the processor gave
the scan, normalised, doubled in time, in the order of the merge blocks -- so the loss on the reading reaches the separator through
the hint's pixels.  The plan and its sources (a recogniser's loss on a restorer: TBSRN, TPGSR, GAN-HTR; what is different here: the
reader that teaches is the reader that reads, it sees the scan beside the hint, and the hint cannot add ink) are in the design of
2026-11-13.
"""
import argparse
import json
from pathlib import Path

from ..util import require

CANVAS = 256


def load(run, upstream):
    """(the run's selected NAFNet on the GPU, its weights file)."""
    import torch
    from ..baselines.restoration import upstream_of
    from ..baselines.restoration_torch import build
    manifest = json.loads((run/'run_manifest.json').read_text())
    require(manifest['config']['model'] == 'NAFNet', 'the separator inside the reader is a NAFNet')
    weights = run/json.loads((run/'selection.json').read_text())['checkpoint']/'weights.pt'
    network, _, _ = build('NAFNet', upstream_of(upstream, 'NAFNet'))
    network.load_state_dict(torch.load(weights, map_location='cuda', weights_only=False)['model'])
    return network, weights


def hint(network, scan):
    """The hint of a crop ([H, W] in [0, 1], on the network's device), or None when the crop does not fit the canvas."""
    import torch
    h, w = scan.shape
    if h > CANVAS or w > CANVAS:
        return None
    top, left = (CANVAS-h)//2, (CANVAS-w)//2
    canvas = torch.nn.functional.pad(scan, (left, CANVAS-w-left, top, CANVAS-h-top), value=1.)
    restored = network(canvas[None, None].repeat(1, 3, 1, 1)).mean(1)[0].clamp(0, 1)[top:top+h, left:left+w]
    return torch.maximum(scan, restored)


def patches_of(image, rows, cols, patch, merge, temporal, mean, std):
    """[rows x cols, 3 x temporal x patch x patch]: an image ([H, W] in [0, 1]) as the patches of a ``rows x cols`` grid."""
    import torch
    x = torch.nn.functional.interpolate(image[None, None], size=(rows*patch, cols*patch), mode='bicubic', antialias=True, align_corners=False)[0, 0].clamp(0, 1)
    x = torch.stack([(x-m)/s for m, s in zip(mean, std)])
    x = x[None].repeat(temporal, 1, 1, 1).reshape(temporal, 3, rows//merge, merge, patch, cols//merge, merge, patch)
    return x.permute(2, 5, 3, 6, 1, 0, 4, 7).reshape(rows*cols, 3*temporal*patch*patch)


def lay_hints(batch, network, image_processor, second=1, per_sample=2):
    """Put the separator's hint of every sample's scan in the place of its ``second`` image (of ``per_sample`` a sample) among the
    batch's patches.  ``batch`` carries ``scans`` (grey uint8 arrays) and ``anchors`` (the crop a person erased, or None), both taken
    out here; returns the mean absolute difference of the hints from the anchors (0 without any)."""
    import torch
    scans, anchors = batch.pop('scans'), batch.pop('anchors')
    grid = batch['image_grid_thw']; counts = (grid[:, 1]*grid[:, 2]).tolist(); starts = [sum(counts[:j]) for j in range(len(counts)+1)]
    require(len(counts) == per_sample*len(scans), 'not that many images a sample')
    pixels = batch['pixel_values'].float(); device = pixels.device; away, made = 0., 0
    p = image_processor
    for b, scan in enumerate(scans):
        x = torch.from_numpy(scan).to(device, torch.float32)/255
        made_hint = hint(network, x)
        if made_hint is None:
            continue
        j = per_sample*b+second
        pixels[starts[j]:starts[j+1]] = patches_of(made_hint, int(grid[j, 1]), int(grid[j, 2]), p.patch_size, p.merge_size, p.temporal_patch_size, p.image_mean, p.image_std)
        if anchors[b] is not None:
            away, made = away+(made_hint-torch.from_numpy(anchors[b]).to(device, torch.float32)/255).abs().mean(), made+1
    batch['pixel_values'] = pixels
    return away/made if made else torch.zeros((), device=device)


def main(argv=None):
    """Check, on a pack's first crops, that ``patches_of`` cuts an image as the reader's image processor does."""
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument('--model', type=Path, required=True); p.add_argument('--pack', type=Path, required=True); p.add_argument('--crops', type=int, default=16)
    args = p.parse_args(argv)
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    q = AutoProcessor.from_pretrained(str(args.model)).image_processor; worst = 0.
    for row in json.loads((args.pack/'pack.json').read_text())['rows'][:args.crops]:
        with Image.open(args.pack/row['images']['raw']) as opened:
            image = opened.convert('RGB')
        made = q(images=[image], return_tensors='pt'); rows, cols = (int(v) for v in made['image_grid_thw'][0, 1:])
        ours = patches_of(torch.from_numpy(np.asarray(image.convert('L'))).float()/255, rows, cols, q.patch_size, q.merge_size, q.temporal_patch_size, q.image_mean, q.image_std)
        require(ours.shape == made['pixel_values'].shape, f"{row['id']}: {tuple(ours.shape)} patches, the processor makes {tuple(made['pixel_values'].shape)}")
        worst = max(worst, float((ours-made['pixel_values']).abs().mean()))
    require(worst < .02, f'patches differ from the processor\'s by {worst:.4f} on average: not the same cut')
    print(json.dumps({'crops': args.crops, 'largest_mean_difference': worst}))


if __name__ == '__main__':
    main()
