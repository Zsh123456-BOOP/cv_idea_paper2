"""The scan's tokens at chosen patches replaced, after the first K blocks of Qwen3-VL's frozen vision tower, by the erased image's
tokens at the same patches and block (the premise check of 2026-11-22, 早期局部修_前提检验, revision one): ``recognize_lines
--transplant-pack PACK --transplant-after K --transplant-select struck|next|all|none [--transplant-places PACK --transplant-rule
any|brighter]``.  Per batch the frozen tower runs once on the transplant pack's images of the same jobs (in the same order, so the
token offsets match; the output is thrown away, only block K's output is kept) and once on the read pack's, a hook on block K
putting, for each image ``--student-images`` names, the companion's rows in place at its chosen patches; the later blocks, the
deepstack levels taken from block K on and the merger then run as usual.  A deepstack level taken before block K stays the scan's.

The chosen patches (one token per 16-pixel patch, in the processor's 2 x 2 merge-group order) come from a strike mask on the scan's
pixels, classed as ``classes``: struck (a mask pixel inside the source rectangle the patch is resized from), next to struck (one of its
eight neighbours struck), far.  ``struck`` takes the struck, ``next`` the struck and the next, ``all`` / ``none`` every patch / none
(the checks).  The mask (``strike_pixels``): ``any`` the pixels where the places image differs from the scan at all (the human erased
crop: its strike map), ``brighter`` the removal metrics' erased pixel (brighter than the scan by more than 0.2 on [0, 1], on ink,
grey below 200: ``data.struck_pack.fixed_pack``, ``baselines.restoration.removal_metrics``) -- a restorer's erased crop."""
import numpy as np

SELECTS = ('struck', 'next', 'all', 'none')
RULES = ('any', 'brighter')
BRIGHTER, INK = .2, 200


def classes(mask, rows, cols):
    """[rows, cols] 0 struck / 1 next / 2 far: a cell struck when the source rectangle it is resized from holds a mask pixel."""
    h0, w0 = mask.shape
    ys, xs = np.nonzero(mask)
    struck = np.zeros((rows, cols), bool)
    if len(ys):
        # a source pixel [y, y + 1) falls in the cells whose source rectangles [i h0 / rows, (i + 1) h0 / rows) it overlaps
        top = np.floor(ys*rows/h0).astype(int); bottom = np.ceil((ys+1)*rows/h0).astype(int)-1
        left = np.floor(xs*cols/w0).astype(int); right = np.ceil((xs+1)*cols/w0).astype(int)-1
        for t, b, l, r in zip(top, bottom, left, right):
            struck[t:b+1, l:r+1] = True
    near = np.zeros_like(struck)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.zeros_like(struck)
            shifted[max(dy, 0):rows+min(dy, 0), max(dx, 0):cols+min(dx, 0)] = struck[max(-dy, 0):rows+min(-dy, 0), max(-dx, 0):cols+min(-dx, 0)]
            near |= shifted
    return np.where(struck, 0, np.where(near, 1, 2))


def merge_ordered(grid, merge):
    """A [rows, cols] class grid in the tokens' 2 x 2 merge-group order (``models.maa.merge_order``)."""
    import torch
    from .maa import merge_order
    rows, cols = grid.shape
    return merge_order(torch.as_tensor(grid).reshape(1, rows, cols, 1), merge).reshape(-1).numpy()


def strike_pixels(scan, places, rule):
    """bool [H, W] of two PIL images of one size: the pixels the places image marks as the strike (see the module's words)."""
    if scan.size != places.size:
        raise ValueError(f'the places image is {places.size}, the scan {scan.size}')
    if rule == 'any':
        a, b = (np.asarray(im.convert('RGB')).astype(np.int16) for im in (scan, places))
        return (a != b).any(-1)
    if rule != 'brighter':
        raise ValueError(f'a rule is one of {RULES}')
    g, h = (np.asarray(im.convert('L'), np.float32)/255 for im in (scan, places))
    return ((h-g) > BRIGHTER) & (g < INK/255)


def patch_labels(mask, rows, cols, merge):
    """Per patch token (merge-group order): 0 struck / 1 next / 2 far."""
    return merge_ordered(classes(mask, rows, cols), merge)


def selection(labels, select):
    """bool per patch token: the patches ``select`` takes of the labels (``patch_labels``)."""
    if select not in SELECTS:
        raise ValueError(f'a selection is one of {SELECTS}')
    return {'struck': labels == 0, 'next': labels <= 1, 'all': np.ones(len(labels), bool), 'none': np.zeros(len(labels), bool)}[select]


class Transplant:
    """What ``transplanted_features`` does: after ``after`` blocks; per image of the next batch ``flags`` (True: transplanted) and
    ``selections`` (a bool per patch token, or None for an image not transplanted), ``companion`` the transplant pack's processed
    images of the same batch (pixel values, grid); ``stats`` per image of the last batch (None, or its tokens and the replaced)."""

    def __init__(self, after, blocks):
        if not 1 <= after <= blocks:
            raise ValueError(f'after 1 to {blocks} blocks, not {after}')
        self.after = after
        self.flags = self.companion = self.selections = None; self.stats = []


def token_rows(grid, flags, selections):
    """The rows of the batch's patch tokens to replace: per flagged image its selected patches, offset by the tokens before it."""
    import torch
    sizes = [int(t*h*w) for t, h, w in grid.tolist()]
    if len(flags) != len(sizes) or len(selections) != len(sizes):
        raise ValueError(f'{len(flags)} flags and {len(selections)} selections for {len(sizes)} images')
    rows, at, stats = [], 0, []
    for size, flag, chosen in zip(sizes, flags, selections):
        if flag:
            chosen = np.asarray(chosen, bool)
            if chosen.shape != (size,):
                raise ValueError(f'a selection of {chosen.shape} for an image of {size} patch tokens')
            rows.append(at+np.flatnonzero(chosen)); stats.append({'tokens': size, 'replaced': int(chosen.sum())})
        else:
            stats.append(None)
        at += size
    return torch.as_tensor(np.concatenate(rows) if rows else np.zeros(0, int), dtype=torch.long), stats


def run_with_transplant(features, visual, plant, pixel_values, grid, **kwargs):
    """``features(pixel_values, grid)`` with block ``plant.after``'s output taking the companion's rows at the chosen patches."""
    import torch
    block = visual.blocks[plant.after-1]
    pixels, companion_grid = plant.companion
    if not torch.equal(companion_grid.to(grid.device), grid):
        raise ValueError('the transplant pack gives other grids')
    rows, plant.stats = token_rows(grid, plant.flags, plant.selections)
    caught = []
    hook = block.register_forward_hook(lambda m, i, o: caught.append(o.detach()))
    try:
        with torch.no_grad():
            features(pixels.to(pixel_values.device, pixel_values.dtype), companion_grid.to(grid.device), **kwargs)
    finally:
        hook.remove()
    if len(caught) != 1:
        raise ValueError(f'block {plant.after} ran {len(caught)} times')
    rows = rows.to(caught[0].device)

    def swap(module, inputs, output):
        output = output.clone(); output[rows] = caught[0][rows]
        return output
    hook = block.register_forward_hook(swap)
    try:
        return features(pixel_values, grid, **kwargs)
    finally:
        hook.remove()


def transplanted_features(features, visual, plant):
    """``get_image_features`` of a Qwen3-VL with the images ``plant.flags`` names transplanted (see the module's words)."""
    import torch

    def chosen(pixel_values, image_grid_thw=None, **kwargs):
        with torch.no_grad():
            return run_with_transplant(features, visual, plant, pixel_values, image_grid_thw, **kwargs)
    return chosen


def install(reader, after):
    """Route a Qwen3-VL ``vlm_tokens.Reader``'s image features (no student) through a ``Transplant``, set as the reader's ``mix``;
    returns it."""
    inner = reader.model.model
    if reader.students is not None:
        raise ValueError('a transplant reads with the frozen tower, no student')
    plant = Transplant(after, len(inner.visual.blocks))
    inner.get_image_features = transplanted_features(inner.get_image_features, inner.visual, plant)
    reader.mix = plant                                                  # Reader.read hands it the flags and the companion images
    return plant


def job_selections(processor, merge, scans, places, rule, select):
    """Per image of a job (PIL scans; ``places`` the places images, or None for an image not transplanted): its selection or None."""
    out = []
    for scan, place in zip(scans, places):
        if place is None:
            out.append(None); continue
        t, h, w = processor.image_processor(images=[scan], return_tensors='pt')['image_grid_thw'][0].tolist()
        labels = patch_labels(strike_pixels(scan, place, rule), h, w, merge) if select in ('struck', 'next') else np.zeros(t*h*w, int)
        out.append(selection(labels, select))
    return out


def features_check(args):
    """Check 1 / 2 of the plan's section four, bit for bit, on the first ``--regions`` rows of a fold in batches of 8 as read: images 2
    and 3 with every patch transplanted after K blocks against the frozen tower's features of the transplant pack (K <= 9 all four
    levels; K > 9 the merged tokens and the deepstack levels taken from block K on, the earlier ones the scan's), with none against
    the scan's own; image 1 always the scan's."""
    import json
    import torch
    from PIL import Image
    from ..recognize_lines import image_keys
    from .vlm_tokens import Reader, image_levels
    reader = Reader(args.model); inner = reader.model.model; keys = image_keys('context_revision')
    deep = list(inner.visual.deepstack_visual_indexes)
    plant = Transplant(1, len(inner.visual.blocks)); original = inner.get_image_features
    inner.get_image_features = transplanted_features(original, inner.visual, plant)
    load = lambda pack: {r['id']: r for r in json.loads((pack/'pack.json').read_text())['rows']}
    read, clean = load(args.pack), load(args.transplant_pack)
    rows = sorted((r for r in read.values() if r['fold'] == args.fold), key=lambda r: r['id'])[:args.regions]
    out = {f'after{k}_{s}': {'images': 0, 'as_wanted': 0, 'differ': []} for k in args.after for s in ('all', 'none')}
    processed = lambda pack, rs: reader.processor.image_processor(images=[Image.open(pack/r['images'][k]).convert('RGB') for r in rs for k in keys], return_tensors='pt')
    for start in range(0, len(rows), 8):
        batch = rows[start:start+8]
        a, b = processed(args.pack, batch), processed(args.transplant_pack, [clean[r['id']] for r in batch])
        pv, grid = a['pixel_values'].to(reader.device, reader.model.dtype), a['image_grid_thw'].to(reader.device)
        with torch.inference_mode():
            scan = image_levels(original(pv, grid))
            erased = image_levels(original(b['pixel_values'].to(reader.device, reader.model.dtype), b['image_grid_thw'].to(reader.device)))
        plant.companion = (b['pixel_values'], b['image_grid_thw']); sizes = [int(t*h*w) for t, h, w in grid.tolist()]
        plant.flags = [n % len(keys)+1 in (2, 3) for n in range(len(sizes))]
        for k in args.after:
            plant.after = k
            for s in ('all', 'none'):
                plant.selections = [np.full(n, s == 'all') if f else None for n, f in zip(sizes, plant.flags)]
                with torch.inference_mode():
                    got = image_levels(inner.get_image_features(pv, grid))
                for i, levels in enumerate(got):
                    if plant.flags[i] and s == 'all':      # level 0 merged; level n the deepstack taken after block deep[n-1] + 1
                        want = [erased[i][n] if n == 0 or deep[n-1]+1 >= k else scan[i][n] for n in range(len(levels))]
                    else:
                        want = scan[i]
                    same = all(torch.equal(x, y) for x, y in zip(levels, want)); e = out[f'after{k}_{s}']
                    e['images'] += 1; e['as_wanted'] += same
                    if not same:
                        e['differ'].append({'id': batch[i//len(keys)]['id'], 'image': i % len(keys)+1,
                                            'levels': [bool(torch.equal(x, y)) for x, y in zip(levels, want)]})
    return {'regions': len(rows), 'deepstack_after_blocks': [d+1 for d in deep], 'checks': out, 'passed': all(not v['differ'] for v in out.values())}


def main(argv=None):
    import argparse
    import json
    from pathlib import Path
    from ..util import make_output_dir, write_json, write_manifest
    p = argparse.ArgumentParser(description='Checks 1 and 2 of 早期局部修_前提检验 (features, bit for bit, on the reader\'s card).')
    p.add_argument('--model', type=Path, required=True); p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--transplant-pack', type=Path, required=True); p.add_argument('--after', type=int, nargs='+', default=[3, 6, 9, 13, 17])
    p.add_argument('--fold', type=int, default=0); p.add_argument('--regions', type=int, default=20)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    out = make_output_dir(args.output_dir); write_manifest(out, config={k: str(v) for k, v in vars(args).items()})
    result = features_check(args)
    write_json(out/'result.json', result); print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, ensure_ascii=False))
    print(json.dumps({k: {'images': v['images'], 'as_wanted': v['as_wanted']} for k, v in result['checks'].items()}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
