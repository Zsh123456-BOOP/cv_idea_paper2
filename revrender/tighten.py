"""Masks cut to the pixels that are not white paper.

A revision is annotated (gold) or delivered (model) as a region, so its mask also lies on blank paper: around a scribble, between the
strokes of a crossed-out word.  Taking the revision out of the page later should take its ink and not the paper beside it.  ``tighten``
keeps of each mask the page's non-white pixels (``non_white``): the ink and the soft edge of its strokes, which has to go with the stroke
or erasing it leaves a grey outline behind.  Nothing is grown: a tightened mask is always a part of the mask it came from, and one with
no ink left is dropped (and counted).

Scans are white-levelled (blank paper is exactly 255) but the JPEG leaves a speckle of 240-254 beside every stroke that the eye takes for
white, so "not pure white" cannot be ``grey < 255``: that keeps nearly the whole region.  Ink is grey below ``eval.ink.INK_BELOW`` (the
ink of the paper-1 ink view); its soft edge is grey below ``EDGE_BELOW`` within ``EDGE_PX`` pixels of ink.

    python -m revrender.tighten --pages-output OUT --images DIR [--images DIR ...] [--ids FILE] --output-dir NEW

OUT is the output of ``revrender.pages`` or of ``data.page_gold`` (the gold, in the same layout); NEW has the same layout, each instance
cut to ink with its box shrunk to what is left and ``from_instance`` naming the instance it was cut from.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from . import pages
from .eval.ink import INK_BELOW
from .util import digest, make_output_dir, require, to_jsonable, write_manifest

EDGE_BELOW = 240           # a pixel this light or lighter is paper
EDGE_PX = 2                # the soft edge of a stroke reaches this far from its ink
QUANTILES = (.1, .25, .5, .75, .9)


def non_white(grey):
    """The page's non-white pixels (a boolean array): ink, and the soft edge of its strokes."""
    grey = np.asarray(grey)
    ink = grey < INK_BELOW
    reach = ndimage.binary_dilation(ink, np.ones((3, 3), bool), iterations=EDGE_PX)
    return ink | (reach & (grey < EDGE_BELOW))


def tighten_instance(instance, white):
    """The instance cut to the page's non-white pixels, its box shrunk to what is left; None when no ink is left."""
    left, top, right, bottom = instance['box']
    mask = instance['mask'] & white[top:bottom, left:right]
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    t, b, l, r = int(ys.min()), int(ys.max())+1, int(xs.min()), int(xs.max())+1
    return {**instance, 'box': [left+l, top+t, left+r, top+b], 'mask': mask[t:b, l:r]}


def tighten_page(instances, grey):
    """The page's instances cut to ink (``tighten_instance``), those with ink left only, each naming its index in ``instances``."""
    white = non_white(grey)
    out = []
    for n, instance in enumerate(instances):
        cut = tighten_instance(instance, white)
        if cut is not None:
            out.append({**cut, 'from_instance': n})
    return out


def listed_ids(directory, ids_file=None):
    """The pages of an output (or the ones a file names, which all have to be there)."""
    found = sorted(p.stem for p in Path(directory).glob('*.json') if p.name != 'run_manifest.json')
    if ids_file is None:
        return found
    wanted = [line.strip() for line in Path(ids_file).read_text().splitlines() if line.strip()]
    missing = sorted(set(wanted)-set(found))
    require(not missing, f'{len(missing)} page(s) not in {directory}, e.g. {missing[:2]}')
    return wanted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--pages-output', type=Path, required=True, help='the output folder of revrender.pages or data.page_gold')
    parser.add_argument('--images', type=Path, action='append', required=True, help='a folder of <image_id>.jpg (repeatable)')
    parser.add_argument('--ids', type=Path, help='a file of image ids, one per line (default: every page of --pages-output)')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    ids = listed_ids(args.pages_output, args.ids)
    files = pages.page_files(args.images, ids)
    out = make_output_dir(args.output_dir)
    shares, totals = [], {'pages': 0, 'instances_in': 0, 'instances_out': 0, 'dropped': 0, 'pixels_in': 0, 'pixels_out': 0}
    for image_id in ids:
        record, instances = pages.read_output(args.pages_output, image_id)
        with Image.open(files[image_id]) as opened:
            grey = np.asarray(opened.convert('L'))
        require(list(grey.shape[::-1]) == record['size'], f'{image_id}: the image is not the page the output was made on')
        kept = tighten_page(instances, grey)
        pages.write_page(out, image_id, record['size'], record['lines'], kept)
        for instance in kept:
            before = float(instances[instance['from_instance']]['mask'].sum())
            shares.append(1.-float(instance['mask'].sum())/before)
        totals['pages'] += 1
        totals['instances_in'] += len(instances)
        totals['instances_out'] += len(kept)
        totals['dropped'] += len(instances)-len(kept)
        totals['pixels_in'] += sum(int(i['mask'].sum()) for i in instances)
        totals['pixels_out'] += sum(int(i['mask'].sum()) for i in kept)
    shares = np.array(shares)
    write_manifest(out, config=to_jsonable(args), inputs={str(args.pages_output/'run_manifest.json'): digest(args.pages_output/'run_manifest.json')},
                   ink_below=INK_BELOW, edge_below=EDGE_BELOW, edge_px=EDGE_PX,
                   removed_share_of_pixels=1.-totals['pixels_out']/max(1, totals['pixels_in']),
                   removed_share_per_instance={f'q{int(q*100)}': float(np.quantile(shares, q)) if len(shares) else None for q in QUANTILES},
                   **totals)
    print(totals)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
