"""A static HTML viewer for cleaned pages: each page as it was scanned, with the box it was kept to, beside the page cleaned.

    python -m revrender.tools.clean_view --before DIR [--before DIR ...] --after CLEAN_DIR [--per-question N] --output-dir SITE

``--before`` is the folder of scans and ``--after`` the output of ``revrender.data.clean_pages`` (the png pages and its frames.json).
SITE/index.html shows ``--per-question`` pages of each question (a fixed random choice, seed 42), each with the ink (grey below
``eval.ink.INK_BELOW``) the cleaning took off, counted outside the frame (the neighbours' lines and writing, the footer, the marks at
the corners) and inside it (dust).  The pages are copied to SITE/assets, so the folder opens from anywhere without a server.
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import pages
from ..data.clean_pages import read_grey
from ..eval.ink import INK_BELOW
from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_manifest
from .pages_view import embed, question_of

TEMPLATE = Path(__file__).with_name('clean_view.html')


def removed_ink(before, after, box):
    """(the pixels of ink the cleaning took off outside ``box`` (left, top, right, bottom), those it took off inside it)."""
    gone = (before < INK_BELOW) & (after >= INK_BELOW)
    left, top, right, bottom = box
    inside = np.zeros(gone.shape, bool)
    inside[top:bottom+1, left:right+1] = True
    return int((gone & ~inside).sum()), int((gone & inside).sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--before', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    parser.add_argument('--after', type=Path, required=True, help='the output folder of revrender.data.clean_pages')
    parser.add_argument('--per-question', type=int, default=10, help='pages shown of each question')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    frames = {k: v for k, v in json.loads((args.after/'frames.json').read_text()).items() if v}
    by_question = defaultdict(list)
    for image_id in sorted(frames):
        by_question[question_of(image_id)[0]].append(image_id)
    rng = random.Random(SEED)
    chosen = sorted(i for ids in by_question.values() for i in rng.sample(ids, min(args.per_question, len(ids))))
    require(chosen, 'no cleaned page with a frame')
    files = pages.page_files(args.before, chosen)
    cleaned = pages.page_files([args.after], chosen)
    out = make_output_dir(args.output_dir)
    (out/'assets').mkdir()
    cards = []
    for image_id in chosen:
        before, after = read_grey(files[image_id]), read_grey(cleaned[image_id])
        require(before.shape == after.shape, f'{image_id}: the cleaned page is not the size of the scan')
        outside, inside = removed_ink(before, after, frames[image_id])
        shutil.copyfile(files[image_id], out/'assets'/f'{image_id}_before{files[image_id].suffix}')
        shutil.copyfile(cleaned[image_id], out/'assets'/f'{image_id}_after.png')
        question, supplementary = question_of(image_id)
        cards.append({'id': image_id, 'ext': files[image_id].suffix[1:], 'w': before.shape[1], 'h': before.shape[0], 'q': question, 'supp': supplementary,
                      'box': frames[image_id], 'outside': outside, 'inside': inside})
    (out/'index.html').write_text(TEMPLATE.read_text(encoding='utf-8').replace('__DATA__', embed({'cards': cards})), encoding='utf-8')
    write_manifest(out, config=to_jsonable(args), inputs={str(args.after/'frames.json'): digest(args.after/'frames.json')}, pages=len(cards))
    print(json.dumps({'pages': len(cards), 'output': str(out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
