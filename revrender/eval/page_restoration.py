"""Restorers on whole questions whose every region is confirmed (user 2026-10-23: should a question go in whole?).

    python -m revrender.eval.page_restoration --export EXPORT --pages DIR --ids held_out.txt --upstream UP \
        --model NAME=RUN:STEP:MODE [...] --output-dir OUT

A model is a restorer run and one of its step directories, read either on the whole question (``page``: every 256 tile, as it would
prefill a question nobody boxed) or only inside the regions the detector proposed (``boxes``: the regions of the export that carry a
``detector_index``, the way the library is prefilled now; a region added by hand is then never reached).  On each question the truth is
the union of its confirmed pixels (``data.restoration_pairs.question_gold``); ink is grey < 200 and a pixel is erased when the model
brightens it by more than 0.2 (``baselines.restoration``).  Per model: precision, recall, F1 and the writing erased, the writing erased
outside every region (where a whole-question model can wrongly strike), and the share of the strike in hand-added regions it erased
(strikes the detector missed).  Nothing is trained.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..baselines.restoration import restore_native
from ..data.restoration_pairs import question_gold
from ..util import digest, make_output_dir, require, write_json, write_manifest, to_jsonable

ERASED = .2      # brightened by more than this: erased (baselines.restoration.removal_counts)


def erase_boxes(predict, grey, revisions):
    """The question restored only inside the detector's regions (each region's box, the values inside its polygon)."""
    from ..data.pixel_masks import region_pixels
    raw = grey.astype(np.float32)/255; out = raw.copy()
    for r in revisions:
        if 'detector_index' not in r or r.get('deleted'): continue
        l, t, rr, b = r['box']; inside = region_pixels(r.get('region'), r['box'])
        value = restore_native(predict, raw[t:b, l:rr]); out[t:b, l:rr][inside] = value[inside]
    return out


def score(grey, gold, regions, hand, restored):
    ink = grey < 200; removed = ((restored - grey.astype(np.float32)/255) > ERASED) & ink
    truth = gold & ink; writing = ink & ~truth
    return np.array([(removed & truth).sum(), (removed & writing).sum(), (~removed & truth).sum(), writing.sum(),
                     (removed & writing & ~regions).sum(), (writing & ~regions).sum(), (removed & truth & hand).sum(), (truth & hand).sum()], np.int64)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('export', 'pages', 'ids', 'upstream', 'output-dir'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--model', action='append', required=True, help='NAME=RUN:STEP:MODE (MODE page or boxes)')
    args = p.parse_args(argv)
    data = json.loads(args.export.read_text()); ids = args.ids.read_text().split()
    models = {}
    for spec in args.model:
        name, rest = spec.split('=', 1); run, step, mode = rest.split(':'); require(mode in ('page', 'boxes'), f'mode {mode}')
        models[name] = (Path(run), step, mode)
    out = make_output_dir(args.output_dir); result = {}; inputs = {str(args.export): digest(args.export), str(args.ids): digest(args.ids)}
    from ..baselines.restoration_export import selected_predictor
    for name, (run, step, mode) in models.items():
        predict, checkpoint, _ = selected_predictor(run, args.upstream, step); inputs[str(checkpoint)] = digest(checkpoint)
        total = np.zeros(8, np.int64)
        for image_id in ids:
            revisions = data['pages'][image_id]['revisions']
            with Image.open(args.pages/(image_id+'.png')) as page:
                grey, gold, regions = question_gold(page, revisions)
                _, hand, _ = question_gold(page, [r for r in revisions if 'detector_index' not in r])
            restored = restore_native(predict, grey.astype(np.float32)/255) if mode == 'page' else erase_boxes(predict, grey, revisions)
            total += score(grey, gold, regions, hand, restored)
        tp, fp, fn, writing, fp_out, writing_out, hand_tp, hand_all = map(int, total)
        result[name] = dict(run=str(run), step=step, mode=mode, precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn), f1=2*tp/max(1, 2*tp+fp+fn),
                            writing_erased=fp/max(1, writing), writing_erased_outside_regions=fp_out/max(1, writing_out),
                            hand_added_strike_erased=hand_tp/max(1, hand_all), strike_pixels=tp+fn, hand_added_strike_pixels=hand_all)
        print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result[name].items() if k not in ('run',)}), flush=True)
    write_json(out/'scores.json', {'questions': ids, 'models': result})
    write_manifest(out, config=to_jsonable(args), inputs=inputs, questions=len(ids))


if __name__ == '__main__':
    main()
