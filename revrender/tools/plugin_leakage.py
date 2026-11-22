"""Check 6 of 插件结构优化 (plan of 2026-11-22, section four): nothing of a held-out fold or of the test pages reaches a tower adapter's
training pairs, and what the readers see holds none of the annotator's erasure.

    python -m revrender.tools.plugin_leakage --pack VIEWS --scan-pack RAW_VIEWS --synthetic SYNTH --output OUT.json

Per fold K, as ``train_clean_view train`` takes them: its real training and validation regions are of the other folds, and none of
their writers (a student's two ids as one) writes in fold K; its synthetic pairs (``SYNTH/fold_K.json``) have host and material
writers neither in fold K nor among the test pages' writers (``data.synth_views.test_writers``, with the dataset, the test halves and
the same-student pairs the synthetic build was made with, from its run manifest).  No region of the pack is a test-page writer's.
Every row of the scan pack shows the scan: its ``erased`` window is its ``raw`` crop pixel for pixel, and its ``context`` line is not
the pack's (the line with the annotator's erased window).
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ..util import require, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key in ('pack', 'scan-pack', 'synthetic', 'output'): p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args(argv)
    from ..data.prior_annotations import twins
    from ..data.synth_views import test_writers
    from ..train_clean_view import regions_of, validation_writers
    built = json.loads((args.synthetic/'run_manifest.json').read_text())['config']
    other = twins(Path(built['same_student'])); canonical = lambda w: min(w, other.get(w, w))
    tests = test_writers(SimpleNamespace(dataset=Path(built['dataset']), test_halves=Path(built['test_halves'])), canonical)
    rows = json.loads((args.pack/'pack.json').read_text())['rows']; scans = {r['id']: r for r in json.loads((args.scan_pack/'pack.json').read_text())['rows']}
    folds = sorted({r['fold'] for r in rows}); fold_of = {r['id']: r['fold'] for r in rows}
    result = {'test_page_writers': len(tests), 'pack_regions_of_test_writers': sorted(r['id'] for r in rows if canonical(r['writer']) in tests), 'folds': {}}
    for k in folds:
        here = {canonical(r['writer']) for r in rows if r['fold'] == k}
        real = regions_of(args.pack, args.scan_pack, lambda r: r['fold'] != k); held = validation_writers(real)
        synthetic = json.loads((args.synthetic/f'fold_{k}.json').read_text())
        bad_synth = [s['id'] for s in synthetic if {canonical(w) for w in (s.get('host_writer'), s.get('material_writer')) if w} & (here | tests)]
        result['folds'][k] = {'real_training': sum(r[1] not in held for r in real), 'validation': sum(r[1] in held for r in real),
                              'regions_of_fold': sorted(r[0] for r in real if fold_of[r[0]] == k), 'writers_of_fold': sorted({canonical(r[1]) for r in real} & here),
                              'synthetic': len(synthetic), 'synthetic_of_fold_or_test_writers': bad_synth}
    scan_raw_differs, line_is_the_packs = [], []
    for r in rows:
        q = scans[r['id']]
        if not np.array_equal(np.asarray(Image.open(args.scan_pack/q['images']['raw'])), np.asarray(Image.open(args.scan_pack/q['images']['erased']))):
            scan_raw_differs.append(r['id'])
        if (args.scan_pack/q['images']['context']).read_bytes() == (args.pack/r['images']['context']).read_bytes():
            line_is_the_packs.append(r['id'])
    result['reading_images'] = {'rows': len(rows), 'erased_not_the_raw_crop': scan_raw_differs, 'line_equals_the_erased_line': line_is_the_packs}
    result['passed'] = (not result['pack_regions_of_test_writers'] and not scan_raw_differs and not line_is_the_packs
                        and all(not v['regions_of_fold'] and not v['writers_of_fold'] and not v['synthetic_of_fold_or_test_writers'] for v in result['folds'].values()))
    require(not args.output.exists(), f'{args.output} exists'); write_json(args.output, result)
    print(json.dumps({'passed': result['passed'], 'test_page_writers': len(tests)}), flush=True)
    raise SystemExit(0 if result['passed'] else 'a leak: see '+str(args.output))


if __name__ == '__main__':
    main()
