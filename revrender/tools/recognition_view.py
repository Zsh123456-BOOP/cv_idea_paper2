"""Build the revision-only two-task comparison and key-character annotation page."""
import argparse
import json
import shutil
from pathlib import Path

from ..util import digest, make_output_dir, require, to_jsonable, write_manifest
from .pages_view import embed

TEMPLATE = Path(__file__).with_name('recognition_view.html')
METRICS = Path(__file__).with_name('recognition_metrics.js')
TASKS = {'verbatim', 'deletion'}


def predictions(files, row_ids):
    out = {}
    for path in files:
        for text in Path(path).read_text().splitlines():
            item = json.loads(text)
            key, task, mode = item['row_id'], item['test'], item['mode']
            require(key in row_ids and mode in ('B', 'C', 'D') and task in TASKS,
                    'prediction does not belong to this pilot')
            results = out.setdefault(key, {}).setdefault(task, {})
            require(mode not in results, f'duplicate prediction: {key}, {task}, {mode}')
            results[mode] = {k: item[k] for k in ('text', 'truncated')}
    return out


def protocols(files, pack_hash):
    out = {}
    for path in files:
        manifest = json.loads((path.parent/'run_manifest.json').read_text())
        require(manifest.get('protocol') == 'revision-only-two-tasks', 'wrong experiment protocol')
        pack_inputs = [v for k, v in manifest['inputs'].items() if Path(k).name == 'pack.json']
        require(pack_inputs == [pack_hash], 'prediction pack hash mismatch')
        mode = manifest['mode']
        require(mode not in out, 'duplicate mode manifest')
        require(set(manifest['messages']) == TASKS, 'missing task prompts')
        out[mode] = {k: manifest[k] for k in ('messages', 'context', 'generation', 'rendered_prompts', 'git')}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--predictions', type=Path, action='append', default=[])
    p.add_argument('--mathjax', type=Path, help='optional local tex-svg.js for offline formula previews')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    pack_path = args.pack/'pack.json'
    pack = json.loads(pack_path.read_text())
    require(pack.get('selection', {}).get('kind') == 'revision-only', 'requires revision-only pack')
    require(all(p['split'] == 'dev' and all(r['masked_pixels'] > 0 for r in p['rows']) for p in pack['pages']),
            'requires positive development cuts')
    row_ids = {r['id'] for page in pack['pages'] for r in page['rows']}
    result = predictions(args.predictions, row_ids)
    require(set(result) == row_ids and all(set(x) == TASKS and all(set(y) == {'B', 'C', 'D'} for y in x.values())
                                         for x in result.values()),
            'a comparison requires B/C/D in both tasks for every cut')
    protocol = protocols(args.predictions, digest(pack_path))
    out = make_output_dir(args.output_dir)
    shutil.copytree(args.pack/'assets', out/'assets')
    if args.mathjax:
        (out/'assets/math').mkdir()
        shutil.copyfile(args.mathjax, out/'assets/math/tex-svg.js')
        license_path = args.mathjax.parent/'LICENSE'
        if license_path.exists():
            shutil.copyfile(license_path, out/'assets/math/LICENSE')
    payload = {'dataset_id': digest(pack_path), 'pages': pack['pages'], 'predictions': result,
               'selection': pack['selection'], 'protocol': protocol,
               'math_preview': bool(args.mathjax), 'model': 'Qwen3-VL-8B-Instruct', 'seed': pack['seed']}
    page = TEMPLATE.read_text().replace('__DATA__', embed(payload)).replace('/*__METRICS__*/', METRICS.read_text())
    if args.mathjax:
        page = page.replace('<!--MATHJAX-->', '<script defer src="assets/math/tex-svg.js"></script>')
    (out/'index.html').write_text(page, encoding='utf-8')
    inputs = {str(path): digest(path) for path in [pack_path, *args.predictions,
              *(p.parent/'run_manifest.json' for p in args.predictions)]}
    if args.mathjax:
        inputs[str(args.mathjax)] = digest(args.mathjax)
    counts = {task: {mode: sum(mode in x[task] for x in result.values()) for mode in ('B', 'C', 'D')}
              for task in sorted(TASKS)}
    write_manifest(out, config=to_jsonable(args), inputs=inputs, pages=len(pack['pages']), rows=len(row_ids), predictions=counts)
    print(json.dumps({'pages': len(pack['pages']), 'rows': len(row_ids), 'predictions': counts, 'output': str(out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
