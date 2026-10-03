"""Build an offline, line-by-line transcription editor and B/C/D comparison page."""
import argparse
import json
import shutil
from pathlib import Path

from ..util import digest, make_output_dir, require, to_jsonable, write_manifest
from .pages_view import embed

TEMPLATE = Path(__file__).with_name('recognition_view.html')


def predictions(files, row_ids):
    out = {}
    for path in files:
        for text in Path(path).read_text().splitlines():
            item = json.loads(text)
            key, mode = item['row_id'], item['mode']
            require(key in row_ids and mode in ('B', 'C', 'D'), 'prediction does not belong to this pilot')
            require(mode not in out.get(key, {}), f'duplicate prediction: {key}, {mode}')
            out.setdefault(key, {})[mode] = {k: item[k] for k in ('text', 'truncated')}
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
    row_ids = {r['id'] for page in pack['pages'] for r in page['rows']}
    result = predictions(args.predictions, row_ids)
    if args.predictions:
        require(set(result) == row_ids and all(set(x) == {'B', 'C', 'D'} for x in result.values()),
                'a comparison requires exactly one B/C/D prediction for every cut')
    out = make_output_dir(args.output_dir)
    shutil.copytree(args.pack/'assets', out/'assets')
    if args.mathjax:
        (out/'assets/math').mkdir()
        shutil.copyfile(args.mathjax, out/'assets/math/tex-svg.js')
        license_path = args.mathjax.parent/'LICENSE'
        if license_path.exists():
            shutil.copyfile(license_path, out/'assets/math/LICENSE')
    payload = {'dataset_id': digest(pack_path), 'pages': pack['pages'], 'predictions': result,
               'math_preview': bool(args.mathjax), 'model': 'Qwen3-VL-8B-Instruct', 'seed': pack['seed']}
    page = TEMPLATE.read_text().replace('__DATA__', embed(payload))
    if args.mathjax:
        page = page.replace('<!--MATHJAX-->', '<script defer src="assets/math/tex-svg.js"></script>')
    (out/'index.html').write_text(page, encoding='utf-8')
    inputs = {str(path): digest(path) for path in [pack_path, *args.predictions]}
    if args.mathjax:
        inputs[str(args.mathjax)] = digest(args.mathjax)
    counts = {mode: sum(mode in x for x in result.values()) for mode in ('B', 'C', 'D')}
    write_manifest(out, config=to_jsonable(args), inputs=inputs, pages=len(pack['pages']), rows=len(row_ids), predictions=counts)
    print(json.dumps({'pages': len(pack['pages']), 'rows': len(row_ids), 'predictions': counts, 'output': str(out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
