"""Fold a re-annotation package folder and the annotation editor (editor.html) into one self-contained HTML.

The folder form (editor + manifest.js + crops/ + refs/) only works when the page
is opened from a browser that loads files next to it.  Opened any other way --
the desktop app's preview renders a local page as a static snapshot -- the
manifest and every crop fail to load and the annotator sees an empty canvas.
One file with the manifest and every image inline has nothing to fail to load.

This script deliberately does NOT: change any mark, any crop or the manifest's
content beyond adding each row's inline image; read any model output; read,
transcribe or otherwise consume the frozen test split; or contact the network.
The output is student handwriting: it belongs under work/, never in Git.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

EDITOR = Path(__file__).with_name('editor.html')
BEGIN, END = '<!--PACK-->', '<!--/PACK-->'


def inline(path, kind):
    return f'data:{kind};base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def read_manifest(folder):
    text = (Path(folder)/'manifest.js').read_text(encoding='utf-8')
    return json.loads(text[text.index('{'):text.rindex('}')+1])


def build(folder, editor_text):
    folder = Path(folder)
    pack = read_manifest(folder)
    for row in pack['rows']:
        row['image_data'] = inline(folder/'crops'/row['file'], 'image/jpeg')
        if row.get('reference'):
            row['reference_data'] = inline(folder/row['reference'], 'image/png')
    if editor_text.count(BEGIN) != 1 or editor_text.count(END) != 1:
        raise SystemExit('single-file editor refused: the editor has no single manifest slot')
    head, rest = editor_text.split(BEGIN)
    _, tail = rest.split(END)
    payload = json.dumps(pack, ensure_ascii=False).replace('</', '<\\/')
    return head + '<script>window.PACK = ' + payload + ';</script>' + tail, pack


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit('single-file editor refused: create-only output exists')
    page, pack = build(args.package, EDITOR.read_text(encoding='utf-8'))
    args.output.write_text(page, encoding='utf-8')
    print(json.dumps({'rows': len(pack['rows']), 'bytes': args.output.stat().st_size,
                      'editor_sha256': hashlib.sha256(EDITOR.read_bytes()).hexdigest(),
                      'output': str(args.output)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
