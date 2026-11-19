"""The hint a restorer is given beside the scan: a text (a reading, or the answer) typeset in print and laid over the region's ink
(plan of 2026-11-20, 前提检验_带提示的描字器, section two).

    python -m revrender.data.hint_image stats --pack PACK [--readings DIR[,DIR...]] --output OUT.json
    python -m revrender.data.hint_image hints --pack PACK --source answer|wrong|white|readings [--readings DIR[,DIR...]] --output HINTS.json
    python -m revrender.data.hint_image windows --sets SETS --answers ANSWERS.json --readings READINGS.json --output-dir HINTED_SETS

``hint_text``: the text as typeset -- the annotator's or a reader's LaTeX spelling, ``?`` dropped, ``$$...$$`` as ``$...$``, an odd
count of ``$`` dropped altogether (plain text); None (a white hint) for a refusal, an empty text or the unreadable mark.  ``render``:
matplotlib's mathtext with DejaVu Sans and WenQuanYi Zen Hei, black on white; a text it cannot typeset (an empty ``\\frac{}{}``, an
``array``, a line break, a character mathtext has no glyph for...) falls back to its ``eval.coverage.prepare`` spelling typeset as plain text.  The typeset ink is cut to its
own box and scaled, each way on its own, to the box of the window's ink (grey below 200) inside the polygon, on a white window of the
scan's size; a window with no ink inside its polygon gets a white hint.  ``wrong_answer``: the answer with one symbol changed by the
plan's fixed rule (section four).  ``stats``: how many answers and readings fall back to the plain spelling and how many are white.
``hints``: {region: text or null} of every region of a pack -- its answer (null where unreadable), its wrong answer (``wrong_answer``;
null where it takes no part: unreadable, nothing to change, a ``\begin``), white, or a reading.  ``windows``: windowed sets
(``data.struck_pack fold-sets --windowed``) again with the hint windows a restorer trains and stops on: per training region
``hint_answer_<k>`` and ``hint_reading_<k>``, per validation and held-out region ``hint_<k>`` from the readings.
"""
import argparse
import io
import json
import re
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from ..util import require, write_json

INK = 200
FONTS = ['DejaVu Sans', 'WenQuanYi Zen Hei']
UNREADABLE = '[读不出]'


def hint_text(text):
    """The text to typeset, or None for a white hint."""
    if text is None or UNREADABLE.strip('[]') in text: return None
    s = text.replace('?', '').strip()
    if not s: return None
    s = re.sub(r'\$\$(.*?)\$\$', r'$\1$', s, flags=re.S)
    if s.count('$') % 2: s = s.replace('$', '')
    return s


def typeset(text):
    """(the text typeset as a grey array, black on white, cut to its ink; whether it fell back to the plain spelling)."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from ..eval.coverage import prepare

    def draw(s, plain):
        figure = plt.figure(figsize=(0.01, 0.01), dpi=100)
        figure.text(0, 0, s, fontsize=40, family=FONTS, usetex=False, parse_math=not plain)
        buffer = io.BytesIO()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                figure.savefig(buffer, format='png', dpi=100, bbox_inches='tight', pad_inches=0.05, facecolor='white')
        finally:
            plt.close(figure)
        # mathtext draws a character its fonts lack (Chinese inside $...$) as an empty box: that is not typeset either
        require(plain or not any('glyph' in str(w.message) for w in caught), 'a character mathtext cannot draw')
        grey = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert('L'))
        ys, xs = np.nonzero(grey < INK)
        require(len(xs), 'nothing typeset')
        return grey[ys.min():ys.max()+1, xs.min():xs.max()+1]
    if '\n' not in text:
        try:
            return draw(text, False), False
        except Exception:
            pass
    plain = prepare(text).replace('$', '')
    return draw(plain or text.replace('$', ''), True), True


def ink_box(image, inner):
    ys, xs = np.nonzero((np.asarray(image) < INK) & np.asarray(inner, bool))
    return (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1) if len(xs) else None


def render(text, image, inner):
    """(the hint window: uint8, the scan window's size, white; whether it fell back to the plain spelling, or None for a white hint)."""
    out = np.full(np.asarray(image).shape, 255, np.uint8); s = hint_text(text); box = ink_box(image, inner)
    if s is None or box is None: return out, None
    grey, plain = typeset(s); l, t, r, b = box
    out[t:b, l:r] = np.asarray(Image.fromarray(grey).resize((r-l, b-t), Image.Resampling.LANCZOS))
    return out, plain


SYMBOLS = re.compile(r'[0-9A-Za-z+\-=]')


def wrong_answer(text):
    """The answer with its ⌊m/2⌋-th changeable symbol changed (plan, section four): command names (a backslash and letters) left out of
    the count; a digit to the next (9 to 0), a letter to the next (z to a, case kept), + to -, - to +, = to -.  None when m = 0 or
    the answer has a ``\\begin``."""
    s = (text or '').replace('?', '')
    if '\\begin' in s: return None
    commands = [m.span() for m in re.finditer(r'\\[A-Za-z]+', s)]
    places = [m.start() for m in SYMBOLS.finditer(s) if not any(a <= m.start() < b for a, b in commands)]
    if not places: return None
    k = places[len(places)//2]; c = s[k]
    new = (str((int(c)+1) % 10) if c.isdigit() else chr((ord(c)-97+1) % 26+97) if c.islower() else chr((ord(c)-65+1) % 26+65) if c.isupper()
           else {'+': '-', '-': '+', '=': '-'}[c])
    return s[:k]+new+s[k+1:]


def stats(args):
    """How many of the pack's answers and of the readings fall back to the plain spelling, and how many hints are white."""
    from ..eval.transcription import read_arm
    rows = json.loads((args.pack/'pack.json').read_text())['rows']
    sources = [('answers', {r['id']: (r['text'] if r['readable'] else UNREADABLE) for r in rows})]
    if args.readings: sources.append(('readings', read_arm('R0='+args.readings)[1]))
    result = {}
    for name, texts in sources:
        counts = {'regions': len(texts), 'white': 0, 'typeset': 0, 'plain': 0}; plain = []
        for k, t in texts.items():
            s = hint_text(t)
            if s is None: counts['white'] += 1; continue
            _, fell = typeset(s); counts['plain' if fell else 'typeset'] += 1
            if fell: plain.append(k)
        counts['plain_share'] = counts['plain']/max(1, counts['regions']); result[name] = {**counts, 'plain_ids': plain}
    require(not args.output.exists(), f'{args.output} exists'); write_json(args.output, result)
    print(json.dumps({k: {x: y for x, y in v.items() if x != 'plain_ids'} for k, v in result.items()}), flush=True)


def hints(args):
    from ..eval.transcription import read_arm
    rows = json.loads((args.pack/'pack.json').read_text())['rows']
    if args.source == 'readings':
        texts = read_arm('R='+args.readings)[1]; fallback = json.loads(args.fallback.read_text()) if args.fallback else {}
        out = {r['id']: texts[r['id']] if r['id'] in texts else fallback[r['id']] for r in rows}
    else: out = {r['id']: None if args.source == 'white' or not r['readable'] else r['text'] if args.source == 'answer' else wrong_answer(r['text']) for r in rows}
    require(not args.output.exists(), f'{args.output} exists'); write_json(args.output, out)
    print(json.dumps({'regions': len(out), 'white': sum(hint_text(t) is None for t in out.values())}), flush=True)


def windows(args):
    import shutil
    from ..util import digest, make_output_dir, write_manifest
    answers, readings = json.loads(args.answers.read_text()), json.loads(args.readings.read_text())
    out = make_output_dir(args.output_dir); fell = {'answer': 0, 'reading': 0}
    for folder in sorted(args.sets.glob('fold_*')):
        dest = out/folder.name; dest.mkdir()
        for name in ('training_native', 'validation_native', 'heldout_native'):
            rows = json.loads((folder/f'{name}.json').read_text()); arrays = dict(np.load(folder/f'{name}.npz'))
            require(all(r.get('windowed') for r in rows), 'not windowed sets')
            for i, r in enumerate(rows):
                image, inner = arrays[f'image_{i}'], arrays[f'inner_{i}']
                if name == 'training_native':
                    arrays[f'hint_answer_{i}'], a = render(answers[r['id']], image, inner); arrays[f'hint_reading_{i}'], b = render(readings[r['id']], image, inner)
                    fell['answer'] += bool(a); fell['reading'] += bool(b)
                else: arrays[f'hint_{i}'], _ = render(readings[r['id']], image, inner)
            np.savez_compressed(dest/f'{name}.npz', **arrays); shutil.copyfile(folder/f'{name}.json', dest/f'{name}.json')
        for name in ('records.json', 'summary.json'):
            if (folder/name).exists(): shutil.copyfile(folder/name, dest/name)
        # each fold is a review set of its own (data.restoration_pairs human reads its manifest)
        write_manifest(dest, config={'sets': str(folder), 'answers': str(args.answers), 'readings': str(args.readings)},
                       inputs={str(p): digest(p) for p in (folder/'run_manifest.json', args.answers, args.readings) if p.exists()})
    write_manifest(out, config={'sets': str(args.sets), 'answers': str(args.answers), 'readings': str(args.readings)},
                   inputs={str(p): digest(p) for p in (args.answers, args.readings)}, plain_typeset=fell)
    print(json.dumps({'folds': len(list(out.glob('fold_*'))), 'plain_typeset': fell}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('stats')
    s.add_argument('--pack', type=Path, required=True); s.add_argument('--readings', help='DIR[,DIR...] of recognize_lines')
    s.add_argument('--output', type=Path, required=True)
    h = sub.add_parser('hints')
    h.add_argument('--pack', type=Path, required=True); h.add_argument('--source', choices=('answer', 'wrong', 'white', 'readings'), required=True)
    h.add_argument('--readings', help='DIR[,DIR...] of recognize_lines, with --source readings'); h.add_argument('--output', type=Path, required=True)
    h.add_argument('--fallback', type=Path, help='hints of the regions the readings lack (a smoke test reads a few)')
    w = sub.add_parser('windows')
    for key in ('sets', 'answers', 'readings', 'output-dir'): w.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args(argv); {'stats': stats, 'hints': hints, 'windows': windows}[args.command](args)


if __name__ == '__main__':
    main()
