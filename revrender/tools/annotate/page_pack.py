"""Fold the questions the user annotates into one self-contained page: the scans and their revisions, each with a region and a starting text.

    python -m revrender.tools.annotate.page_pack --prefill PREFILL_OUT --pages DIR [--pages DIR ...] [--readings predictions.jsonl]
        [--first N] --output-dir OUT

``--prefill`` is the output of ``page_prefill`` (the revisions and their regions).  ``--readings`` is what ``recognize_lines`` read on its
crops under ``raw_struck``: each revision's text starts from it (□ written as ?, line breaks as spaces, the marks a reader draws a strike
with -- an overline, a cancel and the like -- taken off, a formula with no command in it written plainly); a revision without a reading
starts empty.  A blackout starts as unreadable (its text kept, for when some of it can be read after all).

The editor (``page_editor.html``) and its rules (``page_rules.js``) are inlined with the scans, so OUT/标注.html opens from anywhere
without a server (formulas render when KaTeX loads from its CDN, and show as written when it does not).  The pack's id depends only on
the prefill's ``--which``: a rebuilt pack keeps what the annotator already did in the browser.  The output is student handwriting: it
belongs under work/, never in Git.
"""
import argparse
import base64
import json
import re
from pathlib import Path

from ... import pages as pages_module
from ...util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

HERE = Path(__file__).parent
EDITOR, RULES = HERE/'page_editor.html', HERE/'page_rules.js'
CONDITION = 'raw_struck'
WRAPPER = re.compile(r'\\(?:overline|underline|cancel|bcancel|xcancel|sout|st|boxed|text|mathrm)\s*\{([^{}]*)\}')
PLAIN = re.compile(r"\$\$([^$\\^_{}]+)\$\$|\$([^$\\^_{}]+)\$")


def slot(text, name, payload):
    """Replace the <!--NAME-->...<!--/NAME--> block of the editor with an inline script."""
    begin, end = f'<!--{name}-->', f'<!--/{name}-->'
    require(text.count(begin) == 1 and text.count(end) == 1, f'the editor has no single {name} slot')
    head, rest = text.split(begin)
    return head+'<script>'+payload.replace('</', '<\\/')+'</script>'+rest.split(end)[1]


def starting_text(reading):
    """A reading as the page's starting text: □ (the reader's unreadable character) as ?, line breaks as spaces, the marks a strike was
    drawn with taken off (innermost first), and a formula with no command, ^ or _ in it as plain text."""
    text, before = ' '.join(reading.replace('□', '?').split()), None
    while text != before:
        before, text = text, WRAPPER.sub(r'\1', text)
    return PLAIN.sub(lambda m: m.group(1) if m.group(1) is not None else m.group(2), text).strip()


def read_readings(path):
    """{row id: starting text} from recognize_lines' predictions under ``CONDITION``."""
    texts = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['condition'] == CONDITION:
            texts[row['id']] = starting_text(row['text'])
    require(texts, f'{path}: no {CONDITION} reading')
    return texts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--prefill', type=Path, required=True, help='the output of page_prefill')
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    parser.add_argument('--readings', type=Path, help=f'recognize_lines predictions on the prefill\'s pack ({CONDITION})')
    parser.add_argument('--first', type=int, default=0, help='only the first N questions (in the order of the selection)')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    prefill = json.loads((args.prefill/'prefill.json').read_text(encoding='utf-8'))
    questions = prefill['questions'][:args.first or None]
    require(questions, 'no question to pack')
    texts = read_readings(args.readings) if args.readings else {}
    scans = pages_module.page_files(args.pages, [q['id'] for q in questions])
    records = []
    for q in questions:
        revisions = [{**r, 'text': texts.get(f'{q["id"]}#{r["id"]}', ''), 'unreadable': r['kind'] == 'blot'} for r in q['revisions']]
        records.append({**q, 'revisions': revisions, 'image': 'data:image/jpeg;base64,'+base64.b64encode(scans[q['id']].read_bytes()).decode()})
    pack = {'id': f'paper2-revision-boxes-{prefill["which"]}', 'schema': 'paper2-revision-box-pack-v1', 'pages': records}
    html = slot(slot(EDITOR.read_text(encoding='utf-8'), 'RULES', RULES.read_text(encoding='utf-8')), 'PACK', 'window.PACK = '+json.dumps(pack, ensure_ascii=False)+';')
    out = make_output_dir(args.output_dir)
    (out/'标注.html').write_text(html, encoding='utf-8')
    revisions = [r for q in records for r in q['revisions']]
    summary = {'questions': len(records), 'revisions': len(revisions), 'blots': sum(r['kind'] == 'blot' for r in revisions),
               'with_text': sum(bool(r['text']) for r in revisions), 'bytes': (out/'标注.html').stat().st_size}
    inputs = [args.prefill/'prefill.json', EDITOR, RULES, *([args.readings] if args.readings else [])]
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in inputs}, **summary)
    write_json(out/'questions.json', [{'id': q['id'], 'question': q['question'], 'pilot': q['pilot']} for q in records])
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
