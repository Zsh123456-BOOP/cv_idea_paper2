"""The lines of a page, as the OCR model cuts them: the line boxes of the catalogue are these.

Unlimited-OCR reads a whole-question image into layout regions, ``<|det|>type [x0, y0, x1, y1]<|/det|>content``, the box in
thousandths of the page.  A region is split into visual lines at newlines and at ``\\\\``; a region that gives several lines can
only share its box between them (equal heights: ``inferred``).  The catalogue keeps the *strict* lines: handwriting (not page
numbers or figures), one line to its region, a real box.  ``strict_lines`` gives those, in reading order (a second column after
the first), each box once; on the OCR text of 2,238 pages it reproduces the catalogue's logical lines box for box.
"""
import json
import re
import unicodedata
from pathlib import Path

REGION = re.compile(r'<\|det\|>\s*([a-z_]+)\s*\[([^\]]*)\]\s*<\|/det\|>')
PRINTED = ('page_number', 'image', 'image_caption')        # furniture of the page, not a student's line
COLUMN_GAP = 260                                           # thousandths between the centres of two columns
ROW_BAND = 18                                              # lines whose tops fall in the same band read left to right

_NAMES = {'because': '∵', 'therefore': '∴', 'angle': '∠', 'triangle': '△', 'circ': '°', 'times': '×', 'div': '/', 'cdot': '*',
          'ast': '*', 'leq': '≤', 'le': '≤', 'geq': '≥', 'ge': '≥', 'neq': '≠', 'ne': '≠', 'pm': '±', 'mp': '∓', 'perp': '⊥',
          'parallel': '∥', 'cong': '≅', 'sim': '~', 'infty': '∞', 'square': '□', 'pi': 'π', 'alpha': 'α', 'beta': 'β',
          'gamma': 'γ', 'theta': 'θ', 'lambda': 'λ', 'mu': 'μ', 'Delta': 'Δ', 'approx': '≈', 'in': '∈', 'cap': '∩', 'cup': '∪',
          'rightarrow': '→', 'to': '→', 'Rightarrow': '⇒', 'ldots': '…', 'dots': '…'}
_CIRCLED = '①②③④⑤⑥⑦⑧⑨⑩'
_SUPERSCRIPT = {str(i): c for i, c in enumerate('⁰¹²³⁴⁵⁶⁷⁸⁹')}
_LAYOUT = ('left', 'right', 'quad', 'qquad', 'displaystyle', 'limits', 'big', 'Big', 'bigg', 'Bigg', 'mathrm', 'mathbf', 'mathit',
           'operatorname', 'text', 'textbf', 'textit', 'textrm', 'textnormal')


def normalise(text):
    """LaTeX and Unicode spellings of a line of mathematics brought to one notation, whitespace removed (only its length is used)."""
    if not text:
        return ''
    s = unicodedata.normalize('NFKC', text)
    s = re.sub(r'\\(?:' + '|'.join(_LAYOUT) + r')(?![a-zA-Z])', ' ', s)
    s = re.sub(r'\\begin\{[a-zA-Z*]+\}|\\end\{[a-zA-Z*]+\}', ' ', s)
    s = re.sub(r'\\[\(\)\[\]]', ' ', s)
    s = re.sub(r'\\\\', ' ', s)
    for _ in range(6):                                     # fractions, roots and overlines, nested
        t = re.sub(r'\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'(\1)/(\2)', s)
        t = re.sub(r'\\sqrt\s*\{([^{}]*)\}', r'√(\1)', t)
        t = re.sub(r'\\overline\s*\{([^{}]*)\}', r'\1', t)
        if t == s:
            break
        s = t
    s = re.sub(r'\\textcircled\s*\{?\s*(\d+)\s*\}?',
               lambda m: _CIRCLED[int(m.group(1))-1] if m.group(1).isdigit() and 1 <= int(m.group(1)) <= 10 else m.group(1), s)
    s = re.sub(r'\^\s*\{?\s*\\circ\s*\}?', '°', s)
    for name in sorted(_NAMES, key=len, reverse=True):     # whole command names only: \le must not eat \left
        s = re.sub(r'\\' + name + r'(?![a-zA-Z])', _NAMES[name], s)
    s = re.sub(r'\^\s*\{?\s*(\d)\s*\}?', lambda m: _SUPERSCRIPT[m.group(1)], s)
    s = re.sub(r'_\s*\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\[a-zA-Z]+', ' ', s)
    s = s.replace('$', ' ').replace('{', ' ').replace('}', ' ')
    for wide, narrow in (('（', '('), ('）', ')'), ('，', ','), ('。', '.'), ('：', ':'), ('、', ','), ('；', ';')):
        s = s.replace(wide, narrow)
    return ''.join(c for c in unicodedata.normalize('NFKC', s) if not c.isspace())


def split_region(content):
    """The visual lines of one region: split at newlines and at ``\\\\``, pieces shorter than two symbols dropped."""
    lines = []
    for chunk in re.split(r'\n+', content or ''):
        for piece in re.split(r'\\\\', chunk):
            piece = piece.strip()
            if len(normalise(piece)) >= 2:
                lines.append(piece)
    return lines


def regions(text):
    """Every region of the OCR text as (type, 0-1000 box) with a real box: figures and page numbers included."""
    out = []
    parts = REGION.split(text or '')
    for index in range(1, len(parts)-1, 3):
        try:
            box = [float(v) for v in parts[index+1].split(',')]
        except ValueError:
            continue
        if len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
            out.append((parts[index].strip(), box))
    return out


def parse_regions(text):
    """Every visual line of the OCR text with its 0-1000 box: [{ocr_type, box, ocr_text, student, inferred, region, parts}]; the lines
    of a multi-line region (``inferred``) also carry the region's own box and how many lines it holds."""
    out = []
    parts = REGION.split(text or '')
    for index in range(1, len(parts)-1, 3):
        ocr_type = parts[index].strip()
        try:
            box = [float(v) for v in parts[index+1].split(',')]
            box = box if len(box) == 4 else None
        except ValueError:
            box = None
        lines = split_region(parts[index+2])
        for k, line in enumerate(lines):
            part = box
            if box and len(lines) > 1:                     # the region's box shared in equal heights
                height = (box[3]-box[1])/len(lines)
                part = [box[0], box[1]+k*height, box[2], box[1]+(k+1)*height]
            out.append({'ocr_type': ocr_type, 'box': part, 'ocr_text': line, 'student': ocr_type not in PRINTED,
                        'inferred': len(lines) > 1, 'region': box, 'parts': len(lines)})
    return out


def reorder(lines):
    """Reading order: a second column after the first, top to bottom in bands, left to right within a band."""
    boxed = [line for line in lines if line['box']]
    if len(lines) < 2 or len(boxed) < len(lines)*.6:
        return lines
    centres = sorted((line['box'][0]+line['box'][2])/2 for line in boxed)
    gap, cut = 0., None
    for a, b in zip(centres, centres[1:]):
        if b-a > gap:
            gap, cut = b-a, (a+b)/2
    two_columns = gap > COLUMN_GAP and cut is not None

    def key(line):
        box = line['box'] or [0, 0, 0, 0]
        return (1 if two_columns and (box[0]+box[2])/2 > cut else 0, box[1] // ROW_BAND, box[0])

    return sorted(lines, key=key)


def strict_lines(text):
    """The page's lines the catalogue keeps: [{box (thousandths), ocr_type, ocr_text}] in reading order, each box once."""
    seen, out = set(), []
    for line in reorder(parse_regions(text)):
        box = line['box']
        if not box or line['inferred'] or not line['student'] or not (box[2] > box[0] and box[3] > box[1]) or tuple(box) in seen:
            continue
        seen.add(tuple(box))
        out.append({'box': box, 'ocr_type': line['ocr_type'], 'ocr_text': line['ocr_text']})
    return out


def answer_texts(files):
    """{image_id: what the student wrote on the page}: the first non-empty ``text`` among the jsonl files' {image_id, text}
    records, ``''`` for a page whose every record is empty.  The records are the student-only transcription of the annotation
    pipeline (Qwen3-VL-8B: printed question and crossed-out writing left out), which is what the page's answer length is read from."""
    out = {}
    for path in files:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                if not out.get(record['image_id']):
                    out[record['image_id']] = (record.get('text') or '').strip()
    return out


def answer_chars(text):
    """How much a student wrote, in characters: the length of the transcription once its markup and spaces are gone."""
    return len(normalise(text))
