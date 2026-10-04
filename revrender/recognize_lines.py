"""Read the lines of a ``data.line_pack`` pack with one VLM under several conditions, one image per request.

    python -m revrender.recognize_lines --pack PACK --model MODEL_DIR [--conditions NAME ...] [--tp N] --output-dir OUT

A condition is an image of the line and the words that go with it (``CONDITIONS``).  The words are one skeleton -- transcribe, LaTeX for
formulas, □ for what cannot be read, no solving or correcting -- and what a condition adds to it:

  raw_neutral      the scan, nothing more (the model reads what it sees)
  raw_final        the scan, and the instruction to give only the final valid work: what is struck out, blacked out or voided is left out
  erased_neutral   the revision erased (the masks of the pack), nothing more
  erased_final     the revision erased, and the instruction above
  overlay_final    the revision tinted red in the one image, said to be a marker and not writing, and the instruction above
  hint_final       the scan, the places of the masks in words (thousandths of the width and height), and the instruction above

Greedy decoding, seed 42, vLLM; every request carries one image, so nothing in the words can be read back as text.  OUT/predictions.jsonl
has a row per (condition, line): the reading (the text inside <transcription>), the raw output, its length and why it stopped.
"""
import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

from PIL import Image

from .util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

SYSTEM = 'You are an OCR transcription assistant.'
SKELETON = ('请转写图中的手写数学作答。保留原来的换行；公式使用 LaTeX。无法辨认的字符写 □，不猜测、不解题、不纠错、不补全。'
            '没有可转写的内容时输出 [空]。仅返回 <transcription>转写结果</transcription>，不附带解释。')
FINAL = '只转写学生最终保留的有效作答：被划掉、涂黑或作废的内容不要输出，保留替换后的内容。'
OVERLAY = '图中的红色是已检测出的涂改位置，是辅助标记，不是学生的笔迹。'
CONDITIONS = {
    'raw_neutral': ('raw', ()),
    'raw_final': ('raw', (FINAL,)),
    'erased_neutral': ('erased', ()),
    'erased_final': ('erased', (FINAL,)),
    'overlay_final': ('overlay', (OVERLAY, FINAL)),
    'hint_final': ('raw', ('hint', FINAL)),
}


def hint_text(boxes):
    """The places of the masks in words: boxes (left, top, right, bottom) in thousandths of the crop's width and height."""
    if not boxes:
        return '没有检测到涂改区域。'
    places = '；'.join(f'[{l}, {t}, {r}, {b}]' for l, t, r, b in boxes)
    return f'已检测出的涂改区域（相对图像宽高的千分比，[左, 上, 右, 下]）：{places}。'


def prompt(condition, row):
    """The words of a condition for a row of the pack."""
    _, parts = CONDITIONS[condition]
    return '\n'.join([SKELETON, *(hint_text(row['boxes']) if part == 'hint' else part for part in parts)])


def messages(image, text):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    url = 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}}, {'type': 'text', 'text': text}]}]


def reading_of(raw):
    """The text inside <transcription> (all of the output when the tags are missing); [空] is no text."""
    found = re.search(r'<transcription>(.*?)(?:</transcription>|$)', raw, re.S)
    text = (found.group(1) if found else raw).strip()
    return '' if text in ('[空]', '空', '【空】') else text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--conditions', nargs='+', choices=tuple(CONDITIONS), default=list(CONDITIONS))
    p.add_argument('--tp', type=int, default=1, help='tensor parallel size (the 32B model needs two cards)')
    p.add_argument('--batch-size', type=int, default=64, help='requests per progress checkpoint')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='only the first N lines (a smoke test)')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    require(len(set(args.conditions)) == len(args.conditions), 'duplicate condition')
    pack = json.loads((args.pack/'pack.json').read_text())
    require(pack.get('schema') == 'revrender-line-pack', 'not a line pack')
    rows = pack['rows'][:args.limit or None]
    jobs = [(c, row) for c in args.conditions for row in rows]
    out = make_output_dir(args.output_dir)
    from vllm import LLM, SamplingParams
    model = LLM(model=str(args.model), dtype='bfloat16', seed=SEED, enforce_eager=True, max_model_len=4096, max_num_seqs=16, gpu_memory_utilization=.90,
                limit_mm_per_prompt={'image': 1}, tensor_parallel_size=args.tp, trust_remote_code=args.trust_remote_code, disable_log_stats=True)
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=SEED)
    write_manifest(out, config=to_jsonable(args), inputs={str(args.pack/'pack.json'): digest(args.pack/'pack.json'),
                                                           **{str(w): digest(w) for w in sorted(args.model.glob('*.safetensors'))}},
                   backend='vllm', rows=len(jobs), lines=len(rows), conditions={c: {'image': CONDITIONS[c][0], 'example_prompt': prompt(c, rows[0])} for c in args.conditions},
                   system=SYSTEM, generation={'temperature': 0, 'max_new_tokens': args.max_new_tokens, 'dtype': 'bfloat16'})
    started, done = time.monotonic(), 0
    with (out/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start+args.batch_size]
            requests = []
            for condition, row in batch:
                with Image.open(args.pack/row['images'][CONDITIONS[condition][0]]) as opened:
                    requests.append(messages(opened.convert('RGB'), prompt(condition, row)))
            generated = model.chat(requests, sampling, use_tqdm=False)
            require(len(generated) == len(batch), 'missing generation results')
            for (condition, row), result in zip(batch, generated):
                output = result.outputs[0]
                stream.write(json.dumps({'condition': condition, 'id': row['id'], 'line_id': row['line_id'], 'model': args.model.name, 'text': reading_of(output.text),
                                         'raw': output.text, 'generated_tokens': len(output.token_ids), 'finish_reason': output.finish_reason}, ensure_ascii=False)+'\n')
            stream.flush()
            done += len(batch)
            print(json.dumps({'completed': done, 'total': len(jobs), 'seconds': round(time.monotonic()-started, 1)}), flush=True)
    write_json(out/'complete.json', {'rows': done, 'seconds': round(time.monotonic()-started, 1), 'predictions_sha256': digest(out/'predictions.jsonl')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
