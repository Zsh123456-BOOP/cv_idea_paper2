"""Read the lines of a ``data.line_pack`` pack with one VLM under several conditions, one image per request.

    python -m revrender.recognize_lines --pack PACK --model MODEL_DIR [--conditions NAME ...] [--backend vllm|hf] [--adapter DIR] [--tp N] --output-dir OUT

A condition is an image of the line and the words that go with it (``CONDITIONS``).  The words are one skeleton -- transcribe, LaTeX for
formulas, □ for what cannot be read, no solving or correcting -- and what a condition adds to it:

  raw_neutral      the scan, nothing more (the model reads what it sees)
  raw_final        the scan, and the instruction to give only the final valid work: what is struck out, blacked out or voided is left out
  erased_neutral   the revision erased (the masks of the pack), nothing more
  erased_final     the revision erased, and the instruction above
  blanked_neutral  the columns under each mask whited out (the revised words gone), nothing more
  blanked_final    the same, and the instruction above
  overlay_final    the revision tinted red in the one image, said to be a marker and not writing, and the instruction above
  hint_final       the scan, the places of the masks in words (thousandths of the width and height), and the instruction above
  raw_struck       the scan, and the instruction to give only what is struck out or blacked out (the annotation page's starting text)

Token conditions (``--backend hf``, Qwen3-VL; ``models.vlm_tokens``) leave the image alone and hide visual tokens from the text, each
token keeping its position:

  hide_band_neutral          the scan; the tokens with at least ``--cell-threshold`` of their area in the columns of the masks are hidden
  hide_ink_neutral           the scan; the tokens with at least ``--ink-threshold`` of their area on masked ink are hidden
  soft_band_neutral          the scan; the same tokens as hide_band, their attention logits lowered by ``--soft`` instead of removed
  blanked_hide_band_neutral  the columns whited out and the tokens over them hidden as well

Greedy decoding, seed 42; every request carries one image, so nothing in the words can be read back as text.  The backend is vLLM (the
default, for the plain conditions) or transformers (``hf``, for all of them: the plain conditions read the same to the precision of another
kernel).  OUT/predictions.jsonl has a row per (condition, line): the reading (the text inside <transcription>), the raw output, its
length and why it stopped.
"""
import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

from PIL import Image

from .data.line_pack import band_mask, ink_mask
from .util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

SYSTEM = 'You are an OCR transcription assistant.'
SKELETON = ('请转写图中的手写数学作答。保留原来的换行；公式使用 LaTeX。无法辨认的字符写 □，不猜测、不解题、不纠错、不补全。'
            '没有可转写的内容时输出 [空]。仅返回 <transcription>转写结果</transcription>，不附带解释。')
FINAL = '只转写学生最终保留的有效作答：被划掉、涂黑或作废的内容不要输出，保留替换后的内容。'
STRUCK = '只转写被划线划掉或涂黑的那部分内容，能认出多少写多少；没有被划掉的内容不要输出。公式写在 $ $ 之间。'
OVERLAY = '图中的红色是已检测出的涂改位置，是辅助标记，不是学生的笔迹。'
CONDITIONS = {
    'raw_neutral': ('raw', (), None),
    'raw_final': ('raw', (FINAL,), None),
    'erased_neutral': ('erased', (), None),
    'erased_final': ('erased', (FINAL,), None),
    'blanked_neutral': ('blanked', (), None),
    'blanked_final': ('blanked', (FINAL,), None),
    'overlay_final': ('overlay', (OVERLAY, FINAL), None),
    'hint_final': ('raw', ('hint', FINAL), None),
    'raw_struck': ('raw', (STRUCK,), None),
    'hide_band_neutral': ('raw', (), 'band'),
    'hide_ink_neutral': ('raw', (), 'ink'),
    'soft_band_neutral': ('raw', (), 'soft_band'),
    'blanked_hide_band_neutral': ('blanked', (), 'band'),
}
TOKEN_RULES = {c for c, (_, _, rule) in CONDITIONS.items() if rule}


def hint_text(boxes):
    """The places of the masks in words: boxes (left, top, right, bottom) in thousandths of the crop's width and height."""
    if not boxes:
        return '没有检测到涂改区域。'
    places = '；'.join(f'[{l}, {t}, {r}, {b}]' for l, t, r, b in boxes)
    return f'已检测出的涂改区域（相对图像宽高的千分比，[左, 上, 右, 下]）：{places}。'


def prompt(condition, row):
    """The words of a condition for a row of the pack."""
    _, parts, _ = CONDITIONS[condition]
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


def hidden_flags(reader, rule, row, image, root, args):
    """The bool flags of the image's tokens a token rule hides in a row (None when the rule hides nothing there)."""
    from .models.vlm_tokens import cell_shares, hidden_tokens
    rows, cols = reader.grid(image)
    if rule == 'ink':
        with Image.open(root/row['images']['raw']) as raw, Image.open(root/row['images']['erased']) as erased:
            mask, threshold = ink_mask(raw, erased), args.ink_threshold
    else:
        mask, threshold = band_mask(row['boxes'], row['size']), args.cell_threshold
    flags = hidden_tokens(cell_shares(mask, rows, cols), threshold)
    return flags if flags.any() else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--conditions', nargs='+', choices=tuple(CONDITIONS), default=[c for c in CONDITIONS if c not in TOKEN_RULES | {'raw_struck'}])
    p.add_argument('--backend', choices=('vllm', 'hf'), default='vllm')
    p.add_argument('--adapter', type=Path, help='hf: a LoRA adapter of train_reader, merged into the model')
    p.add_argument('--tp', type=int, default=1, help='vllm: tensor parallel size (the 32B model needs two cards)')
    p.add_argument('--gpu-memory', type=float, default=.90, help='vllm: share of each card the engine may take (lower it when others hold memory there)')
    p.add_argument('--batch-size', type=int, default=64, help='requests per progress checkpoint (hf: lines per forward pass, keep it small)')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--cell-threshold', type=float, default=.25, help='token rules: share of a token cell under the band that hides it')
    p.add_argument('--ink-threshold', type=float, default=.05, help='token rules: share of a token cell on masked ink that hides it')
    p.add_argument('--soft', type=float, default=4., help='soft_band: bias subtracted from the attention logits of the hidden tokens')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='only the first N lines (a smoke test)')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    require(len(set(args.conditions)) == len(args.conditions), 'duplicate condition')
    require(args.backend == 'hf' or not TOKEN_RULES & set(args.conditions), 'token conditions need --backend hf')
    require(args.backend == 'hf' or not args.adapter, '--adapter needs --backend hf')
    pack = json.loads((args.pack/'pack.json').read_text())
    require(pack.get('schema') == 'revrender-line-pack', 'not a line pack')
    rows = pack['rows'][:args.limit or None]
    jobs = [(c, row) for c in args.conditions for row in rows]
    out = make_output_dir(args.output_dir)
    if args.backend == 'hf':
        from .models.vlm_tokens import Reader
        reader = Reader(args.model, adapter=args.adapter)
    else:
        from vllm import LLM, SamplingParams
        model = LLM(model=str(args.model), dtype='bfloat16', seed=SEED, enforce_eager=True, max_model_len=4096, max_num_seqs=16, gpu_memory_utilization=args.gpu_memory,
                    limit_mm_per_prompt={'image': 1}, tensor_parallel_size=args.tp, trust_remote_code=args.trust_remote_code, disable_log_stats=True)
        sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=SEED)
    write_manifest(out, config=to_jsonable(args), inputs={str(args.pack/'pack.json'): digest(args.pack/'pack.json'),
                                                           **{str(w): digest(w) for w in sorted(args.model.glob('*.safetensors'))},
                                                           **({str(w): digest(w) for w in sorted(args.adapter.glob('*.safetensors'))} if args.adapter else {})},
                   backend=args.backend, rows=len(jobs), lines=len(rows),
                   conditions={c: {'image': CONDITIONS[c][0], 'token_rule': CONDITIONS[c][2], 'example_prompt': prompt(c, rows[0])} for c in args.conditions},
                   system=SYSTEM, generation={'temperature': 0, 'max_new_tokens': args.max_new_tokens, 'dtype': 'bfloat16'})
    name = args.model.name+(f'+{args.adapter.parent.name}' if args.adapter else '')      # the reading of a model with an adapter is another model's
    started, done = time.monotonic(), 0
    with (out/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start+args.batch_size]
            images = []
            for condition, row in batch:
                with Image.open(args.pack/row['images'][CONDITIONS[condition][0]]) as opened:
                    images.append(opened.convert('RGB'))
            if args.backend == 'hf':
                requests = [(image, prompt(c, row), hidden_flags(reader, CONDITIONS[c][2], row, image, args.pack, args) if CONDITIONS[c][2] else None,
                             args.soft if CONDITIONS[c][2] == 'soft_band' else 0.) for (c, row), image in zip(batch, images)]
                results = reader.read(requests, SYSTEM, args.max_new_tokens)
            else:
                generated = model.chat([messages(image, prompt(c, row)) for (c, row), image in zip(batch, images)], sampling, use_tqdm=False)
                results = [{'text': g.outputs[0].text, 'tokens': len(g.outputs[0].token_ids), 'finish': g.outputs[0].finish_reason} for g in generated]
            require(len(results) == len(batch), 'missing generation results')
            for (condition, row), result in zip(batch, results):
                stream.write(json.dumps({'condition': condition, 'id': row['id'], 'line_id': row['line_id'], 'model': name, 'text': reading_of(result['text']),
                                         'raw': result['text'], 'generated_tokens': result['tokens'], 'finish_reason': result['finish']}, ensure_ascii=False)+'\n')
            stream.flush()
            done += len(batch)
            print(json.dumps({'completed': done, 'total': len(jobs), 'seconds': round(time.monotonic()-started, 1)}), flush=True)
    write_json(out/'complete.json', {'rows': done, 'seconds': round(time.monotonic()-started, 1), 'predictions_sha256': digest(out/'predictions.jsonl')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
