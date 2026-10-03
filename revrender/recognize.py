"""Read the existing page cuts with one VLM in three matched conditions: B, C and D.

Each call sees a full-question context with the target cut outlined in blue, and the
original-size target crop. D additionally sees an auxiliary mask-highlighted crop.
The output is one transcription per original cut, not a re-segmentation of the page.
"""
import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

from .util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

SYSTEM = ('You faithfully transcribe Chinese middle-school mathematics handwriting. '
          'Do not solve, correct, explain, or complete the student\'s work.')
INSTRUCTION = '''第一张图是整道题的上下文，蓝框标出目标切块；第二张图是这个切块的放大图。
只转写第二张图中的、学生最终保留的有效手写作答。上下文只用于判断修改和辨认，不输出其他切块的内容。
明确划掉、涂黑或作废的内容不输出，保留替换后的作答。局部划除不等于整行作废。
学生保留的数学错误必须照录，不计算、不纠正、不补全。印刷题干、题号、边框、图形线条不转写。
保留原来的换行；公式使用 LaTeX。无法辨认的有效字符写 □，不得猜测。
如果本切块没有有效手写文本，只输出 [空]。
仅返回 <transcription>转写文本</transcription>，不附带分析或说明。'''
MASK_INSTRUCTION = '''第三张图是第二张图叠加红色候选涂改掩膜的辅助图。
红色用于提示可能作废的位置，不是原始笔迹，也不代表整行都作废。结合前两张原图判断，仅忽略确认作废的内容，保护仍有效的邻近文字。'''


def messages(mode):
    content = [{'type': 'text', 'text': INSTRUCTION}, {'type': 'image'}, {'type': 'image'}]
    if mode == 'D':
        content += [{'type': 'text', 'text': MASK_INSTRUCTION}, {'type': 'image'}]
    return [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': content}]


def clean_text(text):
    text = text.split('</transcription>', 1)[0].replace('<transcription>', '').strip()
    return '' if text in ('[空]', '空', '【空】') else text


@lru_cache(maxsize=6)
def opened(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def input_images(pack_dir, page, row, mode):
    kind = 'erased' if mode == 'C' else 'raw'
    context = opened(str(pack_dir/page['images'][kind])).copy()
    draw = ImageDraw.Draw(context)
    l, t, r, b = row['box']
    draw.rectangle((l, t, r-1, b-1), outline=(30, 110, 220), width=2)
    context.thumbnail((1024, 1408), Image.Resampling.LANCZOS)
    images = [context, opened(str(pack_dir/row['images'][kind]))]
    if mode == 'D':
        images.append(opened(str(pack_dir/row['images']['hint'])))
    return images


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--mode', choices=('B', 'C', 'D'), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16, help='requests per progress checkpoint; vLLM schedules four at a time')
    p.add_argument('--max-new-tokens', type=int, default=384)
    p.add_argument('--max-pixels', type=int, default=1003520)
    p.add_argument('--limit', type=int, default=0, help='small explicit smoke only; zero means every row')
    args = p.parse_args(argv)
    require(args.batch_size > 0, 'batch size must be positive')
    pack = json.loads((args.pack/'pack.json').read_text())
    require(all(x['split'] == 'dev' for x in pack['pages']), 'this pilot accepts development pages only')
    jobs = [(page, row) for page in pack['pages'] for row in page['rows']]
    if args.limit:
        jobs = jobs[:args.limit]
    out = make_output_dir(args.output_dir)
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = LLM(model=str(args.model), dtype='bfloat16', seed=SEED, enforce_eager=True,
                max_model_len=8192, max_num_seqs=4, max_num_batched_tokens=8192,
                gpu_memory_utilization=.90, limit_mm_per_prompt={'image': 3},
                mm_processor_kwargs={'max_pixels': args.max_pixels}, disable_log_stats=True)
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=SEED, stop=['</transcription>'])
    prompt = processor.apply_chat_template(messages(args.mode), tokenize=False, add_generation_prompt=True) + '<transcription>\n'
    weights = {str(path): digest(path) for path in sorted(args.model.glob('*.safetensors'))}
    write_manifest(out, config=to_jsonable(args), inputs={str(args.pack/'pack.json'): digest(args.pack/'pack.json'), **weights},
                   backend='vllm', rows=len(jobs), mode=args.mode, system=SYSTEM,
                   instruction=INSTRUCTION, mask_instruction=MASK_INSTRUCTION if args.mode == 'D' else None,
                   generation={'temperature': 0, 'max_new_tokens': args.max_new_tokens, 'stop': '</transcription>',
                               'max_model_len': 8192, 'max_num_seqs': 4, 'dtype': 'bfloat16'},
                   context='full question plus original existing cut; D adds auxiliary marked cut')
    started = time.monotonic()
    completed = 0
    with (out/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start+args.batch_size]
            requests = [{'prompt': prompt, 'multi_modal_data': {'image': input_images(args.pack, page, row, args.mode)}}
                        for page, row in batch]
            generated = model.generate(requests, sampling, use_tqdm=False)
            for (page, row), request in zip(batch, generated):
                output = request.outputs[0]
                raw = output.text
                item = {'image_id': page['image_id'], 'row_id': row['id'], 'source_index': row['source_index'],
                        'mode': args.mode, 'text': clean_text(raw), 'raw': raw, 'generated_tokens': len(output.token_ids),
                        'truncated': output.finish_reason == 'length', 'finish_reason': output.finish_reason}
                stream.write(json.dumps(item, ensure_ascii=False)+'\n')
            stream.flush()
            completed += len(batch)
            progress = {'mode': args.mode, 'completed': completed, 'total': len(jobs),
                        'seconds': round(time.monotonic()-started, 1)}
            (out/'progress.json').write_text(json.dumps(progress)+'\n')
            print(json.dumps(progress), flush=True)
    write_json(out/'complete.json', {'rows': completed, 'seconds': round(time.monotonic()-started, 1),
                                     'predictions_sha256': digest(out/'predictions.jsonl')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
