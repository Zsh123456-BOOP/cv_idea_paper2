"""Read the existing page cuts with one VLM in three matched conditions: B, C and D.

Each call sees only an existing detector-positive crop. D additionally sees the same
crop with a mask highlight. Verbatim and deletion-tagging tasks are kept separate.
"""
import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

from PIL import Image

from .util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

SYSTEM = 'You are an OCR transcription assistant.'
INSTRUCTIONS = {
    'verbatim': '请转写图中的所有手写字符和公式。无论是否有划线、涂黑，只要有墨迹就请如实还原，不要主观过滤。',
    'deletion': '请完整识别图像中的全部内容。若内容属于正常作答，直接输出；若文字被划除或涂抹覆盖，请用 <del>被划掉的内容</del> 标签包裹。',
}
OUTPUT_FORMAT = ('保留原来的换行；公式使用 LaTeX。无法辨认的字符写 □，不猜测、不解题、不纠错、不补全。'
                 '没有可转写的内容时输出 [空]。仅返回 <transcription>转写结果</transcription>，不附带解释。')
MASK_INSTRUCTION = ('第二张是第一张同一切块的辅助图，叠加了第一篇模型检测出的候选涂改掩膜。'
                    '红色是辅助标记，不是新增的原始笔迹。')


def messages(mode, task):
    content = [{'type': 'text', 'text': INSTRUCTIONS[task]+'\n'+OUTPUT_FORMAT}, {'type': 'image'}]
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


def input_images(pack_dir, row, mode):
    kind = 'erased' if mode == 'C' else 'raw'
    images = [opened(str(pack_dir/row['images'][kind]))]
    if mode == 'D':
        images.append(opened(str(pack_dir/row['images']['hint'])))
    return images


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--mode', choices=('B', 'C', 'D'), required=True)
    p.add_argument('--tests', nargs='+', choices=tuple(INSTRUCTIONS), default=list(INSTRUCTIONS))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16, help='requests per progress checkpoint; vLLM schedules four at a time')
    p.add_argument('--max-new-tokens', type=int, default=384)
    p.add_argument('--max-pixels', type=int, default=1003520)
    p.add_argument('--limit', type=int, default=0, help='small explicit smoke only; zero means every row')
    args = p.parse_args(argv)
    require(args.batch_size > 0, 'batch size must be positive')
    pack = json.loads((args.pack/'pack.json').read_text())
    require(all(x['split'] == 'dev' for x in pack['pages']), 'this pilot accepts development pages only')
    require(pack.get('selection', {}).get('kind') == 'revision-only', 'use the frozen revision-only subset')
    cuts = [(page, row) for page in pack['pages'] for row in page['rows']]
    require(all(row['masked_pixels'] > 0 for _, row in cuts), 'zero-mask cuts are excluded')
    require(len(set(args.tests)) == len(args.tests), 'duplicate test')
    if args.limit:
        cuts = cuts[:args.limit]
    jobs = [(task, page, row) for task in args.tests for page, row in cuts]
    assets = json.loads((args.pack/'run_manifest.json').read_text())['asset_sha256']
    image_hashes = {str(args.pack/relative): digest(args.pack/relative)
                    for _, row in cuts for relative in row['images'].values()}
    require(all(assets[str(Path(path).relative_to(args.pack))] == value for path, value in image_hashes.items()),
            'input crop hash mismatch')
    out = make_output_dir(args.output_dir)
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = LLM(model=str(args.model), dtype='bfloat16', seed=SEED, enforce_eager=True,
                max_model_len=8192, max_num_seqs=4, max_num_batched_tokens=8192,
                gpu_memory_utilization=.90, limit_mm_per_prompt={'image': 2},
                mm_processor_kwargs={'max_pixels': args.max_pixels}, disable_log_stats=True)
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=SEED, stop=['</transcription>'])
    prompts = {task: processor.apply_chat_template(messages(args.mode, task), tokenize=False, add_generation_prompt=True)
               + '<transcription>\n' for task in args.tests}
    weights = {str(path): digest(path) for path in sorted(args.model.glob('*.safetensors'))}
    write_manifest(out, config=to_jsonable(args), inputs={str(args.pack/'pack.json'): digest(args.pack/'pack.json'), **weights, **image_hashes},
                   backend='vllm', rows=len(jobs), cuts=len(cuts), mode=args.mode, protocol='revision-only-two-tasks',
                   messages={task: messages(args.mode, task) for task in args.tests}, rendered_prompts=prompts,
                   generation={'temperature': 0, 'max_new_tokens': args.max_new_tokens, 'stop': '</transcription>',
                               'max_model_len': 8192, 'max_num_seqs': 4, 'dtype': 'bfloat16'},
                   context='existing crop only; no full page; D adds the same crop with a mask highlight')
    started = time.monotonic()
    completed = 0
    with (out/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start+args.batch_size]
            requests = [{'prompt': prompts[task], 'multi_modal_data': {'image': input_images(args.pack, row, args.mode)}}
                        for task, page, row in batch]
            generated = model.generate(requests, sampling, use_tqdm=False)
            require(len(generated) == len(batch), 'missing generation results')
            for (task, page, row), request in zip(batch, generated):
                output = request.outputs[0]
                raw = output.text
                item = {'test': task, 'image_id': page['image_id'], 'row_id': row['id'], 'source_index': row['source_index'],
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
